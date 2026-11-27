use crossbeam_channel::{unbounded, Receiver};
use egui::{Color32, RichText, Rounding, Stroke};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use sysinfo::{Pid, System};

use crate::audio::analyzer::{analyze_audio, AudioAnalysis, IssueSeverity};
use crate::audio::io::{decode_audio_file, export_audio_file, StereoAudioBuffer};
use crate::audio::player::{format_duration, AbPlayer, ActiveTrack};
use crate::config::{EnhancementPreset, OutputFormat, RestorationConfig, SpatialMode};
use crate::ml::{
    download_model, DownloadProgress, ModelInventory, ModelRegistry, ModelStage, PipelineProgress,
    PipelineResult, PipelineStage, PluggableModelEntry, RestorationPipeline,
};
use crate::ui::visuals::{
    apply_nexgen_theme, draw_spectrogram, draw_waveform, BORDER_COLOR, CARD_BG, ERROR_RED,
    NEON_CYAN, NEON_PURPLE, SUCCESS_GREEN, TEXT_MUTED, WARN_AMBER,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppTab {
    Studio,
    BatchQueue,
    SettingsModels,
    About,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MixerSubTab {
    Eq5Band,
    Stem4Mixer,
}

#[derive(Debug, Clone)]
enum BatchStatus {
    Queued,
    Processing,
    Done(String),
    Failed(String),
}

#[derive(Debug, Clone)]
struct BatchItem {
    path: PathBuf,
    status: BatchStatus,
}

pub struct NexGenDesktopApp {
    active_tab: AppTab,
    mixer_sub_tab: MixerSubTab,
    config: RestorationConfig,
    inventory: ModelInventory,
    registry: ModelRegistry,
    installed_model_files: Vec<String>,
    sys: System,

    // Loaded & restored audio state
    input_path: Option<PathBuf>,
    original_audio: Option<StereoAudioBuffer>,
    original_analysis: Option<AudioAnalysis>,
    restored_result: Option<PipelineResult>,

    // Playback & A-B Loop
    player: Option<AbPlayer>,
    view_restored_visuals: bool,
    loop_a: Option<f32>,
    loop_b: Option<f32>,
    loop_enabled: bool,
    is_preview_mode: bool,
    settings_path: PathBuf,

    // Single-file worker state
    is_processing: bool,
    current_stage: PipelineStage,
    stage_progress: f32,
    overall_progress: f32,
    backend_note: String,
    cancel_flag: Arc<AtomicBool>,
    rx: Option<Receiver<PipelineProgress>>,

    // Batch Queue state
    batch_items: Vec<BatchItem>,
    batch_running: bool,
    batch_rx: Option<Receiver<(usize, BatchStatus)>>,

    // Model Downloader & Custom Registry Form
    download_rx: Option<Receiver<DownloadProgress>>,
    download_progress: HashMap<String, (u64, Option<u64>)>,
    active_download_id: Option<String>,
    custom_name: String,
    custom_filename: String,
    custom_url: String,
    custom_stage: ModelStage,

    // Status banner
    status_message: Option<(String, bool)>,
}

impl NexGenDesktopApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_nexgen_theme(&cc.egui_ctx);

        let models_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("models")))
            .filter(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from("models"));

        let settings_path = models_dir.join("settings.json");
        let config = RestorationConfig::load_or_default(&settings_path);
        let inventory = ModelInventory::scan(&models_dir);
        let registry = ModelRegistry::load_or_create(&models_dir);
        let installed_model_files = ModelRegistry::list_installed_files(&models_dir);
        let player = AbPlayer::try_new().ok();

        Self {
            active_tab: AppTab::Studio,
            mixer_sub_tab: MixerSubTab::Eq5Band,
            config,
            inventory,
            registry,
            installed_model_files,
            sys: System::new_all(),
            input_path: None,
            original_audio: None,
            original_analysis: None,
            restored_result: None,
            player,
            view_restored_visuals: true,
            loop_a: None,
            loop_b: None,
            loop_enabled: false,
            is_preview_mode: false,
            settings_path,
            is_processing: false,
            current_stage: PipelineStage::Preparing,
            stage_progress: 0.0,
            overall_progress: 0.0,
            backend_note: String::new(),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            rx: None,
            batch_items: Vec::new(),
            batch_running: false,
            batch_rx: None,
            download_rx: None,
            download_progress: HashMap::new(),
            active_download_id: None,
            custom_name: String::new(),
            custom_filename: String::new(),
            custom_url: String::new(),
            custom_stage: ModelStage::DenoiseStem,
            status_message: None,
        }
    }

    fn refresh_models(&mut self) {
        let dir = self.inventory.models_dir.clone();
        self.inventory = ModelInventory::scan(&dir);
        self.installed_model_files = ModelRegistry::list_installed_files(&dir);
    }

    fn load_file(&mut self, path: PathBuf) {
        if let Some(p) = &mut self.player {
            p.stop();
        }
        match decode_audio_file(&path) {
            Ok(buf) => {
                let analysis = analyze_audio(&buf);
                self.input_path = Some(path.clone());
                self.original_audio = Some(buf);
                self.original_analysis = Some(analysis);
                self.restored_result = None;
                self.view_restored_visuals = false;
                self.active_tab = AppTab::Studio;
                self.status_message = Some((
                    format!(
                        "Loaded {}",
                        path.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("audio file")
                    ),
                    false,
                ));
            }
            Err(e) => {
                self.status_message = Some((format!("Failed to load audio: {e}"), true));
            }
        }
    }

    fn start_restoration(&mut self) {
        let Some(input) = self.original_audio.clone() else {
            return;
        };
        if self.is_processing {
            return;
        }
        if let Some(p) = &mut self.player {
            p.stop();
        }

        self.is_preview_mode = false;
        self.is_processing = true;
        self.overall_progress = 0.0;
        self.stage_progress = 0.0;
        self.current_stage = PipelineStage::Preparing;
        self.backend_note = "Starting restoration pipeline...".to_string();
        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.config.save(&self.settings_path);

        let (tx, rx) = unbounded();
        self.rx = Some(rx);

        let config = self.config.clone();
        let inventory = self.inventory.clone();
        let cancel = self.cancel_flag.clone();

        thread::spawn(move || {
            RestorationPipeline::run(input, config, inventory, cancel, tx);
        });
    }

    fn start_preview_10s(&mut self) {
        let Some(full) = &self.original_audio else {
            return;
        };
        if self.is_processing {
            return;
        }
        let cur_secs = self
            .player
            .as_mut()
            .map(|p| p.position_secs())
            .unwrap_or(0.0);
        if let Some(p) = &mut self.player {
            p.stop();
        }

        let sr = full.sample_rate as usize;
        let start_sample = ((cur_secs * sr as f32) as usize).min(full.len().saturating_sub(sr));
        let end_sample = (start_sample + sr * 10).min(full.len());
        let slice = StereoAudioBuffer {
            left: full.left[start_sample..end_sample].to_vec(),
            right: full.right[start_sample..end_sample].to_vec(),
            sample_rate: full.sample_rate,
            original_channels: full.original_channels,
            original_sample_rate: full.original_sample_rate,
        };

        self.is_preview_mode = true;
        self.is_processing = true;
        self.overall_progress = 0.0;
        self.stage_progress = 0.0;
        self.current_stage = PipelineStage::Preparing;
        self.backend_note = "Rendering 10-second real-time preview...".to_string();
        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.config.save(&self.settings_path);

        let (tx, rx) = unbounded();
        self.rx = Some(rx);
        let config = self.config.clone();
        let inventory = self.inventory.clone();
        let cancel = self.cancel_flag.clone();

        thread::spawn(move || {
            RestorationPipeline::run(slice, config, inventory, cancel, tx);
        });
    }

    fn start_batch_processing(&mut self, output_dir: PathBuf) {
        if self.batch_running || self.batch_items.is_empty() {
            return;
        }
        self.batch_running = true;
        let items: Vec<PathBuf> = self.batch_items.iter().map(|b| b.path.clone()).collect();
        let config = self.config.clone();
        let inventory = self.inventory.clone();
        let (tx, rx) = unbounded();
        self.batch_rx = Some(rx);

        thread::spawn(move || {
            for (idx, path) in items.into_iter().enumerate() {
                let _ = tx.send((idx, BatchStatus::Processing));
                let res = (|| -> anyhow::Result<PathBuf> {
                    let buf = decode_audio_file(&path)?;
                    let (ptx, prx) = unbounded();
                    let cancel = Arc::new(AtomicBool::new(false));
                    RestorationPipeline::run(buf, config.clone(), inventory.clone(), cancel, ptx);

                    let mut final_res = None;
                    while let Ok(msg) = prx.recv() {
                        match msg {
                            PipelineProgress::Completed(r) => {
                                final_res = Some(r);
                                break;
                            }
                            PipelineProgress::Failed(e) => anyhow::bail!(e),
                            PipelineProgress::Cancelled => anyhow::bail!("Cancelled"),
                            _ => {}
                        }
                    }
                    let r = final_res.ok_or_else(|| anyhow::anyhow!("No pipeline output"))?;
                    let stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("restored");
                    let out_path =
                        output_dir.join(format!("{}_restored.{}", stem, config.output_format.extension()));
                    export_audio_file(
                        &r.restored_audio,
                        &out_path,
                        config.output_format,
                        Some(&path),
                        config.preserve_metadata,
                    )?;
                    Ok(out_path)
                })();

                match res {
                    Ok(out) => {
                        let _ = tx.send((
                            idx,
                            BatchStatus::Done(
                                out.file_name()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("done")
                                    .to_string(),
                            ),
                        ));
                    }
                    Err(e) => {
                        let _ = tx.send((idx, BatchStatus::Failed(format!("{e}"))));
                    }
                }
            }
            let _ = tx.send((usize::MAX, BatchStatus::Done("ALL_DONE".to_string())));
        });
    }

    fn start_model_download(&mut self, entry: PluggableModelEntry) {
        if self.active_download_id.is_some() {
            return;
        }
        self.active_download_id = Some(entry.id.clone());
        self.download_progress
            .insert(entry.id.clone(), (0, None));
        let (tx, rx) = unbounded();
        self.download_rx = Some(rx);
        let dir = self.inventory.models_dir.clone();
        thread::spawn(move || {
            download_model(entry, dir, tx);
        });
    }

    fn poll_workers(&mut self, ctx: &egui::Context) {
        // 1. Single-file pipeline
        let mut finished = false;
        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    PipelineProgress::Update {
                        stage,
                        stage_progress,
                        overall_progress,
                        backend_note,
                    } => {
                        self.current_stage = stage;
                        self.stage_progress = stage_progress;
                        self.overall_progress = overall_progress;
                        self.backend_note = backend_note;
                    }
                    PipelineProgress::Completed(res) => {
                        let label = if self.is_preview_mode {
                            "10s Preview ready"
                        } else {
                            "Restoration complete"
                        };
                        self.status_message = Some((
                            format!(
                                "{} in {:.1}s ({})",
                                label,
                                res.elapsed_secs,
                                res.used_onnx_models.join(" -> ")
                            ),
                            false,
                        ));
                        let auto_play_preview = self.is_preview_mode;
                        self.restored_result = Some(res);
                        self.view_restored_visuals = true;
                        if auto_play_preview {
                            let restored_ref = self.restored_result.as_ref().map(|r| &r.restored_audio);
                            if let Some(p) = &mut self.player {
                                let _ = p.play_from(
                                    ActiveTrack::Restored,
                                    self.original_audio.as_ref(),
                                    restored_ref,
                                    0.0,
                                );
                            }
                        }
                        finished = true;
                    }
                    PipelineProgress::Cancelled => {
                        self.status_message = Some(("Restoration cancelled.".to_string(), true));
                        finished = true;
                    }
                    PipelineProgress::Failed(err) => {
                        self.status_message = Some((format!("Error: {err}"), true));
                        finished = true;
                    }
                }
            }
        }
        if finished {
            self.is_processing = false;
            self.rx = None;
        }

        // 2. Batch queue
        let mut batch_done = false;
        if let Some(rx) = &self.batch_rx {
            while let Ok((idx, status)) = rx.try_recv() {
                if idx == usize::MAX {
                    batch_done = true;
                } else if let Some(item) = self.batch_items.get_mut(idx) {
                    item.status = status;
                }
            }
        }
        if batch_done {
            self.batch_running = false;
            self.batch_rx = None;
            self.status_message = Some(("Batch restoration queue completed!".to_string(), false));
        }

        // 3. Model downloader
        let mut dl_done = false;
        if let Some(rx) = &self.download_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    DownloadProgress::Progress {
                        model_id,
                        downloaded_bytes,
                        total_bytes,
                    } => {
                        self.download_progress
                            .insert(model_id, (downloaded_bytes, total_bytes));
                    }
                    DownloadProgress::Completed { model_id, path } => {
                        self.status_message = Some((
                            format!("Downloaded {} to {}", model_id, path.display()),
                            false,
                        ));
                        dl_done = true;
                    }
                    DownloadProgress::Failed { model_id, error } => {
                        self.status_message =
                            Some((format!("Download failed ({}): {}", model_id, error), true));
                        dl_done = true;
                    }
                }
            }
        }
        if dl_done {
            self.active_download_id = None;
            self.download_rx = None;
            self.refresh_models();
        }

        if self.is_processing || self.batch_running || self.active_download_id.is_some() {
            ctx.request_repaint();
        }
    }

    fn export_restored(&mut self) {
        let Some(res) = &self.restored_result else {
            return;
        };
        let ext = self.config.output_format.extension();
        let default_name = self
            .input_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .map(|s| format!("{}_restored.{}", s, ext))
            .unwrap_or_else(|| format!("restored_stereo.{}", ext));

        let filter_name = if self.config.output_format.is_mp3() {
            "MP3 Audio"
        } else if ext == "flac" {
            "FLAC Lossless Audio"
        } else if ext == "ac3" {
            "AC-3 Dolby Digital Audio"
        } else {
            "WAV Audio"
        };

        if let Some(save_path) = rfd::FileDialog::new()
            .add_filter(filter_name, &[ext])
            .set_file_name(&default_name)
            .save_file()
        {
            match export_audio_file(
                &res.restored_audio,
                &save_path,
                self.config.output_format,
                self.input_path.as_deref(),
                self.config.preserve_metadata,
            ) {
                Ok(_) => {
                    self.status_message =
                        Some((format!("Exported to {}", save_path.display()), false));
                }
                Err(e) => {
                    self.status_message = Some((format!("Export failed: {e}"), true));
                }
            }
        }
    }

    fn current_ram_mb(&mut self) -> u64 {
        let pid = Pid::from_u32(std::process::id());
        self.sys
            .refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        self.sys
            .process(pid)
            .map(|p| p.memory() / (1024 * 1024))
            .unwrap_or(0)
    }

    fn render_batch_tab(&mut self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new("Batch Folder / Multi-File Queue")
                .size(18.0)
                .strong()
                .color(NEON_CYAN),
        );
        ui.label(
            RichText::new(
                "Add multiple retro MP3/WAV/FLAC files or an entire folder, then restore & export them all using your active preset and export format.",
            )
            .color(TEXT_MUTED),
        );
        ui.add_space(10.0);

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.batch_running, egui::Button::new("Add Files..."))
                .clicked()
            {
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter("Audio Files", &["mp3", "wav", "flac", "ogg", "m4a", "aac"])
                    .pick_files()
                {
                    for p in paths {
                        self.batch_items.push(BatchItem {
                            path: p,
                            status: BatchStatus::Queued,
                        });
                    }
                }
            }

            if ui
                .add_enabled(!self.batch_running, egui::Button::new("Add Folder..."))
                .clicked()
            {
                if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                    if let Ok(entries) = std::fs::read_dir(folder) {
                        for e in entries.flatten() {
                            let p = e.path();
                            if p.is_file() {
                                if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                                    let low = ext.to_lowercase();
                                    if matches!(
                                        low.as_str(),
                                        "mp3" | "wav" | "flac" | "ogg" | "m4a" | "aac"
                                    ) {
                                        self.batch_items.push(BatchItem {
                                            path: p,
                                            status: BatchStatus::Queued,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ui
                .add_enabled(!self.batch_running, egui::Button::new("Clear Queue"))
                .clicked()
            {
                self.batch_items.clear();
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let run_btn = egui::Button::new(
                    RichText::new("Process All & Export to Folder...")
                        .strong()
                        .color(Color32::WHITE),
                )
                .fill(NEON_PURPLE)
                .rounding(Rounding::same(8.0));
                if ui
                    .add_enabled(!self.batch_running && !self.batch_items.is_empty(), run_btn)
                    .clicked()
                {
                    if let Some(out_dir) = rfd::FileDialog::new().pick_folder() {
                        self.start_batch_processing(out_dir);
                    }
                }
            });
        });

        ui.add_space(10.0);
        egui::ScrollArea::vertical().show(ui, |ui| {
            for item in &self.batch_items {
                egui::Frame::none()
                    .fill(CARD_BG)
                    .stroke(Stroke::new(1.0_f32, BORDER_COLOR))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                item.path
                                    .file_name()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("file"),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| match &item.status {
                                    BatchStatus::Queued => {
                                        ui.label(RichText::new("Queued").color(TEXT_MUTED));
                                    }
                                    BatchStatus::Processing => {
                                        ui.label(
                                            RichText::new("Processing...").color(NEON_CYAN).strong(),
                                        );
                                    }
                                    BatchStatus::Done(out) => {
                                        ui.label(
                                            RichText::new(format!("Done -> {}", out))
                                                .color(SUCCESS_GREEN),
                                        );
                                    }
                                    BatchStatus::Failed(e) => {
                                        ui.label(
                                            RichText::new(format!("Failed: {}", e)).color(ERROR_RED),
                                        );
                                    }
                                },
                            );
                        });
                    });
                ui.add_space(4.0);
            }
        });
    }

    fn render_settings_models_tab(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            if let Some((msg, is_err)) = &self.status_message {
                let color = if *is_err { ERROR_RED } else { SUCCESS_GREEN };
                egui::Frame::none()
                    .fill(color.linear_multiply(0.15))
                    .stroke(Stroke::new(1.0_f32, color))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new(msg).color(color));
                    });
                ui.add_space(8.0);
            }

            if let Some(dl_id) = &self.active_download_id {
                let (dl, tot) = self
                    .download_progress
                    .get(dl_id)
                    .copied()
                    .unwrap_or((0, None));
                let approx_bytes = self
                    .registry
                    .models
                    .iter()
                    .find(|m| &m.id == dl_id)
                    .map(|m| m.approx_size_mb as u64 * 1024 * 1024)
                    .unwrap_or(100 * 1024 * 1024);
                let denom = tot.unwrap_or(approx_bytes).max(1) as f32;
                let frac = (dl as f32 / denom).clamp(0.0, 1.0);
                let dl_mb = dl as f32 / (1024.0 * 1024.0);
                let tot_mb = denom / (1024.0 * 1024.0);

                egui::Frame::none()
                    .fill(CARD_BG)
                    .stroke(Stroke::new(1.5_f32, NEON_CYAN))
                    .rounding(Rounding::same(10.0))
                    .inner_margin(10.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("Downloading model '{}'...", dl_id))
                                    .strong()
                                    .color(NEON_CYAN),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        RichText::new(format!(
                                            "{:.1} MB / {:.1} MB ({:.0}%)",
                                            dl_mb,
                                            tot_mb,
                                            frac * 100.0
                                        ))
                                        .color(Color32::WHITE),
                                    );
                                },
                            );
                        });
                        ui.add_space(6.0);
                        ui.add(egui::ProgressBar::new(frac).animate(true));
                    });
                ui.add_space(10.0);
            }

            ui.label(
                RichText::new("Settings & Pluggable AI Models")
                    .size(18.0)
                    .strong()
                    .color(NEON_CYAN),
            );
            ui.label(
                RichText::new(format!(
                    "Models directory: {}  (Drop any future .onnx or .bin file here to plug it in)",
                    self.inventory.models_dir.display()
                ))
                .size(12.0)
                .color(TEXT_MUTED),
            );
            ui.add_space(10.0);

            // ---- Active Model Slot Selection ----
            egui::Frame::none()
                .fill(CARD_BG)
                .stroke(Stroke::new(1.0_f32, BORDER_COLOR))
                .rounding(Rounding::same(12.0))
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new("ACTIVE PIPELINE MODEL SLOTS (PLUGGABLE)")
                            .size(11.0)
                            .color(TEXT_MUTED),
                    );
                    ui.add_space(6.0);

                    let files = self.installed_model_files.clone();
                    let render_slot =
                        |ui: &mut egui::Ui, label: &str, selected: &mut Option<String>| {
                            let current = selected
                                .clone()
                                .unwrap_or_else(|| "Auto (Best Installed / DSP Fallback)".to_string());
                            egui::ComboBox::from_label(label)
                                .selected_text(current)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        selected,
                                        None,
                                        "Auto (Best Installed / DSP Fallback)",
                                    );
                                    for f in &files {
                                        ui.selectable_value(selected, Some(f.clone()), f);
                                    }
                                });
                        };

                    render_slot(
                        ui,
                        "Stage 1 (Denoise / Stems)",
                        &mut self.config.selected_denoise_model,
                    );
                    render_slot(
                        ui,
                        "Stage 2 (Bandwidth / Restore)",
                        &mut self.config.selected_recon_model,
                    );

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Refresh Models Folder").clicked() {
                            self.refresh_models();
                        }
                        if ui.button("Import Local .onnx / .bin File...").clicked() {
                            if let Some(src) = rfd::FileDialog::new()
                                .add_filter("ONNX / HRIR Models", &["onnx", "bin"])
                                .pick_file()
                            {
                                if let Some(fname) = src.file_name() {
                                    let dest = self.inventory.models_dir.join(fname);
                                    match std::fs::copy(&src, &dest) {
                                        Ok(_) => {
                                            self.refresh_models();
                                            self.status_message = Some((
                                                format!("Imported model {}", dest.display()),
                                                false,
                                            ));
                                        }
                                        Err(e) => {
                                            self.status_message = Some((
                                                format!("Import failed: {e}"),
                                                true,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    });
                });

            ui.add_space(12.0);
            ui.label(
                RichText::new("1-CLICK MODEL DOWNLOADER")
                    .size(11.0)
                    .color(TEXT_MUTED),
            );
            ui.add_space(6.0);

            let catalog = self.registry.models.clone();
            for entry in catalog {
                let is_installed = self.inventory.models_dir.join(&entry.filename).exists();
                let is_downloading = self.active_download_id.as_deref() == Some(entry.id.as_str());

                egui::Frame::none()
                    .fill(CARD_BG)
                    .stroke(Stroke::new(1.0_f32, BORDER_COLOR))
                    .rounding(Rounding::same(10.0))
                    .inner_margin(10.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new(&entry.name).strong().color(Color32::WHITE));
                                ui.label(
                                    RichText::new(format!(
                                        "{}  •  {}  (~{} MB)",
                                        entry.stage.label(),
                                        entry.filename,
                                        entry.approx_size_mb
                                    ))
                                    .size(11.0)
                                    .color(NEON_PURPLE),
                                );
                                ui.label(
                                    RichText::new(&entry.description)
                                        .size(11.0)
                                        .color(TEXT_MUTED),
                                );
                            });

                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if is_installed {
                                    ui.label(
                                        RichText::new("Installed ✓")
                                            .strong()
                                            .color(SUCCESS_GREEN),
                                    );
                                } else if is_downloading {
                                    let (dl, tot) = self
                                        .download_progress
                                        .get(&entry.id)
                                        .copied()
                                        .unwrap_or((0, None));
                                    let denom = tot
                                        .unwrap_or((entry.approx_size_mb as u64).max(1) * 1024 * 1024)
                                        .max(1) as f32;
                                    let frac = (dl as f32 / denom).clamp(0.0, 1.0);
                                    let mb = dl as f32 / (1024.0 * 1024.0);
                                    ui.add(
                                        egui::ProgressBar::new(frac)
                                            .desired_width(160.0)
                                            .animate(true)
                                            .text(format!("{:.1} MB ({:.0}%)", mb, frac * 100.0)),
                                    );
                                } else if ui
                                    .add_enabled(
                                        self.active_download_id.is_none(),
                                        egui::Button::new("Download Model"),
                                    )
                                    .clicked()
                                {
                                    self.start_model_download(entry.clone());
                                }
                            });
                        });
                    });
                ui.add_space(6.0);
            }

            ui.add_space(12.0);
            // ---- Add Future / Custom Pluggable Model ----
            egui::Frame::none()
                .fill(CARD_BG)
                .stroke(Stroke::new(1.0_f32, BORDER_COLOR))
                .rounding(Rounding::same(12.0))
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new("ADD CUSTOM / FUTURE MODEL TO REGISTRY")
                            .size(11.0)
                            .color(TEXT_MUTED),
                    );
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut self.custom_name);
                        ui.label("Filename:");
                        ui.text_edit_singleline(&mut self.custom_filename);
                    });
                    ui.horizontal(|ui| {
                        ui.label("URL:");
                        ui.text_edit_singleline(&mut self.custom_url);
                    });
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_label("Stage")
                            .selected_text(self.custom_stage.label())
                            .show_ui(ui, |ui| {
                                for s in [
                                    ModelStage::DenoiseStem,
                                    ModelStage::BandwidthRecon,
                                    ModelStage::SpatialHrtf,
                                ] {
                                    ui.selectable_value(&mut self.custom_stage, s, s.label());
                                }
                            });
                        if ui.button("Add to Pluggable Registry").clicked()
                            && !self.custom_name.trim().is_empty()
                            && !self.custom_filename.trim().is_empty()
                            && !self.custom_url.trim().is_empty()
                        {
                            let id = self
                                .custom_name
                                .to_lowercase()
                                .replace(' ', "_");
                            self.registry.models.push(PluggableModelEntry {
                                id,
                                name: self.custom_name.trim().to_string(),
                                stage: self.custom_stage,
                                filename: self.custom_filename.trim().to_string(),
                                download_url: self.custom_url.trim().to_string(),
                                description: "Custom user-added ONNX / HRIR model.".to_string(),
                                approx_size_mb: 100,
                                sha256: None,
                            });
                            let _ = self.registry.save(&self.inventory.models_dir);
                            self.custom_name.clear();
                            self.custom_filename.clear();
                            self.custom_url.clear();
                            self.status_message =
                                Some(("Added custom model to registry.json".to_string(), false));
                        }
                    });
                });
        });
    }

    fn render_about_tab(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label(
                RichText::new("Reviva NexGEN Audio Restorer")
                    .size(22.0)
                    .strong()
                    .color(NEON_CYAN),
            );
            ui.label(
                RichText::new("Version 1.0.0  •  MIT Open-Source License  •  100% Local Processing")
                    .size(12.5)
                    .color(NEON_PURPLE),
            );
            ui.add_space(10.0);

            egui::Frame::none()
                .fill(CARD_BG)
                .stroke(Stroke::new(1.0_f32, BORDER_COLOR))
                .rounding(Rounding::same(10.0))
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new("DEVELOPER & PRIVACY")
                            .size(11.0)
                            .strong()
                            .color(TEXT_MUTED),
                    );
                    ui.add_space(4.0);
                    ui.label(RichText::new("Developer: Dr. Vinayak Ashok Bharadi").strong().color(Color32::WHITE));
                    ui.label(RichText::new("Contact: vinayagaurang@zohomail.in").color(NEON_CYAN));
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(
                            "Privacy Guarantee: Zero data collection. No telemetry, analytics, or audio uploads. Everything runs locally on your PC.",
                        )
                        .color(SUCCESS_GREEN),
                    );
                });

            ui.add_space(10.0);

            egui::Frame::none()
                .fill(CARD_BG)
                .stroke(Stroke::new(1.0_f32, BORDER_COLOR))
                .rounding(Rounding::same(10.0))
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new("USER AGREEMENT, TERMS & CONDITIONS, AND PRIVACY POLICY")
                            .size(11.0)
                            .strong()
                            .color(TEXT_MUTED),
                    );
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(include_str!("../../TERMS_OF_USE.txt"))
                            .size(12.0)
                            .color(Color32::from_rgb(220, 224, 235)),
                    );
                });

            ui.add_space(10.0);

            egui::Frame::none()
                .fill(CARD_BG)
                .stroke(Stroke::new(1.0_f32, BORDER_COLOR))
                .rounding(Rounding::same(10.0))
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new("OPEN-SOURCE LIBRARIES USED")
                            .size(11.0)
                            .strong()
                            .color(TEXT_MUTED),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(
                            "• eframe / egui (MIT/Apache-2.0) — Native Rust Desktop UI\n\
                             • symphonia (MPL-2.0) — Audio Decoding (MP3, WAV, FLAC, OGG, AAC)\n\
                             • ort (MIT/Apache-2.0) — ONNX Runtime DirectML GPU & CPU Inference\n\
                             • mp3lame-encoder / LAME (MIT / LGPL) — High-Quality VBR & CBR MP3 Encoding\n\
                             • flacenc (Apache-2.0) & lofty (MIT/Apache-2.0) — Lossless FLAC & Metadata Tags\n\
                             • ebur128 (MIT) — ITU-R BS.1770 / EBU R128 Broadcast Loudness Mastering\n\
                             • hrtf (MIT) — 3D Binaural Head-Related Transfer Function Spatialization\n\
                             • realfft, rubato, rayon, rodio, hound — High-Speed DSP & Audio Playback",
                        )
                        .size(11.5)
                        .color(TEXT_MUTED),
                    );
                });
        });
    }
}

impl eframe::App for NexGenDesktopApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.config.save(&self.settings_path);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_workers(ctx);

        // Keyboard shortcuts: Space = Play/Pause, Tab = A/B Original vs Restored
        if !ctx.wants_keyboard_input() && self.active_tab == AppTab::Studio && self.original_audio.is_some() {
            if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
                let is_playing = self.player.as_ref().map(|p| p.is_playing()).unwrap_or(false);
                let restored_ref = self.restored_result.as_ref().map(|r| &r.restored_audio);
                if let Some(p) = &mut self.player {
                    if is_playing {
                        p.pause();
                    } else {
                        let pos = p.position_secs();
                        let track = if self.restored_result.is_some() {
                            p.active_track()
                        } else {
                            ActiveTrack::Original
                        };
                        let _ = p.play_from(track, self.original_audio.as_ref(), restored_ref, pos);
                    }
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Tab)) && self.restored_result.is_some() {
                let active = self
                    .player
                    .as_ref()
                    .map(|p| p.active_track())
                    .unwrap_or(ActiveTrack::Restored);
                let next = match active {
                    ActiveTrack::Original => ActiveTrack::Restored,
                    ActiveTrack::Restored => ActiveTrack::Original,
                };
                let restored_ref = self.restored_result.as_ref().map(|r| &r.restored_audio);
                if let Some(p) = &mut self.player {
                    let _ = p.switch_track(next, self.original_audio.as_ref(), restored_ref);
                }
                self.view_restored_visuals = next == ActiveTrack::Restored;
            }
        }

        // Handle drag-and-drop MP3/audio files onto the window
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if let Some(first) = dropped.into_iter().next() {
            if !self.is_processing {
                self.load_file(first);
            }
        }

        // ---- Top Bar ----
        egui::TopBottomPanel::top("top_bar")
            .frame(egui::Frame::none().fill(CARD_BG).inner_margin(12.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Reviva NexGEN Audio Restorer")
                            .size(20.0)
                            .strong()
                            .color(NEON_CYAN),
                    );
                    ui.add_space(12.0);
                    ui.selectable_value(&mut self.active_tab, AppTab::Studio, "Studio");
                    ui.selectable_value(&mut self.active_tab, AppTab::BatchQueue, "Batch Queue");
                    ui.selectable_value(
                        &mut self.active_tab,
                        AppTab::SettingsModels,
                        "Settings & Models",
                    );
                    ui.selectable_value(&mut self.active_tab, AppTab::About, "About & Legal");

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let open_btn = egui::Button::new(
                            RichText::new("Browse File...").strong().color(Color32::WHITE),
                        )
                        .fill(NEON_PURPLE)
                        .rounding(Rounding::same(8.0));
                        if ui.add_enabled(!self.is_processing, open_btn).clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Audio Files", &["mp3", "wav", "flac", "ogg", "m4a", "aac"])
                                .pick_file()
                            {
                                self.load_file(path);
                            }
                        }

                        ui.add_space(12.0);
                        let ram_mb = self.current_ram_mb();
                        ui.label(
                            RichText::new(format!("RAM: {} MB", ram_mb))
                                .size(12.0)
                                .color(TEXT_MUTED),
                        );

                        ui.add_space(12.0);
                        let model_badge = if self.inventory.any_onnx_ready() {
                            RichText::new("ONNX Models: Ready").color(SUCCESS_GREEN)
                        } else {
                            RichText::new("ONNX Models: DSP Fallback").color(WARN_AMBER)
                        };
                        ui.label(model_badge.size(12.0));
                    });
                });
            });

        if self.active_tab == AppTab::BatchQueue {
            egui::CentralPanel::default().show(ctx, |ui| self.render_batch_tab(ui));
            return;
        }
        if self.active_tab == AppTab::SettingsModels {
            egui::CentralPanel::default().show(ctx, |ui| self.render_settings_models_tab(ui));
            return;
        }
        if self.active_tab == AppTab::About {
            egui::CentralPanel::default().show(ctx, |ui| self.render_about_tab(ui));
            return;
        }

        // ---- Left Sidebar: Compact Color-Coded Pipeline Blocks ----
        egui::SidePanel::left("controls_panel")
            .exact_width(345.0)
            .frame(egui::Frame::none().fill(CARD_BG).inner_margin(8.0))
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;

                // Block 1: Presets (2x2 compact grid, Purple)
                egui::Frame::none()
                    .fill(Color32::from_rgb(24, 18, 44))
                    .stroke(Stroke::new(1.0_f32, NEON_PURPLE.linear_multiply(0.55)))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(
                            RichText::new("1. ENHANCEMENT PRESET")
                                .size(10.0)
                                .strong()
                                .color(NEON_PURPLE),
                        );
                        ui.add_enabled_ui(!self.is_processing, |ui| {
                            let presets = [
                                EnhancementPreset::CleanNatural,
                                EnhancementPreset::StudioRemaster,
                                EnhancementPreset::Surround3D,
                                EnhancementPreset::VocalFocus,
                            ];
                            egui::Grid::new("preset_grid")
                                .num_columns(2)
                                .spacing([6.0, 4.0])
                                .show(ui, |ui| {
                                    for (i, preset) in presets.iter().enumerate() {
                                        let selected = self.config.preset == *preset;
                                        let btn = egui::Button::new(
                                            RichText::new(preset.label())
                                                .size(11.5)
                                                .strong()
                                                .color(if selected { Color32::WHITE } else { TEXT_MUTED }),
                                        )
                                        .fill(if selected {
                                            NEON_PURPLE.linear_multiply(0.45)
                                        } else {
                                            Color32::from_rgb(18, 18, 32)
                                        })
                                        .stroke(Stroke::new(
                                            1.0_f32,
                                            if selected { NEON_CYAN } else { BORDER_COLOR },
                                        ))
                                        .min_size(egui::vec2(156.0, 24.0))
                                        .rounding(Rounding::same(6.0));
                                        if ui.add(btn).on_hover_text(preset.description()).clicked() {
                                            self.config.apply_preset(*preset);
                                        }
                                        if i % 2 == 1 {
                                            ui.end_row();
                                        }
                                    }
                                });
                        });
                    });

                // Block 2: Stage 0 & 1 — Clean & Denoise (Cyan)
                let s1_active = self.is_processing
                    && matches!(
                        self.current_stage,
                        PipelineStage::Preparing | PipelineStage::StemDenoise
                    );
                egui::Frame::none()
                    .fill(Color32::from_rgb(14, 28, 38))
                    .stroke(Stroke::new(
                        if s1_active { 1.8_f32 } else { 1.0_f32 },
                        if s1_active {
                            NEON_CYAN
                        } else {
                            NEON_CYAN.linear_multiply(0.45)
                        },
                    ))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("2. CLEAN & DENOISE")
                                    .size(10.0)
                                    .strong()
                                    .color(NEON_CYAN),
                            );
                            if s1_active {
                                ui.label(RichText::new("● ACTIVE").size(9.5).strong().color(NEON_CYAN));
                            }
                        });
                        ui.add_enabled_ui(!self.is_processing, |ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.config.enable_hum_declick, "Hum+Click");
                                if self.config.enable_hum_declick {
                                    ui.selectable_value(&mut self.config.hum_freq_hz, 50.0, "50Hz");
                                    ui.selectable_value(&mut self.config.hum_freq_hz, 60.0, "60Hz");
                                }
                            });
                            ui.checkbox(
                                &mut self.config.enable_stem_denoise,
                                "Stage 1: Deep Denoise / Stems",
                            );
                            if self.config.enable_stem_denoise {
                                ui.add(
                                    egui::Slider::new(&mut self.config.denoise_strength, 0.0..=1.0)
                                        .text("Denoise"),
                                );
                            }
                        });
                    });

                // Block 3: Stage 2 & 3 — Bandwidth & Spatial Stereo (Blue/Indigo)
                let s23_active = self.is_processing
                    && matches!(
                        self.current_stage,
                        PipelineStage::BandwidthReconstruction | PipelineStage::SpatialStereoUpmix
                    );
                let blue_accent = Color32::from_rgb(96, 165, 250);
                egui::Frame::none()
                    .fill(Color32::from_rgb(18, 24, 46))
                    .stroke(Stroke::new(
                        if s23_active { 1.8_f32 } else { 1.0_f32 },
                        if s23_active {
                            NEON_CYAN
                        } else {
                            blue_accent.linear_multiply(0.45)
                        },
                    ))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("3. RECONSTRUCT & SPATIAL")
                                    .size(10.0)
                                    .strong()
                                    .color(blue_accent),
                            );
                            if s23_active {
                                ui.label(RichText::new("● ACTIVE").size(9.5).strong().color(NEON_CYAN));
                            }
                        });
                        ui.add_enabled_ui(!self.is_processing, |ui| {
                            ui.checkbox(
                                &mut self.config.enable_bandwidth_recon,
                                "Stage 2: Bandwidth Reconstruction",
                            );
                            if self.config.enable_bandwidth_recon {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.config.reconstruction_strength,
                                        0.0..=1.0,
                                    )
                                    .text("Clarity"),
                                );
                            }
                            ui.checkbox(
                                &mut self.config.enable_spatial_upmix,
                                "Stage 3: Stereo & 3D HRTF",
                            );
                            if self.config.enable_spatial_upmix {
                                ui.add(
                                    egui::Slider::new(&mut self.config.stereo_width, 0.0..=1.0)
                                        .text("Width"),
                                );
                                egui::ComboBox::from_label("Spatial")
                                    .selected_text(self.config.spatial_mode.label())
                                    .show_ui(ui, |ui| {
                                        for m in [
                                            SpatialMode::OriginalStereo,
                                            SpatialMode::WideStereo,
                                            SpatialMode::BinauralSurround,
                                        ] {
                                            ui.selectable_value(
                                                &mut self.config.spatial_mode,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    });
                            }
                        });
                    });

                // Block 4: Tone & Stem Mixer (Amber)
                egui::Frame::none()
                    .fill(Color32::from_rgb(34, 24, 18))
                    .stroke(Stroke::new(1.0_f32, WARN_AMBER.linear_multiply(0.45)))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("4. TONE & MIXER")
                                    .size(10.0)
                                    .strong()
                                    .color(WARN_AMBER),
                            );
                            ui.selectable_value(
                                &mut self.mixer_sub_tab,
                                MixerSubTab::Eq5Band,
                                "5-Band EQ",
                            );
                            ui.selectable_value(
                                &mut self.mixer_sub_tab,
                                MixerSubTab::Stem4Mixer,
                                "4-Stem",
                            );
                            if ui.small_button("Reset").clicked() && !self.is_processing {
                                match self.mixer_sub_tab {
                                    MixerSubTab::Eq5Band => self.config.eq_gains_db = [0.0; 5],
                                    MixerSubTab::Stem4Mixer => self.config.stem_gains = [1.0; 4],
                                }
                            }
                        });
                        ui.add_enabled_ui(!self.is_processing, |ui| match self.mixer_sub_tab {
                            MixerSubTab::Eq5Band => {
                                let labels = ["60Hz", "250Hz", "1kHz", "4kHz", "12kHz"];
                                for (idx, lbl) in labels.iter().enumerate() {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut self.config.eq_gains_db[idx],
                                            -6.0..=6.0,
                                        )
                                        .text(*lbl)
                                        .suffix(" dB"),
                                    );
                                }
                            }
                            MixerSubTab::Stem4Mixer => {
                                let stem_labels = ["Vocals", "Drums", "Bass", "Other"];
                                for (idx, lbl) in stem_labels.iter().enumerate() {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut self.config.stem_gains[idx],
                                            0.0..=2.0,
                                        )
                                        .text(*lbl)
                                        .suffix("x"),
                                    );
                                }
                            }
                        });
                    });

                // Block 5: Mastering, Export & Actions (Emerald)
                let s4_active =
                    self.is_processing && self.current_stage == PipelineStage::Finalizing;
                egui::Frame::none()
                    .fill(Color32::from_rgb(14, 32, 26))
                    .stroke(Stroke::new(
                        if s4_active { 1.8_f32 } else { 1.0_f32 },
                        if s4_active {
                            SUCCESS_GREEN
                        } else {
                            SUCCESS_GREEN.linear_multiply(0.45)
                        },
                    ))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(
                            RichText::new("5. MASTERING & EXPORT")
                                .size(10.0)
                                .strong()
                                .color(SUCCESS_GREEN),
                        );
                        ui.add_enabled_ui(!self.is_processing, |ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.config.enable_loudness_mastering, "R128");
                                ui.checkbox(&mut self.config.preserve_metadata, "Tags");
                                ui.checkbox(&mut self.config.use_gpu_directml, "GPU");
                            });
                            egui::ComboBox::from_label("Format")
                                .selected_text(self.config.output_format.label())
                                .show_ui(ui, |ui| {
                                    for fmt in [
                                        OutputFormat::Mp3VbrV0,
                                        OutputFormat::Mp3VbrV2,
                                        OutputFormat::Mp3Cbr320,
                                        OutputFormat::Flac24Bit,
                                        OutputFormat::Wav24Bit,
                                        OutputFormat::Wav16Bit,
                                        OutputFormat::Ac3DolbyCompatible,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.config.output_format,
                                            fmt,
                                            fmt.label(),
                                        );
                                    }
                                });
                        });

                        ui.add_space(4.0);
                        if self.is_processing {
                            let cancel_btn = egui::Button::new(
                                RichText::new("Cancel Processing").strong().color(Color32::WHITE),
                            )
                            .fill(ERROR_RED)
                            .min_size(egui::vec2(ui.available_width(), 34.0))
                            .rounding(Rounding::same(8.0));
                            if ui.add(cancel_btn).clicked() {
                                self.cancel_flag.store(true, Ordering::Relaxed);
                            }
                        } else {
                            ui.horizontal(|ui| {
                                let preview_btn = egui::Button::new(
                                    RichText::new("Preview 10s").strong().color(Color32::WHITE),
                                )
                                .fill(Color32::from_rgb(38, 92, 128))
                                .min_size(egui::vec2(96.0, 36.0))
                                .rounding(Rounding::same(8.0));
                                if ui
                                    .add_enabled(self.original_audio.is_some(), preview_btn)
                                    .clicked()
                                {
                                    self.start_preview_10s();
                                }

                                let run_btn = egui::Button::new(
                                    RichText::new("Reconstruct & Enhance")
                                        .strong()
                                        .size(13.0)
                                        .color(Color32::WHITE),
                                )
                                .fill(NEON_PURPLE)
                                .min_size(egui::vec2(ui.available_width(), 36.0))
                                .rounding(Rounding::same(8.0));
                                if ui
                                    .add_enabled(self.original_audio.is_some(), run_btn)
                                    .clicked()
                                {
                                    self.start_restoration();
                                }
                            });
                        }
                    });
            });

        // ---- Main Workspace (Studio Tab) ----
        egui::CentralPanel::default().show(ctx, |ui| {
            // Active Models & Capabilities Summary Card
            let s1_name = self
                .config
                .selected_denoise_model
                .clone()
                .or_else(|| {
                    self.inventory
                        .stem_denoise_model
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| "Spectral Wiener DSP (Built-in)".to_string());
            let s1_cap = self
                .registry
                .models
                .iter()
                .find(|m| m.filename == s1_name)
                .map(|m| m.description.as_str())
                .unwrap_or("Multi-core realfft Wiener denoiser + 50/60Hz hum notch & de-clicker.");

            let s2_name = self
                .config
                .selected_recon_model
                .clone()
                .or_else(|| {
                    self.inventory
                        .apollo_recon_model
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| "Harmonic Exciter + 5-Band EQ (Built-in)".to_string());

            let s3_name = match self.config.spatial_mode {
                SpatialMode::BinauralSurround => {
                    if self.inventory.hrtf_sphere_file.is_some() {
                        "HRTF 3D Binaural (IRC_1002_C.bin)".to_string()
                    } else {
                        "HRTF 3D Binaural (Acoustic Sphere)".to_string()
                    }
                }
                _ => self.config.spatial_mode.label().to_string(),
            };

            egui::Frame::none()
                .fill(CARD_BG)
                .stroke(Stroke::new(1.0_f32, BORDER_COLOR))
                .rounding(Rounding::same(10.0))
                .inner_margin(8.0)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            RichText::new("ACTIVE PIPELINE:")
                                .size(11.0)
                                .strong()
                                .color(NEON_CYAN),
                        );
                        ui.label(
                            RichText::new(format!("1) {}  •  2) {}  •  3) {}", s1_name, s2_name, s3_name))
                                .size(11.5)
                                .color(Color32::WHITE),
                        );
                    });
                    ui.label(
                        RichText::new(format!(
                            "Capabilities: {} | GPU: {} | ORT DLL: {}",
                            s1_cap,
                            if self.config.use_gpu_directml { "DirectML" } else { "CPU" },
                            if self.inventory.ort_dll_present { "Found" } else { "System/Auto" }
                        ))
                        .size(10.5)
                        .color(TEXT_MUTED),
                    );
                    if let Some(res) = &self.restored_result {
                        ui.label(
                            RichText::new(format!(
                                "Executed on audio: {}",
                                res.used_onnx_models.join("  ->  ")
                            ))
                            .size(10.5)
                            .color(SUCCESS_GREEN),
                        );
                    }
                });
            ui.add_space(8.0);

            if let Some((msg, is_err)) = &self.status_message {
                let color = if *is_err { ERROR_RED } else { SUCCESS_GREEN };
                egui::Frame::none()
                    .fill(color.linear_multiply(0.15))
                    .stroke(Stroke::new(1.0_f32, color))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new(msg).color(color));
                    });
                ui.add_space(8.0);
            }

            if self.original_analysis.is_none() {
                ui.centered_and_justified(|ui| {
                    egui::Frame::none()
                        .fill(CARD_BG)
                        .stroke(Stroke::new(1.5_f32, BORDER_COLOR))
                        .rounding(Rounding::same(16.0))
                        .inner_margin(36.0)
                        .show(ui, |ui| {
                            ui.vertical_centered(|ui| {
                                ui.label(
                                    RichText::new("Drop an MP3, WAV, or FLAC file here")
                                        .size(20.0)
                                        .strong()
                                        .color(Color32::WHITE),
                                );
                                ui.add_space(8.0);
                                ui.label(
                                    RichText::new(
                                        "Select or drop an MP3, WAV, FLAC, OGG, or AAC file to reconstruct into clean stereo.",
                                    )
                                    .color(TEXT_MUTED),
                                );
                                ui.add_space(16.0);
                                let browse_center_btn = egui::Button::new(
                                    RichText::new("Browse Files from Drive...")
                                        .strong()
                                        .size(14.0)
                                        .color(Color32::WHITE),
                                )
                                .fill(NEON_PURPLE)
                                .min_size(egui::vec2(220.0, 40.0))
                                .rounding(Rounding::same(10.0));
                                if ui.add(browse_center_btn).clicked() {
                                    if let Some(path) = rfd::FileDialog::new()
                                        .add_filter(
                                            "Audio Files",
                                            &["mp3", "wav", "flac", "ogg", "m4a", "aac"],
                                        )
                                        .pick_file()
                                    {
                                        self.load_file(path);
                                    }
                                }
                                ui.add_space(14.0);
                                ui.label(
                                    RichText::new(format!(
                                        "Models folder: {}",
                                        self.inventory.models_dir.display()
                                    ))
                                    .size(11.0)
                                    .color(TEXT_MUTED),
                                );
                            });
                        });
                });
                return;
            }

            if self.is_processing {
                egui::Frame::none()
                    .fill(CARD_BG)
                    .stroke(Stroke::new(1.0_f32, NEON_CYAN))
                    .rounding(Rounding::same(12.0))
                    .inner_margin(12.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(self.current_stage.label())
                                    .strong()
                                    .color(NEON_CYAN),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    RichText::new(format!("{:.0}%", self.overall_progress * 100.0))
                                        .strong()
                                        .color(Color32::WHITE),
                                );
                            });
                        });
                        ui.add_space(6.0);
                        ui.add(egui::ProgressBar::new(self.overall_progress).animate(true));
                        ui.add_space(4.0);
                        ui.label(RichText::new(&self.backend_note).size(11.0).color(TEXT_MUTED));
                    });
                ui.add_space(10.0);
            }

            let orig_analysis = self.original_analysis.as_ref().unwrap().clone();
            let active_analysis = if self.view_restored_visuals {
                self.restored_result
                    .as_ref()
                    .map(|r| r.restored_analysis.clone())
                    .unwrap_or_else(|| orig_analysis.clone())
            } else {
                orig_analysis.clone()
            };

            ui.horizontal(|ui| {
                if self.restored_result.is_some() {
                    ui.selectable_value(&mut self.view_restored_visuals, false, "Original Audio");
                    ui.selectable_value(&mut self.view_restored_visuals, true, "Restored Stereo");
                } else {
                    ui.label(RichText::new("Original Audio Analysis").strong().color(NEON_CYAN));
                }

                ui.add_space(10.0);
                let (badge_text, badge_color) = if active_analysis.is_true_stereo() {
                    (
                        format!(
                            "((•)) 2.0 TRUE STEREO  •  L/R Side {:.1}%",
                            active_analysis.side_energy_pct
                        ),
                        SUCCESS_GREEN,
                    )
                } else {
                    ("((•)) 1.0 MONO / DUAL-MONO".to_string(), WARN_AMBER)
                };
                egui::Frame::none()
                    .fill(badge_color.linear_multiply(0.18))
                    .stroke(Stroke::new(1.0_f32, badge_color))
                    .rounding(Rounding::same(12.0))
                    .inner_margin(egui::vec2(10.0, 4.0))
                    .show(ui, |ui| {
                        ui.label(RichText::new(badge_text).size(11.5).strong().color(badge_color));
                    });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.restored_result.is_some() {
                        let export_label = match self.config.output_format {
                            OutputFormat::Mp3VbrV0 | OutputFormat::Mp3VbrV2 | OutputFormat::Mp3Cbr320 => {
                                "Export Reconstructed MP3"
                            }
                            OutputFormat::Flac24Bit => "Export Lossless FLAC",
                            OutputFormat::Ac3DolbyCompatible => "Export AC-3 (Dolby Digital)",
                            _ => "Export Reconstructed WAV",
                        };
                        let export_btn = egui::Button::new(
                            RichText::new(export_label).strong().color(Color32::WHITE),
                        )
                        .fill(SUCCESS_GREEN.linear_multiply(0.85))
                        .rounding(Rounding::same(8.0));
                        if ui.add(export_btn).clicked() {
                            self.export_restored();
                        }
                    }
                });
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!(
                        "L: {:.1} dB   |   R: {:.1} dB   |   Side Energy: {:.1}%   |   Phase Corr: {:.2}",
                        active_analysis.left_rms_db,
                        active_analysis.right_rms_db,
                        active_analysis.side_energy_pct,
                        active_analysis.stereo_correlation
                    ))
                    .size(11.0)
                    .color(TEXT_MUTED),
                );
            });
            ui.add_space(6.0);

            let total_dur = orig_analysis.duration_secs.max(0.01);
            let cur_pos = self
                .player
                .as_mut()
                .map(|p| p.position_secs())
                .unwrap_or(0.0);

            // Handle A-B loop wrap-around
            if self.loop_enabled {
                if let (Some(a), Some(b)) = (self.loop_a, self.loop_b) {
                    let lo = a.min(b);
                    let hi = a.max(b);
                    if hi > lo + 0.1 && cur_pos >= hi {
                        let restored_ref = self.restored_result.as_ref().map(|r| &r.restored_audio);
                        if let Some(p) = &mut self.player {
                            if p.is_playing() {
                                let _ = p.seek(lo, self.original_audio.as_ref(), restored_ref);
                            }
                        }
                    }
                }
            }

            let progress_frac = (cur_pos / total_dur).clamp(0.0, 1.0);
            let accent = if self.view_restored_visuals && self.restored_result.is_some() {
                NEON_PURPLE
            } else {
                NEON_CYAN
            };
            let loop_norm = match (self.loop_a, self.loop_b) {
                (Some(a), Some(b)) => Some((a / total_dur, b / total_dur)),
                _ => None,
            };

            if let Some(seek_norm) = draw_waveform(
                ui,
                &active_analysis.waveform_peaks,
                progress_frac,
                accent,
                105.0,
                loop_norm,
            ) {
                let seek_secs = seek_norm * total_dur;
                let restored_ref = self.restored_result.as_ref().map(|r| &r.restored_audio);
                if let Some(p) = &mut self.player {
                    let _ = p.seek(seek_secs, self.original_audio.as_ref(), restored_ref);
                }
            }
            ui.add_space(8.0);
            draw_spectrogram(ui, &active_analysis.spectrogram, 120.0);
            ui.add_space(10.0);

            // ---- A/B Player Bar ----
            egui::Frame::none()
                .fill(CARD_BG)
                .stroke(Stroke::new(1.0_f32, BORDER_COLOR))
                .rounding(Rounding::same(12.0))
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let is_playing = self.player.as_ref().map(|p| p.is_playing()).unwrap_or(false);
                        let play_label = if is_playing { "Pause" } else { "Play" };
                        if ui.button(RichText::new(play_label).strong()).clicked() {
                            let restored_ref = self.restored_result.as_ref().map(|r| &r.restored_audio);
                            if let Some(p) = &mut self.player {
                                if is_playing {
                                    p.pause();
                                } else {
                                    let pos = p.position_secs();
                                    let track = if self.restored_result.is_some() {
                                        p.active_track()
                                    } else {
                                        ActiveTrack::Original
                                    };
                                    let _ = p.play_from(track, self.original_audio.as_ref(), restored_ref, pos);
                                }
                            }
                        }

                        if self.restored_result.is_some() {
                            let active = self
                                .player
                                .as_ref()
                                .map(|p| p.active_track())
                                .unwrap_or(ActiveTrack::Restored);
                            let ab_text = match active {
                                ActiveTrack::Original => "Listening: A (Original)",
                                ActiveTrack::Restored => "Listening: B (Restored)",
                            };
                            if ui
                                .button(RichText::new(ab_text).color(NEON_CYAN).strong())
                                .clicked()
                            {
                                let next = match active {
                                    ActiveTrack::Original => ActiveTrack::Restored,
                                    ActiveTrack::Restored => ActiveTrack::Original,
                                };
                                let restored_ref = self.restored_result.as_ref().map(|r| &r.restored_audio);
                                if let Some(p) = &mut self.player {
                                    let _ = p.switch_track(next, self.original_audio.as_ref(), restored_ref);
                                }
                                self.view_restored_visuals = next == ActiveTrack::Restored;
                            }
                        }

                        ui.add_space(8.0);
                        if ui.small_button("Set A").clicked() {
                            self.loop_a = Some(cur_pos);
                        }
                        if ui.small_button("Set B").clicked() {
                            self.loop_b = Some(cur_pos);
                            if self.loop_a.is_some() {
                                self.loop_enabled = true;
                            }
                        }
                        if self.loop_a.is_some() || self.loop_b.is_some() {
                            ui.checkbox(&mut self.loop_enabled, "Loop A-B");
                            if ui.small_button("Clear").clicked() {
                                self.loop_a = None;
                                self.loop_b = None;
                                self.loop_enabled = false;
                            }
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                RichText::new(format!(
                                    "{} / {}   (Space: Play • Tab: A/B)",
                                    format_duration(cur_pos),
                                    format_duration(total_dur)
                                ))
                                .color(TEXT_MUTED),
                            );
                        });
                    });
                });

            ui.add_space(10.0);

            if let Some(res) = &self.restored_result {
                let ra = &res.restored_analysis;
                let score_delta = ra.quality_score - orig_analysis.quality_score;
                let pct_gain = if orig_analysis.quality_score > 0.1 {
                    (score_delta / orig_analysis.quality_score) * 100.0
                } else {
                    0.0
                };
                let noise_delta = orig_analysis.noise_floor_db - ra.noise_floor_db; // positive = quieter
                let bw_delta_khz = (ra.effective_bandwidth_hz - orig_analysis.effective_bandwidth_hz) / 1000.0;
                let side_delta = ra.side_energy_pct - orig_analysis.side_energy_pct;

                // Overall Quality Improvement Banner
                egui::Frame::none()
                    .fill(SUCCESS_GREEN.linear_multiply(0.12))
                    .stroke(Stroke::new(1.0_f32, SUCCESS_GREEN))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!(
                                    "QUALITY SCORE IMPROVEMENT:  {:.1} -> {:.1} / 10   ({:+.1} pts / {:+.0}%)",
                                    orig_analysis.quality_score,
                                    ra.quality_score,
                                    score_delta,
                                    pct_gain
                                ))
                                .strong()
                                .color(SUCCESS_GREEN),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    RichText::new(format!("Rendered in {:.1}s", res.elapsed_secs))
                                        .size(11.0)
                                        .color(TEXT_MUTED),
                                );
                            });
                        });
                    });
                ui.add_space(6.0);

                ui.columns(2, |cols| {
                    cols[0].group(|ui| {
                        ui.set_width(ui.available_width());
                        ui.label(
                            RichText::new("ORIGINAL AUDIO DETAILS")
                                .size(11.0)
                                .strong()
                                .color(NEON_CYAN),
                        );
                        ui.add_space(4.0);
                        ui.label(format!(
                            "Sample Rate: {} Hz  •  {}",
                            orig_analysis.sample_rate,
                            if orig_analysis.is_true_stereo() { "2.0 Stereo" } else { "1.0 Mono / Flat" }
                        ));
                        ui.label(format!("Noise Floor: {:.1} dB", orig_analysis.noise_floor_db));
                        ui.label(format!(
                            "Bandwidth: {:.1} kHz",
                            orig_analysis.effective_bandwidth_hz / 1000.0
                        ));
                        ui.label(format!(
                            "Stereo Side: {:.1}%  (Corr: {:.2})",
                            orig_analysis.side_energy_pct, orig_analysis.stereo_correlation
                        ));
                        ui.label(
                            RichText::new(format!("Quality Score: {:.1} / 10", orig_analysis.quality_score))
                                .strong()
                                .color(WARN_AMBER),
                        );
                        ui.add_space(4.0);
                        ui.label(RichText::new("DETECTED ISSUES:").size(10.5).strong().color(TEXT_MUTED));
                        for issue in &orig_analysis.issues {
                            let c = match issue.severity {
                                IssueSeverity::Low => SUCCESS_GREEN,
                                IssueSeverity::Medium => WARN_AMBER,
                                IssueSeverity::High => ERROR_RED,
                            };
                            ui.label(RichText::new(format!("• {} — {}", issue.title, issue.detail)).size(11.0).color(c));
                        }
                    });

                    cols[1].group(|ui| {
                        ui.set_width(ui.available_width());
                        ui.label(
                            RichText::new("RECONSTRUCTED AUDIO DETAILS & IMPROVEMENTS")
                                .size(11.0)
                                .strong()
                                .color(SUCCESS_GREEN),
                        );
                        ui.add_space(4.0);
                        ui.label(format!(
                            "Sample Rate: {} Hz  •  {}",
                            ra.sample_rate,
                            if ra.is_true_stereo() { "2.0 True Stereo" } else { "Stereo" }
                        ));
                        ui.label(format!(
                            "Noise Floor: {:.1} dB  ({:+.1} dB cleaner)",
                            ra.noise_floor_db, noise_delta
                        ));
                        ui.label(format!(
                            "Bandwidth: {:.1} kHz  ({:+.1} kHz restored)",
                            ra.effective_bandwidth_hz / 1000.0,
                            bw_delta_khz
                        ));
                        ui.label(format!(
                            "Stereo Side: {:.1}%  ({:+.1}%, Corr: {:.2})",
                            ra.side_energy_pct, side_delta, ra.stereo_correlation
                        ));
                        ui.label(
                            RichText::new(format!(
                                "Quality Score: {:.1} / 10  ({:+.1} pts / {:+.0}%)",
                                ra.quality_score, score_delta, pct_gain
                            ))
                            .strong()
                            .color(SUCCESS_GREEN),
                        );
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new("ISSUES ADDRESSED & NEW FEATURES:")
                                .size(10.5)
                                .strong()
                                .color(TEXT_MUTED),
                        );
                        for issue in &orig_analysis.issues {
                            let resolution = if issue.title.contains("Hiss") || issue.title.contains("Noise") {
                                format!("✓ {} -> Reduced by {:.1} dB", issue.title, noise_delta.max(0.0))
                            } else if issue.title.contains("Bandwidth") {
                                format!(
                                    "✓ {} -> Extended to {:.1} kHz",
                                    issue.title,
                                    ra.effective_bandwidth_hz / 1000.0
                                )
                            } else if issue.title.contains("Mono") || issue.title.contains("Narrow") {
                                format!(
                                    "✓ {} -> Reconstructed 2.0 Stereo ({:.1}% side)",
                                    issue.title, ra.side_energy_pct
                                )
                            } else {
                                format!("✓ {} -> Processed & Mastered", issue.title)
                            };
                            ui.label(RichText::new(resolution).size(11.0).color(SUCCESS_GREEN));
                        }
                        if self.config.enable_hum_declick {
                            ui.label(
                                RichText::new(format!(
                                    "✓ {:.0} Hz Mains Hum & Vinyl Clicks Filtered",
                                    self.config.hum_freq_hz
                                ))
                                .size(11.0)
                                .color(NEON_CYAN),
                            );
                        }
                        if self.config.enable_loudness_mastering {
                            ui.label(
                                RichText::new("✓ Broadcast Loudness Normalized (-14 LUFS EBU R128)")
                                    .size(11.0)
                                    .color(NEON_CYAN),
                            );
                        }
                    });
                });
            } else {
                ui.columns(2, |cols| {
                    cols[0].group(|ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new("AUDIO METRICS").size(11.0).color(TEXT_MUTED));
                        ui.add_space(4.0);
                        ui.label(format!("Sample Rate: {} Hz", active_analysis.sample_rate));
                        ui.label(format!(
                            "Noise Floor: {:.1} dB",
                            active_analysis.noise_floor_db
                        ));
                        ui.label(format!(
                            "Effective Bandwidth: {:.1} kHz",
                            active_analysis.effective_bandwidth_hz / 1000.0
                        ));
                        ui.label(format!(
                            "Stereo Correlation: {:.2}",
                            active_analysis.stereo_correlation
                        ));
                        ui.label(
                            RichText::new(format!(
                                "Quality Score: {:.1} / 10",
                                active_analysis.quality_score
                            ))
                            .strong()
                            .color(NEON_CYAN),
                        );
                    });

                    cols[1].group(|ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new("DETECTED ISSUES").size(11.0).color(TEXT_MUTED));
                        ui.add_space(4.0);
                        for issue in &orig_analysis.issues {
                            let c = match issue.severity {
                                IssueSeverity::Low => SUCCESS_GREEN,
                                IssueSeverity::Medium => WARN_AMBER,
                                IssueSeverity::High => ERROR_RED,
                            };
                            ui.label(RichText::new(&issue.title).strong().color(c));
                            ui.label(RichText::new(&issue.detail).size(11.0).color(TEXT_MUTED));
                            ui.add_space(3.0);
                        }
                    });
                });
            }

            if self.player.as_ref().map(|p| p.is_playing()).unwrap_or(false) {
                ctx.request_repaint();
            }
        });
    }
}
