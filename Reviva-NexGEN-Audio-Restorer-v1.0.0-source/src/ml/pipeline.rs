use anyhow::Result;
use crossbeam_channel::Sender;
use ebur128::{EbuR128, Mode};
use hrtf::{HrirSphere, HrtfContext, HrtfProcessor, Vec3};
use rayon::join;
use realfft::RealFftPlanner;
use std::io::Cursor;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

use crate::audio::analyzer::{analyze_audio, AudioAnalysis};
use crate::audio::io::{resample_stereo, StereoAudioBuffer};
use crate::config::{EnhancementPreset, RestorationConfig, SpatialMode};
use crate::ml::onnx_engine::{ModelInventory, OnnxBackend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Preparing,
    StemDenoise,
    BandwidthReconstruction,
    SpatialStereoUpmix,
    Finalizing,
}

impl PipelineStage {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Preparing => "Preparing & Resampling (44.1 kHz)...",
            Self::StemDenoise => "Stage 1: Deep Denoising & Artifact Removal",
            Self::BandwidthReconstruction => "Stage 2: Bandwidth & Harmonic Reconstruction",
            Self::SpatialStereoUpmix => "Stage 3: Stereo & Spatial Soundstage",
            Self::Finalizing => "Normalizing & Computing Metrics...",
        }
    }
}

#[derive(Debug, Clone)]
pub enum PipelineProgress {
    Update {
        stage: PipelineStage,
        stage_progress: f32,
        overall_progress: f32,
        backend_note: String,
    },
    Completed(PipelineResult),
    Cancelled,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub restored_audio: StereoAudioBuffer,
    pub restored_analysis: AudioAnalysis,
    pub elapsed_secs: f32,
    pub used_onnx_models: Vec<String>,
}

pub struct RestorationPipeline;

impl RestorationPipeline {
    /// Executes the 3-stage restoration pipeline on `input` using `config`.
    /// Runs in a worker thread and sends progress updates via `tx`.
    pub fn run(
        input: StereoAudioBuffer,
        config: RestorationConfig,
        inventory: ModelInventory,
        cancel_flag: Arc<AtomicBool>,
        tx: Sender<PipelineProgress>,
    ) {
        let start = Instant::now();
        let mut used_models = Vec::new();

        let send_update = |stage: PipelineStage, sp: f32, op: f32, note: &str| {
            let _ = tx.send(PipelineProgress::Update {
                stage,
                stage_progress: sp,
                overall_progress: op,
                backend_note: note.to_string(),
            });
        };

        // Step 0: Resample to 44.1 kHz (standard for HTDemucs / Apollo)
        send_update(
            PipelineStage::Preparing,
            0.2,
            0.05,
            "Resampling audio to 44.1 kHz studio rate...",
        );
        let mut audio = match resample_stereo(input, 44_100) {
            Ok(a) => a,
            Err(e) => {
                let _ = tx.send(PipelineProgress::Failed(format!("Resample failed: {e}")));
                return;
            }
        };

        if cancel_flag.load(Ordering::Relaxed) {
            let _ = tx.send(PipelineProgress::Cancelled);
            return;
        }

        // ---- Pre-Stage 0: 50/60Hz Hum Notch & Vinyl De-Clicker ----
        if config.enable_hum_declick {
            send_update(
                PipelineStage::Preparing,
                0.7,
                0.08,
                &format!(
                    "Removing {:.0}Hz mains hum & vinyl clicks (multi-core DSP)...",
                    config.hum_freq_hz
                ),
            );
            apply_hum_notch_and_declick(&mut audio, config.hum_freq_hz);
            used_models.push(format!("{:.0}Hz Hum Notch + De-Clicker", config.hum_freq_hz));
        }

        // ---- Stage 1: Denoise / Stem Cleaning ----
        if config.enable_stem_denoise {
            let mut ran_onnx = false;
            let chosen_denoise = config
                .selected_denoise_model
                .as_ref()
                .map(|name| inventory.models_dir.join(name))
                .filter(|p| p.exists())
                .or_else(|| inventory.stem_denoise_model.clone());
            if let Some(model_path) = &chosen_denoise {
                let model_name = model_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("stem_model.onnx")
                    .to_string();
                match OnnxBackend::load(model_path, config.use_gpu_directml) {
                    Ok(mut backend) => {
                        let ep = if backend.using_directml {
                            "DirectML GPU"
                        } else {
                            "CPU"
                        };
                        send_update(
                            PipelineStage::StemDenoise,
                            0.05,
                            0.12,
                            &format!("Running {} on {}", model_name, ep),
                        );

                        match run_chunked_onnx(
                            &mut backend,
                            &audio,
                            44_100 * 5,
                            config.stem_gains,
                            &cancel_flag,
                            |p| {
                                send_update(
                                    PipelineStage::StemDenoise,
                                    p,
                                    0.10 + p * 0.30,
                                    &format!("Running {} ({})", model_name, ep),
                                );
                            },
                        ) {
                            Ok(cleaned) => {
                                blend_audio(&mut audio, &cleaned, config.denoise_strength);
                                used_models.push(format!("{} ({})", model_name, ep));
                                ran_onnx = true;
                            }
                            Err(e) => {
                                used_models.push(format!(
                                    "{} inference note: {} -> DSP fallback",
                                    model_name, e
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        used_models.push(format!(
                            "{} load note: {} -> DSP fallback",
                            model_name, e
                        ));
                    }
                }
            }

            if !ran_onnx {
                send_update(
                    PipelineStage::StemDenoise,
                    0.1,
                    0.15,
                    "Spectral Wiener Denoiser + 50/60Hz Hum Notch (DSP)",
                );
                spectral_denoise(&mut audio, config.denoise_strength, &cancel_flag, |p| {
                    send_update(
                        PipelineStage::StemDenoise,
                        p,
                        0.15 + p * 0.25,
                        "Spectral Wiener Denoiser + Hum Removal (DSP)",
                    );
                });
                used_models.push("Spectral Wiener DSP".to_string());
            }
        }

        if cancel_flag.load(Ordering::Relaxed) {
            let _ = tx.send(PipelineProgress::Cancelled);
            return;
        }

        // ---- Stage 2: Bandwidth & Harmonic Reconstruction ----
        if config.enable_bandwidth_recon {
            let mut ran_onnx = false;
            let chosen_recon = config
                .selected_recon_model
                .as_ref()
                .map(|name| inventory.models_dir.join(name))
                .filter(|p| p.exists())
                .or_else(|| inventory.apollo_recon_model.clone());
            if let Some(model_path) = &chosen_recon {
                if let Ok(mut backend) = OnnxBackend::load(model_path, config.use_gpu_directml) {
                    let ep = if backend.using_directml {
                        "DirectML GPU"
                    } else {
                        "CPU"
                    };
                    let model_name = model_path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("apollo.onnx")
                        .to_string();
                    send_update(
                        PipelineStage::BandwidthReconstruction,
                        0.05,
                        0.42,
                        &format!("Running {} on {}", model_name, ep),
                    );

                    if let Ok(reconstructed) = run_chunked_onnx(
                        &mut backend,
                        &audio,
                        44_100 * 4,
                        [1.0; 4],
                        &cancel_flag,
                        |p| {
                            send_update(
                                PipelineStage::BandwidthReconstruction,
                                p,
                                0.40 + p * 0.30,
                                &format!("Running {} ({})", model_name, ep),
                            );
                        },
                    ) {
                        blend_audio(&mut audio, &reconstructed, config.reconstruction_strength);
                        used_models.push(format!("{} ({})", model_name, ep));
                        ran_onnx = true;
                    }
                }
            }

            if !ran_onnx {
                send_update(
                    PipelineStage::BandwidthReconstruction,
                    0.1,
                    0.45,
                    "Harmonic Bandwidth Exciter (DSP Reconstruction)",
                );
                harmonic_bandwidth_reconstruct(
                    &mut audio,
                    config.reconstruction_strength,
                    config.preset == EnhancementPreset::VocalFocus,
                );
                send_update(
                    PipelineStage::BandwidthReconstruction,
                    1.0,
                    0.70,
                    "Harmonic Bandwidth Exciter Complete",
                );
                used_models.push("Harmonic Bandwidth Exciter DSP".to_string());
            }
        }

        if cancel_flag.load(Ordering::Relaxed) {
            let _ = tx.send(PipelineProgress::Cancelled);
            return;
        }

        // ---- Stage 3: Stereo Reconstruction & Spatial Enhancement ----
        if config.enable_spatial_upmix {
            send_update(
                PipelineStage::SpatialStereoUpmix,
                0.2,
                0.75,
                &format!("Reconstructing {}...", config.spatial_mode.label()),
            );
            apply_spatial_reconstruction(
                &mut audio,
                config.spatial_mode,
                config.stereo_width,
                &inventory.models_dir,
            );
            send_update(
                PipelineStage::SpatialStereoUpmix,
                1.0,
                0.90,
                "Spatial Soundstage Complete",
            );
            used_models.push(match config.spatial_mode {
                SpatialMode::BinauralSurround => {
                    if inventory.hrtf_sphere_file.is_some() {
                        "HRTF 3D Binaural (IRC_1002_C.bin) + Surround Matrix".to_string()
                    } else {
                        "HRTF 3D Binaural (Acoustic Sphere) + Surround Matrix".to_string()
                    }
                }
                _ => format!("Spatializer ({})", config.spatial_mode.label()),
            });
        }

        // ---- 5-Band Mastering EQ ----
        if config.eq_gains_db.iter().any(|&g| g.abs() > 0.05) {
            apply_5band_mastering_eq(&mut audio, config.eq_gains_db);
            used_models.push("5-Band Mastering EQ".to_string());
        }

        // ---- Finalize: EBU R128 Loudness Mastering & Peak Normalization ----
        send_update(
            PipelineStage::Finalizing,
            0.5,
            0.95,
            if config.enable_loudness_mastering {
                "Mastering to -14 LUFS (EBU R128) & limiting true peaks..."
            } else {
                "Normalizing headroom & computing quality metrics..."
            },
        );
        if config.enable_loudness_mastering {
            apply_ebur128_mastering(&mut audio, -14.0);
            used_models.push("EBU R128 Loudness Master".to_string());
        } else {
            normalize_headroom(&mut audio, 0.94);
        }

        let restored_analysis = analyze_audio(&audio);
        let elapsed_secs = start.elapsed().as_secs_f32();

        let _ = tx.send(PipelineProgress::Completed(PipelineResult {
            restored_audio: audio,
            restored_analysis,
            elapsed_secs,
            used_onnx_models: used_models,
        }));
    }
}

/// Runs chunked overlap-add ONNX inference on a stereo buffer.
fn run_chunked_onnx<F>(
    backend: &mut OnnxBackend,
    audio: &StereoAudioBuffer,
    chunk_len: usize,
    stem_gains: [f32; 4],
    cancel_flag: &AtomicBool,
    mut on_progress: F,
) -> Result<StereoAudioBuffer>
where
    F: FnMut(f32),
{
    let total = audio.len();
    let hop = (chunk_len * 3) / 4; // 25% overlap for smooth crossfading
    let fade_len = chunk_len - hop;

    let mut out_l = vec![0.0_f32; total];
    let mut out_r = vec![0.0_f32; total];
    let mut weight = vec![0.0_f32; total];

    let mut pos = 0usize;
    while pos < total {
        if cancel_flag.load(Ordering::Relaxed) {
            break;
        }
        let end = (pos + chunk_len).min(total);
        let cur_len = end - pos;

        let mut in_l = audio.left[pos..end].to_vec();
        let mut in_r = audio.right[pos..end].to_vec();
        if cur_len < chunk_len {
            in_l.resize(chunk_len, 0.0);
            in_r.resize(chunk_len, 0.0);
        }

        let (res_l, res_r) = backend.run_stereo_chunk(&in_l, &in_r, stem_gains)?;

        for i in 0..cur_len {
            let mut w = 1.0_f32;
            if pos > 0 && i < fade_len {
                w = i as f32 / fade_len as f32;
            } else if end < total && i >= cur_len.saturating_sub(fade_len) {
                let rem = cur_len - 1 - i;
                w = rem as f32 / fade_len as f32;
            }
            out_l[pos + i] += res_l[i] * w;
            out_r[pos + i] += res_r[i] * w;
            weight[pos + i] += w;
        }

        on_progress((end as f32 / total as f32).clamp(0.0, 1.0));
        if end == total {
            break;
        }
        pos += hop;
    }

    for i in 0..total {
        if weight[i] > 1e-6 {
            out_l[i] /= weight[i];
            out_r[i] /= weight[i];
        } else {
            out_l[i] = audio.left[i];
            out_r[i] = audio.right[i];
        }
    }

    Ok(StereoAudioBuffer {
        left: out_l,
        right: out_r,
        sample_rate: audio.sample_rate,
        original_channels: audio.original_channels,
        original_sample_rate: audio.original_sample_rate,
    })
}

fn blend_audio(target: &mut StereoAudioBuffer, processed: &StereoAudioBuffer, wet: f32) {
    let wet = wet.clamp(0.0, 1.0);
    let dry = 1.0 - wet;
    let n = target.len().min(processed.len());
    for i in 0..n {
        target.left[i] = target.left[i] * dry + processed.left[i] * wet;
        target.right[i] = target.right[i] * dry + processed.right[i] * wet;
    }
}

/// Multi-core `realfft` STFT Wiener spectral denoiser.
fn spectral_denoise<F>(
    audio: &mut StereoAudioBuffer,
    strength: f32,
    cancel_flag: &AtomicBool,
    mut on_progress: F,
) where
    F: FnMut(f32),
{
    let fft_size = 2048usize;
    let hop = fft_size / 4;
    let len = audio.len();
    if len < fft_size {
        return;
    }

    let hann: Vec<f32> = (0..fft_size)
        .map(|k| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * k as f32 / (fft_size as f32 - 1.0)).cos())
        })
        .collect();

    // Estimate noise spectrum from the quietest 5% of frames
    let mut frame_energies: Vec<(usize, f32)> = Vec::new();
    for pos in (0..len.saturating_sub(fft_size)).step_by(hop) {
        let e: f32 = audio.left[pos..pos + fft_size]
            .iter()
            .map(|x| x * x)
            .sum();
        frame_energies.push((pos, e));
    }
    frame_energies.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let sample_count = (frame_energies.len() / 20).clamp(1, 20);

    let mut planner = RealFftPlanner::<f32>::new();
    let rfft = planner.plan_fft_forward(fft_size);
    let mut in_buf = rfft.make_input_vec();
    let mut spec_buf = rfft.make_output_vec();
    let bins = spec_buf.len();

    let mut noise_mag = vec![0.0_f32; bins];
    for &(pos, _) in frame_energies.iter().take(sample_count) {
        for k in 0..fft_size {
            in_buf[k] = audio.left[pos + k] * hann[k];
        }
        let _ = rfft.process(&mut in_buf, &mut spec_buf);
        for b in 0..bins {
            noise_mag[b] += spec_buf[b].norm();
        }
    }
    for v in &mut noise_mag {
        *v /= sample_count as f32;
    }

    let alpha = 0.8 + strength * 2.2;
    let floor = 0.08 + (1.0 - strength) * 0.25;

    on_progress(0.2);
    let denoise_channel = |samples: &mut Vec<f32>| {
        let mut p = RealFftPlanner::<f32>::new();
        let fwd = p.plan_fft_forward(fft_size);
        let inv = p.plan_fft_inverse(fft_size);
        let mut inp = fwd.make_input_vec();
        let mut spec = fwd.make_output_vec();
        let mut out_time = inv.make_output_vec();

        let mut out = vec![0.0_f32; len];
        let mut norm = vec![0.0_f32; len];
        let scale = 1.0 / fft_size as f32;

        let mut pos = 0usize;
        while pos + fft_size <= len {
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }
            for k in 0..fft_size {
                inp[k] = samples[pos + k] * hann[k];
            }
            let _ = fwd.process(&mut inp, &mut spec);
            for b in 0..bins {
                let mag = spec[b].norm();
                let n = noise_mag[b] * alpha;
                let gain = ((mag - n) / mag.max(1e-7)).clamp(floor, 1.0);
                spec[b] *= gain;
            }
            // RealFFT requires DC and Nyquist imaginary parts to be 0
            spec[0].im = 0.0;
            if let Some(last) = spec.last_mut() {
                last.im = 0.0;
            }
            let _ = inv.process(&mut spec, &mut out_time);

            for k in 0..fft_size {
                let w = hann[k];
                out[pos + k] += out_time[k] * scale * w;
                norm[pos + k] += w * w;
            }
            pos += hop;
        }

        for i in 0..len {
            if norm[i] > 1e-6 {
                samples[i] = out[i] / norm[i];
            }
        }
    };

    join(|| denoise_channel(&mut audio.left), || denoise_channel(&mut audio.right));
    on_progress(1.0);
}

/// Biquad IIR filter coefficients & state (Direct Form II Transposed).
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn notch(sr: f32, freq: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * freq / sr;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: 1.0 / a0,
            b1: (-2.0 * w0.cos()) / a0,
            b2: 1.0 / a0,
            a1: (-2.0 * w0.cos()) / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn peaking(sr: f32, freq: f32, q: f32, gain_db: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq / sr;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        Self {
            b0: (1.0 + alpha * a) / a0,
            b1: (-2.0 * w0.cos()) / a0,
            b2: (1.0 - alpha * a) / a0,
            a1: (-2.0 * w0.cos()) / a0,
            a2: (1.0 - alpha / a) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Strips 50/60 Hz mains hum (+ 2nd/3rd harmonics) and interpolates vinyl crackle impulses.
fn apply_hum_notch_and_declick(audio: &mut StereoAudioBuffer, hum_hz: f32) {
    let sr = audio.sample_rate as f32;
    let process_ch = |samples: &mut Vec<f32>| {
        let mut n1 = Biquad::notch(sr, hum_hz, 24.0);
        let mut n2 = Biquad::notch(sr, hum_hz * 2.0, 24.0);
        let mut n3 = Biquad::notch(sr, hum_hz * 3.0, 24.0);

        for s in samples.iter_mut() {
            *s = n3.process(n2.process(n1.process(*s)));
        }

        // Simple 5-sample median impulse de-clicker for sharp vinyl pops
        let len = samples.len();
        if len > 6 {
            let orig = samples.clone();
            for i in 2..len - 2 {
                let mut win = [orig[i - 2], orig[i - 1], orig[i], orig[i + 1], orig[i + 2]];
                win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = win[2];
                let diff = (orig[i] - med).abs();
                let local_spread = (win[3] - win[1]).abs().max(0.01);
                if diff > local_spread * 6.5 && diff > 0.12 {
                    samples[i] = 0.5 * (orig[i - 1] + orig[i + 1]);
                }
            }
        }
    };
    join(|| process_ch(&mut audio.left), || process_ch(&mut audio.right));
}

/// 5-Band Mastering Parametric EQ ([60Hz, 250Hz, 1kHz, 4kHz, 12kHz]).
fn apply_5band_mastering_eq(audio: &mut StereoAudioBuffer, gains_db: [f32; 5]) {
    let sr = audio.sample_rate as f32;
    let freqs = [60.0_f32, 250.0, 1000.0, 4000.0, 12000.0];
    let process_ch = |samples: &mut Vec<f32>| {
        let mut bands: Vec<Biquad> = freqs
            .iter()
            .zip(gains_db.iter())
            .filter(|(_, &g)| g.abs() > 0.05)
            .map(|(&f, &g)| Biquad::peaking(sr, f, 0.9, g))
            .collect();
        for s in samples.iter_mut() {
            let mut v = *s;
            for b in &mut bands {
                v = b.process(v);
            }
            *s = v.clamp(-1.0, 1.0);
        }
    };
    join(|| process_ch(&mut audio.left), || process_ch(&mut audio.right));
}

/// Reconstructs upper-frequency air & presence via harmonic excitation + shelf EQ.
fn harmonic_bandwidth_reconstruct(
    audio: &mut StereoAudioBuffer,
    strength: f32,
    vocal_focus: bool,
) {
    let sr = audio.sample_rate as f32;
    let len = audio.len();
    if len < 4 {
        return;
    }

    // 1-pole high-pass at ~3.5 kHz to extract upper-mid harmonics,
    // apply soft odd/even harmonic saturation, and blend back.
    let cutoff = 3500.0_f32;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    let dt = 1.0 / sr;
    let alpha_hp = rc / (rc + dt);

    let drive = 1.5 + strength * 2.5;
    let mix = strength * 0.28;
    let presence_boost = if vocal_focus { 1.18 } else { 1.06 };

    for ch in 0..2 {
        let samples = if ch == 0 {
            &mut audio.left
        } else {
            &mut audio.right
        };
        let mut prev_in = samples[0];
        let mut prev_hp = 0.0_f32;

        for s in samples.iter_mut() {
            let hp = alpha_hp * (prev_hp + *s - prev_in);
            prev_in = *s;
            prev_hp = hp;

            // Asymmetric soft clipper generates both 2nd and 3rd harmonics
            let excited = (hp * drive).tanh() + 0.15 * (hp * hp * drive);
            *s = (*s * presence_boost + excited * mix).clamp(-1.0, 1.0);
        }
    }
}

/// Reconstructs a wide stereo / 3D binaural soundstage from mono or flat stereo.
fn apply_spatial_reconstruction(
    audio: &mut StereoAudioBuffer,
    mode: SpatialMode,
    width: f32,
    models_dir: &Path,
) {
    let len = audio.len();
    let sr = audio.sample_rate as usize;
    if len == 0 {
        return;
    }

    match mode {
        SpatialMode::OriginalStereo => {
            // Subtle M/S polish
            let w = 1.0 + width * 0.2;
            for i in 0..len {
                let m = 0.5 * (audio.left[i] + audio.right[i]);
                let s = 0.5 * (audio.left[i] - audio.right[i]) * w;
                audio.left[i] = (m + s).clamp(-1.0, 1.0);
                audio.right[i] = (m - s).clamp(-1.0, 1.0);
            }
        }
        SpatialMode::WideStereo => {
            // Frequency-aware stereo widening + subtle Haas micro-delay decorrelation
            let delay_samples = (sr as f32 * 0.008) as usize; // ~8 ms
            let decorr_mix = width * 0.22;
            let side_gain = 1.0 + width * 0.85;

            let orig_l = audio.left.clone();
            let orig_r = audio.right.clone();

            for i in 0..len {
                let m = 0.5 * (orig_l[i] + orig_r[i]);
                let s = 0.5 * (orig_l[i] - orig_r[i]) * side_gain;

                let delayed_diff = if i >= delay_samples {
                    0.5 * (orig_l[i - delay_samples] - orig_r[i - delay_samples])
                } else {
                    0.0
                };

                // Add pseudo-stereo side energy if source was mono
                let mono_side = if i >= delay_samples {
                    (orig_l[i] - orig_l[i - delay_samples]) * decorr_mix * 0.5
                } else {
                    0.0
                };

                audio.left[i] = (m + s + delayed_diff * decorr_mix + mono_side).clamp(-1.0, 1.0);
                audio.right[i] = (m - s - delayed_diff * decorr_mix - mono_side).clamp(-1.0, 1.0);
            }
        }
        SpatialMode::BinauralSurround => {
            // Multi-tap early-reflection surround matrix + Haas binaural cues
            let tap1 = (sr as f32 * 0.011) as usize; // 11 ms
            let tap2 = (sr as f32 * 0.023) as usize; // 23 ms
            let tap3 = (sr as f32 * 0.037) as usize; // 37 ms

            let orig_l = audio.left.clone();
            let orig_r = audio.right.clone();
            let side_gain = 1.0 + width * 1.1;
            let room_mix = 0.14 + width * 0.18;

            for i in 0..len {
                let m = 0.5 * (orig_l[i] + orig_r[i]);
                let s = 0.5 * (orig_l[i] - orig_r[i]) * side_gain;

                let r1 = if i >= tap1 { orig_r[i - tap1] } else { 0.0 };
                let l1 = if i >= tap1 { orig_l[i - tap1] } else { 0.0 };
                let r2 = if i >= tap2 { orig_l[i - tap2] } else { 0.0 };
                let l2 = if i >= tap2 { orig_r[i - tap2] } else { 0.0 };
                let r3 = if i >= tap3 { orig_r[i - tap3] } else { 0.0 };
                let l3 = if i >= tap3 { orig_l[i - tap3] } else { 0.0 };

                let surround_l = (r1 * 0.45 - l2 * 0.35 + l3 * 0.25) * room_mix;
                let surround_r = (-l1 * 0.45 + r2 * 0.35 - r3 * 0.25) * room_mix;

                audio.left[i] = (m + s + surround_l).clamp(-1.0, 1.0);
                audio.right[i] = (m - s + surround_r).clamp(-1.0, 1.0);
            }

            // Apply pure-Rust HRTF 3D binaural virtual speaker convolution
            apply_hrtf_virtual_speakers(audio, width, models_dir);
        }
    }
}

/// Loads `models/IRC_1002_C.bin` if present, or synthesizes a 6-point acoustic HRIR sphere.
fn load_or_build_hrir_sphere(sample_rate: u32, models_dir: &Path) -> Option<HrirSphere> {
    let ircam_path = models_dir.join("IRC_1002_C.bin");
    if ircam_path.exists() {
        if let Ok(sphere) = HrirSphere::from_file(&ircam_path, sample_rate) {
            return Some(sphere);
        }
    }

    // Build an in-memory 6-vertex octahedron HRIR sphere (64 samples @ 44100 Hz)
    let sr: u32 = 44_100;
    let ir_len: usize = 64;
    let vertices: [(f32, f32, f32); 6] = [
        (1.0, 0.0, 0.0),  // Right
        (-1.0, 0.0, 0.0), // Left
        (0.0, 1.0, 0.0),  // Top
        (0.0, -1.0, 0.0), // Bottom
        (0.0, 0.0, 1.0),  // Front
        (0.0, 0.0, -1.0), // Back
    ];
    let faces: [u32; 24] = [
        4, 0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 5, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3,
    ];

    let mut buf = Vec::with_capacity(4096);
    buf.extend_from_slice(b"HRIR");
    buf.extend_from_slice(&sr.to_le_bytes());
    buf.extend_from_slice(&(ir_len as u32).to_le_bytes());
    buf.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(faces.len() as u32).to_le_bytes());
    for &idx in &faces {
        buf.extend_from_slice(&idx.to_le_bytes());
    }

    for &(x, y, z) in &vertices {
        buf.extend_from_slice(&x.to_le_bytes());
        buf.extend_from_slice(&y.to_le_bytes());
        buf.extend_from_slice(&z.to_le_bytes());

        // Acoustic ITD (~0.5ms max) and ILD head-shadowing
        let delay_l = ((1.0 + x) * 11.0).round() as usize;
        let delay_r = ((1.0 - x) * 11.0).round() as usize;
        let gain_l = 0.85 - x * 0.15;
        let gain_r = 0.85 + x * 0.15;

        for ear in 0..2 {
            let delay = if ear == 0 { delay_l } else { delay_r };
            let gain = if ear == 0 { gain_l } else { gain_r };
            for s in 0..ir_len {
                let v = if s == delay.min(ir_len - 1) {
                    gain
                } else if s == (delay + 1).min(ir_len - 1) {
                    gain * 0.25
                } else {
                    0.0
                };
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    HrirSphere::new(Cursor::new(buf), sample_rate).ok()
}

/// Applies `hrtf::HrtfProcessor` virtual speaker rendering to enhance 3D headphone immersion.
fn apply_hrtf_virtual_speakers(audio: &mut StereoAudioBuffer, width: f32, models_dir: &Path) {
    let Some(sphere) = load_or_build_hrir_sphere(audio.sample_rate, models_dir) else {
        return;
    };

    let block_len = 512;
    let interp_steps = 4;
    let slice_len = block_len * interp_steps; // 2048 samples
    let len = audio.len();
    if len < slice_len {
        return;
    }

    let mut proc_l = HrtfProcessor::new(sphere.clone(), interp_steps, block_len);
    let mut proc_r = HrtfProcessor::new(sphere, interp_steps, block_len);

    let mut prev_l_l = Vec::new();
    let mut prev_l_r = Vec::new();
    let mut prev_r_l = Vec::new();
    let mut prev_r_r = Vec::new();

    let vec_left = Vec3::new(-0.85, 0.05, 0.52);
    let vec_right = Vec3::new(0.85, 0.05, 0.52);
    let wet = (0.15 + width * 0.25).clamp(0.0, 0.45);
    let dry = 1.0 - wet * 0.5;

    let mut out_l = vec![(0.0_f32, 0.0_f32); slice_len];
    let mut out_r = vec![(0.0_f32, 0.0_f32); slice_len];

    for start in (0..len).step_by(slice_len) {
        let end = (start + slice_len).min(len);
        let cur = end - start;
        if cur < slice_len {
            break;
        }

        out_l.fill((0.0, 0.0));
        out_r.fill((0.0, 0.0));

        proc_l.process_samples(HrtfContext {
            source: &audio.left[start..end],
            output: &mut out_l,
            new_sample_vector: vec_left,
            prev_sample_vector: vec_left,
            prev_left_samples: &mut prev_l_l,
            prev_right_samples: &mut prev_l_r,
            new_distance_gain: 1.0,
            prev_distance_gain: 1.0,
        });

        proc_r.process_samples(HrtfContext {
            source: &audio.right[start..end],
            output: &mut out_r,
            new_sample_vector: vec_right,
            prev_sample_vector: vec_right,
            prev_left_samples: &mut prev_r_l,
            prev_right_samples: &mut prev_r_r,
            new_distance_gain: 1.0,
            prev_distance_gain: 1.0,
        });

        for i in 0..slice_len {
            let hrtf_l = out_l[i].0 + out_r[i].0;
            let hrtf_r = out_l[i].1 + out_r[i].1;
            audio.left[start + i] = (audio.left[start + i] * dry + hrtf_l * wet).clamp(-1.0, 1.0);
            audio.right[start + i] = (audio.right[start + i] * dry + hrtf_r * wet).clamp(-1.0, 1.0);
        }
    }
}

fn apply_ebur128_mastering(audio: &mut StereoAudioBuffer, target_lufs: f64) {
    let len = audio.len();
    if len == 0 {
        return;
    }

    if let Ok(mut ebu) = EbuR128::new(2, audio.sample_rate, Mode::I | Mode::TRUE_PEAK) {
        let mut interleaved = Vec::with_capacity(len * 2);
        for i in 0..len {
            interleaved.push(audio.left[i]);
            interleaved.push(audio.right[i]);
        }
        if ebu.add_frames_f32(&interleaved).is_ok() {
            if let Ok(lufs) = ebu.loudness_global() {
                if lufs.is_finite() && lufs > -60.0 {
                    let gain_db = (target_lufs - lufs).clamp(-12.0, 12.0);
                    let gain = 10.0_f32.powf((gain_db / 20.0) as f32);
                    for i in 0..len {
                        audio.left[i] = (audio.left[i] * gain).tanh();
                        audio.right[i] = (audio.right[i] * gain).tanh();
                    }
                }
            }
        }
    }
    normalize_headroom(audio, 0.94);
}

fn normalize_headroom(audio: &mut StereoAudioBuffer, target_peak: f32) {
    let mut peak = 0.0_f32;
    for (&l, &r) in audio.left.iter().zip(audio.right.iter()) {
        peak = peak.max(l.abs()).max(r.abs());
    }
    if peak > 1e-6 && peak > target_peak {
        let gain = target_peak / peak;
        for l in &mut audio.left {
            *l *= gain;
        }
        for r in &mut audio.right {
            *r *= gain;
        }
    } else if peak > 1e-6 && peak < 0.7 {
        let gain = (target_peak / peak).min(3.0);
        for l in &mut audio.left {
            *l *= gain;
        }
        for r in &mut audio.right {
            *r *= gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;
    use std::path::PathBuf;

    #[test]
    fn dsp_fallback_pipeline_reconstructs_stereo() {
        let sr = 44_100;
        let n = sr; // 1 second
        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let tone = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.4;
            left.push(tone);
            right.push(tone);
        }
        let input = StereoAudioBuffer {
            left,
            right,
            sample_rate: sr as u32,
            original_channels: 1,
            original_sample_rate: sr as u32,
        };

        let mut cfg = RestorationConfig::default();
        cfg.apply_preset(EnhancementPreset::Surround3D);
        let inv = ModelInventory::scan(&PathBuf::from("nonexistent_models_dir"));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = unbounded();

        RestorationPipeline::run(input, cfg, inv, cancel, tx);

        let mut completed = None;
        while let Ok(msg) = rx.try_recv() {
            if let PipelineProgress::Completed(res) = msg {
                completed = Some(res);
            }
        }
        let res = completed.expect("Pipeline should complete in DSP fallback mode");
        assert_eq!(res.restored_audio.sample_rate, 44_100);
        assert!(res.restored_analysis.stereo_correlation < 0.99);
        assert!(res.restored_analysis.is_true_stereo());
    }
}
