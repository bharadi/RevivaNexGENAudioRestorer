use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStage {
    DenoiseStem,
    BandwidthRecon,
    SpatialHrtf,
}

impl ModelStage {
    pub fn label(&self) -> &'static str {
        match self {
            Self::DenoiseStem => "Stage 1: Denoise / Stem Separation",
            Self::BandwidthRecon => "Stage 2: Bandwidth & Music Restoration",
            Self::SpatialHrtf => "Stage 3: Spatial / HRTF Sphere",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluggableModelEntry {
    pub id: String,
    pub name: String,
    pub stage: ModelStage,
    pub filename: String,
    pub download_url: String,
    pub description: String,
    pub approx_size_mb: u32,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: Vec<PluggableModelEntry>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            models: vec![
                PluggableModelEntry {
                    id: "htdemucs_fp16".to_string(),
                    name: "HTDemucs v4 FP16 (4-Stem Denoise)".to_string(),
                    stage: ModelStage::DenoiseStem,
                    filename: "htdemucs.onnx".to_string(),
                    download_url: "https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx".to_string(),
                    description: "Hybrid Transformer stem separator for removing tape hiss & background noise (158 MB).".to_string(),
                    approx_size_mb: 158,
                    sha256: None,
                },
                PluggableModelEntry {
                    id: "bs_roformer_u8".to_string(),
                    name: "BS-RoFormer Quantized (Studio Demix)".to_string(),
                    stage: ModelStage::DenoiseStem,
                    filename: "bs_roformer.onnx".to_string(),
                    download_url: "https://huggingface.co/xycld/BS-RoFormer-ONNX/resolve/main/bs_roformer_ep317_sdr12.9755_quantized_uint8.onnx".to_string(),
                    description: "Band-Split RoFormer for ultra-clean vocal & instrument separation (158 MB).".to_string(),
                    approx_size_mb: 158,
                    sha256: None,
                },
                PluggableModelEntry {
                    id: "deepfilternet3".to_string(),
                    name: "DeepFilterNet 3 (Fast Denoiser)".to_string(),
                    stage: ModelStage::DenoiseStem,
                    filename: "deepfilternet3.onnx".to_string(),
                    download_url: "https://huggingface.co/archgod94/tellcue-models/resolve/main/deepfilternet3.onnx".to_string(),
                    description: "Ultra-fast lightweight neural denoiser for quick cleanup (8.2 MB).".to_string(),
                    approx_size_mb: 8,
                    sha256: None,
                },
                PluggableModelEntry {
                    id: "ircam_hrtf_1002".to_string(),
                    name: "IRCAM Listen HRIR Sphere (1002_C)".to_string(),
                    stage: ModelStage::SpatialHrtf,
                    filename: "IRC_1002_C.bin".to_string(),
                    download_url: "https://github.com/mrDIMAS/hrir_sphere_builder/raw/master/hrtf_base/IRCAM/IRC_1002_C.bin".to_string(),
                    description: "Measured acoustic HRTF sphere for realistic 3D headphone surround sound (0.7 MB).".to_string(),
                    approx_size_mb: 1,
                    sha256: None,
                },
            ],
        }
    }
}

impl ModelRegistry {
    pub fn load_or_create(models_dir: &Path) -> Self {
        let _ = fs::create_dir_all(models_dir);
        let reg_path = models_dir.join("registry.json");
        if reg_path.exists() {
            if let Ok(text) = fs::read_to_string(&reg_path) {
                if let Ok(mut parsed) = serde_json::from_str::<ModelRegistry>(&text) {
                    // Migrate any old placeholder URLs if present
                    let has_placeholder = parsed
                        .models
                        .iter()
                        .any(|m| m.download_url.contains("RevivaNexGen/audio-models"));
                    if has_placeholder {
                        let def = Self::default();
                        parsed.models.retain(|m| !m.download_url.contains("RevivaNexGen/audio-models"));
                        for dm in def.models {
                            if !parsed.models.iter().any(|m| m.id == dm.id) {
                                parsed.models.push(dm);
                            }
                        }
                        let _ = parsed.save(models_dir);
                    }
                    return parsed;
                }
            }
        }
        let def = Self::default();
        let _ = def.save(models_dir);
        def
    }

    pub fn save(&self, models_dir: &Path) -> Result<()> {
        fs::create_dir_all(models_dir)?;
        let reg_path = models_dir.join("registry.json");
        let json = serde_json::to_string_pretty(self)?;
        fs::write(reg_path, json)?;
        Ok(())
    }

    /// Lists all `.onnx` and `.bin` files currently installed in `models_dir`.
    pub fn list_installed_files(models_dir: &Path) -> Vec<String> {
        let mut files = Vec::new();
        if let Ok(entries) = fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if ext.eq_ignore_ascii_case("onnx") || ext.eq_ignore_ascii_case("bin") {
                            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                                files.push(name.to_string());
                            }
                        }
                    }
                }
            }
        }
        files.sort();
        files
    }
}

#[derive(Debug, Clone)]
pub enum DownloadProgress {
    Progress {
        model_id: String,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
    Completed {
        model_id: String,
        path: PathBuf,
    },
    Failed {
        model_id: String,
        error: String,
    },
}

/// Downloads a pluggable model to `models_dir/<filename>` with optional SHA-256 verification.
pub fn download_model(
    entry: PluggableModelEntry,
    models_dir: PathBuf,
    tx: Sender<DownloadProgress>,
) {
    let run = || -> Result<PathBuf> {
        fs::create_dir_all(&models_dir)?;
        let dest_path = models_dir.join(&entry.filename);
        let tmp_path = models_dir.join(format!("{}.part", entry.filename));

        let resp = ureq::get(&entry.download_url)
            .call()
            .with_context(|| format!("HTTP request failed for {}", entry.download_url))?;

        let total_bytes = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok());

        let mut reader = resp.into_reader();
        let mut out = File::create(&tmp_path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0_u8; 64 * 1024];
        let mut downloaded: u64 = 0;

        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            downloaded += n as u64;

            let _ = tx.send(DownloadProgress::Progress {
                model_id: entry.id.clone(),
                downloaded_bytes: downloaded,
                total_bytes,
            });
        }
        out.flush()?;
        drop(out);

        if let Some(expected_hex) = &entry.sha256 {
            let actual_hex = format!("{:x}", hasher.finalize());
            if !actual_hex.eq_ignore_ascii_case(expected_hex.trim()) {
                let _ = fs::remove_file(&tmp_path);
                anyhow::bail!(
                    "SHA-256 mismatch: expected {}, got {}",
                    expected_hex,
                    actual_hex
                );
            }
        }

        fs::rename(&tmp_path, &dest_path)?;
        Ok(dest_path)
    };

    match run() {
        Ok(path) => {
            let _ = tx.send(DownloadProgress::Completed {
                model_id: entry.id,
                path,
            });
        }
        Err(e) => {
            let _ = tx.send(DownloadProgress::Failed {
                model_id: entry.id,
                error: format!("{e}"),
            });
        }
    }
}
