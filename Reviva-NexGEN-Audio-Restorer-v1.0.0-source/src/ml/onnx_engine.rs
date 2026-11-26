use anyhow::{Context, Result};
use ort::environment::Environment;
use ort::ep::DirectML;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static ORT_ENV: OnceLock<Environment> = OnceLock::new();

fn get_or_init_env(models_dir: &Path) -> Result<&'static Environment> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }
    let local_dll = models_dir.join("onnxruntime.dll");
    let exe_dll = PathBuf::from("onnxruntime.dll");
    let builder = if local_dll.exists() {
        ort::init_from(&local_dll)
            .map_err(|e| anyhow::anyhow!("Failed to load {}: {e}", local_dll.display()))?
    } else if exe_dll.exists() {
        ort::init_from(&exe_dll)
            .map_err(|e| anyhow::anyhow!("Failed to load onnxruntime.dll: {e}"))?
    } else {
        ort::init()
    };

    let env = builder
        .with_name("nexgen_desktop_rust")
        .build()
        .map_err(|e| anyhow::anyhow!("ONNX Runtime init failed: {e}"))?;
    let _ = ORT_ENV.set(env);
    Ok(ORT_ENV.get().expect("OnceLock initialized"))
}

/// Discovers available desktop ONNX models and HRIR spheres in the `models/` folder.
#[derive(Debug, Clone, Default)]
pub struct ModelInventory {
    pub models_dir: PathBuf,
    pub stem_denoise_model: Option<PathBuf>,
    pub apollo_recon_model: Option<PathBuf>,
    pub spatial_upmix_model: Option<PathBuf>,
    pub hrtf_sphere_file: Option<PathBuf>,
    pub ort_dll_present: bool,
}

impl ModelInventory {
    pub fn scan(models_dir: &Path) -> Self {
        let find_first = |candidates: &[&str]| -> Option<PathBuf> {
            for name in candidates {
                let p = models_dir.join(name);
                if p.exists() && p.is_file() {
                    return Some(p);
                }
            }
            None
        };

        let stem_denoise_model = find_first(&[
            "bs_roformer.onnx",
            "htdemucs.onnx",
            "demucs_v4.onnx",
            "deepfilternet3.onnx",
        ]);
        let apollo_recon_model = find_first(&[
            "apollo.onnx",
            "apollo_music_restoration.onnx",
            "audiosr.onnx",
        ]);
        let spatial_upmix_model = find_first(&[
            "spatial_tcn.onnx",
            "stereo_reconstructor.onnx",
        ]);
        let hrtf_sphere_file = find_first(&["IRC_1002_C.bin"]);

        let ort_dll_present = Path::new("onnxruntime.dll").exists()
            || models_dir.join("onnxruntime.dll").exists();

        Self {
            models_dir: models_dir.to_path_buf(),
            stem_denoise_model,
            apollo_recon_model,
            spatial_upmix_model,
            hrtf_sphere_file,
            ort_dll_present,
        }
    }

    pub fn any_onnx_ready(&self) -> bool {
        self.stem_denoise_model.is_some()
            || self.apollo_recon_model.is_some()
            || self.spatial_upmix_model.is_some()
    }
}

/// Wrapper around an `ort::Session` configured for DirectML (Windows GPU) or CPU.
pub struct OnnxBackend {
    session: Session,
    pub using_directml: bool,
}

impl OnnxBackend {
    /// Attempts to initialize ONNX Runtime dynamically and load `model_path`.
    pub fn load(model_path: &Path, use_directml: bool) -> Result<Self> {
        let models_dir = model_path.parent().unwrap_or_else(|| Path::new("models"));
        let env = get_or_init_env(models_dir)?;

        let mut builder = Session::builder(env)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut using_directml = false;
        if use_directml {
            if let Ok(b) = Session::builder(env) {
                if let Ok(b) = b.with_optimization_level(GraphOptimizationLevel::Level3) {
                    if let Ok(b) = b.with_execution_providers([DirectML::default().build()]) {
                        builder = b;
                        using_directml = true;
                    }
                }
            }
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model {}: {e}", model_path.display()))?;

        Ok(Self {
            session,
            using_directml,
        })
    }

    /// Runs chunked 3D stereo waveform inference: `[1, 2, samples]` -> `[1, 2, samples]`,
    /// scaling 4-stem outputs `[1, 4, 2, samples]` by `stem_gains`.
    pub fn run_stereo_chunk(
        &mut self,
        left: &[f32],
        right: &[f32],
        stem_gains: [f32; 4],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = left.len().min(right.len());
        let mut flat = Vec::with_capacity(2 * n);
        flat.extend_from_slice(&left[..n]);
        flat.extend_from_slice(&right[..n]);

        let input_tensor = Tensor::from_array(([1_usize, 2, n], flat))
            .context("Failed to create input ONNX tensor")?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .context("ONNX inference execution failed")?;

        let (_, out_val) = outputs
            .iter()
            .next()
            .context("ONNX model produced no outputs")?;
        let (shape, data) = out_val.try_extract_tensor::<f32>()?;

        // Handle [1, 2, N] or [1, stems, 2, N] (e.g., Demucs summing stems)
        let mut out_l = vec![0.0_f32; n];
        let mut out_r = vec![0.0_f32; n];

        if shape.len() == 3 && shape[1] >= 2 {
            let stride = shape[2] as usize;
            let out_len = stride.min(n);
            out_l[..out_len].copy_from_slice(&data[..out_len]);
            out_r[..out_len].copy_from_slice(&data[stride..stride + out_len]);
        } else if shape.len() == 4 && shape[2] >= 2 {
            // Sum all stems with per-stem mixer gains: [1, stems, 2, N]
            let stems = shape[1] as usize;
            let stride = shape[3] as usize;
            let out_len = stride.min(n);
            for s in 0..stems {
                let g = stem_gains.get(s).copied().unwrap_or(1.0);
                let base_l = (s * 2) * stride;
                let base_r = (s * 2 + 1) * stride;
                for i in 0..out_len {
                    out_l[i] += data[base_l + i] * g;
                    out_r[i] += data[base_r + i] * g;
                }
            }
        } else {
            // Fallback passthrough if tensor shape differs
            out_l.copy_from_slice(&left[..n]);
            out_r.copy_from_slice(&right[..n]);
        }

        Ok((out_l, out_r))
    }
}
