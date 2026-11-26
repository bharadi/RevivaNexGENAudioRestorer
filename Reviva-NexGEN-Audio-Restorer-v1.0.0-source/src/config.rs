use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Preset modes tailored for different retro audio restoration goals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnhancementPreset {
    /// Gentle hiss/hum removal + MP3 artifact repair, preserving vintage warmth.
    CleanNatural,
    /// Full stem separation (HTDemucs / BS-RoFormer) + Apollo bandwidth reconstruction + stereo widening.
    StudioRemaster,
    /// Reconstructs wide 3D/surround-like binaural soundstage from mono or flat stereo.
    Surround3D,
    /// Emphasizes vocal clarity & presence while suppressing background tape/room noise.
    VocalFocus,
}

impl EnhancementPreset {
    pub fn label(&self) -> &'static str {
        match self {
            Self::CleanNatural => "Clean & Natural",
            Self::StudioRemaster => "Studio Remaster",
            Self::Surround3D => "3D Surround",
            Self::VocalFocus => "Vocal Focus",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::CleanNatural => "Removes tape hiss & hum while preserving original analog warmth.",
            Self::StudioRemaster => "Full stem cleaning + Apollo high-frequency reconstruction + wide stereo.",
            Self::Surround3D => "Reconstructs a spacious, surround-like soundstage from mono/flat stereo.",
            Self::VocalFocus => "Isolates and clarifies lead vocals while cleaning backing instruments.",
        }
    }
}

/// Spatial output mode for the reconstructed stereo file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialMode {
    OriginalStereo,
    WideStereo,
    BinauralSurround,
}

impl SpatialMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::OriginalStereo => "Original Stereo",
            Self::WideStereo => "Wide Stereo",
            Self::BinauralSurround => "3D Binaural Surround",
        }
    }
}

/// Output export format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Mp3VbrV0,
    Mp3VbrV2,
    Mp3Cbr320,
    Flac24Bit,
    Wav24Bit,
    Wav16Bit,
    Ac3DolbyCompatible,
}

impl OutputFormat {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Mp3VbrV0 => "MP3 VBR V0 (~245 kbps Extreme)",
            Self::Mp3VbrV2 => "MP3 VBR V2 (~190 kbps High)",
            Self::Mp3Cbr320 => "MP3 CBR 320 kbps",
            Self::Flac24Bit => "FLAC (24-bit Lossless + Tags)",
            Self::Wav24Bit => "WAV (24-bit Lossless)",
            Self::Wav16Bit => "WAV (16-bit CD Quality)",
            Self::Ac3DolbyCompatible => "AC-3 448 kbps (Dolby Digital via FFmpeg)",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp3VbrV0 | Self::Mp3VbrV2 | Self::Mp3Cbr320 => "mp3",
            Self::Flac24Bit => "flac",
            Self::Wav24Bit | Self::Wav16Bit => "wav",
            Self::Ac3DolbyCompatible => "ac3",
        }
    }

    pub fn is_mp3(&self) -> bool {
        matches!(self, Self::Mp3VbrV0 | Self::Mp3VbrV2 | Self::Mp3Cbr320)
    }
}

/// User-configurable restoration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestorationConfig {
    pub preset: EnhancementPreset,
    pub enable_hum_declick: bool,
    pub hum_freq_hz: f32,             // 50.0 or 60.0 Hz mains hum
    pub enable_stem_denoise: bool,
    pub enable_bandwidth_recon: bool,
    pub enable_spatial_upmix: bool,
    pub enable_loudness_mastering: bool,
    pub eq_gains_db: [f32; 5],        // [60Hz, 250Hz, 1kHz, 4kHz, 12kHz] in -6..+6 dB
    pub stem_gains: [f32; 4],         // [Vocals, Drums, Bass, Other] for 4-stem ONNX models
    pub denoise_strength: f32,        // 0.0 .. 1.0
    pub reconstruction_strength: f32, // 0.0 .. 1.0
    pub stereo_width: f32,            // 0.0 .. 1.0
    pub spatial_mode: SpatialMode,
    pub use_gpu_directml: bool,
    pub preserve_metadata: bool,
    pub output_format: OutputFormat,
    /// Pluggable model overrides (filename in models/ folder, None = auto-pick best installed)
    pub selected_denoise_model: Option<String>,
    pub selected_recon_model: Option<String>,
    pub selected_spatial_model: Option<String>,
}

impl Default for RestorationConfig {
    fn default() -> Self {
        Self {
            preset: EnhancementPreset::StudioRemaster,
            enable_hum_declick: true,
            hum_freq_hz: 50.0,
            enable_stem_denoise: true,
            enable_bandwidth_recon: true,
            enable_spatial_upmix: true,
            enable_loudness_mastering: true,
            eq_gains_db: [1.5, 0.0, 0.0, 1.5, 2.0],
            stem_gains: [1.0, 1.0, 1.0, 1.0],
            denoise_strength: 0.75,
            reconstruction_strength: 0.70,
            stereo_width: 0.65,
            spatial_mode: SpatialMode::WideStereo,
            use_gpu_directml: true,
            preserve_metadata: true,
            output_format: OutputFormat::Mp3VbrV0,
            selected_denoise_model: None,
            selected_recon_model: None,
            selected_spatial_model: None,
        }
    }
}

impl RestorationConfig {
    pub fn load_or_default(path: &Path) -> Self {
        if let Ok(text) = fs::read_to_string(path) {
            if let Ok(cfg) = serde_json::from_str::<Self>(&text) {
                return cfg;
            }
        }
        Self::default()
    }

    pub fn save(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write(path, json);
        }
    }

    pub fn apply_preset(&mut self, preset: EnhancementPreset) {
        self.preset = preset;
        match preset {
            EnhancementPreset::CleanNatural => {
                self.enable_stem_denoise = true;
                self.enable_bandwidth_recon = true;
                self.enable_spatial_upmix = false;
                self.denoise_strength = 0.55;
                self.reconstruction_strength = 0.40;
                self.stereo_width = 0.25;
                self.spatial_mode = SpatialMode::OriginalStereo;
            }
            EnhancementPreset::StudioRemaster => {
                self.enable_stem_denoise = true;
                self.enable_bandwidth_recon = true;
                self.enable_spatial_upmix = true;
                self.denoise_strength = 0.75;
                self.reconstruction_strength = 0.75;
                self.stereo_width = 0.65;
                self.spatial_mode = SpatialMode::WideStereo;
            }
            EnhancementPreset::Surround3D => {
                self.enable_stem_denoise = true;
                self.enable_bandwidth_recon = true;
                self.enable_spatial_upmix = true;
                self.denoise_strength = 0.70;
                self.reconstruction_strength = 0.70;
                self.stereo_width = 0.90;
                self.spatial_mode = SpatialMode::BinauralSurround;
            }
            EnhancementPreset::VocalFocus => {
                self.enable_stem_denoise = true;
                self.enable_bandwidth_recon = true;
                self.enable_spatial_upmix = true;
                self.denoise_strength = 0.85;
                self.reconstruction_strength = 0.65;
                self.stereo_width = 0.50;
                self.spatial_mode = SpatialMode::WideStereo;
            }
        }
    }
}
