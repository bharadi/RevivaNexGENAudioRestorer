# Reviva NexGEN Audio Restorer — Desktop Studio (Windows `.exe` in Rust)

Native Windows desktop application written in pure Rust (`egui` / `eframe`) for restoring noisy retro MP3/audio files into clean, mastered stereo or 3D binaural surround audio using desktop-scale deep learning models (`ort` + DirectML GPU acceleration) and pure-Rust DSP.

## Features

- **Pure Native Windows `.exe`** using `eframe` (`egui`) — no Electron/WebView overhead.
- **Universal Audio Decoding (`symphonia`)**: Drag & drop or browse MP3, WAV, FLAC, OGG, AAC/M4A files.
- **3-Stage Restoration + Loudness Mastering Pipeline**:
  1. **Deep Denoising / Stem Cleaning**: `BS-RoFormer` or `HTDemucs` via ONNX Runtime (DirectML GPU / CPU), with automatic STFT Wiener Spectral Denoiser fallback.
  2. **Bandwidth & Artifact Reconstruction**: `Apollo` music restoration via ONNX Runtime, with Harmonic Bandwidth Exciter fallback.
  3. **Stereo & Spatial Reconstruction**: Mid/Side widening, Haas micro-delay decorrelation, and 3D Binaural Surround synthesis.
  4. **EBU R128 Broadcast Loudness Mastering (`ebur128`)**: ITU-R BS.1770 loudness normalization to -14 LUFS with true-peak soft limiting.
- **Real-Time A/B Comparison Player (`rodio`)**: Switch instantaneously between **Original** and **Restored** audio at the exact same timestamp, with interactive waveform & spectrogram heatmaps.
- **High-Quality VBR MP3 & Lossless WAV Export**:
  - **MP3 VBR V0** (~245 kbps Extreme Quality via bundled LAME 3.100)
  - **MP3 VBR V2** (~190 kbps High Quality via LAME)
  - **MP3 CBR 320 kbps**
  - **WAV 24-bit Lossless** & **WAV 16-bit CD Quality**

## Open-Source Surround & Enhancement Libraries (Dolby / DTS Alternatives)

Proprietary **Dolby Digital (AC-3)**, **Dolby Atmos**, and **DTS** SDKs are closed-source commercial libraries. Reviva NexGEN Audio Restorer uses and supports open-source equivalents:

| Capability | Proprietary Equivalent | Open-Source Library / Implementation |
| :--- | :--- | :--- |
| **Broadcast Loudness & Normalization** | Dolby Volume / Dialog Norm | **`ebur128`** (ITU-R BS.1770 / EBU R128, *integrated*) |
| **Binaural / 3D Surround Soundstage** | Dolby Headphone / DTS Headphone:X | **Haas + Multi-tap Room Matrix** (*integrated*); optional **`hrtf`** or **Steam Audio (`audionimbus`)** |
| **Stereo Widening & Matrixing** | Dolby Pro Logic II | **M/S + Frequency-Aware Decorrelator** (*integrated*) |
| **High-Quality VBR MP3 Encoding** | — | **`mp3lame-encoder`** (LAME 3.100, *integrated*) |
| **AC-3 / E-AC-3 Container Codec** | Dolby Digital | **FFmpeg (`libavcodec` `ac3`)** *(optional external CLI/lib)* |

## Build & Run

```powershell
cd nexgen_desktop_rust
cargo test
cargo build --release
# Output executable:
# target\release\nexgen_desktop_rust.exe
```
