# Desktop ONNX Models (`models/`)

Place desktop-grade ONNX models next to `nexgen_desktop_rust.exe` inside a `models/` directory (and `onnxruntime.dll` from Microsoft's ONNX Runtime DirectML release for GPU acceleration).

## Supported Model Slots

1. **Stage 1 — Stem Separation & Denoising** (any one of):
   - `bs_roformer.onnx` *(recommended — Band-Split RoFormer)*
   - `htdemucs.onnx` *(Demucs v4 Hybrid Transformer)*
   - `demucs_v4.onnx`

2. **Stage 2 — Music Restoration & Bandwidth Extension** (any one of):
   - `apollo.onnx` *(recommended — Apollo Band-Sequence Music Restoration)*
   - `apollo_music_restoration.onnx`
   - `audiosr.onnx`

3. **Stage 3 — Neural Spatial Upmix** (optional):
   - `spatial_tcn.onnx`
   - `stereo_reconstructor.onnx`

> **Note:** If no `.onnx` files are present, the app automatically runs in **Spectral DSP Fallback Mode** (STFT Wiener Denoiser + Harmonic Bandwidth Exciter + M/S & Binaural Stereo Reconstructor) so you can test and export restored audio immediately.
