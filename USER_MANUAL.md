# Reviva NexGEN Audio Restorer — User Manual & Technical Specifications

**Version:** 1.0.0  
**Platform:** Windows 10 / Windows 11 (64-bit)  
**License:** MIT Open-Source License  
**Developer:** Dr. Vinayak Ashok Bharadi (`vinayagaurang@zohomail.in`)  
**Privacy:** 100% Local Offline Processing • Zero Data Collection

---

## 1. Introduction

**Reviva NexGEN Audio Restorer** is a native Windows desktop application built in **Rust** (`eframe`/`egui`) that restores old, noisy, mono, or low-bitrate retro MP3 and audio files into clean, wide stereo or 3D binaural studio-mastered audio—completely offline on your PC.

It combines **ONNX Runtime AI models** (with DirectML GPU acceleration) and a **multi-core Rust DSP engine** (`realfft`, `rayon`, `ebur128`, `hrtf`) so that every track is cleaned, frequency-extended, spatially reconstructed, and loudness-mastered without ever uploading your audio to the cloud.

---

## 2. System Requirements & Installation

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10 (64-bit) | Windows 11 (64-bit) |
| **CPU** | 64-bit Dual-Core x86_64 | Quad-Core+ (uses `rayon` multi-threading) |
| **RAM** | 4 GB | 8 GB+ |
| **GPU** | DirectX 12 compatible (or CPU fallback) | DirectML-compatible GPU (NVIDIA / AMD / Intel) |
| **Storage** | 100 MB free | 1 GB+ (if downloading additional ONNX models) |

### What Is Included in the Installer
Running `Reviva-NexGEN-Audio-Restorer-Setup-v1.0.0.exe` installs:
* `nexgen_desktop_rust.exe` (v1.0.0 standalone native binary)
* `models/deepfilternet3.onnx` (8.59 MB — bundled offline neural denoiser)
* `models/IRC_1002_C.bin` (772 KB — bundled IRCAM measured 3D HRTF sphere)
* `TERMS_OF_USE.txt` & `LICENSE` (MIT License)

> **100% Ready Offline:** Because `deepfilternet3.onnx`, `IRC_1002_C.bin`, and the multi-core Rust DSP engine are bundled in the installer, you can restore audio immediately with **no internet connection**.

---

## 3. User Manual — Studio Tab

### Step 1: Import an Audio File
You can load any **MP3, WAV, FLAC, OGG, M4A, or AAC** file using:
* **Drag & Drop** directly onto the application window
* **"Browse Files from Drive..."** button in the center drop zone
* **"Browse File..."** button in the top bar

Once loaded, the app automatically decodes and analyzes the track, displaying:
* **Waveform** (click anywhere to seek) and **Spectrogram Heatmap**
* **Stereo Channel Badge:** `((•)) 2.0 TRUE STEREO` (green) or `((•)) 1.0 MONO / DUAL-MONO` (amber), with live **Left/Right RMS (dB)**, **Side Energy (%)**, and **Phase Correlation**
* **Original Audio Details** & **Detected Issues** (e.g., *Elevated Tape Hiss*, *Band-Limited High Frequencies*, *Narrow / Mono Soundstage*)

### Step 2: Configure the 5 Color-Coded Pipeline Blocks (Left Sidebar)
1. **Enhancement Preset (Purple):**
   * **Clean & Natural:** Removes hiss & hum while preserving vintage analog warmth.
   * **Studio Remaster:** Full stem cleaning + bandwidth reconstruction + wide stereo.
   * **3D Surround:** Reconstructs a spacious, surround-like soundstage from mono or flat stereo.
   * **Vocal Focus:** Isolates and clarifies lead vocals while cleaning backing instruments.
2. **Clean & Denoise (Cyan):**
   * **Hum+Click (`50Hz` / `60Hz`):** Multi-harmonic IIR notch filters strip mains electrical hum (50/100/150 Hz or 60/120/180 Hz), while a 5-sample median impulse de-clicker removes vinyl crackle and pops.
   * **Stage 1: Deep Denoise / Stems:** Uses the active ONNX model (e.g., `deepfilternet3.onnx`, `htdemucs.onnx`, `bs_roformer.onnx`) or multi-core `realfft` STFT Wiener denoiser. Adjust strength with the **Denoise** slider (`0.0 – 1.0`).
3. **Reconstruct & Spatial (Blue):**
   * **Stage 2: Bandwidth Reconstruction:** Restores muffled high frequencies using ONNX restoration (e.g., `apollo.onnx`) or a harmonic bandwidth exciter. Adjust with the **Clarity** slider.
   * **Stage 3: Stereo & 3D HRTF:** Choose **Original Stereo**, **Wide Stereo** (Mid/Side + Haas micro-delay decorrelation), or **3D Binaural Surround** (multi-tap early-reflection matrix + `hrtf` Head-Related Transfer Function virtual speaker convolution). Adjust spread with the **Width** slider.
4. **Tone & Mixer (Amber):**
   * **5-Band EQ Tab:** Fine-tune `60Hz`, `250Hz`, `1kHz`, `4kHz`, and `12kHz` (`-6 dB` to `+6 dB`), or click **Reset**.
   * **4-Stem Tab:** Adjust `Vocals`, `Drums`, `Bass`, and `Other` (`0.0x – 2.0x`) when running 4-stem ONNX models (`HTDemucs` / `BS-RoFormer`).
5. **Mastering & Export (Emerald):**
   * **R128:** Enables ITU-R BS.1770 / EBU R128 broadcast loudness normalization to **-14 LUFS** with true-peak limiting.
   * **Tags:** Copies original ID3v2 / Vorbis metadata and album cover art to exported MP3/FLAC files.
   * **GPU:** Enables DirectML hardware acceleration for ONNX models (falls back to CPU automatically if unavailable).
   * **Format:** Choose **MP3 VBR V0 (~245 kbps Extreme)**, **MP3 VBR V2 (~190 kbps High)**, **MP3 CBR 320 kbps**, **FLAC (24-bit Lossless + Tags)**, **WAV (24-bit Lossless)**, **WAV (16-bit CD Quality)**, or **AC-3 448 kbps (Dolby Digital via FFmpeg)**.

### Step 3: Preview or Reconstruct Full Track
* **Preview 10s:** Processes a 10-second slice from the current playhead in ~1 second and auto-plays it so you can audition settings quickly.
* **Reconstruct & Enhance:** Processes the full audio track. While running, the active sidebar block highlights in cyan.

### Step 4: Compare & Verify Side-by-Side
* **A/B Player Controls:** Click **Play/Pause** (or press **`Space`**) and toggle **Listening: A (Original) ↔ B (Restored)** (or press **`Tab`**) at the exact same timestamp.
* **A–B Loop Markers:** Click **Set A** and **Set B** to loop a specific section repeatedly; green/amber markers appear on the waveform.
* **Before vs. After Improvement Report:** Shows the **Quality Score Improvement** banner (`Original -> Restored / 10`), side-by-side Original vs. Reconstructed metrics (`Noise Floor dB`, `Bandwidth kHz`, `Stereo Side %`, `Phase Correlation`), and **Issues Addressed & New Features**.

### Step 5: Export
Click **Export Reconstructed MP3 / Lossless FLAC / WAV / AC-3** to save the enhanced stereo file to disk.

---

## 4. Batch Queue Tab

Use the **Batch Queue** tab to restore multiple tracks or an entire music folder:
1. Click **Add Files...** or **Add Folder...** to queue MP3, WAV, FLAC, OGG, M4A, or AAC tracks.
2. Click **Process All & Export to Folder...** and select a destination folder.
3. Each track is restored using your active Studio preset, EQ, mastering, and export format settings, saved as `<original_name>_restored.<ext>`.

---

## 5. Settings & Models Tab (Pluggable AI Architecture)

* **Active Pipeline Model Slots:** Select `Auto` or explicitly choose any `.onnx` / `.bin` file inside `models/` for Stage 1 and Stage 2.
* **1-Click Model Downloader:** Download additional open-source models (`HTDemucs v4 FP16`, `BS-RoFormer Quantized`, `DeepFilterNet 3`, `IRCAM HRIR Sphere`) with a live progress bar and optional SHA-256 verification.
* **Add Custom / Future Models:** Add new model URLs via the form (stored in `models/registry.json`) or click **Import Local .onnx / .bin File...** to plug in future models without updating the application.

---

## 6. Technical Specifications

| Subsystem | Implementation / Crate | Details |
| :--- | :--- | :--- |
| **Language & Binary** | Rust (Edition 2021) | Native Windows 64-bit `.exe` (~13 MB), zero Electron/WebView |
| **GUI Framework** | `eframe` / `egui 0.29` | GPU-accelerated immediate-mode UI with dark neon theme |
| **Audio Decoding** | `symphonia 0.5` | MP3, WAV, FLAC, OGG (Vorbis), AAC, M4A/MP4 |
| **Resampling** | `rubato 0.15` | 128-tap Blackman-Harris sinc resampling to 44.1 kHz studio rate |
| **Spectral Analysis & DSP** | `realfft 3.4` + `rayon 1.10` | 2048-pt Hann STFT, multi-core L/R Wiener denoiser, 50/60 Hz biquad notches, median de-clicker, 5-band parametric EQ |
| **AI Inference** | `ort 2.0` (ONNX Runtime) | Dynamic DLL loading (`onnxruntime.dll`), DirectML Windows GPU + CPU fallback, chunked overlap-add (`[1, 2, N]` & `[1, 4, 2, N]`) |
| **Spatial & HRTF** | `hrtf 0.9` | Overlap-save frequency-domain HRTF convolution (IRCAM `.bin` or synthesized acoustic sphere) + M/S & Haas matrix |
| **Loudness Mastering** | `ebur128 0.1` | ITU-R BS.1770 / EBU R128 integrated loudness normalization (`-14.0 LUFS`) + headroom limiting |
| **Audio Playback** | `rodio 0.19` | Real-time A/B synchronized playback with seek & A–B looping |
| **Export Encoders** | `mp3lame-encoder`, `flacenc`, `hound` | LAME 3.100 (VBR V0/V2, CBR 320), 24-bit FLAC, 16/24-bit WAV, optional AC-3 via FFmpeg |
| **Metadata** | `lofty 0.22` | Preserves ID3v2 / Vorbis tags and album cover art |
| **Settings & Registry** | `serde` / `serde_json` | Persistent `models/settings.json` and `models/registry.json` |
| **Model Downloader** | `ureq 2.12` + `sha2 0.10` | Streaming HTTPS download with progress tracking & SHA-256 checksum verification |
| **Installer** | Inno Setup 6 | LZMA2/ultra64 compressed installer (`Reviva-NexGEN-Audio-Restorer-Setup-v1.0.0.exe`) |

---

## 7. Frequently Asked Questions

* **Why is an exported VBR MP3 sometimes reported as slightly longer or shorter in Windows Media Player?**  
  MP3 uses 1152-sample frames (~26 ms padding at the end), and some legacy players estimate VBR duration from the first frame instead of reading the Xing/LAME VBR header. The audio itself plays at the exact original tempo and length. Use **MP3 CBR 320**, **FLAC**, or **WAV** for exact sample-accurate duration reporting in every player.
* **Why does AC-3 export ask for FFmpeg?**  
  Proprietary Dolby/DTS encoders cannot be bundled directly. AC-3 export uses open-source FFmpeg if installed (`winget install Gyan.FFmpeg`). All MP3, FLAC, and WAV formats work natively with zero external tools.
* **Does the app send any data over the internet?**  
  No. Audio processing is 100% offline. Network connections only occur if you manually click **Download Model** in the **Settings & Models** tab.
