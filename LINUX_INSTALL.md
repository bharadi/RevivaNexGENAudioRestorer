# Linux Installation & Build Guide — Reviva NexGEN Audio Restorer (v1.0.0)

This guide explains how to build and install **Reviva NexGEN Audio Restorer** from source on **Ubuntu / Debian**, **Fedora**, and **Arch Linux**.

---

## 1. Install System Dependencies

### Ubuntu / Debian (22.04 / 24.04+)
```bash
sudo apt update
sudo apt install -y \
  build-essential pkg-config cmake nasm \
  libasound2-dev libgtk-3-dev libxkbcommon-dev \
  libssl-dev libegl1-mesa-dev libgl1-mesa-dev \
  libwayland-dev libxrandr-dev libxi-dev libxcursor-dev
```

### Fedora (39 / 40+)
```bash
sudo dnf install -y \
  gcc gcc-c++ make pkgconfig cmake nasm \
  alsa-lib-devel gtk3-devel libxkbcommon-devel \
  openssl-devel mesa-libGL-devel mesa-libEGL-devel \
  wayland-devel libXrandr-devel libXi-devel libXcursor-devel
```

### Arch Linux / Manjaro
```bash
sudo pacman -S --needed \
  base-devel pkgconf cmake nasm \
  alsa-lib gtk3 libxkbcommon openssl mesa \
  wayland libxrandr libxi libxcursor
```

> *(Optional)* Install `ffmpeg` if you want **AC-3 (Dolby Digital-compatible)** export in addition to the built-in MP3 VBR, FLAC, and WAV encoders:
> ```bash
> sudo apt install -y ffmpeg   # or dnf/pacman equivalent
> ```

---

## 2. Install Rust Toolchain (if not installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

---

## 3. Build & Run from Source

```bash
git clone https://github.com/<your-username>/reviva-nexgen-audio-restorer.git
cd reviva-nexgen-audio-restorer

# Run unit tests
cargo test

# Build optimized release binary
cargo build --release

# Run the application
./target/release/nexgen_desktop_rust
```

---

## 4. Install System-Wide & Add Desktop Launcher (Optional)

```bash
# Install binary and icon
sudo install -Dm755 target/release/nexgen_desktop_rust /usr/local/bin/reviva-nexgen-audio-restorer
sudo install -Dm644 assets/app_icon.png /usr/local/share/icons/hicolor/256x256/apps/reviva-nexgen.png

# Create .desktop entry
cat << 'EOF' | sudo tee /usr/local/share/applications/reviva-nexgen.desktop
[Desktop Entry]
Name=Reviva NexGEN Audio Restorer
Comment=AI Retro Audio Restoration & Stereo Reconstruction
Exec=/usr/local/bin/reviva-nexgen-audio-restorer
Icon=reviva-nexgen
Terminal=false
Type=Application
Categories=AudioVideo;Audio;AudioVideoEditing;
EOF
```

---

## 5. Optional ONNX Runtime (`libonnxruntime.so`)

The app works **100% offline out of the box** using its multi-core Rust DSP engine (`realfft`, `rayon`, `ebur128`, `hrtf`).  
To enable neural ONNX models (`deepfilternet3.onnx`, `htdemucs.onnx`, `bs_roformer.onnx`) on Linux:

1. Download the Linux x64 **ONNX Runtime** release (`onnxruntime-linux-x64-*.tgz`) from Microsoft's official GitHub Releases.
2. Copy `libonnxruntime.so` into the `models/` folder next to the binary (or `/usr/local/lib/` and run `sudo ldconfig`).
3. Launch the app and use the **Settings & Models** tab to download models with 1 click.
