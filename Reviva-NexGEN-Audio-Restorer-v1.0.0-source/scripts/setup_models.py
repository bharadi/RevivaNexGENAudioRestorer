#!/usr/bin/env python3
"""
NexGen Music Restorer (Desktop Rust) — Model & DirectML Runtime Setup Helper
============================================================================
Prepares the `models/` directory next to `nexgen_desktop_rust.exe` with:
  1. Stage 1: BS-RoFormer or HTDemucs ONNX (stem separation & deep denoising)
  2. Stage 2: Apollo ONNX (music bandwidth & MP3 artifact reconstruction)
  3. onnxruntime.dll (ONNX Runtime DirectML for Windows GPU acceleration)

Usage:
  python scripts/setup_models.py
"""

import os
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def check_environment():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {MODELS_DIR}")

    expected = {
        "Stage 1 (Denoise / Stems)": ["bs_roformer.onnx", "htdemucs.onnx", "demucs_v4.onnx"],
        "Stage 2 (Bandwidth / Restore)": ["apollo.onnx", "apollo_music_restoration.onnx", "audiosr.onnx"],
        "Stage 3 (Neural Spatial, optional)": ["spatial_tcn.onnx", "stereo_reconstructor.onnx"],
        "DirectML Runtime": ["onnxruntime.dll"],
    }

    for stage, candidates in expected.items():
        found = [c for c in candidates if (MODELS_DIR / c).exists()]
        if found:
            size_mb = (MODELS_DIR / found[0]).stat().st_size / (1024 * 1024)
            print(f"  [OK] {stage}: {found[0]} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {stage}: place one of {candidates} in {MODELS_DIR}")

    print("\nNote: Even without .onnx weights, nexgen_desktop_rust.exe runs in")
    print("      Spectral DSP Fallback Mode (Wiener Denoiser + Harmonic Exciter + Spatializer).")


if __name__ == "__main__":
    check_environment()
