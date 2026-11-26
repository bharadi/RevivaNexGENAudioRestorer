use realfft::RealFftPlanner;

use super::io::StereoAudioBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub struct DetectedIssue {
    pub title: String,
    pub detail: String,
    pub severity: IssueSeverity,
}

#[derive(Debug, Clone)]
pub struct AudioAnalysis {
    pub duration_secs: f32,
    pub sample_rate: u32,
    pub original_channels: usize,
    pub noise_floor_db: f32,
    pub effective_bandwidth_hz: f32,
    pub stereo_correlation: f32, // 1.0 = mono/identical, 0.0 = wide stereo
    pub left_rms_db: f32,
    pub right_rms_db: f32,
    pub side_energy_pct: f32,    // % of energy in L-R difference (0% = mono, >5% = true stereo)
    pub quality_score: f32,      // 0..10
    pub waveform_peaks: Vec<f32>,
    pub spectrogram: Vec<Vec<f32>>, // [time][freq_bin] normalized 0..1
    pub issues: Vec<DetectedIssue>,
}

impl AudioAnalysis {
    pub fn is_true_stereo(&self) -> bool {
        self.stereo_correlation < 0.98 && self.side_energy_pct > 1.0
    }
}

/// Performs real DSP/FFT spectral analysis on the decoded stereo buffer.
pub fn analyze_audio(buf: &StereoAudioBuffer) -> AudioAnalysis {
    let len = buf.len();
    let sr = buf.sample_rate as f32;

    // 1. Waveform peaks (downsampled to ~320 bars for crisp UI rendering)
    let target_bars = 320.min(len.max(1));
    let step = (len / target_bars).max(1);
    let mut waveform_peaks = Vec::with_capacity(target_bars);
    for chunk_idx in 0..target_bars {
        let start = chunk_idx * step;
        let end = (start + step).min(len);
        let mut peak = 0.0_f32;
        for i in start..end {
            let v = (buf.left[i].abs() + buf.right[i].abs()) * 0.5;
            if v > peak {
                peak = v;
            }
        }
        waveform_peaks.push(peak.min(1.0));
    }

    // 2. Stereo correlation, L/R RMS, and Side-channel energy %
    let mut mid_sq = 0.0_f64;
    let mut side_sq = 0.0_f64;
    let mut left_sq = 0.0_f64;
    let mut right_sq = 0.0_f64;
    let mut count = 0usize;
    let stride = (len / 50_000).max(1);
    for i in (0..len).step_by(stride) {
        let l = buf.left[i] as f64;
        let r = buf.right[i] as f64;
        let m = 0.5 * (l + r);
        let s = 0.5 * (l - r);
        mid_sq += m * m;
        side_sq += s * s;
        left_sq += l * l;
        right_sq += r * r;
        count += 1;
    }
    let count_f = count.max(1) as f64;
    let left_rms_db = 20.0 * ((left_sq / count_f).sqrt().max(1e-6)).log10() as f32;
    let right_rms_db = 20.0 * ((right_sq / count_f).sqrt().max(1e-6)).log10() as f32;
    let total_ms = mid_sq + side_sq;
    let side_energy_pct = if total_ms > 1e-12 {
        ((side_sq / total_ms) * 100.0).clamp(0.0, 100.0) as f32
    } else {
        0.0
    };
    let stereo_correlation = if total_ms > 1e-12 {
        ((mid_sq - side_sq) / total_ms).clamp(-1.0, 1.0) as f32
    } else {
        1.0
    };

    // 3. Noise floor estimation using lowest-RMS 50ms windows
    let win_samples = ((sr * 0.05) as usize).max(256).min(len.max(1));
    let mut frame_rms = Vec::new();
    for start in (0..len).step_by(win_samples) {
        let end = (start + win_samples).min(len);
        if end - start < 64 {
            continue;
        }
        let mut sum = 0.0_f64;
        for i in start..end {
            let s = 0.5 * (buf.left[i] + buf.right[i]) as f64;
            sum += s * s;
        }
        let rms = (sum / (end - start) as f64).sqrt() as f32;
        if rms > 1e-7 {
            frame_rms.push(rms);
        }
    }
    frame_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quiet_rms = if frame_rms.is_empty() {
        1e-4
    } else {
        frame_rms[(frame_rms.len() / 20).min(frame_rms.len() - 1)]
    };
    let noise_floor_db = 20.0 * quiet_rms.max(1e-6).log10();

    // 4. Real-valued FFT Spectrogram + Effective Bandwidth Roll-off detection
    let fft_size = 2048;
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let nyquist_bins = fft_size / 2 + 1;

    let num_slices = 96.min((len / fft_size).max(1));
    let slice_stride = ((len.saturating_sub(fft_size)) / num_slices.max(1)).max(1);
    let display_bins = 48;

    let mut avg_spectrum = vec![0.0_f64; nyquist_bins];
    let mut spectrogram = Vec::with_capacity(num_slices);
    let mut in_buf = fft.make_input_vec();
    let mut out_buf = fft.make_output_vec();

    for s in 0..num_slices {
        let start = (s * slice_stride).min(len.saturating_sub(fft_size));
        for k in 0..fft_size {
            let idx = start + k;
            let sample = if idx < len {
                0.5 * (buf.left[idx] + buf.right[idx])
            } else {
                0.0
            };
            let w = 0.5
                * (1.0 - (2.0 * std::f32::consts::PI * k as f32 / (fft_size as f32 - 1.0)).cos());
            in_buf[k] = sample * w;
        }

        let _ = fft.process(&mut in_buf, &mut out_buf);

        let mut mags = vec![0.0_f32; nyquist_bins];
        for b in 0..nyquist_bins {
            let mag = out_buf[b].norm() / (fft_size as f32 * 0.5);
            mags[b] = mag;
            avg_spectrum[b] += mag as f64;
        }

        // Downsample bins to `display_bins` in log-dB scale for UI spectrogram
        let bins_per_group = (nyquist_bins / display_bins).max(1);
        let mut row = Vec::with_capacity(display_bins);
        for g in 0..display_bins {
            let b_start = g * bins_per_group;
            let b_end = (b_start + bins_per_group).min(nyquist_bins);
            let max_mag = mags[b_start..b_end]
                .iter()
                .copied()
                .fold(1e-7_f32, f32::max);
            let db = 20.0 * max_mag.log10();
            let norm = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
            row.push(norm);
        }
        spectrogram.push(row);
    }

    // Estimate effective bandwidth (highest bin with energy within 45 dB of peak)
    let peak_spec = avg_spectrum
        .iter()
        .copied()
        .fold(1e-12_f64, f64::max);
    let threshold = peak_spec * 10.0_f64.powf(-45.0 / 20.0);
    let mut max_active_bin = 0usize;
    for (b, &val) in avg_spectrum.iter().enumerate() {
        if val > threshold {
            max_active_bin = b;
        }
    }
    let effective_bandwidth_hz =
        ((max_active_bin as f32 / nyquist_bins as f32) * (sr * 0.5)).clamp(2000.0, sr * 0.5);

    // 5. Issue classification
    let mut issues = Vec::new();
    if noise_floor_db > -45.0 {
        issues.push(DetectedIssue {
            title: "Elevated Tape Hiss / Noise Floor".to_string(),
            detail: format!("Measured noise floor at {:.1} dB", noise_floor_db),
            severity: if noise_floor_db > -32.0 {
                IssueSeverity::High
            } else {
                IssueSeverity::Medium
            },
        });
    }

    if effective_bandwidth_hz < 15500.0 {
        issues.push(DetectedIssue {
            title: "Band-Limited High Frequencies".to_string(),
            detail: format!(
                "Audio rolls off around {:.1} kHz (typical of old MP3/tape)",
                effective_bandwidth_hz / 1000.0
            ),
            severity: if effective_bandwidth_hz < 11500.0 {
                IssueSeverity::High
            } else {
                IssueSeverity::Medium
            },
        });
    }

    if buf.original_channels == 1 || stereo_correlation > 0.92 {
        issues.push(DetectedIssue {
            title: "Narrow / Mono Soundstage".to_string(),
            detail: format!(
                "Stereo correlation {:.2} — prime candidate for stereo reconstruction",
                stereo_correlation
            ),
            severity: IssueSeverity::Medium,
        });
    }

    if issues.is_empty() {
        issues.push(DetectedIssue {
            title: "Clean Baseline Recording".to_string(),
            detail: "Ready for Apollo mastering & subtle spatial widening".to_string(),
            severity: IssueSeverity::Low,
        });
    }

    let bw_score = (effective_bandwidth_hz / 20000.0).clamp(0.0, 1.0) * 4.0;
    let noise_score = ((-noise_floor_db - 20.0) / 45.0).clamp(0.0, 1.0) * 4.0;
    let stereo_score = (1.0 - stereo_correlation.abs()).clamp(0.0, 1.0) * 2.0;
    let quality_score = (bw_score + noise_score + stereo_score).clamp(1.0, 9.8);

    AudioAnalysis {
        duration_secs: buf.duration_secs(),
        sample_rate: buf.sample_rate,
        original_channels: buf.original_channels,
        noise_floor_db,
        effective_bandwidth_hz,
        stereo_correlation,
        left_rms_db,
        right_rms_db,
        side_energy_pct,
        quality_score,
        waveform_peaks,
        spectrogram,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyzes_mono_noisy_tone() {
        let sr = 44_100;
        let n = sr * 2;
        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.4 + 0.02;
            left.push(s);
            right.push(s);
        }
        let buf = StereoAudioBuffer {
            left,
            right,
            sample_rate: sr as u32,
            original_channels: 1,
            original_sample_rate: sr as u32,
        };
        let res = analyze_audio(&buf);
        assert!(res.stereo_correlation > 0.99);
        assert!(!res.waveform_peaks.is_empty());
        assert!(!res.spectrogram.is_empty());
        assert!(res.issues.iter().any(|i| i.title.contains("Mono")));
    }
}
