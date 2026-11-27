use anyhow::{anyhow, Context, Result};
use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::config::Encoder as FlacConfig;
use flacenc::error::Verify as FlacVerify;
use flacenc::source::MemSource;
use hound::{SampleFormat, WavSpec, WavWriter};
use lofty::config::WriteOptions;
use lofty::file::{AudioFile as LoftyAudioFile, TaggedFileExt};
use lofty::probe::Probe;
use mp3lame_encoder::{Bitrate, Builder as LameBuilder, DualPcm, FlushNoGap, Quality, VbrMode};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::config::OutputFormat;

/// Planar stereo audio buffer (Left, Right) normalized to [-1.0, 1.0].
#[derive(Debug, Clone)]
pub struct StereoAudioBuffer {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
    pub original_channels: usize,
    pub original_sample_rate: u32,
}

impl StereoAudioBuffer {
    pub fn len(&self) -> usize {
        self.left.len().min(self.right.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.len() as f32 / self.sample_rate as f32
        }
    }
}

/// Decodes any Symphonia-supported audio file (MP3, WAV, FLAC, OGG, AAC/M4A)
/// into a planar stereo `f32` buffer. Mono files are duplicated to L/R.
pub fn decode_audio_file(path: &Path) -> Result<StereoAudioBuffer> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Unsupported or corrupted audio container")?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("No audio track found in file"))?;
    let track_id = track.id;

    let codec_params = track.codec_params.clone();
    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("Unknown sample rate"))?;
    let channels = codec_params.channels.map(|c| c.count()).unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .context("Failed to initialize audio decoder")?;

    let mut left = Vec::new();
    let mut right = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(anyhow!("Decode error: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow!("Frame decode error: {e}")),
        };

        let spec = *decoded.spec();
        let ch = spec.channels.count().max(1);
        let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        if ch == 1 {
            for &s in samples {
                left.push(s);
                right.push(s);
            }
        } else {
            for frame in samples.chunks_exact(ch) {
                left.push(frame[0]);
                right.push(frame[1]);
            }
        }
    }

    if left.is_empty() {
        return Err(anyhow!("Decoded audio is empty"));
    }

    Ok(StereoAudioBuffer {
        left,
        right,
        sample_rate,
        original_channels: channels,
        original_sample_rate: sample_rate,
    })
}

/// High-quality sinc resampling of a stereo buffer to `target_sr` (e.g., 44100 or 48000 Hz).
pub fn resample_stereo(buf: StereoAudioBuffer, target_sr: u32) -> Result<StereoAudioBuffer> {
    if buf.sample_rate == target_sr {
        return Ok(buf);
    }

    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = target_sr as f64 / buf.sample_rate as f64;
    let chunk_size = 1024;
    let mut resampler =
        SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 2).context("Resampler init failed")?;

    let total = buf.len();
    let mut out_l = Vec::with_capacity((total as f64 * ratio) as usize + 1024);
    let mut out_r = Vec::with_capacity((total as f64 * ratio) as usize + 1024);

    let mut idx = 0;
    while idx < total {
        let end = (idx + chunk_size).min(total);
        let mut in_l = buf.left[idx..end].to_vec();
        let mut in_r = buf.right[idx..end].to_vec();
        if in_l.len() < chunk_size {
            in_l.resize(chunk_size, 0.0);
            in_r.resize(chunk_size, 0.0);
        }
        let waves_in = vec![in_l, in_r];
        let waves_out = resampler.process(&waves_in, None)?;
        out_l.extend_from_slice(&waves_out[0]);
        out_r.extend_from_slice(&waves_out[1]);
        idx += chunk_size;
    }

    let expected_len = ((total as f64) * ratio).round() as usize;
    out_l.truncate(expected_len);
    out_r.truncate(expected_len);

    Ok(StereoAudioBuffer {
        left: out_l,
        right: out_r,
        sample_rate: target_sr,
        original_channels: buf.original_channels,
        original_sample_rate: buf.original_sample_rate,
    })
}

/// Exports a planar stereo buffer to MP3 (LAME), FLAC (24-bit), WAV, or AC-3 (via FFmpeg),
/// optionally copying ID3/Vorbis tags and album art from `source_path`.
pub fn export_audio_file(
    buf: &StereoAudioBuffer,
    output_path: &Path,
    format: OutputFormat,
    source_path: Option<&Path>,
    preserve_metadata: bool,
) -> Result<()> {
    match format {
        OutputFormat::Mp3VbrV0 | OutputFormat::Mp3VbrV2 | OutputFormat::Mp3Cbr320 => {
            export_mp3_file(buf, output_path, format)?;
        }
        OutputFormat::Flac24Bit => {
            export_flac_file(buf, output_path)?;
        }
        OutputFormat::Wav24Bit | OutputFormat::Wav16Bit => {
            export_wav_file(buf, output_path, format)?;
        }
        OutputFormat::Ac3DolbyCompatible => {
            export_ac3_via_ffmpeg(buf, output_path)?;
        }
    }

    if preserve_metadata && !matches!(format, OutputFormat::Ac3DolbyCompatible) {
        if let Some(src) = source_path {
            let _ = copy_audio_metadata(src, output_path);
        }
    }
    Ok(())
}

/// Encodes planar stereo audio to 24-bit lossless FLAC using pure-Rust `flacenc`.
pub fn export_flac_file(buf: &StereoAudioBuffer, output_path: &Path) -> Result<()> {
    let len = buf.len();
    let bits_per_sample = 24usize;
    let scale = 8_388_607.0_f32;
    let mut interleaved = Vec::with_capacity(len * 2);
    for i in 0..len {
        interleaved.push((buf.left[i].clamp(-1.0, 1.0) * scale).round() as i32);
        interleaved.push((buf.right[i].clamp(-1.0, 1.0) * scale).round() as i32);
    }

    let config = FlacConfig::default()
        .into_verified()
        .map_err(|(_, e)| anyhow!("FLAC config error: {e:?}"))?;
    let source = MemSource::from_samples(&interleaved, 2, bits_per_sample, buf.sample_rate as usize);
    let flac_stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| anyhow!("FLAC encode error: {e:?}"))?;

    let mut sink = ByteSink::new();
    flac_stream
        .write(&mut sink)
        .map_err(|e| anyhow!("FLAC write error: {e:?}"))?;

    let mut file = File::create(output_path).context("Failed to create output FLAC file")?;
    file.write_all(sink.as_slice())?;
    Ok(())
}

/// Copies tags and album cover art from `src` to `dst` using `lofty`.
fn copy_audio_metadata(src: &Path, dst: &Path) -> Result<()> {
    let src_tagged = Probe::open(src)?.read()?;
    let mut dst_tagged = Probe::open(dst)?.read()?;
    if let Some(tag) = src_tagged.primary_tag().or_else(|| src_tagged.first_tag()) {
        let cloned = tag.clone();
        dst_tagged.insert_tag(cloned);
        dst_tagged.save_to_path(dst, WriteOptions::default())?;
    }
    Ok(())
}

fn find_ffmpeg_executable() -> PathBuf {
    let local = PathBuf::from("ffmpeg.exe");
    if local.exists() {
        return local;
    }
    let in_models = PathBuf::from("models").join("ffmpeg.exe");
    if in_models.exists() {
        return in_models;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let next_to_exe = dir.join("ffmpeg.exe");
            if next_to_exe.exists() {
                return next_to_exe;
            }
            let models_next = dir.join("models").join("ffmpeg.exe");
            if models_next.exists() {
                return models_next;
            }
        }
    }
    if let Ok(local_app) = std::env::var("LOCALAPPDATA") {
        let winget_link = PathBuf::from(local_app)
            .join("Microsoft")
            .join("WinGet")
            .join("Links")
            .join("ffmpeg.exe");
        if winget_link.exists() {
            return winget_link;
        }
    }
    PathBuf::from("ffmpeg")
}

/// Encodes to open-source AC-3 (Dolby Digital-compatible) by piping a temporary WAV through FFmpeg.
fn export_ac3_via_ffmpeg(buf: &StereoAudioBuffer, output_path: &Path) -> Result<()> {
    let temp_wav = output_path.with_extension("tmp_nexgen.wav");
    export_wav_file(buf, &temp_wav, OutputFormat::Wav24Bit)?;

    let ffmpeg_bin = find_ffmpeg_executable();
    let status = std::process::Command::new(&ffmpeg_bin)
        .args([
            "-y",
            "-i",
            temp_wav.to_str().unwrap_or(""),
            "-c:a",
            "ac3",
            "-b:a",
            "448k",
            output_path.to_str().unwrap_or(""),
        ])
        .status();

    let _ = std::fs::remove_file(&temp_wav);

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(anyhow!("FFmpeg AC-3 encoding failed with exit code {s}")),
        Err(_) => Err(anyhow!(
            "FFmpeg not found on PATH. Install FFmpeg (winget install Gyan.FFmpeg) to export AC-3, or choose MP3 VBR / WAV."
        )),
    }
}

/// Encodes planar stereo audio to high-quality VBR or CBR MP3 using LAME.
pub fn export_mp3_file(
    buf: &StereoAudioBuffer,
    output_path: &Path,
    format: OutputFormat,
) -> Result<()> {
    let mut builder = LameBuilder::new()
        .ok_or_else(|| anyhow!("Failed to initialize LAME MP3 encoder"))?;
    builder
        .set_num_channels(2)
        .map_err(|e| anyhow!("LAME channels error: {e:?}"))?;
    builder
        .set_sample_rate(buf.sample_rate)
        .map_err(|e| anyhow!("LAME sample rate error: {e:?}"))?;
    builder
        .set_quality(Quality::Best)
        .map_err(|e| anyhow!("LAME quality error: {e:?}"))?;

    match format {
        OutputFormat::Mp3VbrV0 => {
            builder
                .set_vbr_mode(VbrMode::Mtrh)
                .map_err(|e| anyhow!("LAME VBR mode error: {e:?}"))?;
            builder
                .set_vbr_quality(Quality::Best)
                .map_err(|e| anyhow!("LAME VBR V0 quality error: {e:?}"))?;
            builder
                .set_to_write_vbr_tag(true)
                .map_err(|e| anyhow!("LAME VBR tag error: {e:?}"))?;
        }
        OutputFormat::Mp3VbrV2 => {
            builder
                .set_vbr_mode(VbrMode::Mtrh)
                .map_err(|e| anyhow!("LAME VBR mode error: {e:?}"))?;
            builder
                .set_vbr_quality(Quality::NearBest)
                .map_err(|e| anyhow!("LAME VBR V2 quality error: {e:?}"))?;
            builder
                .set_to_write_vbr_tag(true)
                .map_err(|e| anyhow!("LAME VBR tag error: {e:?}"))?;
        }
        OutputFormat::Mp3Cbr320 => {
            builder
                .set_vbr_mode(VbrMode::Off)
                .map_err(|e| anyhow!("LAME CBR mode error: {e:?}"))?;
            builder
                .set_brate(Bitrate::Kbps320)
                .map_err(|e| anyhow!("LAME 320kbps bitrate error: {e:?}"))?;
        }
        _ => {}
    }

    let mut encoder = builder
        .build()
        .map_err(|e| anyhow!("Failed to build LAME encoder: {e:?}"))?;

    let len = buf.len();
    let mut left_i16 = Vec::with_capacity(len);
    let mut right_i16 = Vec::with_capacity(len);
    for i in 0..len {
        left_i16.push((buf.left[i].clamp(-1.0, 1.0) * 32_767.0).round() as i16);
        right_i16.push((buf.right[i].clamp(-1.0, 1.0) * 32_767.0).round() as i16);
    }

    let mut mp3_out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(len));
    let chunk_size = 1152 * 16;
    for start in (0..len).step_by(chunk_size) {
        let end = (start + chunk_size).min(len);
        let input = DualPcm {
            left: &left_i16[start..end],
            right: &right_i16[start..end],
        };
        encoder
            .encode_to_vec(input, &mut mp3_out)
            .map_err(|e| anyhow!("MP3 encode failed: {e:?}"))?;
    }

    encoder
        .flush_to_vec::<FlushNoGap>(&mut mp3_out)
        .map_err(|e| anyhow!("MP3 flush failed: {e:?}"))?;

    let mut file = File::create(output_path).context("Failed to create output MP3 file")?;
    file.write_all(&mp3_out)?;
    Ok(())
}

/// Exports a planar stereo buffer to a WAV file (24-bit or 16-bit PCM).
pub fn export_wav_file(
    buf: &StereoAudioBuffer,
    output_path: &Path,
    format: OutputFormat,
) -> Result<()> {
    let bits_per_sample = match format {
        OutputFormat::Wav16Bit => 16,
        _ => 24,
    };

    let spec = WavSpec {
        channels: 2,
        sample_rate: buf.sample_rate,
        bits_per_sample,
        sample_format: SampleFormat::Int,
    };

    let mut writer =
        WavWriter::create(output_path, spec).context("Failed to create output WAV file")?;

    let scale = match bits_per_sample {
        24 => 8_388_607.0_f32, // 2^23 - 1
        _ => 32_767.0_f32,     // 2^15 - 1
    };

    let len = buf.len();
    for i in 0..len {
        let l = (buf.left[i].clamp(-1.0, 1.0) * scale).round() as i32;
        let r = (buf.right[i].clamp(-1.0, 1.0) * scale).round() as i32;
        writer.write_sample(l)?;
        writer.write_sample(r)?;
    }
    writer.finalize()?;
    Ok(())
}
