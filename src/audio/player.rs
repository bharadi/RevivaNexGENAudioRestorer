use anyhow::Result;
use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamHandle, Sink};
use std::time::{Duration, Instant};

use super::io::StereoAudioBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTrack {
    Original,
    Restored,
}

/// Real-time A/B audio player using `rodio` so the user can toggle between
/// Original and Restored audio at the exact same timestamp.
pub struct AbPlayer {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
    sink: Option<Sink>,
    active_track: ActiveTrack,
    is_playing: bool,
    position_secs: f32,
    last_tick: Option<Instant>,
    duration_secs: f32,
}

impl AbPlayer {
    pub fn try_new() -> Result<Self> {
        let (stream, stream_handle) = OutputStream::try_default()?;
        Ok(Self {
            _stream: stream,
            stream_handle,
            sink: None,
            active_track: ActiveTrack::Restored,
            is_playing: false,
            position_secs: 0.0,
            last_tick: None,
            duration_secs: 0.0,
        })
    }

    pub fn active_track(&self) -> ActiveTrack {
        self.active_track
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing
    }

    pub fn position_secs(&mut self) -> f32 {
        if self.is_playing {
            if let Some(last) = self.last_tick {
                self.position_secs += last.elapsed().as_secs_f32();
                self.last_tick = Some(Instant::now());
                if self.position_secs >= self.duration_secs && self.duration_secs > 0.0 {
                    self.position_secs = self.duration_secs;
                    self.is_playing = false;
                    self.last_tick = None;
                }
            }
        }
        self.position_secs
    }

    pub fn stop(&mut self) {
        if let Some(sink) = self.sink.take() {
            sink.stop();
        }
        self.is_playing = false;
        self.position_secs = 0.0;
        self.last_tick = None;
    }

    pub fn pause(&mut self) {
        let _ = self.position_secs();
        if let Some(sink) = &self.sink {
            sink.pause();
        }
        self.is_playing = false;
        self.last_tick = None;
    }

    pub fn play_from(
        &mut self,
        track: ActiveTrack,
        original: Option<&StereoAudioBuffer>,
        restored: Option<&StereoAudioBuffer>,
        start_secs: f32,
    ) -> Result<()> {
        let buf = match track {
            ActiveTrack::Original => original,
            ActiveTrack::Restored => restored.or(original),
        };
        let Some(buf) = buf else {
            return Ok(());
        };

        self.active_track = track;
        self.duration_secs = buf.duration_secs();
        let clamped_start = start_secs.clamp(0.0, (self.duration_secs - 0.05).max(0.0));
        let start_frame = (clamped_start * buf.sample_rate as f32) as usize;

        let mut interleaved = Vec::with_capacity((buf.len().saturating_sub(start_frame)) * 2);
        for i in start_frame..buf.len() {
            interleaved.push(buf.left[i]);
            interleaved.push(buf.right[i]);
        }

        if let Some(old) = self.sink.take() {
            old.stop();
        }

        let sink = Sink::try_new(&self.stream_handle)?;
        let source = SamplesBuffer::new(2, buf.sample_rate, interleaved);
        sink.append(source);
        sink.play();

        self.sink = Some(sink);
        self.position_secs = clamped_start;
        self.is_playing = true;
        self.last_tick = Some(Instant::now());
        Ok(())
    }

    pub fn switch_track(
        &mut self,
        new_track: ActiveTrack,
        original: Option<&StereoAudioBuffer>,
        restored: Option<&StereoAudioBuffer>,
    ) -> Result<()> {
        let pos = self.position_secs();
        let was_playing = self.is_playing;
        self.active_track = new_track;
        if was_playing {
            self.play_from(new_track, original, restored, pos)?;
        }
        Ok(())
    }

    pub fn seek(
        &mut self,
        secs: f32,
        original: Option<&StereoAudioBuffer>,
        restored: Option<&StereoAudioBuffer>,
    ) -> Result<()> {
        let was_playing = self.is_playing;
        self.position_secs = secs.max(0.0);
        if was_playing {
            self.play_from(self.active_track, original, restored, self.position_secs)?;
        }
        Ok(())
    }
}

pub fn format_duration(secs: f32) -> String {
    let total = secs.max(0.0).round() as u64;
    let mins = total / 60;
    let s = total % 60;
    format!("{:02}:{:02}", mins, s)
}

#[allow(dead_code)]
fn _unused_duration_check() -> Duration {
    Duration::from_secs(0)
}
