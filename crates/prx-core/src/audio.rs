//! Minimal WAV reader (PLAN 2.4): 16 kHz mono PCM16 or IEEE f32. Other formats are rejected for now.
use std::path::Path;

use crate::{Error, Result};

pub const SAMPLE_RATE: u32 = 16_000;

pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub fn read_wav(path: impl AsRef<Path>) -> Result<Audio> {
    let b = std::fs::read(path)?;
    let bad = |m: &str| Error::Audio(m.to_string());
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let u16le = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let (mut fmt, mut data) = (None, None);
    let mut pos = 12;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32le(pos + 4) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            fmt = Some(body);
        } else if id == b"data" {
            data = Some((body, size.min(b.len() - body)));
        }
        pos = body + size + (size & 1);
    }
    let (fmt, (d0, dlen)) = match (fmt, data) {
        (Some(f), Some(d)) => (f, d),
        _ => return Err(bad("missing fmt or data chunk")),
    };
    let format = u16le(fmt);
    let channels = u16le(fmt + 2) as usize;
    let sample_rate = u32le(fmt + 4);
    let bits = u16le(fmt + 14) as usize;
    if channels != 1 {
        return Err(Error::Audio(format!(
            "{channels} channels; only mono is supported in this phase"
        )));
    }
    if sample_rate != SAMPLE_RATE {
        return Err(Error::Audio(format!(
            "{sample_rate} Hz; only 16 kHz is supported in this phase"
        )));
    }
    let payload = &b[d0..d0 + dlen];
    let samples: Vec<f32> = match (format, bits) {
        (1, 16) => payload
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        (3, 32) => payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => {
            return Err(Error::Audio(format!(
                "unsupported WAV format tag {format} / {bits} bits"
            )))
        }
    };
    Ok(Audio {
        samples,
        sample_rate,
    })
}
