//! WAV input (PLAN 2.4 / 4.5): PCM 8/16/24/32-bit and IEEE f32/f64, any channel count (averaged to mono),
//! any sample rate (windowed-sinc resampling to 16 kHz).
use std::path::Path;

use crate::{Error, Result};

pub const SAMPLE_RATE: u32 = 16_000;

pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Decode a WAV file to mono f32 at its native rate.
pub fn decode_wav(path: impl AsRef<Path>) -> Result<Audio> {
    decode_wav_bytes(&std::fs::read(path)?)
}

/// Decode WAV bytes (a file read into memory, or stdin) to mono f32 at the native rate.
pub fn decode_wav_bytes(b: &[u8]) -> Result<Audio> {
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
    let mut format = u16le(fmt);
    let channels = u16le(fmt + 2) as usize;
    let sample_rate = u32le(fmt + 4);
    let bits = u16le(fmt + 14) as usize;
    if format == 0xFFFE {
        // WAVE_FORMAT_EXTENSIBLE: sub-format GUID's first two bytes hold the real tag
        format = u16le(fmt + 24);
    }
    if channels == 0 {
        return Err(bad("zero channels"));
    }
    let payload = &b[d0..d0 + dlen];
    let bytes_per = bits / 8;
    let frames = payload.len() / (bytes_per * channels);
    let mut samples = Vec::with_capacity(frames);
    let inv = 1.0 / channels as f32;
    for f in 0..frames {
        let mut acc = 0f32;
        for c in 0..channels {
            let o = (f * channels + c) * bytes_per;
            let s = &payload[o..o + bytes_per];
            acc += match (format, bits) {
                (1, 8) => (s[0] as f32 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
                (1, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
                (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
                (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                (3, 64) => f64::from_le_bytes(s.try_into().unwrap()) as f32,
                _ => {
                    return Err(Error::Audio(format!(
                        "unsupported WAV format tag {format} / {bits} bits"
                    )))
                }
            };
        }
        samples.push(acc * inv);
    }
    Ok(Audio {
        samples,
        sample_rate,
    })
}

/// Decode and convert to 16 kHz mono.
pub fn read_wav(path: impl AsRef<Path>) -> Result<Audio> {
    to_16k(decode_wav(path)?)
}

/// `read_wav` for in-memory WAV bytes.
pub fn read_wav_bytes(b: &[u8]) -> Result<Audio> {
    to_16k(decode_wav_bytes(b)?)
}

fn to_16k(a: Audio) -> Result<Audio> {
    if a.sample_rate == SAMPLE_RATE {
        return Ok(a);
    }
    Ok(Audio {
        samples: resample(&a.samples, a.sample_rate, SAMPLE_RATE),
        sample_rate: SAMPLE_RATE,
    })
}

/// Windowed-sinc (Kaiser β=9, 32 taps per side at the lower rate) polyphase resampler, any rational ratio.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(from, to);
    let (up, down) = ((to / g) as usize, (from / g) as usize);
    // low-pass at min(from, to)/2, expressed at the upsampled rate `from * up`
    let cutoff = 0.5 * (from.min(to) as f64) / (from as f64 * up as f64); // cycles per upsampled sample
    let taps_per_side = 32 * up; // 32 zero crossings at the lower rate
    let n_taps = 2 * taps_per_side + 1;
    let beta = 9.0;
    let i0b = bessel_i0(beta);
    let h: Vec<f64> = (0..n_taps)
        .map(|i| {
            let n = i as f64 - taps_per_side as f64;
            let sinc = if n == 0.0 {
                2.0 * cutoff
            } else {
                libm::sin(2.0 * std::f64::consts::PI * cutoff * n) / (std::f64::consts::PI * n)
            };
            let w = bessel_i0(
                beta * (1.0 - (n / taps_per_side as f64) * (n / taps_per_side as f64))
                    .max(0.0)
                    .sqrt(),
            ) / i0b;
            sinc * w * up as f64
        })
        .collect();
    let out_len = (x.len() * up).div_ceil(down);
    let mut y = Vec::with_capacity(out_len);
    for m in 0..out_len {
        // output sample m sits at upsampled index m*down; input sample k contributes h[m*down - k*up + taps_per_side]
        let pos = m * down;
        let k_lo = (pos + taps_per_side)
            .saturating_sub(n_taps - 1)
            .div_ceil(up);
        let k_hi = ((pos + taps_per_side) / up).min(x.len().saturating_sub(1));
        let mut acc = 0f64;
        let mut k = k_lo;
        while k <= k_hi {
            let idx = pos + taps_per_side - k * up;
            if idx < n_taps {
                acc += x[k] as f64 * h[idx];
            }
            k += 1;
        }
        y.push(acc as f32);
    }
    y
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let y = x * x / 4.0;
    for k in 1..50 {
        term *= y / (k as f64 * k as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}
