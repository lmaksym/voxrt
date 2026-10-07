//! NeMo-exact log-mel frontend (PLAN 2.5). Specification and sources: `docs/frontend.md`.
//!
//! Output: `[n_frames, n_mels]` row-major f32, per-feature normalised, with `n_frames = len / hop`.

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8; // 2^-24
pub const NORM_EPS: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemphasis: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 128,
            preemphasis: 0.97,
        }
    }
}

pub struct Frontend {
    pub cfg: MelConfig,
    window: Vec<f32>,     // symmetric Hann, win_length
    filterbank: Vec<f32>, // [n_mels, n_fft/2 + 1]
    fft: Arc<dyn RealToComplex<f32>>,
}

/// librosa `hz_to_mel(htk=False)` (Slaney).
fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = libm::log(6.4) / 27.0;
    if f >= min_log_hz {
        min_log_mel + libm::log(f / min_log_hz) / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = libm::log(6.4) / 27.0;
    if m >= min_log_mel {
        min_log_hz * libm::exp(logstep * (m - min_log_mel))
    } else {
        f_sp * m
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney", htk=False)`, computed in f64 and cast.
pub fn slaney_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let sr = sample_rate as f64;
    let fftfreqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * sr / n_fft as f64).collect();
    let (m_lo, m_hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut fb = vec![0f32; n_mels * n_bins];
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (k, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - mel_f[i]) / (mel_f[i + 1] - mel_f[i]);
            let upper = (mel_f[i + 2] - f) / (mel_f[i + 2] - mel_f[i + 1]);
            let w = lower.min(upper).max(0.0);
            fb[i * n_bins + k] = (w * enorm) as f32;
        }
    }
    fb
}

impl Frontend {
    pub fn new(cfg: MelConfig) -> Self {
        let n = cfg.win_length;
        // torch.hann_window(n, periodic=False): 0.5 - 0.5 cos(2π k / (n-1))
        let window = (0..n)
            .map(|k| 0.5 - 0.5 * libm::cos(2.0 * std::f64::consts::PI * k as f64 / (n - 1) as f64))
            .map(|v| v as f32)
            .collect();
        let filterbank = slaney_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels);
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(cfg.n_fft);
        Self {
            cfg,
            window,
            filterbank,
            fft,
        }
    }

    pub fn n_frames(&self, n_samples: usize) -> usize {
        n_samples / self.cfg.hop_length
    }

    /// Log-mel before normalisation: `[n_frames, n_mels]`.
    pub fn log_mel(&self, samples: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let n_frames = self.n_frames(samples.len());
        let n_bins = c.n_fft / 2 + 1;
        // pre-emphasis
        let mut x = Vec::with_capacity(samples.len());
        if let Some(&s0) = samples.first() {
            x.push(s0);
            for i in 1..samples.len() {
                x.push(samples[i] - c.preemphasis * samples[i - 1]);
            }
        }
        // centre padding (constant 0) of n_fft/2 on each side
        let pad = c.n_fft / 2;
        let mut padded = vec![0f32; x.len() + 2 * pad];
        padded[pad..pad + x.len()].copy_from_slice(&x);
        // torch.stft pads the win_length window to n_fft symmetrically
        let woff = (c.n_fft - c.win_length) / 2;
        // frames are independent: parallel over blocks of frames, each with its own FFT scratch
        use rayon::prelude::*;
        let mut out = vec![0f32; n_frames * c.n_mels];
        let (padded, window, fb, fft) = (&padded, &self.window, &self.filterbank, &self.fft);
        let block = 64usize;
        out.par_chunks_mut(block * c.n_mels)
            .enumerate()
            .for_each(|(bi, ob)| {
                let mut frame = fft.make_input_vec();
                let mut spec = fft.make_output_vec();
                let mut scratch = fft.make_scratch_vec();
                let mut power = vec![0f32; n_bins];
                for (li, row) in ob.chunks_exact_mut(c.n_mels).enumerate() {
                    let t = bi * block + li;
                    frame.iter_mut().for_each(|v| *v = 0.0);
                    let start = t * c.hop_length + woff;
                    for k in 0..c.win_length {
                        frame[woff + k] = padded[start + k] * window[k];
                    }
                    fft.process_with_scratch(&mut frame, &mut spec, &mut scratch)
                        .expect("fft");
                    for (p, s) in power.iter_mut().zip(spec.iter()) {
                        let Complex { re, im } = *s;
                        *p = re * re + im * im;
                    }
                    for (m, o) in row.iter_mut().enumerate() {
                        let fbm = &fb[m * n_bins..(m + 1) * n_bins];
                        let mut acc = 0f64;
                        for (w, p) in fbm.iter().zip(&power) {
                            acc += *w as f64 * *p as f64;
                        }
                        *o = libm::logf(acc as f32 + LOG_ZERO_GUARD);
                    }
                }
            });
        out
    }

    /// Per-feature normalisation over all frames (mean, unbiased std, `std + 1e-5`), in place.
    pub fn normalize(&self, feats: &mut [f32], n_frames: usize) {
        let m = self.cfg.n_mels;
        if n_frames < 2 {
            return;
        }
        for f in 0..m {
            let mut mean = 0f64;
            for t in 0..n_frames {
                mean += feats[t * m + f] as f64;
            }
            mean /= n_frames as f64;
            let mut var = 0f64;
            for t in 0..n_frames {
                let d = feats[t * m + f] as f64 - mean;
                var += d * d;
            }
            let std = (var / (n_frames as f64 - 1.0)).sqrt() as f32 + NORM_EPS;
            let mean = mean as f32;
            for t in 0..n_frames {
                feats[t * m + f] = (feats[t * m + f] - mean) / std;
            }
        }
    }

    /// Full pipeline: normalised features `[n_frames, n_mels]`.
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let n = self.n_frames(samples.len());
        let mut f = self.log_mel(samples);
        self.normalize(&mut f, n);
        (f, n)
    }
}
