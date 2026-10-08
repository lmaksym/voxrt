//! VAD head on the subsampler output: `sigmoid(out(relu(ctx(relu(proj(x))))))`, 80 ms frames.
#![allow(clippy::needless_range_loop)]
use crate::format::ModelFile;
use crate::tensor::load_f32_shaped;
use crate::Result;

pub struct VadHead {
    proj_w: Vec<f32>, // [hidden, d_model]
    proj_b: Vec<f32>,
    ctx_w: Vec<f32>, // [hidden, hidden, k]
    ctx_b: Vec<f32>,
    out_w: Vec<f32>, // [hidden]
    out_b: f32,
    pub d_model: usize,
    pub hidden: usize,
    pub kernel: usize,
    pub frame_seconds: f64,
}

impl VadHead {
    pub fn load(m: &ModelFile) -> Result<Self> {
        let v = &m.ternary.vad_head;
        let (d, h, k) = (v.d_model, v.hidden, v.kernel);
        Ok(Self {
            proj_w: load_f32_shaped(m, "vad_head.proj.weight", &[h, d, 1])?,
            proj_b: load_f32_shaped(m, "vad_head.proj.bias", &[h])?,
            ctx_w: load_f32_shaped(m, "vad_head.ctx.weight", &[h, h, k])?,
            ctx_b: load_f32_shaped(m, "vad_head.ctx.bias", &[h])?,
            out_w: load_f32_shaped(m, "vad_head.out.weight", &[1, h, 1])?,
            out_b: load_f32_shaped(m, "vad_head.out.bias", &[1])?[0],
            d_model: d,
            hidden: h,
            kernel: k,
            frame_seconds: v.frame_seconds,
        })
    }

    /// Speech probability per encoder frame from the subsampler output `[t, d_model]`.
    pub fn probs(&self, sub: &[f32], t: usize) -> Vec<f32> {
        let (d, h, k) = (self.d_model, self.hidden, self.kernel);
        let pad = (k - 1) / 2;
        // proj + relu: [t, h]
        let mut a = vec![0f32; t * h];
        for i in 0..t {
            let x = &sub[i * d..(i + 1) * d];
            for j in 0..h {
                let w = &self.proj_w[j * d..(j + 1) * d];
                let mut acc = self.proj_b[j];
                for c in 0..d {
                    acc += w[c] * x[c];
                }
                a[i * h + j] = acc.max(0.0);
            }
        }
        // ctx conv over time (k taps, same padding) + relu, then out + sigmoid
        let mut p = vec![0f32; t];
        let mut hbuf = vec![0f32; h];
        for i in 0..t {
            for o in 0..h {
                let mut acc = self.ctx_b[o];
                for ci in 0..h {
                    let w = &self.ctx_w[(o * h + ci) * k..(o * h + ci + 1) * k];
                    for (kk, &wv) in w.iter().enumerate() {
                        let j = i as isize + kk as isize - pad as isize;
                        if j >= 0 && (j as usize) < t {
                            acc += wv * a[j as usize * h + ci];
                        }
                    }
                }
                hbuf[o] = acc.max(0.0);
            }
            let mut z = self.out_b;
            for o in 0..h {
                z += self.out_w[o] * hbuf[o];
            }
            p[i] = 1.0 / (1.0 + libm::expf(-z));
        }
        p
    }
}

/// A speech segment in encoder frames `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct SegmentConfig {
    pub threshold: f32,
    pub min_pause_frames: usize, // 3 → 240 ms
    pub max_frames: usize,       // 750 → 60 s
    pub min_frames: usize,       // 125 → 10 s: shorter segments merge with a neighbour
    pub pad_frames: usize,       // speech padding at both ends, if available
}

impl Default for SegmentConfig {
    fn default() -> Self {
        // tuned on TED-LIUM 3 long-form: pause 640 ms, pad 160 ms, min 10 s, cap 60 s. The second
        // round (2026-10-06) showed the residual gap to Photon was segment *context*, not cut placement: WER plateaus
        // at 2.55 once segments are ≥ 10 s with a 60 s cap (2.61 at 5 s / 30 s). Costs: first output after ≥ 10 s of
        // speech when streaming, ~1.6 GB peak RSS on a 20 min talk (segment-sized buffers) instead of ~1.0 GB.
        Self {
            threshold: 0.5,
            min_pause_frames: 8,
            max_frames: 750,
            min_frames: 125,
            pad_frames: 2,
        }
    }
}

impl SegmentConfig {
    /// Default, overridable with `VOXRT_VAD_THRESHOLD`, `VOXRT_VAD_MIN_PAUSE`, `VOXRT_VAD_MAX`, `VOXRT_VAD_MIN`, `VOXRT_VAD_PAD` (frames).
    pub fn from_env() -> Self {
        let mut c = Self::default();
        let get = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
        if let Some(v) = get("VOXRT_VAD_THRESHOLD") {
            c.threshold = v as f32;
        }
        if let Some(v) = get("VOXRT_VAD_MIN_PAUSE") {
            c.min_pause_frames = v as usize;
        }
        if let Some(v) = get("VOXRT_VAD_MAX") {
            c.max_frames = v as usize;
        }
        if let Some(v) = get("VOXRT_VAD_MIN") {
            c.min_frames = v as usize;
        }
        if let Some(v) = get("VOXRT_VAD_PAD") {
            c.pad_frames = v as usize;
        }
        c
    }
}

/// Cut at pauses: speech runs separated by ≥ `min_pause_frames` non-speech frames; runs longer than
/// `max_frames` are split at the lowest-probability frame in their last 5 s; short runs merge with the previous one.
pub fn segment(p: &[f32], cfg: &SegmentConfig) -> Vec<Segment> {
    let t = p.len();
    if t == 0 {
        return Vec::new();
    }
    // 1. speech runs
    let mut runs: Vec<Segment> = Vec::new();
    let mut i = 0;
    while i < t {
        if p[i] >= cfg.threshold {
            let s = i;
            let mut last_speech = i;
            while i < t {
                if p[i] >= cfg.threshold {
                    last_speech = i;
                    i += 1;
                } else {
                    // tolerate pauses shorter than min_pause
                    let mut j = i;
                    while j < t && p[j] < cfg.threshold {
                        j += 1;
                    }
                    if j - i >= cfg.min_pause_frames || j == t {
                        i = j;
                        break;
                    }
                    i = j;
                }
            }
            runs.push(Segment {
                start: s,
                end: last_speech + 1,
            });
        } else {
            i += 1;
        }
    }
    if runs.is_empty() {
        return vec![Segment { start: 0, end: t }];
    }
    // 2. merge short runs into the previous one (or the next, for the first)
    let mut merged: Vec<Segment> = Vec::new();
    for r in runs {
        match merged.last_mut() {
            Some(prev)
                if (r.end - r.start < cfg.min_frames || prev.end - prev.start < cfg.min_frames)
                    && r.end - prev.start <= cfg.max_frames =>
            {
                prev.end = r.end
            }
            _ => merged.push(r),
        }
    }
    // 3. enforce the hard cap by splitting at the weakest frame in the last 5 s of each window
    let mut out = Vec::new();
    for r in merged {
        let mut s = r.start;
        while r.end - s > cfg.max_frames {
            let win_end = s + cfg.max_frames;
            let look = 63.min(cfg.max_frames / 2); // ~5 s
            let (mut best, mut best_p) = (win_end, f32::MAX);
            for f in (win_end - look)..win_end {
                if p[f] < best_p {
                    best_p = p[f];
                    best = f;
                }
            }
            out.push(Segment {
                start: s,
                end: best.max(s + 1),
            });
            s = best.max(s + 1);
        }
        out.push(Segment {
            start: s,
            end: r.end,
        });
    }
    // 4. pad with context frames, never past the midpoint of the gap to a neighbour (no duplicated audio)
    let n = out.len();
    let bounds: Vec<(usize, usize)> = (0..n)
        .map(|i| {
            let lo = if i == 0 {
                0
            } else {
                out[i - 1].end + (out[i].start - out[i - 1].end).div_ceil(2)
            };
            let hi = if i + 1 == n {
                t
            } else {
                out[i].end + (out[i + 1].start - out[i].end) / 2
            };
            (lo, hi)
        })
        .collect();
    for (seg, (lo, hi)) in out.iter_mut().zip(bounds) {
        seg.start = seg.start.saturating_sub(cfg.pad_frames).max(lo);
        seg.end = (seg.end + cfg.pad_frames).min(hi);
    }
    out
}
