//! Long-form / streaming transcription (PLAN 4.2, 4.6): VAD on fixed absolute-time blocks, a segment automaton that
//! only closes segments whose extent cannot change with future audio, and per-segment transcription.
//! `Transcriber::transcribe_long` feeds everything through the same state machine, so streaming == offline by construction.
use crate::decoder::Emission;
use crate::transcriber::{Segment, Transcriber};
use crate::vad::{segment, SegmentConfig};

pub struct LongForm<'a> {
    tr: &'a Transcriber,
    cfg: SegmentConfig,
    block_samples: usize,
    ctx_samples: usize, // VAD context on each side of a block (removes block-edge effects; still chunk-independent)
    pending: Vec<f32>,  // audio from `base_sample` onward
    base_sample: usize, // absolute sample index of pending[0]
    probs: Vec<f32>, // VAD probs for frames from `base_frame` to the end of the last complete block
    next_block: usize, // absolute sample index of the next VAD block
    total: usize,    // absolute samples received
    pub segments: Vec<Segment>,
    pub emissions: Vec<Emission>,
    /// When set, closed segments are recorded as absolute sample ranges in `ranges` instead of being transcribed.
    pub collect_only: bool,
    pub ranges: Vec<(usize, usize)>,
    pub max_pending: usize, // high-water mark of the audio buffer, samples (diagnostics)
}

impl<'a> LongForm<'a> {
    pub fn new(tr: &'a Transcriber) -> Self {
        Self {
            tr,
            cfg: tr.segment_cfg.clone(),
            block_samples: 10 * 16_000,
            ctx_samples: 16 * 1280, // 16 frames = 1.28 s
            pending: Vec::new(),
            base_sample: 0,
            probs: Vec::new(),
            next_block: 0,
            total: 0,
            segments: Vec::new(),
            emissions: Vec::new(),
            collect_only: false,
            ranges: Vec::new(),
            max_pending: 0,
        }
    }

    fn spf(&self) -> usize {
        self.tr.samples_per_frame()
    }

    /// Feed audio; returns segments finalised by this call.
    pub fn push(&mut self, samples: &[f32]) -> Vec<Segment> {
        self.pending.extend_from_slice(samples);
        self.total += samples.len();
        self.max_pending = self.max_pending.max(self.pending.len());
        while self.next_block + self.block_samples + self.ctx_samples <= self.total {
            self.probs
                .extend(self.vad_block_at(self.next_block, self.next_block + self.block_samples));
            self.next_block += self.block_samples;
        }
        self.drain(false)
    }

    /// End of stream: VAD on the remaining partial block, close everything.
    pub fn flush(&mut self) -> Vec<Segment> {
        if std::env::var_os("VOXRT_PROFILE").is_some() {
            eprintln!(
                "longform: max pending buffer {:.1} s, {} segments",
                self.max_pending as f64 / 16000.0,
                self.segments.len()
            );
        }
        while self.next_block < self.total {
            let end = (self.next_block + self.block_samples).min(self.total);
            if end - self.next_block >= self.spf() {
                self.probs.extend(self.vad_block_at(self.next_block, end));
            }
            self.next_block = end;
        }
        self.drain(true)
    }

    /// VAD probs for absolute samples `[start, end)` computed with up to `ctx_samples` of audio on each side
    /// (clipped to what is available), keeping only the block's own frames.
    fn vad_block_at(&self, start: usize, end: usize) -> Vec<f32> {
        let spf = self.spf();
        let lo = start.saturating_sub(self.ctx_samples).max(self.base_sample);
        let lo = start - ((start - lo) / spf) * spf; // whole frames of left context
        let hi = (end + self.ctx_samples).min(self.total);
        let block = &self.pending[lo - self.base_sample..hi - self.base_sample];
        let (feats, t) = self.tr.frontend.compute(block);
        let sub = self.tr.encoder.subsampling.forward(&feats, t);
        let te = self.tr.encoder.subsampling.out_frames(t);
        let p = self.tr.vad.probs(&sub, te);
        let skip = (start - lo) / spf;
        let want = (end - start) / spf;
        p.into_iter().skip(skip).take(want).collect()
    }

    fn drain(&mut self, fin: bool) -> Vec<Segment> {
        let mut out = Vec::new();
        if self.probs.is_empty() {
            if fin && self.pending.len() >= self.spf() {
                out.push(self.emit(0, self.pending.len() / self.spf()));
            }
            return out;
        }
        let segs = segment(&self.probs, &self.cfg);
        let known = self.probs.len();
        let mut closed = 0;
        for i in 0..segs.len() {
            let is_closed = fin
                || (i + 1 < segs.len() && segs[i + 1].end + self.cfg.min_pause_frames <= known)
                || (segs[i].end - segs[i].start >= self.cfg.max_frames
                    && known >= segs[i].end + 63);
            if !is_closed {
                break;
            }
            closed = i + 1;
        }
        if closed == 0 {
            return out;
        }
        for s in &segs[..closed] {
            out.push(self.emit(s.start, s.end));
        }
        // drop audio and probs before the next segment's start (or the last closed end when final)
        let cut_frame = if closed < segs.len() {
            segs[closed].start
        } else {
            segs[closed - 1].end
        };
        let cut_sample = (cut_frame * self.spf()).min(self.pending.len());
        self.pending.drain(..cut_sample);
        self.base_sample += cut_sample;
        self.probs.drain(..cut_frame.min(self.probs.len()));
        out
    }

    /// Transcribe pending frames `[a, b)` (relative to the pending buffer) and record the segment.
    fn emit(&mut self, a: usize, b: usize) -> Segment {
        let spf = self.spf();
        let (s0, s1) = (a * spf, (b * spf).min(self.pending.len()));
        let offset_s = (self.base_sample + s0) as f64 / 16_000.0;
        if self.collect_only {
            self.ranges
                .push((self.base_sample + s0, self.base_sample + s1));
            return Segment {
                start: offset_s,
                end: (self.base_sample + s1) as f64 / 16_000.0,
                text: String::new(),
                words: Vec::new(),
            };
        }
        let (seg, em, _) = self.tr.transcribe_one(&self.pending[s0..s1], offset_s);
        let base_frame = (self.base_sample + s0) / spf;
        self.emissions.extend(em.into_iter().map(|e| Emission {
            token: e.token,
            frame: e.frame + base_frame,
            duration: e.duration,
        }));
        if !seg.text.is_empty() {
            self.segments.push(seg.clone());
        }
        seg
    }

    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}
