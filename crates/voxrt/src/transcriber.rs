//! End-to-end transcription: single utterance and VAD-segmented long-form with timestamps.
use std::path::Path;

use crate::decoder::{Decoder, Emission};
use crate::encoder::Encoder;
use crate::format::ModelFile;
use crate::frontend::{Frontend, MelConfig};
use crate::ternary_linear::Precision;
use crate::tokenizer::Tokenizer;
use crate::vad::{SegmentConfig, VadHead};
use crate::Result;

pub struct Transcriber {
    pub frontend: Frontend,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub tokenizer: Tokenizer,
    pub vad: VadHead,
    pub frame_seconds: f64,
    pub hop: usize,
    pub subsampling_factor: usize,
    /// Audio longer than this goes through VAD segmentation automatically.
    pub long_form_threshold_s: f64,
    pub segment_cfg: SegmentConfig,
    /// Encoder layers on the Apple GPU (`VOXRT_DEVICE=gpu`, feature `metal`).
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    pub gpu: Option<crate::metal::GpuEncoder>,
}

#[derive(Debug, Clone)]
pub struct Word {
    pub start: f64,
    pub end: f64,
    pub word: String,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub words: Vec<Word>,
}

pub struct Transcript {
    pub text: String,
    pub emissions: Vec<Emission>,
    pub n_frames: usize,
    pub segments: Vec<Segment>,
}

impl Transcriber {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with(dir, Precision::from_env())
    }

    pub fn load_with(dir: impl AsRef<Path>, precision: Precision) -> Result<Self> {
        let m = ModelFile::open(&dir)?;
        let frontend = Frontend::new(MelConfig {
            n_mels: m.config.encoder_config.num_mel_bins,
            ..MelConfig::default()
        });
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        let gpu = match std::env::var("VOXRT_DEVICE").as_deref() {
            // GPUs without Apple's simdgroup matrix units (e.g. Intel Macs) cannot build the kernels: fall back to CPU
            Ok("gpu") => match crate::metal::GpuEncoder::load(&m) {
                Ok(g) => {
                    g.warm_up(64);
                    Some(g)
                }
                Err(e) => {
                    eprintln!("voxrt: GPU unavailable ({e}); using the CPU");
                    None
                }
            },
            _ => None,
        };
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        let encoder = if gpu.is_some() {
            Encoder::load_subsampling_only(&m, precision)?
        } else {
            Encoder::load_with(&m, precision)?
        };
        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        let encoder = Encoder::load_with(&m, precision)?;
        let decoder = Decoder::load(&m)?;
        let vad = VadHead::load(&m)?;
        let tokenizer = Tokenizer::load(dir.as_ref().join("tokenizer.json"))?;
        Ok(Self {
            frontend,
            encoder,
            decoder,
            tokenizer,
            vad,
            frame_seconds: m.ternary.vad_head.frame_seconds,
            hop: 160,
            subsampling_factor: m.config.encoder_config.subsampling_factor,
            long_form_threshold_s: 30.0,
            segment_cfg: SegmentConfig::from_env(),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            gpu,
        })
    }

    /// Subsampling output `([t_e, d], t_e)` for log-mel `feats` on the configured device (VAD input).
    pub fn subsample(&self, feats: &[f32], t: usize) -> (Vec<f32>, usize) {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        if let Some(g) = &self.gpu {
            return g.subsample(feats, t);
        }
        (
            self.encoder.subsampling.forward(feats, t),
            self.encoder.subsampling.out_frames(t),
        )
    }

    /// Encoder on the configured device: the CPU, or the GPU (subsampling and layers in one command buffer).
    pub fn encode(&self, feats: &[f32], t: usize) -> (Vec<f32>, usize) {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        if let Some(g) = &self.gpu {
            let t0 = std::time::Instant::now();
            let (out, te) = g.forward_feats(feats, t);
            if std::env::var_os("VOXRT_PROFILE").is_some() {
                eprintln!("gpu encoder (T_e={te}): {:.3}s", t0.elapsed().as_secs_f64());
            }
            return (out, te);
        }
        self.encoder.forward(feats, t, None, None)
    }

    /// Samples per encoder frame (hop × subsampling factor = 1280 → 80 ms).
    pub fn samples_per_frame(&self) -> usize {
        self.hop * self.subsampling_factor
    }

    /// One utterance, no segmentation. `offset_s` shifts timestamps.
    pub fn transcribe_samples(&self, samples: &[f32]) -> Transcript {
        let (seg, emissions, te) = self.transcribe_one(samples, 0.0);
        Transcript {
            text: seg.text.clone(),
            emissions,
            n_frames: te,
            segments: vec![seg],
        }
    }

    pub(crate) fn transcribe_one(
        &self,
        samples: &[f32],
        offset_s: f64,
    ) -> (Segment, Vec<Emission>, usize) {
        let (proj, te) = self.encode_segment(samples);
        self.decode_segment(proj, te, offset_s)
    }

    /// Frontend + encoder + encoder projection: the multi-threaded half of a segment.
    pub(crate) fn encode_segment(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let profiling = std::env::var_os("VOXRT_PROFILE").is_some();
        let t0 = std::time::Instant::now();
        let (feats, t) = self.frontend.compute(samples);
        let t1 = std::time::Instant::now();
        let (enc, te) = self.encode(&feats, t);
        let proj = self.decoder.project_encoder(&enc, te);
        if profiling {
            eprintln!(
                "pipeline profile: frontend {:.3}s · encoder {:.3}s",
                (t1 - t0).as_secs_f64(),
                t1.elapsed().as_secs_f64()
            );
        }
        (proj, te)
    }

    /// Greedy TDT decode + detokenise + timestamps: the single-threaded half of a segment.
    pub(crate) fn decode_segment(
        &self,
        proj: Vec<f32>,
        te: usize,
        offset_s: f64,
    ) -> (Segment, Vec<Emission>, usize) {
        let t2 = std::time::Instant::now();
        let emissions = self.decoder.greedy(&proj, te, None);
        if std::env::var_os("VOXRT_PROFILE").is_some() {
            eprintln!(
                "pipeline profile: decoder {:.3}s ({} steps)",
                t2.elapsed().as_secs_f64(),
                emissions.len()
            );
        }
        let ids: Vec<usize> = emissions.iter().map(|e| e.token).collect();
        let text = self.tokenizer.decode(&ids);
        let end_s = offset_s + te as f64 * self.frame_seconds;
        let words = self.words(&emissions, offset_s, end_s);
        (
            Segment {
                start: offset_s,
                end: end_s,
                text,
                words,
            },
            emissions,
            te,
        )
    }

    /// Word timestamps from emission frames: a word starts at its first piece's frame; it ends where the next word starts
    /// (or at the segment end). Pieces group into words at `▁`.
    fn words(&self, emissions: &[Emission], offset_s: f64, end_s: f64) -> Vec<Word> {
        let mut words: Vec<Word> = Vec::new();
        let mut cur_ids: Vec<usize> = Vec::new();
        let mut cur_start = 0f64;
        let flush = |ids: &mut Vec<usize>, start: f64, words: &mut Vec<Word>, tok: &Tokenizer| {
            if !ids.is_empty() {
                let w = tok.decode(ids);
                if !w.is_empty() {
                    words.push(Word {
                        start,
                        end: start,
                        word: w,
                    });
                }
                ids.clear();
            }
        };
        for e in emissions {
            let piece = self.tokenizer.piece(e.token).unwrap_or("");
            let starts_word = piece.starts_with('\u{2581}');
            if starts_word {
                flush(&mut cur_ids, cur_start, &mut words, &self.tokenizer);
                cur_start = offset_s + e.frame as f64 * self.frame_seconds;
            } else if cur_ids.is_empty() {
                cur_start = offset_s + e.frame as f64 * self.frame_seconds;
            }
            cur_ids.push(e.token);
        }
        flush(&mut cur_ids, cur_start, &mut words, &self.tokenizer);
        let n = words.len();
        for i in 0..n {
            words[i].end = if i + 1 < n { words[i + 1].start } else { end_s };
        }
        words
    }

    /// VAD-segmented long-form transcription. Segment boundaries come from the streaming state machine
    /// (`longform::LongForm`, fed in 10 s chunks, so cuts are identical to a live stream); the segments are then run as a
    /// two-stage pipeline: the multi-threaded encoder on this thread, the single-threaded decoder on a helper thread.
    pub fn transcribe_long(&self, samples: &[f32]) -> Transcript {
        let secs = samples.len() as f64 / 16000.0;
        if secs <= self.long_form_threshold_s {
            return self.transcribe_samples(samples);
        }
        let mut lf = crate::longform::LongForm::new(self);
        lf.collect_only = true;
        for chunk in samples.chunks(10 * 16_000) {
            lf.push(chunk);
        }
        lf.flush();
        let ranges = lf.ranges.clone();
        let spf = self.samples_per_frame();
        let n_frames = samples.len() / spf;
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Vec<f32>, usize, f64)>(2);
        let decoded: Vec<(Segment, Vec<Emission>, usize)> = std::thread::scope(|sc| {
            let dec = sc.spawn(move || {
                let mut out = Vec::new();
                while let Ok((start, proj, te, offset_s)) = rx.recv() {
                    out.push((start, self.decode_segment(proj, te, offset_s)));
                }
                out
            });
            for &(a, b) in &ranges {
                let (proj, te) = self.encode_segment(&samples[a..b]);
                tx.send((a, proj, te, a as f64 / 16_000.0))
                    .expect("decoder thread alive");
            }
            drop(tx);
            dec.join().expect("decoder thread")
        })
        .into_iter()
        .map(|(start, (seg, em, te))| {
            let base_frame = start / spf;
            (
                seg,
                em.into_iter()
                    .map(|e| Emission {
                        token: e.token,
                        frame: e.frame + base_frame,
                        duration: e.duration,
                    })
                    .collect(),
                te,
            )
        })
        .collect();
        let mut segments = Vec::new();
        let mut emissions = Vec::new();
        for (seg, em, _) in decoded {
            emissions.extend(em);
            if !seg.text.is_empty() {
                segments.push(seg);
            }
        }
        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Transcript {
            text,
            emissions,
            n_frames,
            segments,
        }
    }

    pub fn transcribe_wav(&self, path: impl AsRef<Path>) -> Result<Transcript> {
        let a = crate::audio::read_wav(path)?;
        Ok(self.transcribe_long(&a.samples))
    }
}
