//! End-to-end single-utterance transcription (PLAN 2.12).
use std::path::Path;

use crate::decoder::{Decoder, Emission};
use crate::encoder::Encoder;
use crate::format::ModelFile;
use crate::frontend::{Frontend, MelConfig};
use crate::ternary_linear::Precision;
use crate::tokenizer::Tokenizer;
use crate::Result;

pub struct Transcriber {
    pub frontend: Frontend,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub tokenizer: Tokenizer,
    pub frame_seconds: f64,
}

pub struct Transcript {
    pub text: String,
    pub emissions: Vec<Emission>,
    pub n_frames: usize,
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
        let encoder = Encoder::load_with(&m, precision)?;
        let decoder = Decoder::load(&m)?;
        let tokenizer = Tokenizer::load(dir.as_ref().join("tokenizer.json"))?;
        Ok(Self {
            frontend,
            encoder,
            decoder,
            tokenizer,
            frame_seconds: m.ternary.vad_head.frame_seconds,
        })
    }

    pub fn transcribe_samples(&self, samples: &[f32]) -> Transcript {
        let profiling = std::env::var_os("PRX_PROFILE").is_some();
        let t0 = std::time::Instant::now();
        let (feats, t) = self.frontend.compute(samples);
        let t1 = std::time::Instant::now();
        let (enc, te) = self.encoder.forward(&feats, t, None, None);
        let t2 = std::time::Instant::now();
        let proj = self.decoder.project_encoder(&enc, te);
        let emissions = self.decoder.greedy(&proj, te, None);
        if profiling {
            eprintln!(
                "pipeline profile: frontend {:.3}s · encoder {:.3}s · decoder {:.3}s ({} steps)",
                (t1 - t0).as_secs_f64(),
                (t2 - t1).as_secs_f64(),
                t2.elapsed().as_secs_f64(),
                emissions.len()
            );
        }
        let ids: Vec<usize> = emissions.iter().map(|e| e.token).collect();
        Transcript {
            text: self.tokenizer.decode(&ids),
            emissions,
            n_frames: te,
        }
    }

    pub fn transcribe_wav(&self, path: impl AsRef<Path>) -> Result<Transcript> {
        let a = crate::audio::read_wav(path)?;
        Ok(self.transcribe_samples(&a.samples))
    }
}
