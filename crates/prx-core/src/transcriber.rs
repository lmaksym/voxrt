//! End-to-end single-utterance transcription (PLAN 2.12).
use std::path::Path;

use crate::decoder::{Decoder, Emission};
use crate::encoder::Encoder;
use crate::format::ModelFile;
use crate::frontend::{Frontend, MelConfig};
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
        let m = ModelFile::open(&dir)?;
        let frontend = Frontend::new(MelConfig {
            n_mels: m.config.encoder_config.num_mel_bins,
            ..MelConfig::default()
        });
        let encoder = Encoder::load(&m)?;
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
        let (feats, t) = self.frontend.compute(samples);
        let (enc, te) = self.encoder.forward(&feats, t, None, None);
        let proj = self.decoder.project_encoder(&enc, te);
        let emissions = self.decoder.greedy(&proj, te, None);
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
