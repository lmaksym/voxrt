//! Neural Engine encoder (feature `coreml`) vs the CPU f32 reference on golden utterances. The Neural Engine computes
//! in fp16 with int8 weights, so the bar is direction (cosine), not bit-level agreement; the WER gate is the real
//! accuracy check. Skipped without golden/, models/ or a converted bundle in `<model>/coreml`.
#![cfg(all(feature = "coreml", target_vendor = "apple"))]
mod common;
use common::*;
use voxrt::coreml::AneEncoder;
use voxrt::encoder::Encoder;
use voxrt::format::{default_model_dir, ModelFile};
use voxrt::frontend::{Frontend, MelConfig};
use voxrt::ternary_linear::Precision;

#[test]
fn ane_encoder_tracks_cpu_f32() {
    let bundle = AneEncoder::bundle_dir(&default_model_dir());
    if !have_golden() || !bundle.join("encoder.json").exists() {
        eprintln!("golden/ or the Core ML bundle missing; skipped");
        return;
    }
    let m = ModelFile::open(default_model_dir()).unwrap();
    let cpu = Encoder::load_with(&m, Precision::F32).unwrap();
    let ane = AneEncoder::load_upto(&bundle, 128).unwrap();
    let fe = Frontend::new(MelConfig {
        n_mels: m.config.encoder_config.num_mel_bins,
        ..MelConfig::default()
    });
    for id in golden_ids().iter().take(3) {
        let audio = voxrt::audio::read_wav(golden_dir().join(id).join("audio.wav")).unwrap();
        let (feats, t) = fe.compute(&audio.samples);
        let (want, te) = cpu.forward(&feats, t, None, None);
        let x = cpu.subsampling.forward(&feats, t);
        let got = ane.forward(&x, te).expect("fits a bucket");
        let dot: f64 = want
            .iter()
            .zip(&got)
            .map(|(w, g)| (*w as f64) * (*g as f64))
            .sum();
        let nw: f64 = want.iter().map(|w| (*w as f64).powi(2)).sum::<f64>().sqrt();
        let ng: f64 = got.iter().map(|g| (*g as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (nw * ng);
        eprintln!("{id}: T_e={te} cosine {cos:.6}");
        assert!(cos > 0.999, "{id}: cosine {cos:.6}");
    }
}
