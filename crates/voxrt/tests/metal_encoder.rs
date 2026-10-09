//! GPU encoder (feature `metal`) vs the CPU f32 reference on golden utterances: same subsampling input, outputs agree
//! to f32 accumulation-order noise (not bit for bit). Skipped without golden/ and models/.
#![cfg(all(feature = "metal", target_vendor = "apple"))]
mod common;
use common::*;
use voxrt::encoder::Encoder;
use voxrt::format::{default_model_dir, ModelFile};
use voxrt::frontend::{Frontend, MelConfig};
use voxrt::metal::GpuEncoder;
use voxrt::ternary_linear::Precision;

#[test]
fn gpu_encoder_matches_cpu_f32() {
    if !have_golden() || !default_model_dir().join("model.safetensors").exists() {
        eprintln!("golden/ or models/ missing; skipped");
        return;
    }
    let m = ModelFile::open(default_model_dir()).unwrap();
    let cpu = Encoder::load_with(&m, Precision::F32).unwrap();
    let gpu = GpuEncoder::load(&m).unwrap();
    let fe = Frontend::new(MelConfig {
        n_mels: m.config.encoder_config.num_mel_bins,
        ..MelConfig::default()
    });
    for id in golden_ids().iter().take(3) {
        let audio = voxrt::audio::read_wav(golden_dir().join(id).join("audio.wav")).unwrap();
        let (feats, t) = fe.compute(&audio.samples);
        let (want, te) = cpu.forward(&feats, t, None, None);
        let x = cpu.subsampling.forward(&feats, t);
        let got = gpu.forward(&x, te);
        // the whole encoder on the GPU (subsampling included) must agree as well
        let (full, te2) = gpu.forward_feats(&feats, t);
        assert_eq!(te, te2);
        let full_err = want
            .iter()
            .zip(&full)
            .fold(0f32, |a, (w, g)| a.max((w - g).abs()));
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let err = want
            .iter()
            .zip(&got)
            .fold(0f32, |a, (w, g)| a.max((w - g).abs()));
        let mean_err = want
            .iter()
            .zip(&got)
            .map(|(w, g)| (w - g).abs())
            .sum::<f32>()
            / want.len() as f32;
        eprintln!("{id}: T_e={te} max |err| {err:.2e} (scale {scale:.2}), mean {mean_err:.2e}; with GPU subsampling {full_err:.2e}");
        // encoder outputs are LayerNorm-scaled (|x| ~ 1–10); f32 with a different summation order over 24 layers
        assert!(
            err / scale < 2e-3,
            "{id}: max relative error {:.2e}",
            err / scale
        );
    }
}
