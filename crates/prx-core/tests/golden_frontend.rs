//! PLAN 2.4 / 2.5: audio reader and frontend vs the golden set (skipped when golden/ is absent).
mod common;
use common::*;
use prx_core::frontend::{slaney_mel_filterbank, Frontend, MelConfig};

#[test]
fn mel_filterbank_matches_librosa() {
    let want: Vec<f32> =
        std::fs::read(root().join("crates/prx-core/tests/fixtures/mel_fb_128x257.f32.bin"))
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
    let got = slaney_mel_filterbank(16_000, 512, 128);
    assert_eq!(got.len(), want.len());
    let d = max_abs_diff(&got, &want);
    assert!(d < 1e-6, "filterbank max abs diff {d}");
}

#[test]
fn wav_reader_matches_golden_waveform() {
    if !have_golden() {
        return;
    }
    for id in golden_ids() {
        let d = golden_dir().join(&id);
        let a = prx_core::audio::read_wav(d.join("audio.wav")).unwrap();
        let g = read_npy(d.join("waveform.npy"));
        assert_eq!(a.samples.len(), g.data.len(), "{id}");
        assert_eq!(a.samples, g.data, "{id}: samples differ");
    }
}

#[test]
fn log_mel_and_features_match_golden() {
    if !have_golden() {
        return;
    }
    let fe = Frontend::new(MelConfig::default());
    for id in golden_ids() {
        let d = golden_dir().join(&id);
        let wav = read_npy(d.join("waveform.npy")).data;
        let n = fe.n_frames(wav.len());
        let logmel_g = read_npy(d.join("logmel.npy")); // [T_hf, 128] with T_hf = n + 1 (HF keeps a masked extra frame)
        let feats_g = read_npy(d.join("features.npy"));
        assert_eq!(logmel_g.shape[1], 128);
        assert!(
            logmel_g.shape[0] == n || logmel_g.shape[0] == n + 1,
            "{id}: golden frames {} vs ours {n}",
            logmel_g.shape[0]
        );
        let lm = fe.log_mel(&wav);
        let d1 = max_abs_diff(&lm, &logmel_g.data[..n * 128]);
        let (f, _) = fe.compute(&wav);
        let d2 = max_abs_diff(&f, &feats_g.data[..n * 128]);
        eprintln!("{id}: frames={n} logmel max|Δ|={d1:.2e} features max|Δ|={d2:.2e}");
        assert!(d1 < 1e-3, "{id}: log-mel diff {d1}");
        assert!(d2 < 5e-4, "{id}: features diff {d2}"); // f32 accumulation-order noise vs torch; see docs/frontend.md
        if logmel_g.shape[0] == n + 1 {
            assert!(
                feats_g.data[n * 128..].iter().all(|&v| v == 0.0),
                "{id}: HF's extra frame should be masked to 0"
            );
        }
    }
}
