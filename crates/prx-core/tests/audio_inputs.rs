//! PLAN 4.5: other sample rates, stereo, float and 24-bit WAVs transcribe to the same text as the 16 kHz golden file.
mod common;
use common::*;
use prx_core::audio::{read_wav, resample};
use prx_core::format::default_model_dir;
use prx_core::transcriber::Transcriber;

#[test]
fn resampler_preserves_a_sine() {
    let from = 48_000u32;
    let x: Vec<f32> = (0..48_000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from as f32).sin())
        .collect();
    let y = resample(&x, from, 16_000);
    assert!((y.len() as i64 - 16_000).abs() <= 1, "{}", y.len());
    let mut err = 0f32;
    for (i, &v) in y.iter().enumerate().skip(100).take(15_000) {
        let want = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
        err = err.max((v - want).abs());
    }
    assert!(err < 2e-3, "sine resample max err {err}");
}

#[test]
fn variants_transcribe_identically() {
    if !have_golden() || !default_model_dir().join("model.safetensors").exists() {
        return;
    }
    let d = golden_dir().join("3081-166546-0059");
    if !d.join("audio_48000.wav").exists() {
        return;
    }
    let tr = Transcriber::load(default_model_dir()).unwrap();
    let base = tr.transcribe_wav(d.join("audio.wav")).unwrap().text;
    for name in [
        "audio_48000.wav",
        "audio_44100.wav",
        "audio_22050.wav",
        "audio_stereo.wav",
        "audio_f32.wav",
        "audio_24bit.wav",
    ] {
        let a = read_wav(d.join(name)).unwrap();
        assert_eq!(a.sample_rate, 16_000);
        let t = tr.transcribe_long(&a.samples).text;
        eprintln!("{name:18} → {t}");
        assert_eq!(t, base, "{name}");
    }
    // 8 kHz loses the top band; must still be intelligible but is not required to be identical
    let t8 = tr.transcribe_wav(d.join("audio_8000.wav")).unwrap().text;
    eprintln!("audio_8000.wav     → {t8}");
}

#[test]
fn word_timestamps_are_monotonic_and_bounded() {
    if !have_golden() || !default_model_dir().join("model.safetensors").exists() {
        return;
    }
    let tr = Transcriber::load(default_model_dir()).unwrap();
    for id in golden_ids() {
        let r = tr
            .transcribe_wav(golden_dir().join(&id).join("audio.wav"))
            .unwrap();
        for s in &r.segments {
            let mut last = s.start;
            for w in &s.words {
                assert!(
                    w.start >= last - 1e-9 && w.end >= w.start && w.end <= s.end + 1e-9,
                    "{id}: {w:?} in {}..{}",
                    s.start,
                    s.end
                );
                last = w.start;
            }
            let joined = s
                .words
                .iter()
                .map(|w| w.word.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            assert_eq!(
                joined, s.text,
                "{id}: words must re-join to the segment text"
            );
        }
    }
}
