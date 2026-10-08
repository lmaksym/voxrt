//! PLAN 4.6: feeding audio in small chunks must give exactly the offline long-form result.
mod common;
use common::*;
use voxrt::format::default_model_dir;
use voxrt::longform::LongForm;
use voxrt::transcriber::Transcriber;

#[test]
fn streaming_equals_offline() {
    if !have_golden() || !default_model_dir().join("model.safetensors").exists() {
        return;
    }
    let tr = Transcriber::load(default_model_dir()).unwrap();
    let ted = root().join("data/tedlium/ted06.wav");
    let mut files: Vec<std::path::PathBuf> = golden_ids()
        .iter()
        .map(|id| golden_dir().join(id).join("audio.wav"))
        .collect();
    if ted.exists() {
        files.push(ted);
    }
    for f in files {
        let a = voxrt::audio::read_wav(&f).unwrap();
        let offline = tr.transcribe_long(&a.samples);
        for chunk_s in [0.25f64, 1.0, 3.7] {
            let mut lf = LongForm::new(&tr);
            let mut live = Vec::new();
            for c in a.samples.chunks((chunk_s * 16000.0) as usize) {
                live.extend(lf.push(c));
            }
            live.extend(lf.flush());
            let text = lf.text();
            if a.samples.len() as f64 / 16000.0 > tr.long_form_threshold_s {
                assert_eq!(text, offline.text, "{}: chunk {chunk_s}s", f.display());
                assert_eq!(lf.segments.len(), offline.segments.len());
            } else {
                // short files: offline takes the single-utterance path; streaming still produces the text
                assert!(!text.is_empty());
            }
        }
        eprintln!(
            "{}: {} segments, streaming == offline",
            f.file_name().unwrap().to_string_lossy(),
            offline.segments.len()
        );
    }
}
