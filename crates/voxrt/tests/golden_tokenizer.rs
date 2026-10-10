mod common;
use common::*;
use voxrt::tokenizer::Tokenizer;

#[test]
fn decode_matches_reference_text() {
    if !have_golden() {
        return;
    }
    let tok = Tokenizer::load(root().join("models/parakeet-redux/tokenizer.json")).unwrap();
    assert_eq!(tok.vocab_size(), 8192);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(golden_dir().join("manifest.json")).unwrap())
            .unwrap();
    for id in golden_ids() {
        let em = read_npy(golden_dir().join(&id).join("emissions.npy"));
        let ids: Vec<usize> = em
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| c[0] as usize)
            .collect();
        let want = manifest["utterances"][&id]["hyp_text"].as_str().unwrap();
        let got = tok.decode(&ids);
        assert_eq!(got, want, "{id}");
    }
    assert_eq!(tok.decode(&[8192, 8192]), "");
}
