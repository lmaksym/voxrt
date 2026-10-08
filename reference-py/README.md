# reference-py — dev-only Python reference

Never ships. Exists to (1) prove the dequantised model reproduces the card's WER, (2) dump golden tensors for the Rust tests,
(3) answer design questions (activation ranges, VAD head) before kernels are written.

```sh
cd reference-py && uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e .
.venv/bin/python data.py                  # LibriSpeech manifests + fixed dev50 (needs data/librispeech/LibriSpeech/{test,dev}-clean)
.venv/bin/python inspect_safetensors.py   # → docs/tensors.md, structural checks
.venv/bin/python trits.py                 # unpacker tests incl. zero-fraction match on all 264 modules
.venv/bin/python dequant.py               # → models/parakeet-redux-fp/ (HF checkpoint, ternary weights as exact F16)
.venv/bin/python voxrt_ref.py [cpu|mps]     # smoke: 3 dev utterances
.venv/bin/python eval.py --set dev50 --device mps     # → results/ref_<set>_<device>_<date>.jsonl + .wer.txt
.venv/bin/python dump_golden.py           # → golden/<id>/*.npy + golden/manifest.json (CPU, deterministic)
.venv/bin/python vad_reference.py         # VAD head probe → golden/vad/
.venv/bin/python act_stats.py             # activation ranges → results/act_stats_dev50.json
```
`voxrt_ref.py` runs the HF encoder and an explicit TDT greedy loop with NeMo semantics (`docs/tdt.md`) and decodes pieces directly (HF's `ParakeetTokenizer.decode` collapses repeated ids, D-2.11); `score.py` uses the vendored
Open ASR Leaderboard normaliser (`third_party/`). Model: HF `transformers` 5.18 `ParakeetForTDT`; MPS and CPU give identical WER on dev50.
