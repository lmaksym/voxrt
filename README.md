# parakeet-redux-rs — open-source Rust runtime for Parakeet Redux

> Working directory name. The public project name is still open (the brief says `<name>`).
> Everything for this project lives in this folder. Knowledge (research notes, decisions,
> benchmark write-ups) goes to the vault: `02-Projects/Parakeet Redux Rust Runtime.md`.

## Status (2026-10-05)
- **Phase 0–1 done.** Python reference reproduces the card: LibriSpeech test-clean **WER 1.96** (leaderboard-exact scoring). Golden tensors dumped. See `PLAN.md`, `docs/decisions.md`.
- **Phase 2 in progress.** Rust scalar fp32 core (`crates/prx-core`): frontend, subsampling, 24-layer FastConformer, TDT greedy decoder and detokeniser all match the golden set (encoder rel. error ≤ 7e-6, decoder emissions exact). `prx transcribe file.wav` works at ~8× real time on an M4 Max with dequantised f32 weights.
- Two findings worth sharing: the leaderboard's compound-merging alignment is worth 0.4 WER on this model, and HF's `ParakeetTokenizer.decode` collapses repeated tokens (a CTC habit that is wrong for TDT; +0.07 WER).

```sh
cargo build --release
./target/release/prx inspect                      # validates models/parakeet-redux against ternary.json
./target/release/prx transcribe golden/5694-64025-0000/audio.wav
scripts/wer.sh dev50                              # Rust CLI over a manifest + leaderboard-exact WER
cargo test --release --workspace                  # unit tests + golden parity tests (skip if golden/ or models/ absent)
```

## What
A fast, dependency-light, Apache-2.0 inference engine in Rust for
[moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux),
a ternary-weight (1.58-bit) quantization-aware distillation of
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(25-language ASR, ~178 MB).

## Why
The weights are CC-BY-4.0, but the only runtime that executes them at full speed (Photon)
depends on proprietary, non-redistributable kernels. We want an open engine that
can be embedded in desktop, mobile and server apps, with CPU-first performance
competitive with existing Parakeet runtimes.

## Goals
- Bit-exact-enough parity with the reference (target: LibriSpeech test-clean WER ~1.96)
- Native ternary kernels: NEON (SDOT/I8MM), AVX2, AVX-512 VNNI; Metal later
- Long-form transcription via the bundled VAD head, word timestamps, streaming
- Pure Rust core, C ABI, Python and Swift bindings, CLI
- No dependency on Python or PyTorch at runtime

## Non-goals
Training, fine-tuning, other model families (for now).

## Model format
Full spec with verified numbers: [`docs/model-format.md`](docs/model-format.md).
Raw metadata from the model repo (fetched 2026-10-05): [`reference/`](reference/).

Summary (`ternary.json`, format `thrush-ternary-v2`):
- Ternary layers: `qweight` (U8, 5 trits/byte, base 3, LSD first) + `scales` (F16)
- `w = scales[row, col // 128] * (code - 1)`, code ∈ {0,1,2}
- 264 ternary tensors (11 per encoder layer × 24 layers: FF1/FF2 linears, q/k/v/o and
  relative-k projections, pointwise convs 1/2), 603,979,776 params;
  subsampling, depthwise convs, LSTM decoder, joint head and VAD head stay dense (23.3M params)

## Plan
Step-by-step checklist with gates: [`PLAN.md`](PLAN.md). Phase summary:
1. **Reference** (Python, dev-only): dequantize to fp32, reproduce published WER,
   dump per-layer golden tensors. → `reference-py/`
2. **Rust core**: safetensors loader + format validation, NeMo-exact log-mel frontend,
   FastConformer encoder (scalar fp32), TDT greedy decoder, tokenizer.
   Gate: golden-tensor parity + WER match.
3. **Kernels**: repack trits to 2-bit/int8 tiles at load, int8 activations per token,
   SIMD matmuls via `std::arch` with runtime feature detection; evaluate LUT (T-MAC style).
4. **Product features**: VAD segmentation, timestamps, streaming API, multithreading.
5. **Bindings & release**: C ABI, PyO3, Swift package, CLI, reproducible benchmarks
   vs parakeet.cpp, sherpa-onnx, parakeet-mlx.

Targets to beat (from the model card, single utterance, LibriSpeech): Photon 113× real time
on 8 Zen 5 cores (WER 1.94); 38× CPU / 43× GPU on an M2 Air. Incumbents on the same x86 box:
parakeet.cpp q8_0 45×, sherpa-onnx int8 42×, onnx-asr int8 28×.

## Clean-room policy
Built only from the CC-BY model repo, NVIDIA NeMo and HF transformers (Apache-2.0),
and public papers. No code, binaries or reverse engineering of proprietary runtimes.
See [`PROVENANCE.md`](PROVENANCE.md).

## Layout
```
parakeet-redux-rs/
├── README.md          # this brief
├── PLAN.md            # step-by-step checklist with phase gates (tick as you go)
├── PROVENANCE.md      # clean-room policy and allowed sources
├── docs/              # specs and design notes that belong with the code
│   └── model-format.md
├── reference/         # metadata fetched from the model repo (config.json, ternary.json, model card)
├── reference-py/      # phase 1: Python dequant, eval, golden-tensor dumper (dev-only)
├── crates/            # Rust workspace: prx-core (engine), prx-cli (`prx`)
├── scripts/           # wer.sh
├── third_party/       # vendored Open ASR Leaderboard normalizer (Apache-2.0)
├── results/           # eval outputs (jsonl + .wer.txt)
├── golden/            # per-layer golden tensors (gitignored)
└── models/            # downloaded weights (gitignored)
```

## Local environment (this Mac)
Apple M4 Max (arm64, NEON with SDOT/I8MM), rustc/cargo 1.95.0, Python 3.12.7.
x86 (AVX2 / AVX-512 VNNI) kernels need a separate box or CI runner.

## References
- Model: https://huggingface.co/moondream/parakeet-redux
- Teacher: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Release post: https://moondream.ai/blog/introducing-parakeet-redux-and-ultra
- NeMo: https://github.com/NVIDIA/NeMo
- TDT: Xu et al., "Efficient Sequence Transduction by Jointly Predicting Tokens and Durations", arXiv:2304.06795
- FastConformer: Rekesh et al., arXiv:2305.05084
- BitNet b1.58: Ma et al., arXiv:2402.17764
- T-MAC: Wei et al., arXiv:2407.00088

## Credits
Model weights © NVIDIA and M87 Labs (Moondream), CC-BY-4.0.
