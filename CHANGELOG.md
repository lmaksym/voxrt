# Changelog

## 0.1.0 — unreleased (target: first public release)
- Rust engine for `moondream/parakeet-redux`: NeMo-exact log-mel frontend, FastConformer encoder, TDT greedy decoder
  with NeMo semantics, tokenizer; fp32 path token-identical to the Python reference on LibriSpeech test-clean.
- Native ternary kernels: NEON I8MM (`smmla`) and SDOT tiles, 2-bit packed weights, int8 per-group activations,
  int8 attention core, int8 decoder with an `sdot` GEMV; every SIMD kernel bit-identical to the scalar reference.
- Long-form: VAD-head segmentation (10–60 s at pauses), segment and word timestamps, encoder/decoder pipeline,
  streaming API equal to offline by construction. Any-rate / any-format WAV input.
- CLI `prx` (txt/srt/json, stdin, threads), C ABI (`include/prx.h`), Python wheel (abi3), Swift package (XCFramework).
- Benchmarks: 79× real time on LibriSpeech test-clean and 92× on TED-LIUM long-form on an M4 Max CPU, WER 1.95 / 2.55.
