# Changelog

## 0.1.0 — 2026-10-09
- Rust engine for `moondream/parakeet-redux`: NeMo-exact log-mel frontend, FastConformer encoder, TDT greedy decoder
  with NeMo semantics, tokenizer; fp32 path token-identical to the Python reference on LibriSpeech test-clean.
- Native ternary kernels: NEON I8MM (`smmla`) and SDOT tiles, 2-bit packed weights, int8 per-group activations,
  int8 attention core, int8 decoder with an `sdot` GEMV; every SIMD kernel bit-identical to the scalar reference.
- Long-form: VAD-head segmentation (10–60 s at pauses), segment and word timestamps, encoder/decoder pipeline,
  streaming API equal to offline by construction. Any-rate / any-format WAV input.
- CLI `voxrt` (txt/srt/json, stdin, threads), C ABI (`include/voxrt.h`), Python wheel (abi3), Swift package (XCFramework).
- x86-64 kernels: AVX-512 VNNI and AVX2 (picked at run time), AVX2+FMA/F16C elementwise paths; output byte-identical
  across aarch64 and x86-64 (portable math, NEON-order reductions).
- Python: `voxrt.Transcriber()` downloads the model into the Hugging Face cache on first use; FLAC/OGG/MP3 via
  `soundfile` (`voxrt[audio]`), other formats via ffmpeg; results as dataclasses (`Transcript`, `Segment`, `Word`);
  type hints; pytest suite in CI.
- Apple GPU (Metal), the default on Apple Silicon: subsampling and the encoder layers as Metal kernels (2-bit ternary
  weights decoded in-shader), decoder on the CPU. M4 Max: dev50 202× (CPU 72×, Photon GPU 100×), test-clean 197× at
  WER 1.96, 2 s clip 17.5 ms, ~8× less energy than the CPU path. `VOXRT_DEVICE=cpu|gpu|auto`, Python `device=`; CPU
  fallback where the GPU lacks simdgroup matrices.
- macOS artefacts are universal (Apple Silicon + Intel Macs): CLI, C library, Python wheel (universal2), XCFramework.
- Streaming equals offline also for streams up to 30 s that end before a segment is finalised (whole-clip path).
- Benchmarks: 79× real time on LibriSpeech test-clean and 92× on TED-LIUM long-form on an M4 Max CPU, WER 1.95 / 2.55;
  105× on test-clean on an AMD Turin (GCP c4d-standard-16) vs Photon 97×.
