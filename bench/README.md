# Same-machine benchmark

`bench/run_all.py` runs every runtime on the same audio, one utterance at a time, model loaded once per runtime,
two warm-up utterances, **processing time only** (model load excluded), one number per runtime, and scores every
output with the leaderboard-exact scorer (`reference-py/score.py`: Open ASR normaliser + compound-merging alignment).

```
bench/.venv/bin/python bench/run_all.py --set dev50 \
  --runtimes voxrt-int8,voxrt-f32,sherpa,onnx-asr,parakeet.cpp,parakeet-mlx,photon-cpu,photon-mps
```

## Results, 2026-10-07 (dev50: 50 LibriSpeech dev-clean utterances, 371 s)

Machine: Apple M4 Max (14 cores), 36 GB, macOS 26.6.2, rustc 1.95.0. Thermal state moves numbers ~15 %;
only back-to-back pairs are comparable — the pairs below were measured together.

| runtime | model | device | × real time | WER % dev50 |
|---|---|---|---|---|
| **voxrt int8 (this repo)** | parakeet-redux (ternary) | CPU | **78** | 1.68 |
| Photon CPU (moondream 2.6.1, black box) | parakeet-redux | CPU | 70 | 1.68 |
| Photon Apple GPU (`mps`) | parakeet-redux | GPU | 33 | 1.78 |
| sherpa-onnx 1.13.8 int8 | parakeet-tdt-0.6b-v3 (teacher) | CPU | 28 | 1.88 |
| onnx-asr 0.12.0 int8 | teacher | CPU (+CoreML EP partial) | 18 | 1.78 |
| parakeet.cpp 2de154c q8_0 | teacher | CPU | 13 | 1.28 |
| parakeet-mlx 0.5.3 bf16 | teacher | Apple GPU (MLX 0.32.3) | 60‡ | 1.48 |
| voxrt f32 (reference path) | parakeet-redux dequantised | CPU | 11 | 1.78 |

Longer audio (single file, best of 3):

| | voxrt int8 (CPU) | Photon CPU | parakeet-mlx (GPU, teacher) |
|---|---|---|---|
| 26.6 s utterance | 0.26 s (104×) | 0.37 s (72×) | 0.23 s (114×) |
| 5.8 min talk | 3.9 s (90×) | 5.1 s (68×) | 5.2 s (67×) |
| TED-LIUM 3 long-form, 148 min | 92×, WER 2.55 | 65×, WER 2.52 | — |
| LibriSpeech test-clean, 2,620 utts | 79×, WER 1.95 | 51×*, WER 1.96 | — |

\* Photon's test-clean pass ran while other jobs were active; its dev50 number is the clean one.
‡ Corrected 2026-10-09: the harness's per-utterance timer (`timed()`) computed total audio ÷ Σ(time ÷ clip length)
instead of total audio ÷ total time, which inflated parakeet-mlx (the only runtime measured with it after the change)
to 357–367×. Recomputed from the same per-utterance timings: 60× (95× when timed from decoded samples, without its
ffmpeg-based file loading). sherpa-onnx and onnx-asr were measured before the change; rerun with the fixed timer they
give 30× and 19×.

Reading the table: the teacher-model runtimes score a different model (the distilled ternary model is ~0.4 WER
behind its teacher on LibriSpeech), so compare them on speed only. parakeet-mlx on the GPU is on par with the CPU
runtimes (its greedy decoding runs step by step through MLX). Among CPU runtimes of the ternary model, voxrt and
Photon are the only two at this speed; the open ONNX/GGUF runtimes of the teacher are 3–6× slower.

## Results, Apple GPU, 2026-10-09 (same M4 Max)

Encoder (subsampling + 24 layers) on the GPU via Metal, decoder on the CPU; the default on Apple Silicon. Back to back
with `run_all.py --runtimes voxrt-int8,voxrt-gpu,photon-cpu,photon-mps,parakeet-mlx` (dev50; parakeet-mlx and the
Neural Engine row from the corrected rerun the same day, `--runtimes voxrt-ane,voxrt-gpu,voxrt-int8,parakeet-mlx,…`):

| runtime | × real time | WER % dev50 |
|---|---|---|
| **voxrt Neural Engine + GPU** (with a Core ML bundle) | **214** | 1.78 |
| **voxrt GPU** | **202** | 1.78 |
| Photon GPU (`mps`) | 100 | 1.78 |
| voxrt CPU | 72 | 1.68 |
| Photon CPU | 71 | 1.68 |
| parakeet-mlx (teacher, bf16) | 60 (95 from decoded samples) | 1.48 |
| sherpa-onnx int8 (teacher, CPU) | 30 | 1.88 |
| onnx-asr int8 (teacher) | 19 | 1.78 |
| parakeet.cpp q8_0 (teacher, CPU) | 13 | 1.28 |

Photon GPU measures ~100× on its own but 33× when it runs after parakeet-mlx in the same harness session (the MLX
model stays resident on the GPU); the table has the clean number, and voxrt was not affected (202× in both orders).


test-clean 197× · WER 1.96 (CPU 79× · 1.95); TED-LIUM long-form 174× · WER 2.54. GPU encoder output equals the CPU f32
reference to ~2·10⁻⁶ relative (`tests/metal_encoder.rs`); ternary GEMMs reach 8–9 TFLOP/s f32 on long inputs
(`tests/metal_bench.rs`; MLX's tuned 2-bit kernel reaches ~10 on the same GPU). Production view (`prod_bench.py`):
2 s clip 17.5 ms warm and on the first call, 0.6–0.9 CPU cores busy (CPU path ~10), energy impact per audio-minute
75 vs 586.

## Results, FluidAudio, 2026-10-09 (same M4 Max)

[FluidAudio](https://github.com/FluidInference/FluidAudio) (`0b0fa2a`, Swift, Core ML) is the runtime most Mac
dictation and meeting apps use for Parakeet. It ships a Core ML build of the same ternary model
(`FluidInference/parakeet-redux-coreml`, exact 2-bit weights) and of the 6-bit teacher (`v3`), encoder on the Neural
Engine by default or on the GPU. Driven by `bench/fluid_driver` (model loaded once, two warm-up files, each file timed
from URL to text, as for the other runtimes); `run_all.py --runtimes voxrt-ane,voxrt-gpu,fluid-v3-ane,fluid-v3-gpu,fluid-redux-ane,fluid-redux-gpu`,
back to back:

| runtime | model | × real time dev50 | WER % dev50 | × real time test-clean | WER % test-clean |
|---|---|---|---|---|---|
| **voxrt Neural Engine + GPU** | parakeet-redux | **213** | 1.78 | **210** | 1.95 |
| **voxrt GPU** | parakeet-redux | **201** | 1.78 | **201** | 1.96 |
| FluidAudio v3, Neural Engine | teacher, 6-bit | 141 | 1.68 | 138 | 1.63 |
| FluidAudio v3, GPU | teacher, 6-bit | 124 | 1.58 | — | — |
| FluidAudio redux, GPU | parakeet-redux | 111 | 1.88 | — | — |
| FluidAudio redux, Neural Engine | parakeet-redux | 94 | 1.88 | 92 | 2.00 |

Median time per clip (test-clean for the Neural Engine rows, dev50 for the GPU rows) vs voxrt's `prod_bench.py`:

| clip length | voxrt (Neural Engine + GPU) | FluidAudio v3, ANE | FluidAudio redux, ANE | FluidAudio redux, GPU |
|---|---|---|---|---|
| < 4 s | 17 ms (2 s clip) | 39 ms | 62 ms | 51 ms |
| 4–8 s | 22 ms (5 s clip) | 46 ms | 71 ms | 61 ms |
| 8–15 s | 33–37 ms (10 s clip) | 59 ms | 84 ms | 74 ms |
| > 15 s | 107 ms (27 s clip) | 112 ms | 159 ms | 131 ms |

FluidAudio runs the encoder on a fixed 15 s window, so a 2 s clip costs about as much as a 15 s one; voxrt sizes the
work to the audio (length buckets on the Neural Engine, exact lengths on the GPU), which is where most of the gap on
short clips comes from. Its Neural Engine build of parakeet-redux compiles for 5–25 minutes on first load (cached
after). The teacher runtimes score a different model, so compare those rows on speed only.

## Results, x86, 2026-10-08 (Google Cloud c4d-standard-16)

Machine: AMD EPYC 9B45 (Turin, Zen 5), 8 cores / 16 threads, Ubuntu 24.04, rustc 1.99.0; voxrt with the AVX-512 VNNI
kernel (picked at run time), parakeet.cpp 2de154c built `-march=native`, Photon (moondream 2.6.1) black box. Same
method as above; dev50 measured twice in opposite orders (both rounds shown in prose, the table has round 1).

| runtime | model | × real time dev50 | WER % dev50 | × real time test-clean | WER % test-clean |
|---|---|---|---|---|---|
| **voxrt int8 (this repo)** | parakeet-redux (ternary) | **104** | 1.68 | **105** | 1.95 |
| Photon CPU | parakeet-redux | 98 | 1.68 | 97 | 1.94 |
| parakeet.cpp q8_0 | teacher | 42 | 1.18 | — | — |

Round 2 (reversed order): voxrt 104, Photon 99, parakeet.cpp 41. voxrt's test-clean output on this machine is byte-identical
to the Mac run (2,620/2,620 transcripts and emissions). Per core (`tests/x86_bench.rs`): VNNI 337–359 GMAC/s on
the FF shapes, AVX2 178–181, decoder head GEMV 54 µs.

## Install

- voxrt: `cargo build --release` (models in `models/parakeet-redux`, see the top-level README).
- Python runtimes: `uv venv bench/.venv && uv pip install --python bench/.venv/bin/python sherpa-onnx onnx-asr soundfile kaldialign parakeet-mlx`;
  sherpa-onnx int8 model from the sherpa-onnx releases into `bench/models/`.
- parakeet.cpp: clone into `bench/parakeet.cpp`, `cmake -B build && cmake --build build`, q8_0 GGUF into `bench/models/`.
- Photon: separate venv `bench/.venv-photon` with `moondream`, driven only through the model card's usage
  (`bench/photon_driver.py`); its source is never read.
- The dev50 / test-clean / TED-LIUM manifests: `reference-py/` scripts.
