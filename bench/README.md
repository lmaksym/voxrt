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
| parakeet-mlx 0.5.3 bf16 | teacher | Apple GPU (MLX 0.32.3) | 357 | 1.48 |
| voxrt f32 (reference path) | parakeet-redux dequantised | CPU | 11 | 1.78 |

Longer audio (single file, best of 3):

| | voxrt int8 (CPU) | Photon CPU | parakeet-mlx (GPU, teacher) |
|---|---|---|---|
| 26.6 s utterance | 0.26 s (104×) | 0.37 s (72×) | 0.23 s (114×) |
| 5.8 min talk | 3.9 s (90×) | 5.1 s (68×) | 5.2 s (67×) |
| TED-LIUM 3 long-form, 148 min | 92×, WER 2.55 | 65×, WER 2.52 | — |
| LibriSpeech test-clean, 2,620 utts | 79×, WER 1.95 | 51×*, WER 1.96 | — |

\* Photon's test-clean pass ran while other jobs were active; its dev50 number is the clean one.

Reading the table: the teacher-model runtimes score a different model (the distilled ternary model is ~0.4 WER
behind its teacher on LibriSpeech), so compare them on speed only. parakeet-mlx on the GPU is the fastest thing on
this machine for short clips (the GPU is idle otherwise and the per-utterance work is tiny) and on par with the CPU
runtimes for long audio, where its per-step decoding dominates. Among CPU runtimes of the ternary model, voxrt and
Photon are the only two at this speed; the open ONNX/GGUF runtimes of the teacher are 3–6× slower.

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
