---
license: cc-by-4.0
base_model: moondream/parakeet-redux
pipeline_tag: automatic-speech-recognition
library_name: coreml
language: [en, de, fr, es, it, pt, ru, uk, hr, sl, lv, lt, et, fi, sv, da, nl, pl, cs, sk, hu, ro, bg, el, mt]
tags: [parakeet, tdt, ternary, coreml, apple-neural-engine, apple-silicon, speech-recognition, voxrt]
---

# voxrt-parakeet-redux-coreml

The encoder of [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux), the ternary (every
encoder weight ∈ {-1, 0, +1}) re-training of `nvidia/parakeet-tdt-0.6b-v3`, as a Core ML model for the Apple Neural
Engine. It is the Neural Engine path of [voxrt](https://github.com/lmaksym/voxrt), a Rust runtime for the model:
voxrt runs the mel frontend, subsampling, TDT decoder and VAD itself and sends the 24 encoder layers here. It is not a
standalone speech-recognition model.

The weights are the checkpoint's exact ternary values, stored in 2 bits. Each linear layer is a chain of 1×1
convolutions, one per 128-column block: 2-bit codes into a single {-1, 0, +1} table (`constexpr_lut_to_dense`), with
that block's own fp16 per-row scales applied to its output. The whole encoder runs in the Neural Engine's native
`[1, C, 1, T]` layout. No weight is re-quantised; the Neural Engine computes in fp16.

## Usage

The Python package downloads this repository on first use on Apple silicon Macs:

```python
import voxrt
t = voxrt.Transcriber(device="ane")  # the default, "auto", runs the encoder on the GPU
print(t.transcribe("clip.wav").text)
```

For the CLI or the Swift/C bindings, place it next to the model:

```sh
git lfs clone https://huggingface.co/moondream/parakeet-redux models/parakeet-redux
git lfs clone https://huggingface.co/MaksL/voxrt-parakeet-redux-coreml models/parakeet-redux/coreml
VOXRT_DEVICE=ane voxrt transcribe clip.wav
```

## Files

| File | Notes |
|------|-------|
| `encoder.mlmodelc` | 304 MB, 2-bit ternary, macOS 15+ / iOS 18+. One function per length bucket, `t64` … `t768` (≈ 5–61 s of audio), over one shared copy of the weights. Inputs: `x [T, 1024]` fp16 (the subsampling output), `key_mask [1, 1, T]` and `row_mask [T, 1]` (mark the valid frames); output: `[T, 1024]` |
| `encoder.json` | the buckets and the model width |

Inputs are padded to the smallest bucket that fits; the masks keep padded frames out of the attention and the
convolution, so valid frames never see padding.

## Compute units

The encoder runs on the **Neural Engine**. The first load compiles the 2-bit program for it: ≈ 4.7 min for all seven
buckets on an M4 Max. Core ML caches the result, and later loads take ≈ 3 s per bucket (≈ 20 s for all seven).
`VOXRT_ANE_MAX_FRAMES` limits the buckets loaded and sent to the Neural Engine.

Warm, on an M4 Max: 8.7 ms per 64-frame (≈ 5 s) window, 23 / 42 / 70 / 119 ms at 128 / 192 / 256 / 384 frames.

voxrt's GPU path does not use this repository: its Metal kernels decode the same 2-bit weights straight from
`model.safetensors`. On a Mac with a large GPU (the M4 Max has 40 cores) that path is faster on inputs over ≈ 5 s.
The Neural Engine is the better fit on chips with a small GPU, and on iOS, which does not allow GPU work in the
background.

Requires macOS 15 / iOS 18 (the 2-bit encoding uses iOS 18 Core ML ops).

## Accuracy

voxrt on an M4 Max, Neural Engine (this model) against voxrt's own GPU and CPU paths. WER is scored with the Open
ASR Leaderboard normaliser (the upstream card reports 1.96 on test-clean); speed is total audio over processing time,
model load excluded.

| Set | Neural Engine (this model) | GPU | CPU |
|-----|---:|---:|---:|
| LibriSpeech test-clean, 2,620 utterances | 1.96 % · 168× | 1.96 % · 197× | 1.95 % · 79× |
| TED-LIUM 3 long-form, 148 min | 2.53 % · 110× | 2.54 % · 174× | 2.55 % · 92× |
| LibriSpeech dev-clean, 50 short clips | 1.78 % · 171× | 1.78 % · 202× | 1.68 % · 72× |

Encoder output against voxrt's f32 CPU path: cosine ≈ 0.9999. The differences from the CPU path come from fp16
compute, not from the weights.

## License

CC-BY-4.0, the same as [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) and
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3); © NVIDIA and M87 Labs (Moondream).
Core ML conversion by voxrt (`tools/coreml/convert_encoder.py` in the
[voxrt repository](https://github.com/lmaksym/voxrt)); the changes are described above.
