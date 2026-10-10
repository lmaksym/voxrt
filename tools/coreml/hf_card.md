---
license: cc-by-4.0
base_model: moondream/parakeet-redux
pipeline_tag: automatic-speech-recognition
library_name: coreml
language: [en, de, fr, es, it, pt, ru, uk, hr, sl, lv, lt, et, fi, sv, da, nl, pl, cs, sk, hu, ro, bg, el, mt]
tags: [parakeet, coreml, apple-neural-engine, apple-silicon, voxrt]
---

# Parakeet Redux encoder for the Apple Neural Engine (voxrt)

The 24 encoder layers of [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) as a Core ML
model, for [voxrt](https://github.com/lmaksym/voxrt)'s Neural Engine path on Apple silicon Macs. It is not a
standalone model: voxrt runs the mel frontend, subsampling, decoder and VAD itself and only sends the encoder layers
here.

| File | What |
|---|---|
| `encoder.mlmodelc` | compiled multi-function Core ML model, one function `t{T}` per length bucket (64 … 768 encoder frames, up to ≈ 61 s) over one shared copy of the weights (304 MB) |
| `encoder.json` | the buckets and model width |

## Use

The Python package downloads it automatically on Apple silicon Macs (`device="auto"` or `"ane"`). For the CLI or the
Swift/C bindings, place it next to the model:

```sh
git lfs clone https://huggingface.co/moondream/parakeet-redux models/parakeet-redux
git lfs clone https://huggingface.co/MaksL/voxrt-parakeet-redux-coreml models/parakeet-redux/coreml
voxrt transcribe clip.wav        # auto: GPU at once, Neural Engine once compiled
```

macOS compiles the model for the Neural Engine on its first load (a few minutes for all buckets on an M4 Max) and
caches it; voxrt serves from the GPU meanwhile. Requires macOS 15.

## How it was made

`tools/coreml/convert_encoder.py` in the voxrt repository, from the parakeet-redux checkpoint. The ternary weights are
stored exactly, in 2 bits: each linear is a chain of 1×1 convs, one per 128-column block, whose weights are 2-bit
codes into a {-1, 0, +1} table, with that block's per-row scales applied to its output. The model computes in fp16
on the Neural Engine. Measured against voxrt's f32 CPU path: encoder output cosine ≈ 0.9999; LibriSpeech test-clean
WER 1.96, TED-LIUM long-form 2.53 (GPU path: 1.96 and 2.54).

## License and attribution

Derived from [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux), itself derived from
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3); © NVIDIA and M87 Labs (Moondream),
licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). This conversion is distributed under the same
license; the changes are described above.
