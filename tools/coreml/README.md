# Core ML encoder (Apple Neural Engine)

`convert_encoder.py` builds the 24 encoder layers as a Core ML program for `VOXRT_DEVICE=ane`, with the ternary
weights stored exactly in 2 bits. A prebuilt bundle is on Hugging Face,
[`MaksL/voxrt-parakeet-redux-coreml`](https://huggingface.co/MaksL/voxrt-parakeet-redux-coreml) (its model card is
`hf_card.md` here); build your own only to change the buckets.

```sh
python3 -m venv tools/coreml/.venv
tools/coreml/.venv/bin/pip install coremltools numpy safetensors
tools/coreml/.venv/bin/python tools/coreml/convert_encoder.py            # → models/parakeet-redux/coreml/
VOXRT_DEVICE=ane target/release/voxrt transcribe clip.wav
```

The output, in `<model>/coreml/`:

| File | What |
|---|---|
| `encoder.mlmodelc` | the compiled multi-function model voxrt loads: function `t{T}` per length bucket |
| `encoder.json` | the buckets and model width |
| `encoder.mlpackage` | the source package (not needed at run time; add it to an Xcode project to ship it in an app) |

`VOXRT_COREML_DIR` points voxrt at a bundle somewhere else. With a bundle present, `VOXRT_DEVICE=auto` (the default)
uses it once loaded. `VOXRT_ANE_MAX_FRAMES` (default: every bucket) caps which inputs go to the Neural Engine; longer
ones run on the GPU. On an M4 Max, whose 40-core GPU runs the same 2-bit weights, the GPU is faster beyond ≈ 5 s
(`VOXRT_ANE_MAX_FRAMES=64`); on chips with a smaller GPU the Neural Engine keeps the lead to longer inputs.

## Design

- **Static shapes.** The Neural Engine compiles fixed shapes, so there is one function per bucket (default 64, 128,
  192, 256, 384, 512, 768 encoder frames, i.e. up to 61 s). An input is padded to the smallest bucket that fits; an
  additive key mask keeps padded frames out of the softmax and a row mask zeroes them before the depthwise conv, so
  valid frames do not see padding. Longer inputs fall back to the GPU.
- **Exact 2-bit weights.** A ternary linear is `w = scale[row, col // 128] · trit`. Each one becomes a chain of 1×1
  convs, one per 128-column block: the conv weight is 2-bit codes into a single {-1, 0, +1} table
  (`constexpr_lut_to_dense`), and its output is multiplied by that block's per-row scales and accumulated. The
  weights are the checkpoint's values bit for bit; the Neural Engine computes in fp16 (cosine ≈ 0.9999 to the f32
  reference).
- **Native layout.** Activations stay in the Neural Engine's `[1, C, 1, T]` layout from input to output, so the
  graph has no layout changes around the linears.
- **Shared weights.** The buckets are merged with `save_multifunction`, which stores identical constants once
  (304 MB for seven buckets). The position projection (`relative_k_proj(pos)`, 24 tables of `[2T-1, 1024]`) is built
  at the largest bucket for the buckets from 256 up, so they share it and slice their window from the position scores;
  the small buckets keep their own small tables.

Encodings measured on an M4 Max, 64-frame bucket (≈ 5 s), Neural Engine, all exact unless noted:

| Weights | Placement | Latency |
|---|---|---|
| **2-bit codes, per-block conv chain, native layout (this converter)** | all ops on the Neural Engine | **8.7 ms** |
| same chain, `[T, C]` layout with a layout change around each linear | all ops on the Neural Engine | 16.5 ms |
| same chain as `linear` ops in the `[T, C]` layout | all ops on the Neural Engine | 28 ms |
| one grouped conv per linear, then scale and sum over the blocks | all ops on the Neural Engine | 140 ms |
| per-(row, block) table, or 2-bit codes × `constexpr_blockwise_shift_scale` | rejected by the Neural Engine | — |
| int8 per row (re-quantised, not exact; the previous bundle) | all ops on the Neural Engine | 8.5 ms |

Each block's conv writes a full-width partial result, so the chain's cost grows with the input length faster than a
single int8 pass: 23 / 42 / 70 / 119 ms at 128 / 192 / 256 / 384 frames, against 14 / 25 / 40 / 70 ms for int8 per row.

- **Compile cost.** Core ML compiles each function for the Neural Engine on its first load (≈ 4.7 min for all seven
  buckets on an M4 Max) and caches the result for that `.mlmodelc`; later loads take ≈ 3 s per bucket. With `auto`, voxrt loads the bundle
  on a background thread and serves from the GPU until it is ready. In an iOS app, Xcode compiles the `.mlpackage` at
  build time.

`check_encoder.py` compares a single-bucket package against the golden fp32 encoder outputs and times it per compute
unit (`tools/coreml/.venv/bin/python tools/coreml/check_encoder.py <package> ANE,GPU,CPU`; build one with
`--separate --lengths 384`).
