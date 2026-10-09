# Core ML encoder (Apple Neural Engine)

`convert_encoder.py` builds the 24 encoder layers as a Core ML program for `VOXRT_DEVICE=ane`.

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
| `encoder.mlpackage` | the source package (not needed at run time) |

`VOXRT_COREML_DIR` points voxrt at a bundle somewhere else. With a bundle present, `VOXRT_DEVICE=auto` (the default)
uses it; `VOXRT_ANE_MAX_FRAMES` (default 256 ≈ 20 s) sets which inputs go to the Neural Engine, longer ones run on
the GPU. On an M4 Max the Neural Engine is faster than Metal up to ≈ 20 s (5 s: 22 vs 26 ms end to end, 10 s: 33–37
vs 42 ms) and slower beyond, where padding to the bucket and the T² attention cost more; on Macs with fewer GPU cores
the crossover should move up.

## Design

- **Static shapes.** The Neural Engine compiles fixed shapes, so there is one function per bucket (default 64, 128,
  192, 256, 384, 512, 768 encoder frames, i.e. up to 61 s). An input is padded to the smallest bucket that fits; an
  additive key mask keeps padded frames out of the softmax and a row mask zeroes them before the depthwise conv, so
  valid frames do not see padding. Longer inputs fall back to the GPU.
- **Shared weights.** The buckets are merged with `save_multifunction`, which stores identical constants once. The
  folded position projection (`relative_k_proj(pos)`, 24 tables of `[2T-1, 1024]`) is built at the largest bucket for
  the buckets from 256 up, so they share it and slice their window from the position scores (810 → 660 MB for seven
  buckets); the small buckets keep their own small tables, since the wider score matmul would cost them ~20% latency.
- **Weights: int8 per output row.** Measured on an M4 Max, 384-frame bucket, Neural Engine:

  | Format | Placement | Latency |
  |---|---|---|
  | int8 per row (`linear_quantize_weights`) | all ops on the Neural Engine | 70 ms |
  | fp16 dense | all ops on the Neural Engine | 79 ms |
  | per-row palettes (exact ternary × scale) | all ops on the Neural Engine | 307 ms |
  | int4 blockwise (`constexpr_blockwise_shift_scale`) | linears and LayerNorm fall back to the CPU | — |

  int8 per row is not bit-exact to the ternary weights (each row's group scales are re-quantised), and the Neural
  Engine computes in fp16; together they keep cosine ≈ 0.9999 to the f32 reference and leave WER unchanged
  (test-clean 1.95, TED-LIUM 2.52, the same as the CPU path).
- **Compile cost.** Core ML compiles each function for the Neural Engine on its first load (≈ 7–25 s per bucket,
  ≈ 1.5 min for all seven) and caches the result for that `.mlmodelc`; later loads take ≈ 0.1 s per bucket. Apps
  should ship the compiled bundle and load it once at startup or in the background.

`check_encoder.py` compares a single-bucket package against the golden fp32 encoder outputs and times it per compute
unit (`tools/coreml/.venv/bin/python tools/coreml/check_encoder.py <package> ANE,GPU,CPU`; build one with
`--separate --lengths 384`).
