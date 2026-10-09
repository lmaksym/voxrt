"""Build the parakeet-redux encoder layers as Core ML programs for the Apple Neural Engine (one static-shape model per
length bucket; the ANE compiles fixed shapes).

Input: the subsampling output x [T, 1024] (fp16), an additive key mask [1, 1, T] (0 for frames < t, -inf-like for
padding) and a row mask [T, 1] (1 / 0). Output: the encoder output [T, 1024] (rows ≥ t are padding).

The graph follows the Rust f32 encoder operation by operation: macaron feed-forwards with ½ residuals,
relative-position attention (bias_u / bias_v, content + position scores, the Transformer-XL shift done with
pad / reshape / slice), the convolution module with BatchNorm folded into the depthwise conv, and the output LayerNorm.
Padding never leaks into valid frames: keys are masked in the softmax and padded rows are zeroed before the depthwise
conv. Ternary weights are exact: each row's values are {0, ±scale of each 128-column block} (≤ 2·groups + 1 distinct
values), stored as a per-row lookup table (palettisation, "unique" mode). Measured on the M4 Max: palettised linears run
on the Neural Engine, while int4 blockwise weights (`constexpr_blockwise_shift_scale`) fall back to the CPU and drag
their neighbours (LayerNorm) with them.

Weights default to int8 per output row: all ops run on the Neural Engine and it is the fastest format measured (the
exact palettes run at ~4× the latency). The default output is the runtime bundle in `<model>/coreml/`: one
multi-function package with a function per length bucket sharing one copy of the weights, compiled to
`encoder.mlmodelc`, plus `encoder.json` (the buckets). voxrt picks it up with `VOXRT_DEVICE=ane`.

usage: tools/coreml/.venv/bin/python tools/coreml/convert_encoder.py [--model DIR] [--out DIR] [--lengths 64,128,...]"""
import argparse, json, shutil, sys, tempfile, time
from pathlib import Path

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference-py"))
from trits import unpack  # noqa: E402  (our own unpacker, shared with the Python reference)

F16 = np.float16
WEIGHTS = "int8"  # int8 (per-row, the runtime default) | lut (exact, per-row palettes) | fp16 (dense)


def rel_pos_table(t, d):
    """[2t-1, d], positions t-1 … -(t-1), interleaved (sin, cos) per frequency — as `encoder::rel_pos_table`."""
    pos = (t - 1 - np.arange(2 * t - 1, dtype=np.float32))[:, None]
    inv = 1.0 / np.power(np.float32(10000.0), 2.0 * np.arange(d // 2, dtype=np.float32) / d)
    ang = pos * inv[None, :]
    out = np.empty((2 * t - 1, d), dtype=np.float32)
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out


class Weights:
    def __init__(self, model_dir: Path):
        self.t = load_file(str(model_dir / "model.safetensors"))
        tj = json.loads((model_dir / "ternary.json").read_text())
        self.mods = {q["name"]: q for q in tj["quantized_modules"]}
        self.cfg = json.loads((model_dir / "config.json").read_text())["encoder_config"]

    def f(self, name):
        return self.t[name].astype(np.float32)

    def ternary(self, names):
        trits, scales = [], []
        for n in names:
            q = self.mods[n]
            trits.append(unpack(self.t[f"{n}.qweight"], q["in_features"]))
            scales.append(self.t[f"{n}.scales"].astype(F16).reshape(q["out_features"], -1))
        return np.concatenate(trits).astype(np.int8), np.concatenate(scales)


def build(w: Weights, T: int, t_max: int = 0):
    c = w.cfg
    D, H, K, L = c["hidden_size"], c["num_attention_heads"], c["conv_kernel_size"], c["num_hidden_layers"]
    # The position projection relative_k_proj(pos) depends on constants only, so the converter folds it into a
    # [2·t_max-1, D] table per layer. Built at the largest bucket for the large buckets, the folded tables are
    # identical across those functions and the multi-function package stores them once; each bucket slices its window
    # out of the position scores instead.
    t_max = max(t_max, T)
    DH, NP, pad = D // H, 2 * t_max - 1, (K - 1) // 2
    pos_const = rel_pos_table(t_max, D).astype(F16)

    def lin(h, names):
        t, s = w.ternary(names)
        dense = (t.astype(np.float32) * np.repeat(s.astype(np.float32), t.shape[1] // s.shape[1], axis=1)).astype(F16)
        return mb.linear(x=h, weight=dense)  # palettised per row after conversion (exact: trit · f16 scale)

    def ln(h, name):
        return mb.layer_norm(x=h, axes=[-1], gamma=w.f(f"{name}.weight").astype(F16),
                             beta=w.f(f"{name}.bias").astype(F16), epsilon=F16(1e-5))

    def heads(t, n):
        return mb.transpose(x=mb.reshape(x=t, shape=[n, H, DH]), perm=[1, 0, 2])  # [H, n, DH]

    def rel_shift(a):
        # a [H, T, 2T-1] → out[h, i, j] = a[h, i, T-1-i+j]
        b = mb.pad(x=a, pad=[0, 0, 0, 0, 1, 0])
        b = mb.reshape(x=b, shape=[H, 2 * T, T])
        b = mb.slice_by_index(x=b, begin=[0, 1, 0], end=[H, 2 * T, T])
        b = mb.reshape(x=b, shape=[H, T, 2 * T - 1])
        return mb.slice_by_index(x=b, begin=[0, 0, 0], end=[H, T, T])

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(T, D), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, T), dtype=types.fp16),
            mb.TensorSpec(shape=(T, 1), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(x, key_mask, row_mask):
        pos = mb.const(val=pos_const)
        for i in range(L):
            p = f"encoder.layers.{i}"

            def ff(x, name):
                h = ln(x, f"{p}.norm_{name}")
                h = mb.silu(x=lin(h, [f"{p}.{name}.linear1"]))
                return mb.add(x=x, y=mb.mul(x=lin(h, [f"{p}.{name}.linear2"]), y=F16(0.5)))

            x = ff(x, "feed_forward1")
            # relative-position self-attention
            a = f"{p}.self_attn"
            h = ln(x, f"{p}.norm_self_att")
            q, k, v = mb.split(x=lin(h, [f"{a}.q_proj", f"{a}.k_proj", f"{a}.v_proj"]), num_splits=3, axis=1)
            qh, kh, vh = heads(q, T), heads(k, T), heads(v, T)
            bu = w.f(f"{a}.bias_u").reshape(H, 1, DH).astype(F16)
            bv = w.f(f"{a}.bias_v").reshape(H, 1, DH).astype(F16)
            rel = heads(lin(pos, [f"{a}.relative_k_proj"]), NP)
            s1 = mb.matmul(x=mb.add(x=qh, y=bu), y=kh, transpose_y=True)
            s2 = mb.matmul(x=mb.add(x=qh, y=bv), y=rel, transpose_y=True)  # [H, T, 2·t_max-1]
            if t_max > T:  # columns for positions T-1 … -(T-1)
                s2 = mb.slice_by_index(x=s2, begin=[0, 0, t_max - T], end=[H, T, t_max + T - 1])
            s2 = rel_shift(s2)
            sc = mb.add(x=mb.mul(x=mb.add(x=s1, y=s2), y=F16(DH ** -0.5)), y=key_mask)
            ctx = mb.matmul(x=mb.softmax(x=sc, axis=-1), y=vh)
            ctx = mb.reshape(x=mb.transpose(x=ctx, perm=[1, 0, 2]), shape=[T, D])
            x = mb.add(x=x, y=lin(ctx, [f"{a}.o_proj"]))
            # convolution module (BatchNorm folded into the depthwise conv, as in the Rust encoder)
            cv = f"{p}.conv"
            h = lin(ln(x, f"{p}.norm_conv"), [f"{cv}.pointwise_conv1"])
            ga, gb = mb.split(x=h, num_splits=2, axis=1)
            h = mb.mul(x=mb.mul(x=ga, y=mb.sigmoid(x=gb)), y=row_mask)  # padded rows → 0 before the time conv
            dw = w.f(f"{cv}.depthwise_conv.weight")  # [D, 1, K]
            g, bt = w.f(f"{cv}.norm.weight"), w.f(f"{cv}.norm.bias")
            mu, var = w.f(f"{cv}.norm.running_mean"), w.f(f"{cv}.norm.running_var")
            s = g / np.sqrt(var + 1e-5)
            h = mb.expand_dims(x=mb.transpose(x=h, perm=[1, 0]), axes=[0])  # [1, D, T]
            h = mb.conv(x=h, weight=(dw * s[:, None, None]).astype(F16), bias=(bt - mu * s).astype(F16),
                        groups=D, pad_type="custom", pad=[pad, pad])
            h = mb.transpose(x=mb.squeeze(x=mb.silu(x=h), axes=[0]), perm=[1, 0])  # [T, D]
            x = mb.add(x=x, y=lin(h, [f"{cv}.pointwise_conv2"]))
            x = ff(x, "feed_forward2")
            x = ln(x, f"{p}.norm_out")
        return x

    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16)
    if WEIGHTS == "lut":
        per_row = cto.OpPalettizerConfig(mode="unique", granularity="per_grouped_channel", group_size=1)
        return cto.palettize_weights(m, cto.OptimizationConfig(op_type_configs={"linear": per_row}))
    if WEIGHTS == "int8":
        q = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
        return cto.linear_quantize_weights(m, cto.OptimizationConfig(op_type_configs={"linear": q}))
    return m  # fp16


SHARE_FROM = 256  # buckets from this length up share the position tables of the largest bucket


def merge(parts, out: Path):
    """One multi-function package (function `t{T}` per bucket); identical weight constants are stored once."""
    d = ct.utils.MultiFunctionDescriptor()
    for T, path in parts:
        d.add_function(str(path), src_function_name="main", target_function_name=f"t{T}")
    d.default_function_name = f"t{parts[0][0]}"
    ct.utils.save_multifunction(d, str(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/parakeet-redux"))
    ap.add_argument("--out", default=None, help="default: <model>/coreml")
    ap.add_argument("--lengths", default="64,128,192,256,384,512,768")  # 768 covers the 60 s VAD segments
    ap.add_argument("--weights", default="int8", choices=["lut", "int8", "fp16"])
    ap.add_argument("--separate", action="store_true", help="one package per bucket instead of the runtime bundle")
    a = ap.parse_args()
    WEIGHTS = a.weights
    w = Weights(Path(a.model))
    out = Path(a.out or Path(a.model) / "coreml")
    out.mkdir(parents=True, exist_ok=True)
    lengths = sorted(int(v) for v in a.lengths.split(","))
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for T in lengths:
        t0 = time.time()
        # small buckets keep their own (small) position tables: slicing the shared one costs them ~20% latency
        m = build(w, T, 0 if a.separate or T < SHARE_FROM else max(lengths))
        path = (out if a.separate else tmp) / f"encoder_{T}{'' if a.weights == 'lut' else '_' + a.weights}.mlpackage"
        m.save(str(path))
        parts.append((T, path))
        print(f"T={T}: {path} in {time.time() - t0:.0f}s", flush=True)
    if not a.separate:
        # the runtime bundle: encoder.mlpackage (multi-function) → encoder.mlmodelc (compiled once here) + encoder.json
        pkg, compiled = out / "encoder.mlpackage", out / "encoder.mlmodelc"
        for p in (pkg, compiled):
            shutil.rmtree(p, ignore_errors=True)
        merge(parts, pkg)
        ct.utils.compile_model(str(pkg), str(compiled))
        (out / "encoder.json").write_text(json.dumps(
            {"buckets": lengths, "weights": a.weights, "d_model": w.cfg["hidden_size"]}, indent=1) + "\n")
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"bundle: {compiled} ({sum(f.stat().st_size for f in compiled.rglob('*') if f.is_file()) / 1e6:.0f} MB)")
