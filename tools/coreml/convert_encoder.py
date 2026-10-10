"""Build the parakeet-redux encoder layers as Core ML programs for the Apple Neural Engine (one static-shape model per
length bucket; the ANE compiles fixed shapes), with the ternary weights stored exactly in 2 bits.

Input: the subsampling output x [T, 1024] (fp16), an additive key mask [1, 1, T] (0 for frames < t, -inf-like for
padding) and a row mask [T, 1] (1 / 0). Output: the encoder output [T, 1024] (rows ≥ t are padding).

The graph follows the Rust f32 encoder operation by operation: macaron feed-forwards with ½ residuals,
relative-position attention (bias_u / bias_v, content + position scores, the Transformer-XL shift done with
pad / reshape / slice), the convolution module with BatchNorm folded into the depthwise conv, and the output LayerNorm.
Padding never leaks into valid frames: keys are masked in the softmax and padded rows are zeroed before the depthwise
conv.

Activations live in the Neural Engine's native [1, C, 1, T] layout from input to output. Each ternary linear
(w = scale[row, col // 128] · trit) is a chain of 1×1 convs, one per 128-column block: the conv weight is 2-bit codes
into one {-1, 0, +1} table (`constexpr_lut_to_dense`), its output is multiplied by that block's per-row scales and
accumulated. The weights are the checkpoint's own values, bit for bit; the Neural Engine computes in fp16. Encodings
that put the block scales inside the weight (blockwise tables or `constexpr_blockwise_shift_scale`) do not run on the
Neural Engine, and the same chain in the [T, C] layout costs 3× the latency (measurements in README.md).

The default output is the runtime bundle in `<model>/coreml/`: one multi-function package with a function per length
bucket sharing one copy of the weights, compiled to `encoder.mlmodelc`, plus `encoder.json` (the buckets). voxrt
picks it up with `VOXRT_DEVICE=auto` or `ane`.

usage: tools/coreml/.venv/bin/python tools/coreml/convert_encoder.py [--model DIR] [--out DIR] [--lengths 64,128,...]"""
import argparse, json, shutil, sys, tempfile, time
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference-py"))
from trits import unpack  # noqa: E402  (our own unpacker, shared with the Python reference)

F16 = np.float16
TRIT_LUT = np.array([-1, 0, 1, 0], F16).reshape(1, 1, 1, 1, 4, 1)  # code = trit + 1 (code 3 unused)


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
        """Trits [out, in] (int8) and scales [out, in / 128] (fp16) of the given linears, stacked along out."""
        trits, scales = [], []
        for n in names:
            q = self.mods[n]
            trits.append(unpack(self.t[f"{n}.qweight"], q["in_features"]))
            scales.append(self.t[f"{n}.scales"].astype(F16).reshape(q["out_features"], -1))
        return np.concatenate(trits).astype(np.int8), np.concatenate(scales)


def build(w: Weights, T: int, t_max: int = 0):
    c = w.cfg
    D, H, K, L = c["hidden_size"], c["num_attention_heads"], c["conv_kernel_size"], c["num_hidden_layers"]
    # The position projection relative_k_proj(pos) depends on constants only, so it is computed here in f32 as a
    # [2·t_max-1, D] table per layer. Built at the largest bucket for the large buckets, the tables are identical
    # across those functions and the multi-function package stores them once; each bucket slices its window out of the
    # position scores instead.
    t_max = max(t_max, T)
    DH, NP, pad = D // H, 2 * t_max - 1, (K - 1) // 2
    pos_tab = rel_pos_table(t_max, D)

    def lin(x, names):  # x [1, in, 1, T] → [1, out, 1, T]
        t, s = w.ternary(names)
        n, k = t.shape
        g = s.shape[1]
        b = k // g
        xs = mb.split(x=x, num_splits=g, axis=1)
        y = None
        for j in range(g):
            idx = (t[:, j * b:(j + 1) * b] + 1).reshape(n, b, 1, 1).astype(types.np_uint2_dtype)
            yj = mb.mul(x=mb.conv(x=xs[j], weight=mb.constexpr_lut_to_dense(indices=idx, lut=TRIT_LUT)),
                        y=s[:, j].reshape(1, n, 1, 1).astype(F16))
            y = yj if y is None else mb.add(x=y, y=yj)
        return y

    def ln(x, name):
        return mb.layer_norm(x=x, axes=[1], gamma=w.f(f"{name}.weight").astype(F16),
                             beta=w.f(f"{name}.bias").astype(F16), epsilon=F16(1e-5))

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
        x = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, D, 1, T])
        rm = mb.reshape(x=row_mask, shape=[1, 1, 1, T])
        for i in range(L):
            p = f"encoder.layers.{i}"

            def ff(x, name):
                h = mb.silu(x=lin(ln(x, f"{p}.norm_{name}"), [f"{p}.{name}.linear1"]))
                return mb.add(x=x, y=mb.mul(x=lin(h, [f"{p}.{name}.linear2"]), y=F16(0.5)))

            x = ff(x, "feed_forward1")
            # relative-position self-attention, heads as [H, DH, T]
            a = f"{p}.self_attn"
            qkv = lin(ln(x, f"{p}.norm_self_att"), [f"{a}.q_proj", f"{a}.k_proj", f"{a}.v_proj"])
            q, k, v = [mb.reshape(x=z, shape=[H, DH, T]) for z in mb.split(x=qkv, num_splits=3, axis=1)]
            bu = w.f(f"{a}.bias_u").reshape(H, DH, 1).astype(F16)
            bv = w.f(f"{a}.bias_v").reshape(H, DH, 1).astype(F16)
            pt, ps = w.ternary([f"{a}.relative_k_proj"])
            pw = pt.astype(np.float32) * np.repeat(ps.astype(np.float32), pt.shape[1] // ps.shape[1], axis=1)
            rel = (pos_tab @ pw.T).astype(F16).reshape(NP, H, DH).transpose(1, 0, 2)  # [H, NP, DH]
            s1 = mb.matmul(x=mb.add(x=q, y=bu), y=k, transpose_x=True)  # [H, T, T]
            s2 = mb.matmul(x=mb.add(x=q, y=bv), y=rel, transpose_x=True, transpose_y=True)  # [H, T, NP]
            if t_max > T:  # columns for positions T-1 … -(T-1)
                s2 = mb.slice_by_index(x=s2, begin=[0, 0, t_max - T], end=[H, T, t_max + T - 1])
            s2 = rel_shift(s2)
            sc = mb.add(x=mb.mul(x=mb.add(x=s1, y=s2), y=F16(DH ** -0.5)), y=key_mask)
            ctx = mb.matmul(x=v, y=mb.softmax(x=sc, axis=-1), transpose_y=True)  # [H, DH, T]
            x = mb.add(x=x, y=lin(mb.reshape(x=ctx, shape=[1, D, 1, T]), [f"{a}.o_proj"]))
            # convolution module (BatchNorm folded into the depthwise conv, as in the Rust encoder)
            cv = f"{p}.conv"
            ga, gb = mb.split(x=lin(ln(x, f"{p}.norm_conv"), [f"{cv}.pointwise_conv1"]), num_splits=2, axis=1)
            h = mb.mul(x=mb.mul(x=ga, y=mb.sigmoid(x=gb)), y=rm)  # padded frames → 0 before the time conv
            dw = w.f(f"{cv}.depthwise_conv.weight")  # [D, 1, K]
            g, bt = w.f(f"{cv}.norm.weight"), w.f(f"{cv}.norm.bias")
            mu, var = w.f(f"{cv}.norm.running_mean"), w.f(f"{cv}.norm.running_var")
            s = g / np.sqrt(var + 1e-5)
            h = mb.conv(x=h, weight=(dw * s[:, None, None]).reshape(D, 1, 1, K).astype(F16),
                        bias=(bt - mu * s).astype(F16), groups=D, pad_type="custom", pad=[0, 0, pad, pad])
            x = mb.add(x=x, y=lin(mb.silu(x=h), [f"{cv}.pointwise_conv2"]))
            x = ff(x, "feed_forward2")
            x = ln(x, f"{p}.norm_out")
        return mb.transpose(x=mb.reshape(x=x, shape=[D, T]), perm=[1, 0])

    return ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                      compute_precision=ct.precision.FLOAT16)


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
    ap.add_argument("--separate", action="store_true", help="one package per bucket instead of the runtime bundle")
    a = ap.parse_args()
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
        path = (out if a.separate else tmp) / f"encoder_{T}.mlpackage"
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
            {"buckets": lengths, "weights": "ternary-2bit", "d_model": w.cfg["hidden_size"]}, indent=1) + "\n")
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"bundle: {compiled} ({sum(f.stat().st_size for f in compiled.rglob('*') if f.is_file()) / 1e6:.0f} MB)")
