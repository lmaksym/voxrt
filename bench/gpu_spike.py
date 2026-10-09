"""GPU spike: how fast could the encoder run on the Apple GPU? A synthetic encoder with parakeet-redux's exact shapes
(24 FastConformer layers, d=1024, 8 heads, FF 4096, conv kernel 9, relative-position attention) in MLX, timed per
sequence length. Weights are random (speed only). Two weight formats:
  fp16  — ternary weights unpacked to fp16 (dense GEMMs)
  2bit  — MLX 2-bit grouped quantisation (group 128); represents ternary exactly: codes {0,1,2} with bias = -scale
usage: bench/.venv/bin/python bench/gpu_spike.py"""
import time

import mlx.core as mx
import mlx.nn as nn

D, H, DH, FF, K, LAYERS = 1024, 8, 128, 4096, 9, 24


class Lin:
    def __init__(self, i, o, fmt):
        w = (mx.random.randint(-1, 2, (o, i)).astype(mx.float16) * 0.02)
        self.fmt = fmt
        if fmt == "2bit":
            self.wq, self.s, self.b = mx.quantize(w, group_size=128, bits=2)
        else:
            self.w = w

    def __call__(self, x):
        if self.fmt == "2bit":
            return mx.quantized_matmul(x, self.wq, self.s, self.b, transpose=True, group_size=128, bits=2)
        return x @ self.w.T


class Layer:
    def __init__(self, fmt):
        self.ff1 = (Lin(D, FF, fmt), Lin(FF, D, fmt))
        self.ff2 = (Lin(D, FF, fmt), Lin(FF, D, fmt))
        self.qkv, self.pos, self.out = Lin(D, 3 * D, fmt), Lin(D, D, fmt), Lin(D, D, fmt)
        self.pw1, self.pw2 = Lin(D, 2 * D, fmt), Lin(D, D, fmt)
        self.dw = (mx.random.normal((D, K, 1)) * 0.1).astype(mx.float16)  # depthwise conv weight [C, K, 1]

    def ff(self, x, p):
        return x + 0.5 * p[1](nn.silu(p[0](mx.fast.layer_norm(x, None, None, 1e-5))))

    def __call__(self, x, pe):
        t = x.shape[0]
        x = self.ff(x, self.ff1)
        # relative-position self-attention: content scores + position scores (banded via a T x (2T-1) product)
        h = mx.fast.layer_norm(x, None, None, 1e-5)
        q, k, v = mx.split(self.qkv(h).reshape(t, 3, H, DH).transpose(1, 2, 0, 3), 3)
        q, k, v = q[0], k[0], v[0]                                    # [H, T, DH]
        p = self.pos(pe).reshape(-1, H, DH).transpose(1, 0, 2)        # [H, 2T-1, DH]
        s1 = q @ k.transpose(0, 2, 1)                                 # [H, T, T]
        s2 = (q @ p.transpose(0, 2, 1))[:, :, :t]                     # [H, T, T] (shift approximated by a slice)
        a = mx.softmax((s1 + s2) * (DH ** -0.5), axis=-1)
        x = x + self.out((a @ v).transpose(1, 0, 2).reshape(t, D))
        # convolution module: pointwise → GLU → depthwise k=9 → SiLU → pointwise
        h = self.pw1(mx.fast.layer_norm(x, None, None, 1e-5))
        h = h[:, :D] * mx.sigmoid(h[:, D:])
        h = mx.conv1d(h[None], self.dw, padding=K // 2, groups=D)[0]
        x = x + self.pw2(nn.silu(h))
        x = self.ff(x, self.ff2)
        return mx.fast.layer_norm(x, None, None, 1e-5)


def encoder(fmt):
    layers = [Layer(fmt) for _ in range(LAYERS)]

    def fwd(x, pe):
        for l in layers:
            x = l(x, pe)
        return x

    return fwd


def bench(fwd, t, reps=15):
    x = mx.random.normal((t, D)).astype(mx.float16)
    pe = mx.random.normal((2 * t - 1, D)).astype(mx.float16)
    for _ in range(3):
        mx.eval(fwd(x, pe))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fwd(x, pe))
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


if __name__ == "__main__":
    mx.random.seed(0)
    lens = [27, 61, 119, 175, 333, 750]  # encoder frames (80 ms): 2 s, 5 s, 9.5 s, 14 s, 27 s, 60 s of audio
    print(f"{'T_e':>5} {'audio':>7} {'fp16 ms':>9} {'2bit ms':>9} {'fp16+compile':>13}")
    enc16, enc2 = encoder("fp16"), encoder("2bit")
    comp16 = mx.compile(enc16)
    for t in lens:
        a = bench(enc16, t)
        b = bench(enc2, t)
        c = bench(comp16, t)
        print(f"{t:5d} {t * 0.08:6.1f}s {a * 1e3:9.1f} {b * 1e3:9.1f} {c * 1e3:13.1f}")
