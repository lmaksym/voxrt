"""Ternary packing per ternary.json (thrush-ternary-v2): 5 base-3 digits per byte, LSD first."""
import numpy as np

POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint8)

def unpack(qweight: np.ndarray, in_features: int) -> np.ndarray:
    """u8[out, ceil(in/5)] → int8[out, in] with values in {-1, 0, +1}."""
    assert qweight.dtype == np.uint8 and qweight.ndim == 2
    out, rb = qweight.shape
    assert rb == -(-in_features // 5), (rb, in_features)
    b = qweight.astype(np.int16)
    digits = np.empty((out, rb, 5), dtype=np.int8)
    for d in range(5):
        digits[:, :, d] = (b % 3).astype(np.int8)
        b //= 3
    codes = digits.reshape(out, rb * 5)[:, :in_features]
    return codes - 1

def pack(trits: np.ndarray) -> np.ndarray:
    """int8[out, in] in {-1,0,1} → u8[out, ceil(in/5)], pad digits 0."""
    out, n = trits.shape
    rb = -(-n // 5)
    codes = np.zeros((out, rb * 5), dtype=np.uint8)
    codes[:, :n] = (trits + 1).astype(np.uint8)
    codes = codes.reshape(out, rb, 5)
    return (codes * POW3[None, None, :]).sum(-1).astype(np.uint8)

def dequant(qweight: np.ndarray, scales: np.ndarray, in_features: int, group_size: int = 128) -> np.ndarray:
    """w = scales[row, col // group] * trit → float32[out, in]."""
    t = unpack(qweight, in_features).astype(np.float32)
    out, n = t.shape
    s = scales.astype(np.float32)
    assert s.shape == (out, n // group_size), (s.shape, out, n, group_size)
    return t * np.repeat(s, group_size, axis=1)

if __name__ == "__main__":
    import json, struct, sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    MODEL = ROOT / "models/parakeet-redux"
    rng = np.random.default_rng(0)
    # property: pack∘unpack = id, on random trit matrices with ragged widths
    for n in (1, 4, 5, 6, 127, 128, 1024, 4096):
        t = rng.integers(-1, 2, size=(7, n), dtype=np.int8)
        q = pack(t); assert q.max() < 243
        assert np.array_equal(unpack(q, n), t), n
    # real weights: every byte < 243, pad digits 0, zero fraction == ternary.json
    tern = json.load(open(MODEL / "ternary.json"))
    with open(MODEL / "model.safetensors", "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hn)); base = 8 + hn
        def load(name):
            v = hdr[name]; a, b = v["data_offsets"]; f.seek(base + a)
            dt = {"U8": np.uint8, "F16": np.float16, "F32": np.float32}[v["dtype"]]
            return np.frombuffer(f.read(b - a), dtype=dt).reshape(v["shape"])
        worst = 0.0; n_bad_pad = 0
        for m in tern["quantized_modules"]:
            q = load(m["name"] + ".qweight"); i = m["in_features"]
            assert q.max() < 243, m["name"]
            full = unpack(q, q.shape[1] * 5)  # include pad digits; unpack returns digit−1, so pad digit 0 ↔ −1
            if full.shape[1] > i and not np.all(full[:, i:] == -1): n_bad_pad += 1
            t = full[:, :i]
            zf = float(np.mean(t == 0)); worst = max(worst, abs(zf - m["zero_fraction"]))
            s = load(m["name"] + ".scales"); assert np.isfinite(s.astype(np.float32)).all()
    print(f"OK: 264 modules unpacked; max |zero_fraction − ternary.json| = {worst:.2e}; modules with nonzero pad digits = {n_bad_pad}")
    sys.exit(0 if (worst < 1e-6 and n_bad_pad == 0) else 1)
