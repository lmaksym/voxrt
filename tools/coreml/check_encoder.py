"""Core ML encoder vs the golden fp32 encoder outputs (HF reference): feed each golden utterance's subsampling output,
padded to the bucket with masks, compare the valid frames. Also times the model per compute unit.
usage: tools/coreml/.venv/bin/python tools/coreml/check_encoder.py [models/coreml/encoder_384.mlpackage]"""
import sys, time
from pathlib import Path

import numpy as np
import coremltools as ct

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/coreml/encoder_384.mlpackage"
T = int(PKG.stem.split("_")[1])
UNITS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["ANE", "GPU", "CPU"]
D = 1024


def inputs(x):
    t = x.shape[0]
    xp = np.zeros((T, D), np.float16)
    xp[:t] = x
    km = np.full((1, 1, T), -30000.0, np.float16)
    km[..., :t] = 0
    rm = np.zeros((T, 1), np.float16)
    rm[:t] = 1
    return {"x": xp, "key_mask": km, "row_mask": rm}


ids = [l.strip() for l in (ROOT / "reference-py/golden_ids.txt").read_text().splitlines() if l.strip()]
ALL = {"ANE": ct.ComputeUnit.CPU_AND_NE, "GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU": ct.ComputeUnit.CPU_ONLY}
for name, cu in [(u, ALL[u]) for u in UNITS]:
    t0 = time.time()
    m = ct.models.MLModel(str(PKG), compute_units=cu)
    load = time.time() - t0
    out_key = m.get_spec().description.output[0].name
    print(f"\n{name}: load {load:.1f}s")
    for gid in ids:
        g = ROOT / "golden" / gid
        want = np.load(g / "encoder_out.npy")
        t = want.shape[0]  # golden encoder_out is sliced to the valid length; subsampling_out may have one more frame
        x = np.load(g / "subsampling_out.npy")[:t]
        if t > T:
            continue
        inp = inputs(x)
        got = m.predict(inp)[out_key][:t].astype(np.float32)
        ts = []
        for _ in range(5):
            t1 = time.perf_counter()
            m.predict(inp)
            ts.append(time.perf_counter() - t1)
        scale = np.abs(want).max()
        err = np.abs(got - want)
        cos = (got * want).sum() / (np.linalg.norm(got) * np.linalg.norm(want))
        print(f"  {gid}: t={t:3d} max|err| {err.max():.3e} (scale {scale:.2f}, rel {err.max() / scale:.2e}) "
              f"mean {err.mean():.2e} cos {cos:.6f} · {np.median(ts) * 1e3:.1f} ms")
