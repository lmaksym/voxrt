"""Neural Engine spike: how fast and how efficient is an encoder-shaped Core ML model on the ANE? A synthetic
24-layer FastConformer with parakeet-redux's shapes (random ternary weights, fp16 compute, weights palettised to
2-bit lookup tables), one fixed-shape model per sequence length (the ANE needs static shapes). Reports where Core ML
places the ops, latency per compute-unit choice, and the macOS energy impact under a sustained loop.
usage: <venv with coremltools>/bin/python bench/ane_spike.py"""
import os, re, subprocess, sys, tempfile, threading, time

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

D, H, DH, FF, K, LAYERS = 1024, 8, 128, 4096, 9, 24
rng = np.random.default_rng(0)


def tern(out, inp):
    return (rng.integers(-1, 2, (out, inp)) * 0.02).astype(np.float16)


def build(T):
    NP = 2 * T - 1
    pos = (rng.standard_normal((NP, D)) * 0.1).astype(np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(T, D), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def prog(x):
        for _ in range(LAYERS):
            def ff(x):
                h = mb.layer_norm(x=x, axes=[-1], epsilon=np.float16(1e-5))
                h = mb.silu(x=mb.linear(x=h, weight=tern(FF, D)))
                return mb.add(x=x, y=mb.mul(x=mb.linear(x=h, weight=tern(D, FF)), y=np.float16(0.5)))
            x = ff(x)
            # relative-position self-attention (position band approximated by a slice, as in gpu_spike.py)
            h = mb.layer_norm(x=x, axes=[-1], epsilon=np.float16(1e-5))
            qkv = mb.linear(x=h, weight=tern(3 * D, D))
            q, k, v = mb.split(x=qkv, num_splits=3, axis=1)
            heads = lambda t, n: mb.transpose(x=mb.reshape(x=t, shape=[n, H, DH]), perm=[1, 0, 2])
            q, k, v = heads(q, T), heads(k, T), heads(v, T)
            rel = heads(mb.linear(x=pos, weight=tern(D, D)), NP)
            s1 = mb.matmul(x=q, y=k, transpose_y=True)
            s2 = mb.slice_by_size(x=mb.matmul(x=q, y=rel, transpose_y=True), begin=[0, 0, 0], size=[H, T, T])
            a = mb.softmax(x=mb.mul(x=mb.add(x=s1, y=s2), y=np.float16(DH ** -0.5)), axis=-1)
            ctx = mb.reshape(x=mb.transpose(x=mb.matmul(x=a, y=v), perm=[1, 0, 2]), shape=[T, D])
            x = mb.add(x=x, y=mb.linear(x=ctx, weight=tern(D, D)))
            # convolution module
            h = mb.linear(x=mb.layer_norm(x=x, axes=[-1], epsilon=np.float16(1e-5)), weight=tern(2 * D, D))
            ga, gb = mb.split(x=h, num_splits=2, axis=1)
            h = mb.mul(x=ga, y=mb.sigmoid(x=gb))
            h = mb.expand_dims(x=mb.transpose(x=h, perm=[1, 0]), axes=[0])  # [1, D, T]
            w = (rng.standard_normal((D, 1, K)) * 0.1).astype(np.float16)
            h = mb.conv(x=h, weight=w, groups=D, pad_type="same")
            h = mb.transpose(x=mb.squeeze(x=mb.silu(x=h), axes=[0]), perm=[1, 0])
            x = mb.add(x=x, y=mb.linear(x=h, weight=tern(D, D)))
            x = ff(x)
            x = mb.layer_norm(x=x, axes=[-1], epsilon=np.float16(1e-5))
        return x

    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16)
    # ternary weights × one scale per tensor in this synthetic model → 3 unique values: exact 2-bit palettes
    cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode="unique", nbits=2))
    return cto.palettize_weights(m, cfg)


def placement(path):
    try:
        from coremltools.models.compute_plan import MLComputePlan
        from coremltools.models.compute_device import MLNeuralEngineComputeDevice, MLGPUComputeDevice
        plan = MLComputePlan.load_from_path(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        counts = {"ane": 0, "gpu": 0, "cpu": 0}
        fn = plan.model_structure.program.functions["main"]
        for op in fn.block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is None:
                continue
            dev = u.preferred_compute_device
            k = "ane" if isinstance(dev, MLNeuralEngineComputeDevice) else "gpu" if isinstance(dev, MLGPUComputeDevice) else "cpu"
            counts[k] += 1
        return counts
    except Exception as e:  # API differences across versions: placement is informative only
        return f"n/a ({e.__class__.__name__})"


def latency(model, T, reps=15):
    x = {"x": rng.standard_normal((T, D)).astype(np.float16)}
    for _ in range(3):
        model.predict(x)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model.predict(x)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def energy(model, T, seconds=12):
    """Mean macOS energy impact of this process while predicting in a loop."""
    x = {"x": rng.standard_normal((T, D)).astype(np.float16)}
    samples, stop = [], threading.Event()

    def sample():
        top = subprocess.Popen(["top", "-l", "0", "-s", "1", "-pid", str(os.getpid()), "-stats", "pid,cpu,power"],
                               stdout=subprocess.PIPE, text=True)
        for line in top.stdout:
            m = re.match(rf"\s*{os.getpid()}\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                samples.append((float(m.group(1)), float(m.group(2))))
            if stop.is_set():
                break
        top.kill()

    th = threading.Thread(target=sample)
    th.start()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        model.predict(x)
        n += 1
    stop.set()
    th.join()
    s = samples[1:-1] or samples
    wall = (time.perf_counter() - t0) / n
    return sum(c for c, _ in s) / len(s), sum(p for _, p in s) / len(s), wall


if __name__ == "__main__":
    lens = [int(a) for a in sys.argv[1:]] or [63, 333, 750]
    tmp = tempfile.mkdtemp()
    for T in lens:
        t0 = time.time()
        m = build(T)
        path = os.path.join(tmp, f"enc_{T}.mlpackage")
        m.save(path)
        print(f"\nT_e={T} ({T * 0.08:.1f} s audio): built + palettised in {time.time() - t0:.0f}s; placement {placement(path)}")
        for name, cu in [("ANE", ct.ComputeUnit.CPU_AND_NE), ("GPU", ct.ComputeUnit.CPU_AND_GPU)]:
            t1 = time.time()
            mm = ct.models.MLModel(path, compute_units=cu)
            load = time.time() - t1
            lat = latency(mm, T)
            cpu, power, wall = energy(mm, T)
            per_min = power * (60.0 / (T * 0.08 / wall))
            print(f"  {name}: load {load:5.1f}s · latency {lat * 1e3:7.1f} ms ({T * 0.08 / lat:5.0f}× rt) · "
                  f"CPU {cpu:4.0f}% · energy impact {power:6.1f} · per audio-minute {per_min:6.1f}")
