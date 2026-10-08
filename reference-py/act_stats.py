"""Activation range survey at every ternary-module input over dev50 → results/act_stats.json + docs table."""
import json, sys
from pathlib import Path
import numpy as np, torch
from voxrt_ref import Reference, ROOT

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    ref = Reference(device=device)
    tern = json.load(open(ROOT / "models/parakeet-redux/ternary.json"))
    names = [m["name"] for m in tern["quantized_modules"]]
    mods = dict(ref.model.named_modules())
    stats = {n: {"absmax": 0.0, "p999": [], "tok_absmax_ratio": []} for n in names}
    def pre(name):
        def f(mod, inp):
            x = inp[0].detach()
            x = x.transpose(1, 2) if x.ndim == 3 and x.shape[1] in (1024, 2048) and name.endswith(("pointwise_conv1", "pointwise_conv2")) else x
            x = x.reshape(-1, x.shape[-1]).float().cpu().numpy()      # [tokens, in]
            a = np.abs(x)
            s = stats[name]
            s["absmax"] = max(s["absmax"], float(a.max()))
            s["p999"].append(float(np.quantile(a, 0.999)))
            tok_max = a.max(1); s["tok_absmax_ratio"].append(float(np.median(tok_max / (np.median(a, axis=1) + 1e-9))))
        return f
    hs = [mods[n].register_forward_pre_hook(pre(n)) for n in names]
    rows = [json.loads(l) for l in (ROOT / "data/manifests/dev50.jsonl").read_text().splitlines()]
    for r in rows:
        wav = ref.load_audio(ROOT / r["audio"]); feats, mask = ref.features(wav)
        with torch.no_grad(): ref.encode(feats, mask)
    for h in hs: h.remove()
    out = {n: {"absmax": s["absmax"], "p999_median": float(np.median(s["p999"])), "tok_absmax_over_median": float(np.median(s["tok_absmax_ratio"]))} for n, s in stats.items()}
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results/act_stats_dev50.json", "w"), indent=1)
    # summary by module type
    by = {}
    for n, s in out.items():
        t = n.split(".", 3)[-1]; by.setdefault(t, []).append(s)
    print("module type | absmax (max over layers) | p99.9 (median) | per-token absmax/median (median)")
    for t, ss in by.items():
        print(f"{t:28s} | {max(x['absmax'] for x in ss):8.2f} | {np.median([x['p999_median'] for x in ss]):7.2f} | {np.median([x['tok_absmax_over_median'] for x in ss]):6.1f}")
    worst = sorted(out.items(), key=lambda kv: -kv[1]["absmax"])[:5]
    print("largest absmax:", [(n, round(s["absmax"], 1)) for n, s in worst])

if __name__ == "__main__":
    main()
