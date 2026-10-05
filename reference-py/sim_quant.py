"""PLAN 3.2: simulate int8 activation quantisation at every ternary-module input (weights stay exact) and measure
the effect on dev50: WER and token-level agreement with the fp32 run. Schemes: per-token absmax, per-group-G absmax."""
import json, glob, sys, time
from pathlib import Path
import torch
from prx_ref import Reference, ROOT
from score import score

def quant_fn(scheme, group):
    def q(x):
        # x: [..., in]; symmetric int8, absmax scale, round-to-nearest-even like torch.round
        if scheme == "token":
            s = x.abs().amax(dim=-1, keepdim=True) / 127.0
        else:
            shp = x.shape
            xg = x.reshape(*shp[:-1], shp[-1] // group, group)
            s = (xg.abs().amax(dim=-1, keepdim=True) / 127.0).expand_as(xg).reshape(shp)
        s = torch.where(s == 0, torch.ones_like(s), s)
        return torch.round(x / s).clamp(-127, 127) * s
    return q

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    sets = sys.argv[2] if len(sys.argv) > 2 else "dev50"
    ref = Reference(device=device)
    tern = json.load(open(ROOT / "models/parakeet-redux/ternary.json"))
    mods = dict(ref.model.named_modules())
    names = [m["name"] for m in tern["quantized_modules"]]
    rows = [json.loads(l) for l in (ROOT / f"data/manifests/{sets}.jsonl").read_text().splitlines()]
    base = {json.loads(l)["id"]: json.loads(l) for l in open(glob.glob(str(ROOT / f"results/ref_{sets}_mps_*.jsonl"))[0])}
    schemes = [("group", 128)] if sets == "test-clean" else [("token", 0), ("group", 128), ("group", 64), ("group", 32)]
    print(f"{sets}: {len(rows)} utts; baseline fp32 WER {100*score(glob.glob(str(ROOT / f'results/ref_{sets}_mps_*.jsonl'))[0], ROOT / f'data/manifests/{sets}.jsonl', quiet=True):.2f}")
    for scheme, g in schemes:
        q = quant_fn(scheme, g)
        def pre(mod, inp):
            x = inp[0]
            if x.ndim == 3 and x.shape[1] in (1024, 2048) and x.shape[-1] != 1024 and x.shape[-1] != 4096:
                # Conv1d input [B, C, T]: quantise along C
                return (q(x.transpose(1, 2)).transpose(1, 2),)
            return (q(x),)
        hs = [mods[n].register_forward_pre_hook(pre) for n in names]
        out = ROOT / f"results/sim_{scheme}{g}_{sets}.jsonl"; t0 = time.time()
        same = 0
        with open(out, "w") as f:
            for r in rows:
                res = ref.transcribe(ROOT / r["audio"])
                em = [[e.token, e.frame, e.duration] for e in res.emissions]
                same += em == base[r["id"]]["emissions"]
                f.write(json.dumps({"id": r["id"], "text": res.text, "emissions": em}) + "\n")
        for h in hs: h.remove()
        w = score(out, ROOT / f"data/manifests/{sets}.jsonl", quiet=True)
        print(f"{scheme:5s} g={g:3d}: WER {100*w:.2f}  identical token sequences {same}/{len(rows)}  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
