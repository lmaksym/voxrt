"""Black-box Photon driver (D-5.5b). Runs in bench/.venv-photon only. Usage copied from the public model card:
    with md.photon("moondream/parakeet-redux", device=...) as speech: speech.transcribe(audio="x.wav")
usage: photon_driver.py MANIFEST.jsonl OUT.jsonl cpu|mps"""
import json, sys, time
import moondream as md

man, out, device = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(man)]
t_load = time.time()
with md.photon("moondream/parakeet-redux", device=device) as speech:
    load = time.time() - t_load
    for r in rows[:2]:  # warm-up
        speech.transcribe(audio=r["wav"])
    proc = 0.0; audio = 0.0
    with open(out, "w") as f:
        for r in rows:
            t0 = time.time(); res = speech.transcribe(audio=r["wav"]); dt = time.time() - t0
            proc += dt; audio += r["seconds"]
            f.write(json.dumps({"id": r["id"], "text": res["text"], "rtf": r["seconds"] / dt}) + "\n")
print(f"processing only: {audio/proc:.1f}x (load {load:.2f}s excluded; wall column includes it)")
