"""Same-machine, same-audio benchmark of Parakeet runtimes (PLAN 5.5, used early for Gate 3).
One utterance at a time, model loaded once per runtime, 2 warm-up utterances, wall clock per utterance.
usage: bench/.venv/bin/python bench/run_all.py [--set dev50] [--runtimes prx-int8,prx-f32,sherpa,onnx-asr,parakeet.cpp]"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference-py"))

def manifest(name):
    return [json.loads(l) for l in (ROOT / f"data/manifests/{name}.jsonl").read_text().splitlines()]

def wav_of(r):
    return str((ROOT / r["audio"]).with_suffix(".wav"))

def run_prx(rows, precision):
    exe = ROOT / "target/release/prx"
    out = ROOT / f"results/bench_prx-{precision}.jsonl"
    out.unlink(missing_ok=True)
    tmp = ROOT / "data/manifests/_bench.jsonl"  # prx batch resolves audio relative to <root>/data/manifests
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    env = dict(os.environ, PRX_PRECISION=precision)
    # warm-up
    subprocess.run([exe, "transcribe", wav_of(rows[0]), "--precision", precision], capture_output=True, env=env)
    t0 = time.time()
    p = subprocess.run([exe, "batch", str(tmp), str(out), "--precision", precision], capture_output=True, text=True, env=env)
    wall = time.time() - t0
    tmp.unlink()
    return out, wall, p.stderr.strip().splitlines()[-1] if p.stderr else ""

def run_sherpa(rows):
    import sherpa_onnx, soundfile as sf
    d = ROOT / "bench/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"), joiner=str(d / "joiner.int8.onnx"),
        tokens=str(d / "tokens.txt"), num_threads=os.cpu_count(), model_type="nemo_transducer", decoding_method="greedy_search")
    def one(r):
        x, sr = sf.read(wav_of(r), dtype="float32")
        s = rec.create_stream(); s.accept_waveform(sr, x); rec.decode_stream(s); return s.result.text
    return timed(rows, one, "sherpa-int8")

def run_onnx_asr(rows):
    import onnx_asr
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")
    return timed(rows, lambda r: model.recognize(wav_of(r)), "onnx-asr-int8")

def run_parakeet_cpp(rows):
    exe = ROOT / "bench/parakeet.cpp/build/examples/cli/parakeet-cli"
    gguf = ROOT / "bench/models/tdt-0.6b-v3-q8_0.gguf"
    def one(r):
        p = subprocess.run([exe, "transcribe", "--model", gguf, "--input", wav_of(r)], capture_output=True, text=True)
        return p.stdout.strip()
    return timed(rows, one, "parakeet.cpp-q8_0", note="includes model load per utterance (CLI); see bench note")

def run_parakeet_cpp_bench(rows):
    """parakeet-cli's own bench mode: model loaded once, TSV manifest (path<TAB>reference)."""
    exe = ROOT / "bench/parakeet.cpp/build/examples/cli/parakeet-cli"
    gguf = ROOT / "bench/models/tdt-0.6b-v3-q8_0.gguf"
    tsv = ROOT / "results/_bench_manifest.tsv"
    tsv.write_text("".join(f"{wav_of(r)}\t{r['text']}\n" for r in rows))
    js = ROOT / "results/bench_parakeet.cpp-q8_0.json"
    t0 = time.time()
    p = subprocess.run([exe, "bench", "--model", gguf, "--manifest", tsv, "--decoder", "tdt", "--threads", str(os.cpu_count()), "--json", js], capture_output=True, text=True)
    wall = time.time() - t0
    tsv.unlink(missing_ok=True)
    (ROOT / "results/bench_parakeet.cpp-q8_0.stdout.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    # parakeet-cli JSON: {"load_ms", "files": [{"path", "audio_sec", "proc_ms", "text"}]} → processing-only RTF + our scorer
    out = ROOT / "results/bench_parakeet.cpp-q8_0.jsonl"
    data = json.loads(js.read_text())
    files = data["files"]
    by_path = {f["path"]: f for f in files}
    with open(out, "w") as f:
        for r in rows:
            it = by_path[wav_of(r)]
            f.write(json.dumps({"id": r["id"], "text": it["text"], "rtf": it["audio_sec"] / (it["proc_ms"] / 1000)}) + "\n")
    proc = sum(f["proc_ms"] for f in files) / 1000
    audio = sum(f["audio_sec"] for f in files)
    note = f"processing only: {audio/proc:.1f}× (load {data['load_ms']/1000:.2f}s excluded; wall column includes it)"
    return out, wall, note

def run_photon(rows, device):
    """Moondream Photon, black box (D-5.5b): driven only via the usage on the model card, in bench/.venv-photon."""
    exe = ROOT / "bench/.venv-photon/bin/python"
    script = ROOT / "bench/photon_driver.py"
    tmp = ROOT / "results/_photon_manifest.jsonl"
    tmp.write_text("".join(json.dumps({"id": r["id"], "wav": wav_of(r), "seconds": r["seconds"]}) + "\n" for r in rows))
    out = ROOT / f"results/bench_photon-{device}.jsonl"
    t0 = time.time()
    p = subprocess.run([exe, script, str(tmp), str(out), device], capture_output=True, text=True)
    wall = time.time() - t0
    tmp.unlink()
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "photon driver failed")
    note = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    return out, wall, note

def timed(rows, fn, name, note=""):
    out = ROOT / f"results/bench_{name}.jsonl"
    for r in rows[:2]: fn(r)
    t0 = time.time(); per = []
    with open(out, "w") as f:
        for r in rows:
            t1 = time.time(); text = fn(r); dt = time.time() - t1
            per.append(r["seconds"] / dt)
            f.write(json.dumps({"id": r["id"], "text": text, "rtf": r["seconds"] / dt}) + "\n")
    return out, time.time() - t0, note

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev50"); ap.add_argument("--runtimes", default="prx-int8,prx-f32,sherpa,onnx-asr,parakeet.cpp")
    a = ap.parse_args(); rows = manifest(a.set); audio = sum(r["seconds"] for r in rows)
    from score import score
    print(f"set={a.set} utts={len(rows)} audio={audio:.0f}s  cpu={os.cpu_count()} threads\n")
    print(f"{'runtime':20s} {'wall s':>8s} {'× real time':>12s} {'WER %':>7s}  note")
    for rt in a.runtimes.split(","):
        try:
            if rt == "prx-int8": out, wall, note = run_prx(rows, "int8")
            elif rt == "prx-f32": out, wall, note = run_prx(rows, "f32")
            elif rt == "sherpa": out, wall, note = run_sherpa(rows)
            elif rt == "onnx-asr": out, wall, note = run_onnx_asr(rows)
            elif rt == "parakeet.cpp": out, wall, note = run_parakeet_cpp_bench(rows)
            elif rt == "photon-cpu": out, wall, note = run_photon(rows, "cpu")
            elif rt == "photon-mps": out, wall, note = run_photon(rows, "mps")
            elif rt == "parakeet.cpp-cli": out, wall, note = run_parakeet_cpp(rows)
            else: print("unknown", rt); continue
            w = score(out, ROOT / f"data/manifests/{a.set}.jsonl", quiet=True) if out else float("nan")
            print(f"{rt:20s} {wall:8.1f} {audio/wall:12.1f} {100*w:7.2f}  {note}")
        except Exception as e:
            print(f"{rt:20s} FAILED: {e!r}"[:300])

if __name__ == "__main__":
    main()
