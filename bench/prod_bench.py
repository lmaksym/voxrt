"""Production-style benchmark (CPU vs Apple GPU vs Neural Engine; `DEVICES=cpu,gpu,ane`, `SUSTAIN_S=60`): load time, first-call and warm latency per clip length, CPU time per
run, and — under a sustained load — the process's CPU % and macOS energy impact (`top -stats power`, the figure
Activity Monitor shows; a relative score, not watts — for watts run `sudo powermetrics --samplers gpu_power,cpu_power`
alongside).
usage: python3 bench/prod_bench.py [AUDIO.wav]   (default data/tedlium/ted06.wav)"""
import json, os, re, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "target/release/voxrt"
AUDIO = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/tedlium/ted06.wav")
DEVICES = os.environ.get("DEVICES", "cpu,gpu,ane").split(",")
SUSTAIN_S = int(os.environ.get("SUSTAIN_S", "60"))  # clip length for the sustained load


def latency(device):
    env = dict(os.environ, VOXRT_DEVICE=device)
    out = subprocess.run([EXE, "bench", AUDIO, "--seconds", "2,5,10,27,60", "--repeat", "10"],
                         capture_output=True, text=True, env=env, check=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.startswith("{")]


def sustained(device, seconds=SUSTAIN_S, repeat=30):
    """Run a sustained load and sample the process with top: mean CPU % and energy impact."""
    env = dict(os.environ, VOXRT_DEVICE=device)
    p = subprocess.Popen([EXE, "bench", AUDIO, "--seconds", str(seconds), "--repeat", str(repeat)],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    samples = []

    def sample():
        time.sleep(3)  # skip model load
        top = subprocess.Popen(["top", "-l", "0", "-s", "1", "-pid", str(p.pid), "-stats", "pid,cpu,power"],
                               stdout=subprocess.PIPE, text=True)
        for line in top.stdout:
            m = re.match(rf"\s*{p.pid}\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                samples.append((float(m.group(1)), float(m.group(2))))
            if p.poll() is not None:
                break
        top.kill()

    th = threading.Thread(target=sample)
    th.start()
    out, _ = p.communicate()
    th.join()
    run = [json.loads(l) for l in out.splitlines() if l.startswith("{")][0]
    s = samples[1:-1] or samples  # drop partial first/last intervals
    cpu = sum(c for c, _ in s) / len(s)
    power = sum(w for _, w in s) / len(s)
    return run, cpu, power


if __name__ == "__main__":
    print(f"audio: {AUDIO}")
    print(f"\n{'device':6} {'clip':>6} {'load':>7} {'first':>8} {'warm':>8} {'× rt':>6} {'cpu/run':>9} {'cores busy':>10}")
    for d in DEVICES:
        for r in latency(d):
            print(f"{d:6} {r['audio_s']:5.0f}s {r['load_ms']:6.0f}ms {r['first_ms']:7.1f}ms {r['warm_ms']:7.1f}ms "
                  f"{r['rtf']:6.0f} {r['cpu_ms_per_run']:8.1f}ms {r['cpu_cores_busy']:10.1f}")
    print(f"\nsustained 60 s clips, 30 runs back to back:")
    print(f"{'device':6} {'× rt':>6} {'process CPU %':>14} {'energy impact':>14} {'energy / audio-minute':>22}")
    for d in DEVICES:
        run, cpu, power = sustained(d)
        # energy impact is a rate; per audio-minute = rate × (wall seconds per audio minute)
        per_min = power * (60.0 / run["rtf"])
        print(f"{d:6} {run['rtf']:6.0f} {cpu:14.0f} {power:14.1f} {per_min:22.1f}")
