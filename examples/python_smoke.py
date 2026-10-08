"""Python smoke test: path input, array input at a foreign rate, streaming == offline. usage: python examples/python_smoke.py MODEL_DIR"""
import sys, time
import numpy as np, soundfile as sf
import voxrt

model = sys.argv[1] if len(sys.argv) > 1 else None
t = voxrt.Transcriber(model)
print("voxrt", voxrt.__version__, t)
r = t.transcribe("golden/5694-64025-0000/audio.wav")
print("path:", r)
x, sr = sf.read("golden/1919-142785-0007/audio.wav", dtype="float32")
x48 = np.interp(np.arange(0, len(x), sr / 48000), np.arange(len(x)), x).astype(np.float32)  # crude upsample for the test
t0 = time.time(); r = t.transcribe(x48, sample_rate=48000); dt = time.time() - t0
print(f"array @48k: {len(r.segments)} segments, {len(x)/sr/dt:.0f}x real time:", r.text[:60], "…")
print("first word:", r.segments[0].words[0])
y, sr = sf.read("data/tedlium/ted06.wav", dtype="float32")
off = t.transcribe(y)
st = t.stream(); segs = []
for i in range(0, len(y), 16000):
    segs += st.push(y[i:i + 16000])
segs += st.flush()
stream_text = " ".join(s.text for s in segs)
print(f"ted06 ({len(y)/sr/60:.1f} min): offline {len(off.segments)} segments, streaming {len(segs)}; texts equal: {stream_text == off.text}")
assert stream_text == off.text
