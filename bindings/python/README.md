# prx (Python)

Python bindings for the Parakeet Redux runtime: `pip install prx` (wheel) or `maturin develop` from this directory.

```python
import prx
t = prx.Transcriber("/path/to/parakeet-redux")      # or set PRX_MODEL_DIR
r = t.transcribe("talk.wav")                         # WAV, any rate; long audio is segmented automatically
print(r.text)
for s in r.segments:
    print(f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}")   # s['words'] has word timestamps
r = t.transcribe(samples, sample_rate=48000)         # 1-D float32 numpy array
st = t.stream(); st.push(chunk_16k); ...; st.flush()  # streaming == offline by construction
```

The GIL is released during transcription; one `Transcriber` can serve several threads.
