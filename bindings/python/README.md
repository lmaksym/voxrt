# voxrt (Python)

Python bindings for the Parakeet Redux runtime: `pip install voxrt` (wheel) or `maturin develop` from this directory.

```python
import voxrt
t = voxrt.Transcriber()                        # downloads moondream/parakeet-redux into the HF cache on first use,
                                               # (device="ane": plus the Neural Engine bundle, ≈ 300 MB)
                                               # (or Transcriber("/path/to/model"), or set VOXRT_MODEL_DIR)
r = t.transcribe("talk.mp3")                   # WAV natively; FLAC/OGG/MP3 with `pip install voxrt[audio]`;
print(r.text)                                  # other formats (M4A, AAC, video) when ffmpeg is on PATH
for s in r.segments:                           # Segment(start, end, text, words=[Word(word, start, end)])
    print(f"[{s.start:.2f}-{s.end:.2f}] {s.text}")
r = t.transcribe(samples, sample_rate=48000)   # numpy array: mono, or [n, channels] (averaged)
st = t.stream(); st.push(chunk_16k); ...; st.flush()  # streaming == offline (same segment cuts)
```

The GIL is released during transcription; one `Transcriber` can serve several threads.

Results are frozen dataclasses (`Transcript`, `Segment`, `Word`); the package ships type hints (`py.typed`).
Tests: `pip install voxrt[test]` then `VOXRT_TEST_DOWNLOAD=1 pytest bindings/python/tests` (or point `VOXRT_MODEL_DIR`
at a local copy). The test clip is LibriSpeech test-clean `1089-134686-0001` (CC BY 4.0, see `tests/audio/ATTRIBUTION.md`).

