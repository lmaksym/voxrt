"""voxrt — open-source runtime for moondream/parakeet-redux (ternary Parakeet-TDT).

    import voxrt
    t = voxrt.Transcriber("/path/to/parakeet-redux")
    print(t.transcribe("talk.wav").text)
    r = t.transcribe(samples, sample_rate=44100)   # 1-D float32 numpy array, any rate
    for s in r.segments: print(s["start"], s["end"], s["text"])
    st = t.stream(); st.push(chunk); ...; st.flush()
"""
from . import _voxrt
from ._voxrt import Transcript, __version__


def _as_bytes(audio):
    """Path → str; array-like → little-endian float32 bytes (numpy if present, else array/memoryview)."""
    if isinstance(audio, (str, bytes)) and not isinstance(audio, bytes):
        return audio
    if isinstance(audio, bytes):
        return audio
    try:
        import numpy as np
        a = np.ascontiguousarray(audio, dtype="<f4")
        if a.ndim != 1:
            raise ValueError("audio must be 1-D mono samples")
        return a.tobytes()
    except ImportError:
        import array
        return array.array("f", audio).tobytes()


class Transcriber:
    """A loaded model. `Transcriber(model_dir=None)` uses VOXRT_MODEL_DIR or the repo default when None."""

    def __init__(self, model_dir=None):
        self._t = _voxrt.Transcriber(model_dir)

    def transcribe(self, audio, sample_rate=16000):
        """WAV path, or 1-D mono samples (numpy / array / list) at `sample_rate`; long audio is segmented."""
        return self._t.transcribe(_as_bytes(audio), sample_rate)

    def stream(self):
        """Streaming session: `push(samples_16k)` → finalised segments; `flush()` → the rest."""
        return Stream(self._t.stream())

    def __repr__(self):
        return repr(self._t)


class Stream:
    def __init__(self, inner):
        self._s = inner

    def push(self, samples):
        return self._s.push(_as_bytes(samples))

    def flush(self):
        return self._s.flush()


__all__ = ["Transcriber", "Transcript", "Stream", "__version__"]
