"""voxrt — fast, deterministic runtime for moondream/parakeet-redux (ternary Parakeet-TDT speech-to-text).

    import voxrt
    t = voxrt.Transcriber()                          # downloads moondream/parakeet-redux on first use
    r = t.transcribe("talk.mp3")                     # WAV natively; FLAC/OGG/MP3 via soundfile, others via ffmpeg
    print(r.text)
    for s in r.segments:
        print(f"[{s.start:.2f}-{s.end:.2f}] {s.text}")  # s.words: word timestamps
    r = t.transcribe(samples, sample_rate=44100)    # numpy array (mono, or [n, channels] downmixed)
    st = t.stream(); st.push(chunk_16k); ...; st.flush()
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union

from . import _voxrt
from ._audio import load_audio, to_f32_bytes
from ._model import DEFAULT_REPO, resolve_model

__version__: str = _voxrt.__version__

AudioInput = Union[str, "os.PathLike[str]", Sequence[float], Any]


@dataclass(frozen=True)
class Word:
    """A word with its start and end time in seconds."""

    word: str
    start: float
    end: float


@dataclass(frozen=True)
class Segment:
    """A speech segment (VAD-delimited for long audio) with its words."""

    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass(frozen=True)
class Transcript:
    """Result of `Transcriber.transcribe`: full text, segments, and the audio duration in seconds."""

    text: str
    segments: List[Segment]
    duration: float

    def __str__(self) -> str:
        return self.text


def _segment(d: dict) -> Segment:
    return Segment(
        start=d["start"],
        end=d["end"],
        text=d["text"],
        words=[Word(word=w["word"], start=w["start"], end=w["end"]) for w in d["words"]],
    )


class Transcriber:
    """A loaded model.

    `model` is a local model directory or a Hugging Face repo id; when omitted, `VOXRT_MODEL_DIR` is used if set,
    otherwise `moondream/parakeet-redux` is downloaded into the Hugging Face cache (≈ 180 MB, once).
    `device`: "auto" (default; the Apple GPU when available, plus the Neural Engine for short inputs when a Core ML
    bundle is in `<model>/coreml`), "gpu", "ane" or "cpu" (the byte-identical cross-platform path); `VOXRT_DEVICE` sets the default. One instance can serve several threads (the GIL is released while
    transcribing).
    """

    def __init__(
        self,
        model: Optional[Union[str, "os.PathLike[str]"]] = None,
        *,
        revision: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_dir: str = resolve_model(model, revision=revision)
        self._t = _voxrt.Transcriber(self.model_dir, device)

    @property
    def device(self) -> str:
        """Where the encoder runs: "ane" (Neural Engine), "gpu" (Apple silicon, Metal) or "cpu"."""
        return self._t.device

    @classmethod
    def from_pretrained(
        cls, repo_id: str = DEFAULT_REPO, *, revision: Optional[str] = None, device: Optional[str] = None
    ) -> "Transcriber":
        """Load from a Hugging Face repo (downloaded into the HF cache on first use)."""
        return cls(repo_id, revision=revision, device=device)

    def transcribe(self, audio: AudioInput, sample_rate: int = 16000) -> Transcript:
        """Transcribe a file path or an array of samples at `sample_rate`. Long audio is segmented automatically.

        Files: WAV is decoded natively; other formats need `soundfile` (FLAC, OGG, MP3) or `ffmpeg` on PATH
        (M4A, AAC, video containers, …). Arrays: 1-D mono, or `[n, channels]` (averaged to mono); any rate.
        """
        if isinstance(audio, (str, os.PathLike)):
            path = os.fspath(audio)
            if path.lower().endswith(".wav"):
                return self._wrap(self._t.transcribe(path, 16000))
            samples, sample_rate = load_audio(path)
            audio = samples
        return self._wrap(self._t.transcribe(to_f32_bytes(audio), int(sample_rate)))

    def stream(self) -> "Stream":
        """Streaming session: `push(samples_16k)` returns the segments finalised so far, `flush()` the rest.
        Streaming output equals offline output (segments close only once the following pause is known)."""
        return Stream(self._t.stream())

    @staticmethod
    def _wrap(r: Any) -> Transcript:
        return Transcript(text=r.text, segments=[_segment(d) for d in r.segments], duration=r.duration)

    def __repr__(self) -> str:
        return f"Transcriber({self.model_dir!r}, device={self.device!r})"


class Stream:
    """Streaming session over a `Transcriber` (16 kHz mono input)."""

    def __init__(self, inner: Any):
        self._s = inner

    def push(self, samples: AudioInput) -> List[Segment]:
        return [_segment(d) for d in self._s.push(to_f32_bytes(samples))]

    def flush(self) -> List[Segment]:
        return [_segment(d) for d in self._s.flush()]


__all__ = ["Transcriber", "Transcript", "Segment", "Word", "Stream", "DEFAULT_REPO", "__version__"]
