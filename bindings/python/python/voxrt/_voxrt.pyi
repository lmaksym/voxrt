"""Type stub for the native extension (`voxrt._voxrt`). Use the `voxrt` package API instead."""
from typing import Any, Dict, List, Optional, Union

__version__: str

_SegmentDict = Dict[str, Any]  # {"start": float, "end": float, "text": str, "words": [{"word", "start", "end"}]}

class Transcript:
    text: str
    duration: float
    @property
    def segments(self) -> List[_SegmentDict]: ...

class Stream:
    def push(self, samples: bytes) -> List[_SegmentDict]: ...
    def flush(self) -> List[_SegmentDict]: ...

class Transcriber:
    def __init__(self, model_dir: Optional[str] = None) -> None: ...
    def transcribe(self, audio: Union[str, bytes], sample_rate: int = 16000) -> Transcript: ...
    def stream(self) -> Stream: ...
