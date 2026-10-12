<p align="center">
  <img alt="voxrt" src="https://raw.githubusercontent.com/lmaksym/voxrt/main/assets/logo-light.svg" width="300">
</p>

<p align="center"><b>Fast, deterministic speech-to-text for Python — a Rust runtime for
<a href="https://huggingface.co/moondream/parakeet-redux">Parakeet Redux</a>.</b></p>

<p align="center">
  <a href="https://pypi.org/project/voxrt/"><img alt="PyPI" src="https://img.shields.io/pypi/v/voxrt?color=32d058&labelColor=353a41&logo=pypi&logoColor=969da4"></a>
  <a href="https://pypi.org/project/voxrt/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/voxrt?color=32d058&labelColor=353a41&logo=python&logoColor=969da4"></a>
  <a href="https://github.com/lmaksym/voxrt/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/lmaksym/voxrt/ci.yml?branch=main&label=CI&labelColor=353a41&logo=githubactions&logoColor=969da4"></a>
  <a href="https://github.com/lmaksym/voxrt/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-32d058?labelColor=353a41"></a>
  <a href="https://github.com/lmaksym/voxrt"><img alt="Platforms" src="https://img.shields.io/badge/platforms-macOS%20%7C%20Linux-32d058?labelColor=353a41"></a>
  <a href="https://huggingface.co/moondream/parakeet-redux"><img alt="Model" src="https://img.shields.io/badge/model-moondream%2Fparakeet--redux-32d058?logo=huggingface&logoColor=969da4&labelColor=353a41"></a>
</p>

**voxrt** runs [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux), the 1.58-bit ternary
distillation of NVIDIA's Parakeet-TDT 0.6B (25 European languages, 178 MB), with native ternary kernels written in
Rust. No PyTorch, ONNX or ML framework: one small wheel and `huggingface_hub`.

- **Fast** — 197× real time on LibriSpeech test-clean on an M4 Max GPU, 105× on an 8-core AMD server CPU
- **Apple GPU by default** on Apple silicon (Metal); the **Neural Engine** on request; **CPU** everywhere else
- **Same bits everywhere** — the CPU path gives byte-identical transcripts on Apple silicon and x86
- **Long audio built in** — the model's own VAD head cuts talks and meetings at pauses; segment and word timestamps
- **Streaming** — push audio as it arrives; the streamed transcript equals the offline one
- **Accurate** — WER 1.95 on LibriSpeech test-clean (the model card reports 1.96)

## Install

```sh
pip install voxrt            # WAV input
pip install "voxrt[audio]"   # plus FLAC / OGG / MP3 and numpy arrays
```

Wheels for Apple silicon and Intel Macs (macOS 10.12+) and Linux x86-64, Python 3.9 and later. The model
(≈ 180 MB) downloads into the Hugging Face cache on first use.

## Quick start

```python
import voxrt

t = voxrt.Transcriber()                       # or voxrt.Transcriber("/path/to/parakeet-redux")
r = t.transcribe("talk.mp3")
print(r.text)

for s in r.segments:                          # Segment(start, end, text, words=[Word(word, start, end)])
    print(f"[{s.start:6.2f} → {s.end:6.2f}] {s.text}")
```

Audio: WAV is decoded natively, FLAC / OGG / MP3 with `voxrt[audio]`, anything else (M4A, AAC, video) when `ffmpeg` is
on `PATH`. Arrays work too: `t.transcribe(samples, sample_rate=48000)` takes mono or `[n, channels]`.

### Streaming

```python
st = t.stream()
for chunk in microphone_chunks_16k():         # float32, 16 kHz mono
    for seg in st.push(chunk):                # segments are emitted as soon as their end is certain
        print(seg.text)
for seg in st.flush():
    print(seg.text)
```

### Devices

```python
voxrt.Transcriber(device="auto")   # default: the Apple GPU on Apple silicon, the CPU elsewhere
voxrt.Transcriber(device="cpu")    # byte-identical across machines
voxrt.Transcriber(device="ane")    # Apple Neural Engine (macOS 15+): downloads a Core ML bundle, ≈ 300 MB
```

`t.device` reports where the encoder runs. The Neural Engine bundle,
[`MaksL/voxrt-parakeet-redux-coreml`](https://huggingface.co/MaksL/voxrt-parakeet-redux-coreml), keeps the ternary
weights exact in 2 bits; macOS compiles it on the first load (≈ 5 min, once).

Results are frozen dataclasses (`Transcript`, `Segment`, `Word`) with type hints. The GIL is released while
transcribing, so one `Transcriber` can serve several threads.

## How it works

<img alt="Pipeline: audio → log-mel front end with VAD → ternary encoder (on the CPU, the Apple GPU or the Neural Engine) → TDT decoder → text with segment and word timestamps" src="https://raw.githubusercontent.com/lmaksym/voxrt/main/assets/pipeline-light.svg" width="760">

## Benchmarks

Apple M4 Max, the same method for every runtime (processing time, model load excluded):

| Workload | voxrt GPU (default) | voxrt CPU | Photon CPU (Moondream) |
|---|---|---|---|
| LibriSpeech test-clean, 2,620 utterances | **197× · WER 1.96** | 79× · WER 1.95 | 51× · WER 1.96 |
| TED-LIUM 3 long-form, 148 min | **174× · WER 2.54** | 92× · WER 2.55 | 65× · WER 2.52 |

<img alt="Speed on LibriSpeech dev50 on the CPU: voxrt 78× vs Photon 70× on an Apple M4 Max; voxrt 104× vs Photon 98× on AMD Turin" src="https://raw.githubusercontent.com/lmaksym/voxrt/main/assets/bench-light.svg" width="760">

More machines, the method and the comparison with other runtimes:
[benchmarks in the repository](https://github.com/lmaksym/voxrt#benchmarks).

## Links

- **Source, CLI, Swift and C bindings:** [github.com/lmaksym/voxrt](https://github.com/lmaksym/voxrt)
- **Model:** [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) ·
  **Neural Engine bundle:** [MaksL/voxrt-parakeet-redux-coreml](https://huggingface.co/MaksL/voxrt-parakeet-redux-coreml)
- **Changes:** [CHANGELOG](https://github.com/lmaksym/voxrt/blob/main/CHANGELOG.md)

## Development

Build from source with `maturin develop --release` in `bindings/python`. Tests: `pip install "voxrt[test]"`, then
`VOXRT_TEST_DOWNLOAD=1 pytest bindings/python/tests` (or point `VOXRT_MODEL_DIR` at a local copy). The test clip is
LibriSpeech test-clean `1089-134686-0001` (CC BY 4.0, see `tests/audio/ATTRIBUTION.md`).

## License

voxrt is Apache-2.0. The model weights are © NVIDIA and M87 Labs (Moondream), CC-BY-4.0.
