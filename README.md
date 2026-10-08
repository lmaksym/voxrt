# voxrt — fast, open runtime for Parakeet Redux

**voxrt** runs [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux), the 1.58-bit ternary
distillation of NVIDIA's `parakeet-tdt-0.6b-v3` (25 languages, 178 MB), on the CPU with native ternary kernels.
Pure Rust, six small dependencies, Apache-2.0. C ABI, Python and Swift bindings, CLI. Long audio is segmented by
the model's own VAD head, with segment and word timestamps and a streaming API whose output equals the offline one.

| Apple M4 Max, same audio, processing time only | voxrt (CPU) | Photon (Moondream's runtime, CPU) |
|---|---|---|
| LibriSpeech test-clean, 2,620 utterances | **79× real time, WER 1.95** | 51×\*, WER 1.96 |
| dev50 short clips | **78×** | 70× |
| one 27 s utterance | **0.26 s (104×)** | 0.37 s (72×) |
| TED-LIUM 3 long-form, 148 min | **92×, WER 2.55** | 65×, WER 2.52 |

Same weights, same scoring (Open ASR Leaderboard normaliser + compound-merging alignment). The open runtimes of the
teacher model (sherpa-onnx int8, onnx-asr int8, parakeet.cpp q8_0) are 3–6× slower on this machine. Method,
versions and the full table: [`bench/README.md`](bench/README.md). \* Photon's test-clean pass ran on a loaded machine.

## Install

```sh
# CLI: release downloads are voxrt-<v>-darwin-universal (Apple Silicon + Intel Macs) and voxrt-<v>-linux-x86_64;
# the model: git lfs clone https://huggingface.co/moondream/parakeet-redux models/parakeet-redux
cargo install --path crates/voxrt-cli          # or: cargo build --release → target/release/voxrt
export VOXRT_MODEL_DIR=/path/to/parakeet-redux

# Python (wheels: macOS universal2, Linux x86-64; the model downloads on first use)
pip install voxrt                               # or: cd bindings/python && maturin develop --release

# Swift: scripts/build_xcframework.sh, then add bindings/swift as a local package
# C: cargo build --release -p voxrt-ffi → target/release/libvoxrt.{a,dylib}, header include/voxrt.h
```

## Use

```sh
voxrt transcribe talk.wav                                   # WAV: any rate, any channels, PCM or float
voxrt transcribe talk.wav --timestamps word --format json   # segments + words; --format srt for subtitles
cat talk.wav | voxrt transcribe - --quiet                   # stdin
voxrt transcribe talk.wav --threads 4 --precision f32       # exact fp32 reference path
```

```python
import voxrt                                   # pip install voxrt   (voxrt[audio] adds FLAC/OGG/MP3 via soundfile)
t = voxrt.Transcriber()                        # downloads moondream/parakeet-redux once (or a path / VOXRT_MODEL_DIR)
r = t.transcribe("talk.mp3")                   # WAV natively, FLAC/OGG/MP3 via soundfile, anything else via ffmpeg
print(r.text)
for s in r.segments:
    print(f"[{s.start:.2f}–{s.end:.2f}] {s.text}")   # s.words: word timestamps
r = t.transcribe(samples, sample_rate=48000)   # numpy array, mono or [n, channels]
st = t.stream()
for chunk in audio_chunks_16k: segments += st.push(chunk)   # same cuts as offline, by construction
segments += st.flush()
```

```swift
import Voxrt
let t = try Transcriber(modelDir: "/path/to/parakeet-redux")
print(try t.transcribe(path: "talk.wav").text)
```

```c
VoxrtTranscriber *t = voxrt_new("/path/to/parakeet-redux");
VoxrtTranscript *r = voxrt_transcribe_file(t, "talk.wav");
printf("%s\n", r->text);  voxrt_transcript_free(r);  voxrt_free(t);
```

## How it works

- **Kernels.** Ternary weights are repacked at load into 2-bit planes (429 MB resident), activations are quantised
  to int8 per token and 128-column group, and the GEMMs run on NEON `smmla` (I8MM) or `sdot` tiles. The attention
  core, softmax and context product are int8 too. Every SIMD kernel is bit-identical to a scalar reference (tested).
  x86 kernels (AVX-512 VNNI / AVX2) are the next port; the scalar path works everywhere.
- **Decoder.** TDT greedy with NeMo's exact semantics; int8 joint head and LSTM through an `sdot` GEMV.
- **Long-form.** The model's VAD head cuts at pauses into 10–60 s segments; the encoder (all cores) overlaps with
  the decoder (one core). Streaming feeds the same segment automaton, so streaming output equals offline output.
- **Accuracy.** fp32 path: token-identical to the Python reference on all of LibriSpeech test-clean (WER 1.96 = card).
  int8: WER 1.95; the texts that differ from fp32 are near-ties the quantised model gets right as often as fp32,
  the same deviation pattern Photon shows. Details: [`docs/kernels.md`](docs/kernels.md), [`docs/vad.md`](docs/vad.md),
  [`docs/decisions.md`](docs/decisions.md).

## Limits

- The model: a distilled 1.58-bit student. On clean read speech it is ~0.4 WER behind its teacher; the card reports a
  larger gap on noisy audio. voxrt does not change that.
- English-centric evaluation so far (LibriSpeech, TED-LIUM); the model covers 25 languages and voxrt decodes them,
  but we have not scored them.
- Streaming latency: the first segment arrives once the following pause is known (≥ 10 s of speech).
- Peak memory ~1.6 GB on a 20-minute talk (segment-sized buffers; to trim), 0.45 GB on short clips.
- Platforms: macOS (universal: Apple Silicon and Intel), iOS, Linux x86-64; measured on Apple Silicon and AMD
  Genoa / Turin. Intel CPUs run the same AVX2 / AVX-512 VNNI kernels but have not been benchmarked yet; Intel's 256-bit
  AVX-VNNI (12th gen and later) is not used yet (those CPUs take the AVX2 path). Windows and Linux arm64 builds: not yet.

## Development

```sh
cargo test --release                          # unit + golden-parity tests (skip without golden/ and models/)
scripts/wer.sh dev50 | test-clean             # leaderboard-exact WER through the CLI
bench/.venv/bin/python bench/run_all.py       # same-machine benchmark of all runtimes
scripts/c_smoke.sh · examples/python_smoke.py · swift run VoxrtSmoke    # binding smoke tests
```

voxrt is an independent implementation, built only from the CC-BY model repo, NeMo, HF transformers and public
papers; Moondream's Photon runtime is used only as a black-box benchmark. Design decisions: [`docs/decisions.md`](docs/decisions.md).

## Credits

Model weights © NVIDIA and M87 Labs (Moondream), CC-BY-4.0. Text normaliser vendored from the Open ASR Leaderboard
(Apache-2.0). Papers: TDT (arXiv:2304.06795), FastConformer (arXiv:2305.05084), BitNet b1.58 (arXiv:2402.17764).
