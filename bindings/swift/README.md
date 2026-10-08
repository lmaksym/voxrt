# Voxrt (Swift)

Swift package over the C ABI. Build the framework once (`scripts/build_xcframework.sh`, needs the Rust iOS targets
for the iOS slices), then add this directory as a local package or copy `CVoxrt.xcframework` + `Sources/Voxrt`.

```swift
import Voxrt
let t = try Transcriber(modelDir: "/path/to/parakeet-redux")
let r = try t.transcribe(path: "talk.wav")            // or transcribe(samples:sampleRate:)
print(r.text); for s in r.segments { print(s.start, s.end, s.text) }   // s.words has word timestamps
let st = try t.stream(); let segs = try st.push(chunk16k); let rest = try st.flush()
```

`swift run VoxrtSmoke MODEL_DIR FILE.wav` is the smoke test. On macOS the package links CoreFoundation and libc++
(needed by the Rust static library).
