// `swift run VoxrtSmoke MODEL_DIR FILE.wav` — transcribes a file and prints segments with the first word of each.
import Foundation
import Voxrt

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: VoxrtSmoke MODEL_DIR FILE.wav"); exit(2) }
do {
    let t0 = Date()
    let t = try Transcriber(modelDir: args[1])
    print("voxrt \(Transcriber.version) · loaded in \(String(format: "%.2f", Date().timeIntervalSince(t0))) s")
    let t1 = Date()
    let r = try t.transcribe(path: args[2])
    let dt = Date().timeIntervalSince(t1)
    print("\(String(format: "%.1f", r.duration)) s of audio in \(String(format: "%.2f", dt)) s (\(String(format: "%.0f", r.duration / dt))× real time) · \(r.segments.count) segments")
    print(r.text)
    for s in r.segments.prefix(3) {
        let w = s.words.first.map { "\($0.word) \(String(format: "%.2f", $0.start))–\(String(format: "%.2f", $0.end))" } ?? ""
        print("[\(String(format: "%.2f", s.start))–\(String(format: "%.2f", s.end))] \(s.words.count) words · first: \(w)")
    }
} catch {
    print("error: \(error)"); exit(1)
}
