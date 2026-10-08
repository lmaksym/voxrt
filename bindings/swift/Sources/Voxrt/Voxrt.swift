// Swift wrapper over the C ABI (include/voxrt.h). Thread-safe: a Transcriber may be used from several threads.
import CVoxrt
import Foundation

public struct VoxrtError: Error, CustomStringConvertible {
    public let message: String
    public var description: String { message }
    static func last() -> VoxrtError { VoxrtError(message: String(cString: voxrt_last_error())) }
}

public struct Word { public let start: Double, end: Double, word: String }
public struct Segment { public let start: Double, end: Double, text: String, words: [Word] }
public struct Transcript {
    public let text: String
    public let segments: [Segment]
    public let duration: Double

    init(_ p: UnsafeMutablePointer<VoxrtTranscript>) {
        let r = p.pointee
        text = String(cString: r.text)
        duration = r.duration
        var segs: [Segment] = []
        for i in 0..<r.n_segments {
            let s = r.segments[i]
            var words: [Word] = []
            for j in 0..<s.n_words {
                let w = s.words[j]
                words.append(Word(start: w.start, end: w.end, word: String(cString: w.word)))
            }
            segs.append(Segment(start: s.start, end: s.end, text: String(cString: s.text), words: words))
        }
        segments = segs
        voxrt_transcript_free(p)
    }
}

public final class Transcriber {
    let handle: OpaquePointer

    public static var version: String { String(cString: voxrt_version()) }

    /// Load a model directory (the `moondream/parakeet-redux` layout).
    public init(modelDir: String) throws {
        guard let h = voxrt_new(modelDir) else { throw VoxrtError.last() }
        handle = h
    }
    deinit { voxrt_free(handle) }

    /// Transcribe a WAV file (any rate / channels / PCM format). Long audio is segmented automatically.
    public func transcribe(path: String) throws -> Transcript {
        guard let r = voxrt_transcribe_file(handle, path) else { throw VoxrtError.last() }
        return Transcript(r)
    }

    /// Transcribe mono float samples at `sampleRate` (resampled to 16 kHz inside).
    public func transcribe(samples: [Float], sampleRate: UInt32 = 16_000) throws -> Transcript {
        let r = samples.withUnsafeBufferPointer { voxrt_transcribe(handle, $0.baseAddress, $0.count, sampleRate) }
        guard let r else { throw VoxrtError.last() }
        return Transcript(r)
    }

    /// Streaming session; keep the transcriber alive while it is in use (the stream holds a reference).
    public func stream() throws -> Stream { try Stream(self) }
}

public final class Stream {
    let owner: Transcriber
    let handle: OpaquePointer

    init(_ t: Transcriber) throws {
        owner = t
        guard let h = voxrt_stream_new(t.handle) else { throw VoxrtError.last() }
        handle = h
    }
    deinit { voxrt_stream_free(handle) }

    /// Push 16 kHz mono samples; returns the segments finalised by this call (streaming == offline cuts).
    public func push(_ samples: [Float]) throws -> [Segment] {
        let r = samples.withUnsafeBufferPointer { voxrt_stream_push(handle, $0.baseAddress, $0.count) }
        guard let r else { throw VoxrtError.last() }
        return Transcript(r).segments
    }

    public func flush() throws -> [Segment] {
        guard let r = voxrt_stream_flush(handle) else { throw VoxrtError.last() }
        return Transcript(r).segments
    }
}
