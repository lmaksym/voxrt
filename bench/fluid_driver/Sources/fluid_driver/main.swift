// usage: fluid_driver MANIFEST.jsonl OUT.jsonl redux|v3 ane|gpu
import CoreML
import FluidAudio
import Foundation

struct Row: Decodable { let id: String; let wav: String; let seconds: Double }

let args = CommandLine.arguments
let rows = try String(contentsOfFile: args[1], encoding: .utf8).split(separator: "\n")
    .map { try JSONDecoder().decode(Row.self, from: Data($0.utf8)) }
let version: AsrModelVersion = args[3] == "v3" ? .v3 : .redux
let units: MLComputeUnits = args[4] == "gpu" ? .cpuAndGPU : .cpuAndNeuralEngine

let t0 = Date()
let models = try await AsrModels.downloadAndLoad(version: version, encoderComputeUnits: units)
let manager = AsrManager(config: ASRConfig(tdtConfig: TdtConfig(blankId: version.blankId),
                                           encoderHiddenSize: version.encoderHiddenSize))
try await manager.loadModels(models)
let load = Date().timeIntervalSince(t0)

func run(_ r: Row) async throws -> (String, Double) {
    var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
    let t = Date()
    let res = try await manager.transcribe(URL(fileURLWithPath: r.wav), decoderState: &state)
    return (res.text, Date().timeIntervalSince(t))
}
for r in rows.prefix(2) { _ = try await run(r) }
var proc = 0.0, audio = 0.0, out = ""
for r in rows {
    let (text, dt) = try await run(r)
    proc += dt; audio += r.seconds
    let line = try JSONSerialization.data(withJSONObject: ["id": r.id, "text": text, "rtf": r.seconds / dt])
    out += String(decoding: line, as: UTF8.self) + "\n"
}
try out.write(toFile: args[2], atomically: true, encoding: .utf8)
print(String(format: "processing only: %.1fx (load %.2fs excluded)", audio / proc, load))
