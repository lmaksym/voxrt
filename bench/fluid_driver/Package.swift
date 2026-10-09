// swift-tools-version:6.0
// FluidAudio driver for bench/run_all.py: model loaded once, two warm-up files, then each file timed from URL to text
// (file decode included, as for the other runtimes). FluidAudio is a local clone in bench/FluidAudio (gitignored).
import PackageDescription

let package = Package(
    name: "fluid_driver",
    platforms: [.macOS(.v15)],
    dependencies: [.package(path: "../FluidAudio")],
    targets: [
        .executableTarget(name: "fluid_driver", dependencies: [.product(name: "FluidAudio", package: "FluidAudio")])
    ]
)
