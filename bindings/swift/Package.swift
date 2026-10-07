// swift-tools-version:5.9
// Swift package over the C ABI (PLAN 5.3). The binary target is built by scripts/build_xcframework.sh
// (macOS arm64 + iOS arm64 + iOS simulator arm64) from crates/prx-ffi.
import PackageDescription

let package = Package(
    name: "Prx",
    platforms: [.macOS(.v11), .iOS(.v14)],
    products: [
        .library(name: "Prx", targets: ["Prx"]),
        .executable(name: "PrxSmoke", targets: ["PrxSmoke"]),
    ],
    targets: [
        .binaryTarget(name: "CPrx", path: "CPrx.xcframework"),
        .target(name: "Prx", dependencies: ["CPrx"], linkerSettings: [.linkedLibrary("c++"), .linkedFramework("CoreFoundation")]),
        .executableTarget(name: "PrxSmoke", dependencies: ["Prx"]),
    ]
)
