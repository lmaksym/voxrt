// swift-tools-version:5.9
// Swift package over the C ABI. The binary target is built by scripts/build_xcframework.sh
// (macOS arm64 + iOS arm64 + iOS simulator arm64) from crates/voxrt-ffi.
import PackageDescription

let package = Package(
    name: "Voxrt",
    platforms: [.macOS(.v11), .iOS(.v14)],
    products: [
        .library(name: "Voxrt", targets: ["Voxrt"]),
        .executable(name: "VoxrtSmoke", targets: ["VoxrtSmoke"]),
    ],
    targets: [
        .binaryTarget(name: "CVoxrt", path: "CVoxrt.xcframework"),
        .target(name: "Voxrt", dependencies: ["CVoxrt"], linkerSettings: [.linkedLibrary("c++"), .linkedFramework("CoreFoundation")]),
        .executableTarget(name: "VoxrtSmoke", dependencies: ["Voxrt"]),
    ]
)
