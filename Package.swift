// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LLMRuntimeKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        // Library only — the runnable demo lives in a separate repository
        // (on-device-llm-runtime-kit-demo-app) that consumes this package
        // as a remote dependency, the way any real consumer would.
        .library(name: "LLMRuntimeKit", targets: ["LLMRuntimeKit"])
    ],
    targets: [
        .target(name: "LLMRuntimeKit"),
        .testTarget(name: "LLMRuntimeKitTests", dependencies: ["LLMRuntimeKit"])
    ]
)
