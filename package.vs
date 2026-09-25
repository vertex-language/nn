// The 'nn' repository: the layers models are made of. See README.md.
import PackageDescription

let package = Package(
    name: "nn",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(name: "nn", targets: ["nn"]),
        .executable(name: "test-nn", targets: ["test_nn"]),
    ],
    targets: [
        .target(
            name: "nn",
            path: "nn"
        ),
        .executableTarget(
            name: "test_nn",
            dependencies: ["nn"],
            path: "tests/nn"
        ),
    ]
)
