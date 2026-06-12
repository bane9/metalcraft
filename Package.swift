// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Metalcraft",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "Metalcraft",
            path: "Sources/Metalcraft",
            resources: [
                .copy("Resources/terrain.png"),
                .copy("Resources/mob"),
                .copy("Resources/gui"),
                .copy("Resources/environment"),
            ]
        )
    ]
)
