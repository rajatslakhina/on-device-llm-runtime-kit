import Foundation
@testable import LLMRuntimeKit

let MB: Int64 = 1_048_576

enum Fixtures {
    static func quant(
        name: String,
        memoryMB: Int64,
        diskMB: Int64 = 0,
        quality: Double,
        bits: Double = 4
    ) -> QuantizationOption {
        QuantizationOption(
            name: name,
            bitsPerWeight: bits,
            estimatedMemoryBytes: memoryMB * MB,
            estimatedDiskBytes: diskMB * MB,
            qualityScore: quality
        )
    }

    static func manifest(
        id: String = "atlas-3b",
        format: ModelFormat = .gguf,
        contextTokens: Int = 4096,
        kvBytesPerToken: Int64 = 1024,
        quants: [QuantizationOption]
    ) -> ModelManifest {
        ModelManifest(
            id: id,
            displayName: id,
            parameterCount: 3_000_000_000,
            format: format,
            contextWindowTokens: contextTokens,
            kvBytesPerToken: kvBytesPerToken,
            quantizations: quants
        )
    }

    static func runtime(
        id: String,
        formats: Set<ModelFormat> = [.gguf],
        minOS: Int = 17,
        requiresANE: Bool = false,
        overhead: Double = 1.0,
        throughput: Double = 0.5,
        streaming: Bool = true
    ) -> RuntimeDescriptor {
        RuntimeDescriptor(
            id: id,
            displayName: id,
            supportedFormats: formats,
            minimumOSMajorVersion: minOS,
            requiresNeuralEngine: requiresANE,
            memoryOverheadFactor: overhead,
            throughputScore: throughput,
            supportsStreaming: streaming
        )
    }

    static func device(
        usableMB: Int64 = 4096,
        totalMB: Int64? = nil,
        ane: Bool = true,
        os: Int = 27,
        freeDiskMB: Int64 = 100_000,
        thermal: ThermalState = .nominal
    ) -> DeviceProfile {
        DeviceProfile(
            totalMemoryBytes: (totalMB ?? usableMB * 2) * MB,
            usableMemoryBytes: usableMB * MB,
            hasNeuralEngine: ane,
            osMajorVersion: os,
            freeDiskBytes: freeDiskMB * MB,
            thermalState: thermal
        )
    }
}

/// Deterministic time source. `autoAdvance` moves the clock forward on every
/// `now()` read, which makes latency stats exactly predictable in tests.
///
/// `@unchecked Sendable` is justified narrowly: the only mutable state
/// (`current`) is accessed exclusively inside `lock`.
final class ManualClock: NowProviding, @unchecked Sendable {
    private let lock = NSLock()
    private var current: TimeInterval
    private let autoAdvance: TimeInterval

    init(start: TimeInterval = 0, autoAdvance: TimeInterval = 0) {
        self.current = start
        self.autoAdvance = autoAdvance
    }

    func now() -> TimeInterval {
        lock.lock()
        defer { lock.unlock() }
        let value = current
        current += autoAdvance
        return value
    }

    func advance(by delta: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        current += delta
    }
}
