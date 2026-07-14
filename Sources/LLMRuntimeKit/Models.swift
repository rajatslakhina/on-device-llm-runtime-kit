import Foundation

// MARK: - Model format

/// The on-disk / in-memory format a distributable model artifact ships in.
///
/// A *logical* model (e.g. "Atlas 3B Instruct") may be distributed in several
/// formats; each format is a separate ``ModelManifest``. Runtimes advertise
/// the formats they can execute via ``RuntimeDescriptor/supportedFormats``,
/// and the selector matches on this. There is deliberately no "conversion"
/// concept in this layer: converting between weight formats is an offline
/// pipeline concern, not something to attempt on a phone at request time.
public enum ModelFormat: String, Sendable, Hashable, CaseIterable {
    /// MLX-native safetensors shards (Apple-silicon GPU via MLX).
    case mlxSafetensors
    /// A compiled Core ML model package (ANE/GPU dispatch decided by Core ML).
    case coreMLPackage
    /// llama.cpp's GGUF single-file format (CPU, or GPU via Metal).
    case gguf
    /// Apple's system-provided Foundation Models. No weights ship with the
    /// app, so disk cost is zero and memory cost is only the session working
    /// set the OS attributes to the process.
    case foundationModels
}

// MARK: - Quantization

/// One quantization variant of a model, with the resource envelope the
/// selection layer reasons about.
///
/// `estimatedMemoryBytes` is the *working set* (weights plus fixed runtime
/// buffers), not just the file size — on iOS the number that gets a process
/// jetsammed is resident memory, so that is the number selection must use.
public struct QuantizationOption: Sendable, Hashable {
    public let name: String
    public let bitsPerWeight: Double
    /// Expected resident memory once loaded (weights + fixed buffers).
    public let estimatedMemoryBytes: Int64
    /// Bytes required on disk to store the artifact (0 for system models).
    public let estimatedDiskBytes: Int64
    /// Relative answer-quality score in `0...1` against the unquantized
    /// reference (task-level eval score, not perplexity — perplexity deltas
    /// are famously misleading across quantization schemes).
    public let qualityScore: Double

    public init(
        name: String,
        bitsPerWeight: Double,
        estimatedMemoryBytes: Int64,
        estimatedDiskBytes: Int64,
        qualityScore: Double
    ) {
        self.name = name
        self.bitsPerWeight = max(0, bitsPerWeight)
        self.estimatedMemoryBytes = max(0, estimatedMemoryBytes)
        self.estimatedDiskBytes = max(0, estimatedDiskBytes)
        self.qualityScore = min(max(qualityScore, 0), 1)
    }
}

// MARK: - Model manifest

/// A distributable model artifact: one logical model in one concrete format,
/// with the quantization variants available for it.
public struct ModelManifest: Sendable, Hashable {
    public let id: String
    public let displayName: String
    /// Absolute parameter count (e.g. `3_000_000_000`).
    public let parameterCount: Int64
    public let format: ModelFormat
    /// Maximum context length the model supports. Also used by the KV-cache
    /// layer as the per-session sliding-window bound.
    public let contextWindowTokens: Int
    /// KV-cache growth per token held in context. This is what makes context
    /// length a *memory* problem and not just a latency problem on device.
    public let kvBytesPerToken: Int64
    public let quantizations: [QuantizationOption]

    public init(
        id: String,
        displayName: String,
        parameterCount: Int64,
        format: ModelFormat,
        contextWindowTokens: Int,
        kvBytesPerToken: Int64,
        quantizations: [QuantizationOption]
    ) {
        self.id = id
        self.displayName = displayName
        self.parameterCount = max(0, parameterCount)
        self.format = format
        self.contextWindowTokens = max(0, contextWindowTokens)
        self.kvBytesPerToken = max(0, kvBytesPerToken)
        self.quantizations = quantizations
    }
}

// MARK: - Thermal state

/// Mirror of `ProcessInfo.ThermalState`, defined locally so the selection and
/// governance layers stay platform-agnostic and headlessly testable.
public enum ThermalState: Int, Sendable, Hashable, Comparable, CaseIterable {
    case nominal = 0
    case fair = 1
    case serious = 2
    case critical = 3

    public static func < (lhs: ThermalState, rhs: ThermalState) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

// MARK: - Device profile

/// A snapshot of the device the selection decision is being made for.
///
/// `usableMemoryBytes` is deliberately distinct from `totalMemoryBytes`:
/// iOS will jetsam a process long before it approaches physical RAM, and the
/// per-process ceiling varies by device and OS. Callers populate this from
/// `os_proc_available_memory()` (or a conservative fraction of physical RAM);
/// the selector treats it as the hard truth.
public struct DeviceProfile: Sendable, Hashable {
    public let totalMemoryBytes: Int64
    public let usableMemoryBytes: Int64
    public let hasNeuralEngine: Bool
    public let osMajorVersion: Int
    public let freeDiskBytes: Int64
    public let thermalState: ThermalState

    public init(
        totalMemoryBytes: Int64,
        usableMemoryBytes: Int64,
        hasNeuralEngine: Bool,
        osMajorVersion: Int,
        freeDiskBytes: Int64,
        thermalState: ThermalState
    ) {
        let total = max(0, totalMemoryBytes)
        self.totalMemoryBytes = total
        // Usable memory can never exceed physical memory; clamp rather than
        // trust the caller, because an inverted pair here would silently
        // corrupt every downstream memory decision.
        self.usableMemoryBytes = min(max(0, usableMemoryBytes), total)
        self.hasNeuralEngine = hasNeuralEngine
        self.osMajorVersion = max(0, osMajorVersion)
        self.freeDiskBytes = max(0, freeDiskBytes)
        self.thermalState = thermalState
    }
}
