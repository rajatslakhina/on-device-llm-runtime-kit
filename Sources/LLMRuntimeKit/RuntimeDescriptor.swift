import Foundation

// MARK: - Runtime descriptor

/// A declarative description of one inference runtime (MLX, Core ML,
/// llama.cpp, Foundation Models, …) — everything the selection layer needs
/// to reason about a runtime *without* linking it.
///
/// This is the seam that keeps the decision layer testable and the app's
/// binary size honest: the selector works over descriptors, and only the
/// chosen runtime's real backend ever has to be instantiated.
public struct RuntimeDescriptor: Sendable, Hashable {
    public let id: String
    public let displayName: String
    public let supportedFormats: Set<ModelFormat>
    public let minimumOSMajorVersion: Int
    public let requiresNeuralEngine: Bool
    /// Multiplier applied to a quantization's estimated memory to account
    /// for runtime overhead (allocator slack, graph buffers, ANE compilation
    /// copies). Always ≥ 1. Core ML's ANE compilation, for example, can
    /// transiently need noticeably more than the weights themselves.
    public let memoryOverheadFactor: Double
    /// Coarse relative decode-throughput rank in `0...1`, measured offline
    /// on representative hardware. Deliberately a rank, not a tokens/sec
    /// claim — absolute numbers vary too much across devices to encode here.
    public let throughputScore: Double
    /// Whether generation can be served token-by-token.
    public let supportsStreaming: Bool

    public init(
        id: String,
        displayName: String,
        supportedFormats: Set<ModelFormat>,
        minimumOSMajorVersion: Int,
        requiresNeuralEngine: Bool,
        memoryOverheadFactor: Double,
        throughputScore: Double,
        supportsStreaming: Bool
    ) {
        self.id = id
        self.displayName = displayName
        self.supportedFormats = supportedFormats
        self.minimumOSMajorVersion = max(0, minimumOSMajorVersion)
        self.requiresNeuralEngine = requiresNeuralEngine
        self.memoryOverheadFactor = max(1, memoryOverheadFactor)
        self.throughputScore = min(max(throughputScore, 0), 1)
        self.supportsStreaming = supportsStreaming
    }
}

// MARK: - Backend protocol

/// An opaque handle to a model instance a backend has actually loaded.
public struct BackendModelInstance: Sendable, Hashable {
    public let id: String

    public init(id: String) {
        self.id = id
    }
}

/// The execution seam: the one protocol a real MLX / Core ML / llama.cpp
/// adapter implements. Everything above this line (selection, loading policy,
/// KV budgeting, session semantics, governance) is runtime-agnostic and is
/// exactly the code this package exists to get right once.
public protocol InferenceBackend: Sendable {
    var descriptor: RuntimeDescriptor { get }

    /// Load (or compile) the given model + quantization. Implementations may
    /// take significant wall time; callers must expect suspension.
    func loadModel(
        manifest: ModelManifest,
        quantization: QuantizationOption
    ) async throws -> BackendModelInstance

    /// Release everything held for the instance. Must be idempotent.
    func unloadModel(_ instance: BackendModelInstance) async

    /// Begin one generation. The returned stream yields decoded tokens and
    /// finishes when the model stops (or throws on failure). Implementations
    /// should honor task cancellation promptly — an abandoned decode loop is
    /// pure battery drain.
    func generate(
        instance: BackendModelInstance,
        prompt: String,
        maxTokens: Int
    ) async -> AsyncThrowingStream<String, Error>
}
