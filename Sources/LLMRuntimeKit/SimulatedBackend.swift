import Foundation

// MARK: - Simulated backend

public enum SimulatedBackendError: Error, Sendable, Hashable {
    case loadFailed
    case generationFailed
}

/// A deterministic, script-driven `InferenceBackend`.
///
/// This ships in the library (not just the test target) on purpose, and its
/// docs say what it is: a **simulation**. It exists so that (a) the test
/// suite can exercise every failure mode of the layers above the backend
/// seam — coalescing, eviction, rollback, governance — deterministically,
/// and (b) the demo app can run the full pipeline on a Simulator with no
/// model weights. It makes no pretense of measuring real inference; real
/// MLX / Core ML / llama.cpp adapters implement `InferenceBackend` in their
/// own modules and slot in unchanged.
public actor SimulatedInferenceBackend: InferenceBackend {
    public struct Behavior: Sendable {
        /// Artificial latency per load call.
        public var loadDelayNanoseconds: UInt64
        /// Fail this many subsequent load calls, then succeed.
        public var failLoadsRemaining: Int
        /// Artificial latency before each emitted token.
        public var tokenDelayNanoseconds: UInt64
        /// If set, the stream throws after emitting this many tokens.
        public var failAfterTokens: Int?
        /// Maps a prompt to the reply tokens the "model" produces.
        public var replyProvider: @Sendable (String) -> [String]

        public init(
            loadDelayNanoseconds: UInt64 = 0,
            failLoadsRemaining: Int = 0,
            tokenDelayNanoseconds: UInt64 = 0,
            failAfterTokens: Int? = nil,
            replyProvider: @escaping @Sendable (String) -> [String] = Behavior.defaultReply
        ) {
            self.loadDelayNanoseconds = loadDelayNanoseconds
            self.failLoadsRemaining = max(0, failLoadsRemaining)
            self.tokenDelayNanoseconds = tokenDelayNanoseconds
            self.failAfterTokens = failAfterTokens
            self.replyProvider = replyProvider
        }

        /// Default script: a short deterministic acknowledgement that echoes
        /// the prompt's words back, so streaming is visible in the demo.
        public static let defaultReply: @Sendable (String) -> [String] = { prompt in
            let words = prompt.split(separator: " ").map(String.init)
            let echo = words.isEmpty ? ["(empty", "prompt)"] : words
            return ["Considering"] + echo.map { "'\($0)'" } + ["—", "done."]
        }
    }

    public nonisolated let descriptor: RuntimeDescriptor

    private var behavior: Behavior
    public private(set) var loadCallCount = 0
    public private(set) var unloadCallCount = 0
    public private(set) var activeInstanceIDs: Set<String> = []

    public init(descriptor: RuntimeDescriptor, behavior: Behavior = Behavior()) {
        self.descriptor = descriptor
        self.behavior = behavior
    }

    public func setBehavior(_ newBehavior: Behavior) {
        behavior = newBehavior
    }

    // MARK: InferenceBackend

    public func loadModel(
        manifest: ModelManifest,
        quantization: QuantizationOption
    ) async throws -> BackendModelInstance {
        loadCallCount += 1
        if behavior.loadDelayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: behavior.loadDelayNanoseconds)
        }
        if behavior.failLoadsRemaining > 0 {
            behavior.failLoadsRemaining -= 1
            throw SimulatedBackendError.loadFailed
        }
        let instanceID = "\(manifest.id)/\(quantization.name)/\(UUID().uuidString)"
        activeInstanceIDs.insert(instanceID)
        return BackendModelInstance(id: instanceID)
    }

    public func unloadModel(_ instance: BackendModelInstance) async {
        unloadCallCount += 1
        activeInstanceIDs.remove(instance.id)
    }

    public func generate(
        instance: BackendModelInstance,
        prompt: String,
        maxTokens: Int
    ) async -> AsyncThrowingStream<String, Error> {
        // Snapshot behavior so the stream's detached work sees a consistent
        // configuration even if `setBehavior` runs mid-stream.
        let tokens = behavior.replyProvider(prompt)
        let failAfter = behavior.failAfterTokens
        let delay = behavior.tokenDelayNanoseconds
        let cap = max(0, maxTokens)

        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let task = Task {
            var emitted = 0
            for token in tokens.prefix(cap) {
                if let failAfter, emitted >= failAfter {
                    continuation.finish(throwing: SimulatedBackendError.generationFailed)
                    return
                }
                if delay > 0 {
                    do {
                        try await Task.sleep(nanoseconds: delay)
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                if Task.isCancelled {
                    continuation.finish(throwing: CancellationError())
                    return
                }
                continuation.yield(token)
                emitted += 1
            }
            continuation.finish()
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }
}
