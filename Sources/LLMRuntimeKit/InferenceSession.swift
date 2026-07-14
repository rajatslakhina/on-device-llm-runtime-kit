import Foundation

// MARK: - Events and stats

public enum SessionError: Error, Sendable, Hashable {
    /// A turn is already streaming. Sessions are deliberately one-turn-at-a-
    /// time: interleaving two generations into one KV cache would corrupt
    /// its positional state. Rejected alternative — implicit queuing — hides
    /// unbounded latency behind an innocent-looking call; the caller should
    /// decide whether to wait, cancel, or open another session.
    case generationInProgress
}

public enum StopReason: String, Sendable, Hashable {
    /// The backend's stream ended on its own.
    case completed
    /// The session-side token cap ended the turn.
    case maxTokensReached
}

public struct GenerationStats: Sendable, Hashable {
    public let promptTokens: Int
    public let generatedTokens: Int
    /// `nil` when the turn produced no tokens at all.
    public let timeToFirstToken: TimeInterval?
    /// Decode throughput over the whole turn. Reported as 0 when elapsed
    /// time is 0 (possible under injected clocks) — never infinity.
    public let tokensPerSecond: Double
    public let stopReason: StopReason
}

public enum InferenceEvent: Sendable, Hashable {
    case started(promptTokens: Int)
    case token(String)
    case finished(GenerationStats)
}

// MARK: - Session

/// One conversation's execution state: a pinned model, a KV-cache identity,
/// and the turn loop that streams tokens out while accounting for every
/// token that enters the cache.
///
/// The invariant this type exists to defend: **turns are transactional with
/// respect to the KV cache.** A turn that completes keeps its tokens (the
/// context genuinely grew); a turn that fails or is cancelled rolls back
/// exactly the tokens it appended. Without this, every mid-stream failure
/// leaks phantom tokens into the budget until the numbers drift far enough
/// to evict innocent sessions.
public actor InferenceSession {
    public let id: String

    private let model: LoadedModel
    private let backend: any InferenceBackend
    private let kvStore: KVCacheStore
    private let estimator: any TokenEstimating
    private let nowProvider: any NowProviding
    private let defaultMaxTokens: Int
    private var isGenerating = false

    /// Creates a session and registers its KV-cache identity. The model's
    /// context window doubles as the per-session KV sliding window — the
    /// cache must never claim more context than the model can attend to.
    public init(
        id: String = UUID().uuidString,
        model: LoadedModel,
        backend: any InferenceBackend,
        kvStore: KVCacheStore,
        estimator: any TokenEstimating = HeuristicTokenEstimator(),
        nowProvider: any NowProviding = SystemNowProvider(),
        defaultMaxTokens: Int = 512
    ) async {
        self.id = id
        self.model = model
        self.backend = backend
        self.kvStore = kvStore
        self.estimator = estimator
        self.nowProvider = nowProvider
        self.defaultMaxTokens = max(1, defaultMaxTokens)
        await kvStore.register(
            sessionID: id,
            bytesPerToken: model.manifest.kvBytesPerToken,
            windowTokens: model.manifest.contextWindowTokens
        )
    }

    /// Starts one turn. Throws `SessionError.generationInProgress` if a turn
    /// is already streaming. Cancelling the returned stream's consumer
    /// cancels the turn and rolls back its KV contribution.
    public func respond(
        to prompt: String,
        maxTokens: Int? = nil
    ) throws -> AsyncThrowingStream<InferenceEvent, Error> {
        guard !isGenerating else { throw SessionError.generationInProgress }
        isGenerating = true

        let cap = max(1, maxTokens ?? defaultMaxTokens)
        let promptTokens = max(0, estimator.estimateTokens(for: prompt))

        let (stream, continuation) = AsyncThrowingStream<InferenceEvent, Error>.makeStream()
        let task = Task {
            await self.runTurn(
                prompt: prompt,
                promptTokens: promptTokens,
                cap: cap,
                continuation: continuation
            )
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    // MARK: Internals

    private func runTurn(
        prompt: String,
        promptTokens: Int,
        cap: Int,
        continuation: AsyncThrowingStream<InferenceEvent, Error>.Continuation
    ) async {
        defer { isGenerating = false }

        var turnAppendedTokens = 0
        continuation.yield(.started(promptTokens: promptTokens))

        // Prefill: the prompt enters the KV cache before decoding starts.
        if promptTokens > 0 {
            _ = await kvStore.append(sessionID: id, tokens: promptTokens)
            turnAppendedTokens += promptTokens
        }

        let startedAt = nowProvider.now()
        var firstTokenAt: TimeInterval?
        var generated = 0
        var stopReason = StopReason.completed

        let backendStream = await backend.generate(
            instance: model.instance,
            prompt: prompt,
            maxTokens: cap
        )

        do {
            for try await token in backendStream {
                try Task.checkCancellation()
                if firstTokenAt == nil {
                    firstTokenAt = nowProvider.now()
                }
                generated += 1
                _ = await kvStore.append(sessionID: id, tokens: 1)
                turnAppendedTokens += 1
                continuation.yield(.token(token))
                if generated >= cap {
                    stopReason = .maxTokensReached
                    break
                }
            }
            try Task.checkCancellation()

            let elapsed = max(nowProvider.now() - startedAt, 0)
            let tokensPerSecond = elapsed > 0 ? Double(generated) / elapsed : 0
            let stats = GenerationStats(
                promptTokens: promptTokens,
                generatedTokens: generated,
                timeToFirstToken: firstTokenAt.map { max($0 - startedAt, 0) },
                tokensPerSecond: tokensPerSecond,
                stopReason: stopReason
            )
            continuation.yield(.finished(stats))
            continuation.finish()
        } catch {
            // Transactional rollback: the failed/cancelled turn contributes
            // nothing to the cache. The prompt tokens roll back too — the
            // model never attended to them in a completed turn.
            await kvStore.rollback(sessionID: id, tokens: turnAppendedTokens)
            continuation.finish(throwing: error)
        }
    }
}
