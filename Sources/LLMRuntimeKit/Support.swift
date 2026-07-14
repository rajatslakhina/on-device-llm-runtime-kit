import Foundation

// MARK: - Time

/// Injectable time source so latency statistics (time-to-first-token,
/// tokens/sec) are deterministic under test instead of flaky wall-clock
/// assertions.
public protocol NowProviding: Sendable {
    /// Monotonic-enough "now" in seconds. Only differences are ever used.
    func now() -> TimeInterval
}

public struct SystemNowProvider: NowProviding {
    public init() {}

    public func now() -> TimeInterval {
        Date().timeIntervalSinceReferenceDate
    }
}

// MARK: - Token estimation

/// Injectable prompt-token estimator. Real integrations plug the model's own
/// tokenizer in here; the default heuristic is good enough for budgeting and
/// is honest about being a heuristic.
public protocol TokenEstimating: Sendable {
    func estimateTokens(for text: String) -> Int
}

/// The classic ~4-characters-per-token heuristic. Chosen over shipping a real
/// tokenizer because (a) the correct tokenizer is model-specific and belongs
/// behind the `TokenEstimating` seam, and (b) KV budgeting only needs the
/// right order of magnitude — being 20% off on token count changes eviction
/// timing, never correctness.
public struct HeuristicTokenEstimator: TokenEstimating {
    public init() {}

    public func estimateTokens(for text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        return max(1, Int((Double(text.count) / 4.0).rounded(.up)))
    }
}
