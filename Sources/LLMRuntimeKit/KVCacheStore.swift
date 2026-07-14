import Foundation

// MARK: - Outcome

/// What one append actually did to the store — auditable, so callers (and
/// the demo UI) can show *why* memory moved.
public struct KVAppendOutcome: Sendable, Hashable {
    /// Tokens the caller asked to append (after input clamping).
    public let appendedTokens: Int
    /// Own-session tokens dropped by the per-session sliding window.
    public let windowTrimmedTokens: Int
    /// Other sessions whose caches were dropped entirely (LRU order).
    public let evictedSessionIDs: [String]
    /// Own oldest tokens dropped because this session alone exceeded the
    /// global budget.
    public let selfTrimmedTokens: Int

    public static let none = KVAppendOutcome(
        appendedTokens: 0, windowTrimmedTokens: 0,
        evictedSessionIDs: [], selfTrimmedTokens: 0
    )
}

// MARK: - Store

/// Byte-budgeted bookkeeping for per-session KV caches.
///
/// The KV cache is the quiet killer of on-device inference: it grows linearly
/// with context (`kvBytesPerToken × tokens`) and, unlike weights, it grows
/// *while the user watches*. This store makes that growth explicit, bounded,
/// and observable.
///
/// Eviction semantics, and why:
///
/// * **Across sessions, eviction is all-or-nothing.** A KV cache with holes
///   in the middle is useless — attention state is positional — so dropping
///   "some" of another session's cache is equivalent to dropping all of it
///   while pretending otherwise. Whole-session eviction is the only honest
///   granularity. An evicted session is not destroyed: it re-prefills on its
///   next turn and pays that cost visibly.
/// * **Within a session, the oldest tokens go first.** Dropping the oldest
///   context is exactly the sliding-window behavior the model's finite
///   context imposes anyway, so the mechanism reuses a semantic the model
///   already has, rather than inventing a new failure mode.
/// * **The appender pays.** When an append pushes the store over budget,
///   other sessions are evicted LRU-first; if the appender *alone* exceeds
///   the budget, its own oldest tokens are trimmed. The cost lands on the
///   actor causing the pressure, never silently on an idle session first
///   unless LRU says so.
public actor KVCacheStore {
    private struct Entry {
        var tokenCount: Int
        let bytesPerToken: Int64
        let windowTokens: Int
        var lastUsed: UInt64
    }

    private let budgetBytes: Int64
    private var sessions: [String: Entry] = [:]
    private var useCounter: UInt64 = 0

    public init(budgetBytes: Int64) {
        self.budgetBytes = max(0, budgetBytes)
    }

    // MARK: Introspection

    public var usageBytes: Int64 {
        sessions.values.reduce(0) { $0 + Int64($1.tokenCount) * $1.bytesPerToken }
    }

    public func tokenCount(sessionID: String) -> Int {
        sessions[sessionID]?.tokenCount ?? 0
    }

    public func contains(sessionID: String) -> Bool {
        sessions[sessionID] != nil
    }

    // MARK: Lifecycle

    /// Registers (or re-registers) a session. `windowTokens` of 0 means "no
    /// per-session window" — the global budget is then the only bound.
    public func register(sessionID: String, bytesPerToken: Int64, windowTokens: Int) {
        sessions[sessionID] = Entry(
            tokenCount: 0,
            bytesPerToken: max(0, bytesPerToken),
            windowTokens: max(0, windowTokens),
            lastUsed: nextUse()
        )
    }

    public func remove(sessionID: String) {
        sessions[sessionID] = nil
    }

    // MARK: Append / rollback

    /// Appends `tokens` to the session's cache and enforces both bounds
    /// (per-session window, global budget). Appending to an unknown session
    /// or appending a non-positive count is a no-op returning `.none` — the
    /// store never traps on caller mistakes.
    @discardableResult
    public func append(sessionID: String, tokens: Int) -> KVAppendOutcome {
        guard tokens > 0, var entry = sessions[sessionID] else { return .none }

        entry.tokenCount += tokens
        entry.lastUsed = nextUse()

        // 1. Per-session sliding window.
        var windowTrimmed = 0
        if entry.windowTokens > 0, entry.tokenCount > entry.windowTokens {
            windowTrimmed = entry.tokenCount - entry.windowTokens
            entry.tokenCount = entry.windowTokens
        }
        sessions[sessionID] = entry

        // 2. Global budget: evict other sessions, LRU first, whole caches.
        var evicted: [String] = []
        while usageBytes > budgetBytes {
            guard let victimID = lruSessionID(excluding: sessionID, requireNonEmpty: true) else { break }
            if var victim = sessions[victimID] {
                victim.tokenCount = 0
                sessions[victimID] = victim
                evicted.append(victimID)
            }
        }

        // 3. Still over budget → the appender alone exceeds it; trim self.
        var selfTrimmed = 0
        if usageBytes > budgetBytes, var own = sessions[sessionID], own.bytesPerToken > 0 {
            let overBytes = usageBytes - budgetBytes
            let tokensToDrop = Int(
                min(Int64(own.tokenCount),
                    (overBytes + own.bytesPerToken - 1) / own.bytesPerToken)
            )
            own.tokenCount -= tokensToDrop
            selfTrimmed = tokensToDrop
            sessions[sessionID] = own
        }

        return KVAppendOutcome(
            appendedTokens: tokens,
            windowTrimmedTokens: windowTrimmed,
            evictedSessionIDs: evicted,
            selfTrimmedTokens: selfTrimmed
        )
    }

    /// Removes `tokens` from the session's count, clamping at zero. Used by
    /// sessions to keep turns transactional: a failed or cancelled turn rolls
    /// back exactly what it appended.
    public func rollback(sessionID: String, tokens: Int) {
        guard tokens > 0, var entry = sessions[sessionID] else { return }
        entry.tokenCount = max(0, entry.tokenCount - tokens)
        sessions[sessionID] = entry
    }

    // MARK: Pressure response

    /// Evicts whole sessions (LRU first) until usage is at or below
    /// `fraction` of the budget. Returns bytes freed. Registration survives —
    /// evicted sessions re-prefill on their next turn.
    @discardableResult
    public func trim(toFraction fraction: Double) -> Int64 {
        let clamped = min(max(fraction, 0), 1)
        let target = Int64((Double(budgetBytes) * clamped).rounded(.down))
        var freed: Int64 = 0
        while usageBytes > target {
            guard let victimID = lruSessionID(excluding: nil, requireNonEmpty: true) else { break }
            if var victim = sessions[victimID] {
                freed += Int64(victim.tokenCount) * victim.bytesPerToken
                victim.tokenCount = 0
                sessions[victimID] = victim
            }
        }
        return freed
    }

    // MARK: Internals

    private func lruSessionID(excluding excluded: String?, requireNonEmpty: Bool) -> String? {
        sessions
            .filter { key, value in
                key != excluded && (!requireNonEmpty || value.tokenCount > 0)
            }
            .min { $0.value.lastUsed < $1.value.lastUsed }?
            .key
    }

    private func nextUse() -> UInt64 {
        useCounter += 1
        return useCounter
    }
}
