import XCTest
@testable import LLMRuntimeKit

final class KVCacheStoreTests: XCTestCase {
    // MARK: Input guards

    func testAppendToUnknownSessionIsNoop() async {
        let store = KVCacheStore(budgetBytes: 10_000)
        let outcome = await store.append(sessionID: "ghost", tokens: 100)
        XCTAssertEqual(outcome, .none)
        let usage = await store.usageBytes
        XCTAssertEqual(usage, 0)
    }

    func testNonPositiveAppendIsNoop() async {
        let store = KVCacheStore(budgetBytes: 10_000)
        await store.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 0)
        let zero = await store.append(sessionID: "s1", tokens: 0)
        let negative = await store.append(sessionID: "s1", tokens: -5)
        XCTAssertEqual(zero, .none)
        XCTAssertEqual(negative, .none)
        let count = await store.tokenCount(sessionID: "s1")
        XCTAssertEqual(count, 0)
    }

    func testAppendAfterRemoveIsNoop() async {
        let store = KVCacheStore(budgetBytes: 10_000)
        await store.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 0)
        await store.remove(sessionID: "s1")
        let outcome = await store.append(sessionID: "s1", tokens: 10)
        XCTAssertEqual(outcome, .none)
    }

    // MARK: Sliding window

    func testPerSessionWindowTrimsOldestTokens() async {
        let store = KVCacheStore(budgetBytes: 1_000_000)
        await store.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 100)
        let outcome = await store.append(sessionID: "s1", tokens: 150)
        XCTAssertEqual(outcome.appendedTokens, 150)
        XCTAssertEqual(outcome.windowTrimmedTokens, 50)
        let count = await store.tokenCount(sessionID: "s1")
        XCTAssertEqual(count, 100)
    }

    // MARK: Global budget

    func testBudgetEvictsLRUOtherSessionWholesale() async {
        let store = KVCacheStore(budgetBytes: 10_000)
        await store.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 0)
        await store.register(sessionID: "s2", bytesPerToken: 10, windowTokens: 0)

        _ = await store.append(sessionID: "s1", tokens: 500)   // 5000 B
        let outcome = await store.append(sessionID: "s2", tokens: 700) // 7000 B → over

        XCTAssertEqual(outcome.evictedSessionIDs, ["s1"])
        XCTAssertEqual(outcome.selfTrimmedTokens, 0)
        let s1Count = await store.tokenCount(sessionID: "s1")
        let s2Count = await store.tokenCount(sessionID: "s2")
        XCTAssertEqual(s1Count, 0, "evicted session drops its whole cache")
        XCTAssertEqual(s2Count, 700, "the appender keeps everything")
        let stillRegistered = await store.contains(sessionID: "s1")
        XCTAssertTrue(stillRegistered, "eviction drops content, not registration")
    }

    func testSoleSessionOverBudgetTrimsItself() async {
        let store = KVCacheStore(budgetBytes: 1000)
        await store.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 0)
        let outcome = await store.append(sessionID: "s1", tokens: 150) // 1500 B
        // Over by 500 B → ceil(500/10) = 50 tokens dropped.
        XCTAssertEqual(outcome.selfTrimmedTokens, 50)
        let count = await store.tokenCount(sessionID: "s1")
        XCTAssertEqual(count, 100)
        let usage = await store.usageBytes
        XCTAssertEqual(usage, 1000)
    }

    func testZeroBudgetKeepsStoreEmpty() async {
        let store = KVCacheStore(budgetBytes: 0)
        await store.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 0)
        _ = await store.append(sessionID: "s1", tokens: 100)
        let usage = await store.usageBytes
        XCTAssertEqual(usage, 0)
        let count = await store.tokenCount(sessionID: "s1")
        XCTAssertEqual(count, 0)
    }

    func testUsageBytesAccountsForDifferentTokenSizes() async {
        let store = KVCacheStore(budgetBytes: 1_000_000)
        await store.register(sessionID: "small", bytesPerToken: 8, windowTokens: 0)
        await store.register(sessionID: "big", bytesPerToken: 64, windowTokens: 0)
        _ = await store.append(sessionID: "small", tokens: 100) // 800
        _ = await store.append(sessionID: "big", tokens: 100)   // 6400
        let usage = await store.usageBytes
        XCTAssertEqual(usage, 7200)
    }

    // MARK: Rollback

    func testRollbackRemovesExactly() async {
        let store = KVCacheStore(budgetBytes: 1_000_000)
        await store.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 0)
        _ = await store.append(sessionID: "s1", tokens: 100)
        await store.rollback(sessionID: "s1", tokens: 30)
        let count = await store.tokenCount(sessionID: "s1")
        XCTAssertEqual(count, 70)
    }

    func testRollbackClampsAtZero() async {
        let store = KVCacheStore(budgetBytes: 1_000_000)
        await store.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 0)
        _ = await store.append(sessionID: "s1", tokens: 10)
        await store.rollback(sessionID: "s1", tokens: 50)
        let count = await store.tokenCount(sessionID: "s1")
        XCTAssertEqual(count, 0, "rollback past zero must clamp, never underflow")
    }

    // MARK: Pressure trim

    func testTrimEvictsLRUSessionsUntilTarget() async {
        let store = KVCacheStore(budgetBytes: 10_000)
        await store.register(sessionID: "old", bytesPerToken: 10, windowTokens: 0)
        await store.register(sessionID: "new", bytesPerToken: 10, windowTokens: 0)
        _ = await store.append(sessionID: "old", tokens: 300) // 3000 B, older use
        _ = await store.append(sessionID: "new", tokens: 400) // 4000 B

        let freed = await store.trim(toFraction: 0.5) // target 5000 B, usage 7000 B

        XCTAssertEqual(freed, 3000, "evicting the LRU session suffices")
        let oldCount = await store.tokenCount(sessionID: "old")
        let newCount = await store.tokenCount(sessionID: "new")
        XCTAssertEqual(oldCount, 0)
        XCTAssertEqual(newCount, 400)
    }

    func testTrimToZeroFlushesEverything() async {
        let store = KVCacheStore(budgetBytes: 10_000)
        await store.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 0)
        await store.register(sessionID: "s2", bytesPerToken: 10, windowTokens: 0)
        _ = await store.append(sessionID: "s1", tokens: 100)
        _ = await store.append(sessionID: "s2", tokens: 200)

        let freed = await store.trim(toFraction: 0)

        XCTAssertEqual(freed, 3000)
        let usage = await store.usageBytes
        XCTAssertEqual(usage, 0)
    }
}
