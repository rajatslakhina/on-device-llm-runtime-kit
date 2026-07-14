import XCTest
@testable import LLMRuntimeKit

final class InferenceSessionTests: XCTestCase {
    private struct Harness {
        let backend: SimulatedInferenceBackend
        let kvStore: KVCacheStore
        let session: InferenceSession
    }

    /// kvBytesPerToken = 8, context window 1000, default reply = 5 tokens.
    private func makeHarness(
        behavior: SimulatedInferenceBackend.Behavior = .init(replyProvider: { _ in ["a", "b", "c", "d", "e"] }),
        nowProvider: any NowProviding = SystemNowProvider(),
        defaultMaxTokens: Int = 512
    ) async -> Harness {
        let backend = SimulatedInferenceBackend(
            descriptor: Fixtures.runtime(id: "sim"),
            behavior: behavior
        )
        let kvStore = KVCacheStore(budgetBytes: 1_000_000)
        let quant = Fixtures.quant(name: "q4", memoryMB: 100, quality: 0.8)
        let manifest = Fixtures.manifest(id: "m", contextTokens: 1000, kvBytesPerToken: 8, quants: [quant])
        let model = LoadedModel(
            key: ModelKey(manifestID: "m", quantizationName: "q4", runtimeID: "sim"),
            instance: BackendModelInstance(id: "m/q4/test"),
            manifest: manifest,
            quantization: quant
        )
        let session = await InferenceSession(
            id: "session-under-test",
            model: model,
            backend: backend,
            kvStore: kvStore,
            nowProvider: nowProvider,
            defaultMaxTokens: defaultMaxTokens
        )
        return Harness(backend: backend, kvStore: kvStore, session: session)
    }

    private func collect(_ stream: AsyncThrowingStream<InferenceEvent, Error>) async throws -> [InferenceEvent] {
        var events: [InferenceEvent] = []
        for try await event in stream {
            events.append(event)
        }
        return events
    }

    // MARK: Happy path

    func testTurnStreamsAllEventsInOrderAndGrowsKV() async throws {
        let harness = await makeHarness()
        // "hi there" = 8 chars → ceil(8/4) = 2 prompt tokens.
        let stream = try await harness.session.respond(to: "hi there")
        let events = try await collect(stream)

        XCTAssertEqual(events.first, .started(promptTokens: 2))
        let tokens = events.compactMap { if case .token(let t) = $0 { return t } else { return nil } }
        XCTAssertEqual(tokens, ["a", "b", "c", "d", "e"])
        guard case .finished(let stats)? = events.last else {
            return XCTFail("last event must be .finished, got \(String(describing: events.last))")
        }
        XCTAssertEqual(stats.promptTokens, 2)
        XCTAssertEqual(stats.generatedTokens, 5)
        XCTAssertEqual(stats.stopReason, .completed)

        let kvCount = await harness.kvStore.tokenCount(sessionID: "session-under-test")
        XCTAssertEqual(kvCount, 7, "completed turn keeps prompt + generated tokens in the cache")
    }

    func testMaxTokensCapStopsGeneration() async throws {
        let harness = await makeHarness()
        let stream = try await harness.session.respond(to: "hi there", maxTokens: 3)
        let events = try await collect(stream)

        let tokens = events.compactMap { if case .token(let t) = $0 { return t } else { return nil } }
        XCTAssertEqual(tokens, ["a", "b", "c"])
        guard case .finished(let stats)? = events.last else {
            return XCTFail("expected .finished")
        }
        XCTAssertEqual(stats.stopReason, .maxTokensReached)
        let kvCount = await harness.kvStore.tokenCount(sessionID: "session-under-test")
        XCTAssertEqual(kvCount, 5, "2 prompt + 3 generated")
    }

    func testEmptyPromptContributesZeroPromptTokens() async throws {
        let harness = await makeHarness()
        let stream = try await harness.session.respond(to: "")
        let events = try await collect(stream)
        XCTAssertEqual(events.first, .started(promptTokens: 0))
        let kvCount = await harness.kvStore.tokenCount(sessionID: "session-under-test")
        XCTAssertEqual(kvCount, 5)
    }

    func testEmptyReplyFinishesWithNilTimeToFirstToken() async throws {
        let harness = await makeHarness(
            behavior: .init(replyProvider: { _ in [] }),
            nowProvider: ManualClock(start: 0, autoAdvance: 1)
        )
        let stream = try await harness.session.respond(to: "hi")
        let events = try await collect(stream)
        guard case .finished(let stats)? = events.last else {
            return XCTFail("expected .finished")
        }
        XCTAssertEqual(stats.generatedTokens, 0)
        XCTAssertNil(stats.timeToFirstToken)
        XCTAssertEqual(stats.tokensPerSecond, 0)
    }

    // MARK: Turn transactionality

    func testMidStreamFailureRollsBackKV() async throws {
        let harness = await makeHarness(
            behavior: .init(failAfterTokens: 2, replyProvider: { _ in ["a", "b", "c", "d", "e"] })
        )
        let stream = try await harness.session.respond(to: "hi there")

        var received: [InferenceEvent] = []
        do {
            for try await event in stream {
                received.append(event)
            }
            XCTFail("stream must throw on backend failure")
        } catch let error as SimulatedBackendError {
            XCTAssertEqual(error, .generationFailed)
        }

        let tokens = received.compactMap { if case .token(let t) = $0 { return t } else { return nil } }
        XCTAssertEqual(tokens, ["a", "b"], "tokens before the failure were delivered")
        let kvCount = await harness.kvStore.tokenCount(sessionID: "session-under-test")
        XCTAssertEqual(kvCount, 0, "failed turn must contribute nothing to the KV cache")
    }

    func testConsumerCancellationRollsBackKV() async throws {
        let harness = await makeHarness(
            behavior: .init(
                tokenDelayNanoseconds: 30_000_000,
                replyProvider: { _ in (0..<20).map(String.init) }
            )
        )
        let stream = try await harness.session.respond(to: "hello")

        // Note: merely `break`-ing out of a `for try await` loop drops the
        // iterator but does NOT terminate an AsyncThrowingStream. Cancelling
        // the consuming task does — that is the path a real UI takes when
        // the user taps "stop".
        let consumer = Task {
            var seen = 0
            do {
                for try await event in stream {
                    if case .token = event { seen += 1 }
                }
            } catch {
                // Cancellation surfaces as a throw; expected.
            }
            return seen
        }
        try await Task.sleep(nanoseconds: 100_000_000) // ~3 tokens at 30 ms each
        consumer.cancel()
        let seen = await consumer.value
        XCTAssertLessThan(seen, 20, "cancellation must land mid-stream to test anything")

        // Rollback is asynchronous with respect to the consumer; poll.
        var rolledBack = false
        for _ in 0..<200 {
            let count = await harness.kvStore.tokenCount(sessionID: "session-under-test")
            if count == 0 {
                rolledBack = true
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(rolledBack, "cancelled turn must roll back every token it appended")
    }

    // MARK: Concurrency contract

    func testSecondTurnWhileStreamingThrows() async throws {
        let harness = await makeHarness(
            behavior: .init(
                tokenDelayNanoseconds: 30_000_000,
                replyProvider: { _ in ["a", "b", "c", "d", "e"] }
            )
        )
        let stream = try await harness.session.respond(to: "first")
        // Inline collection: capturing `self` (an XCTestCase) in a Task is a
        // Swift 6 strict-concurrency violation.
        let consumer = Task {
            var events: [InferenceEvent] = []
            for try await event in stream {
                events.append(event)
            }
            return events
        }

        do {
            _ = try await harness.session.respond(to: "second")
            XCTFail("expected generationInProgress")
        } catch let error as SessionError {
            XCTAssertEqual(error, .generationInProgress)
        }

        _ = try await consumer.value
    }

    func testSequentialTurnsBothCompleteAndAccumulateContext() async throws {
        let harness = await makeHarness()
        let first = try await harness.session.respond(to: "hi there")   // 2 + 5
        _ = try await collect(first)
        let second = try await harness.session.respond(to: "and then?") // ceil(9/4)=3 + 5
        _ = try await collect(second)

        let kvCount = await harness.kvStore.tokenCount(sessionID: "session-under-test")
        XCTAssertEqual(kvCount, 15, "context accumulates across completed turns")
    }

    // MARK: Stats

    func testStatsAreDeterministicUnderManualClock() async throws {
        // autoAdvance 1s per read; reads happen at: turn start, first token,
        // turn end → start 0, first token 1, end 2.
        let harness = await makeHarness(nowProvider: ManualClock(start: 0, autoAdvance: 1))
        let stream = try await harness.session.respond(to: "hi there")
        let events = try await collect(stream)
        guard case .finished(let stats)? = events.last else {
            return XCTFail("expected .finished")
        }
        XCTAssertEqual(stats.timeToFirstToken ?? -1, 1.0, accuracy: 0.0001)
        XCTAssertEqual(stats.tokensPerSecond, 2.5, accuracy: 0.0001, "5 tokens over 2 elapsed seconds")
    }
}
