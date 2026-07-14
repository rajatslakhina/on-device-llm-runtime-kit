import XCTest
@testable import LLMRuntimeKit

final class ResourceGovernorTests: XCTestCase {
    private struct Harness {
        let kvStore: KVCacheStore
        let loader: ModelLoader
        let backend: SimulatedInferenceBackend
        let governor: ResourceGovernor
    }

    /// KV budget 10 000 B; one session holding 8 000 B; loader budget
    /// 10 000 MB with one unpinned 1 000 MB model resident.
    private func makeHarness(policy: GovernorPolicy = GovernorPolicy()) async throws -> Harness {
        let kvStore = KVCacheStore(budgetBytes: 10_000)
        await kvStore.register(sessionID: "s1", bytesPerToken: 10, windowTokens: 0)
        _ = await kvStore.append(sessionID: "s1", tokens: 800)

        let backend = SimulatedInferenceBackend(descriptor: Fixtures.runtime(id: "sim"))
        let loader = ModelLoader(budgetBytes: 10_000 * MB)
        let quant = Fixtures.quant(name: "q4", memoryMB: 1000, quality: 0.8)
        let manifest = Fixtures.manifest(id: "m", quants: [quant])
        let model = try await loader.acquire(manifest: manifest, quantization: quant, backend: backend)
        await loader.release(model) // resident but unpinned

        let governor = ResourceGovernor(kvStore: kvStore, loader: loader, policy: policy)
        return Harness(kvStore: kvStore, loader: loader, backend: backend, governor: governor)
    }

    func testMemoryWarningTrimsKVToPolicyFraction() async throws {
        let harness = try await makeHarness()
        await harness.governor.handle(.memoryPressure(.warning))

        let usage = await harness.kvStore.usageBytes
        XCTAssertLessThanOrEqual(usage, 5000, "usage must be at or below half the KV budget")
        let modelCount = await harness.loader.loadedCount
        XCTAssertEqual(modelCount, 1, "a warning must not unload models")

        let log = await harness.governor.drainActionLog()
        XCTAssertEqual(log, [.trimmedKVCache(toFraction: 0.5, freedBytes: 8000)])
    }

    func testCriticalMemoryFlushesKVAndUnloadsIdleModels() async throws {
        let harness = try await makeHarness()
        await harness.governor.handle(.memoryPressure(.critical))

        let usage = await harness.kvStore.usageBytes
        XCTAssertEqual(usage, 0)
        let modelCount = await harness.loader.loadedCount
        XCTAssertEqual(modelCount, 0, "idle models must be evicted at critical pressure")
        let unloads = await harness.backend.unloadCallCount
        XCTAssertEqual(unloads, 1)

        let log = await harness.governor.drainActionLog()
        XCTAssertEqual(log, [
            .trimmedKVCache(toFraction: 0.0, freedBytes: 8000),
            .unloadedIdleModels(freedBytes: 1000 * MB)
        ])
    }

    func testSubCriticalThermalIsObservedButUntouched() async throws {
        let harness = try await makeHarness()
        await harness.governor.handle(.thermal(.serious))

        let usage = await harness.kvStore.usageBytes
        XCTAssertEqual(usage, 8000, "sub-critical thermal must not touch running state")
        let log = await harness.governor.drainActionLog()
        XCTAssertEqual(log, [.noAction(.thermal(.serious))])
    }

    func testCriticalThermalTrimsKVOnly() async throws {
        let harness = try await makeHarness()
        await harness.governor.handle(.thermal(.critical))

        let usage = await harness.kvStore.usageBytes
        XCTAssertLessThanOrEqual(usage, 5000)
        let modelCount = await harness.loader.loadedCount
        XCTAssertEqual(modelCount, 1, "thermal pressure must never unload models")

        let log = await harness.governor.drainActionLog()
        XCTAssertEqual(log, [.trimmedKVCache(toFraction: 0.5, freedBytes: 8000)])
    }

    func testSignalPumpProcessesStreamedSignals() async throws {
        let harness = try await makeHarness()
        let (stream, continuation) = AsyncStream<ResourceSignal>.makeStream()
        await harness.governor.start(signals: stream)
        continuation.yield(.memoryPressure(.warning))

        var actions: [GovernorAction] = []
        for _ in 0..<200 {
            actions += await harness.governor.drainActionLog()
            if !actions.isEmpty { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(actions, [.trimmedKVCache(toFraction: 0.5, freedBytes: 8000)])

        await harness.governor.stop()
        continuation.finish()
    }

    func testDrainEmptiesTheLog() async throws {
        let harness = try await makeHarness()
        await harness.governor.handle(.memoryPressure(.warning))
        let first = await harness.governor.drainActionLog()
        XCTAssertFalse(first.isEmpty)
        let second = await harness.governor.drainActionLog()
        XCTAssertTrue(second.isEmpty)
    }
}
