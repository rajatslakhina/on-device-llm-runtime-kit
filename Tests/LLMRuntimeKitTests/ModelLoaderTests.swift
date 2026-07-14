import XCTest
@testable import LLMRuntimeKit

final class ModelLoaderTests: XCTestCase {
    private func makeBackend(
        id: String = "sim",
        overhead: Double = 1.0,
        behavior: SimulatedInferenceBackend.Behavior = .init()
    ) -> SimulatedInferenceBackend {
        SimulatedInferenceBackend(
            descriptor: Fixtures.runtime(id: id, overhead: overhead),
            behavior: behavior
        )
    }

    private func makeManifest(id: String, memoryMB: Int64 = 1000) -> (ModelManifest, QuantizationOption) {
        let quant = Fixtures.quant(name: "q4", memoryMB: memoryMB, quality: 0.8)
        return (Fixtures.manifest(id: id, quants: [quant]), quant)
    }

    // MARK: Basic residency

    func testAcquireLoadsOnceAndPins() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 10_000 * MB)
        let (manifest, quant) = makeManifest(id: "a")

        let model = try await loader.acquire(manifest: manifest, quantization: quant, backend: backend)

        let loadCalls = await backend.loadCallCount
        XCTAssertEqual(loadCalls, 1)
        let resident = await loader.isResident(model.key)
        XCTAssertTrue(resident)
        let pins = await loader.pinCount(for: model.key)
        XCTAssertEqual(pins, 1)
        let bytes = await loader.loadedBytes
        XCTAssertEqual(bytes, 1000 * MB)
    }

    func testSecondAcquireIsACacheHit() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 10_000 * MB)
        let (manifest, quant) = makeManifest(id: "a")

        let first = try await loader.acquire(manifest: manifest, quantization: quant, backend: backend)
        let second = try await loader.acquire(manifest: manifest, quantization: quant, backend: backend)

        XCTAssertEqual(first, second)
        let loadCalls = await backend.loadCallCount
        XCTAssertEqual(loadCalls, 1)
        let pins = await loader.pinCount(for: first.key)
        XCTAssertEqual(pins, 2)
    }

    // MARK: Single-flight

    func testConcurrentAcquiresCoalesceIntoOneLoad() async throws {
        let backend = makeBackend(behavior: .init(loadDelayNanoseconds: 50_000_000))
        let loader = ModelLoader(budgetBytes: 10_000 * MB)
        let (manifest, quant) = makeManifest(id: "a")

        async let m1 = loader.acquire(manifest: manifest, quantization: quant, backend: backend)
        async let m2 = loader.acquire(manifest: manifest, quantization: quant, backend: backend)
        async let m3 = loader.acquire(manifest: manifest, quantization: quant, backend: backend)
        async let m4 = loader.acquire(manifest: manifest, quantization: quant, backend: backend)
        async let m5 = loader.acquire(manifest: manifest, quantization: quant, backend: backend)
        let models = try await [m1, m2, m3, m4, m5]

        XCTAssertEqual(Set(models).count, 1, "all callers must share one instance")
        let loadCalls = await backend.loadCallCount
        XCTAssertEqual(loadCalls, 1, "five concurrent acquires must perform exactly one load")
        let pins = await loader.pinCount(for: models[0].key)
        XCTAssertEqual(pins, 5)
    }

    func testFailedLoadDoesNotPoisonNextAttempt() async throws {
        let backend = makeBackend(behavior: .init(failLoadsRemaining: 1))
        let loader = ModelLoader(budgetBytes: 10_000 * MB)
        let (manifest, quant) = makeManifest(id: "a")

        do {
            _ = try await loader.acquire(manifest: manifest, quantization: quant, backend: backend)
            XCTFail("first load should fail")
        } catch let error as SimulatedBackendError {
            XCTAssertEqual(error, .loadFailed)
        }

        // The failure must be forgotten: the next acquire starts fresh.
        let model = try await loader.acquire(manifest: manifest, quantization: quant, backend: backend)
        let resident = await loader.isResident(model.key)
        XCTAssertTrue(resident)
        let loadCalls = await backend.loadCallCount
        XCTAssertEqual(loadCalls, 2)
    }

    // MARK: Eviction

    func testLRUEvictionFreesOldestUnpinned() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 2500 * MB)
        let (mA, qA) = makeManifest(id: "a")
        let (mB, qB) = makeManifest(id: "b")
        let (mC, qC) = makeManifest(id: "c")

        let a = try await loader.acquire(manifest: mA, quantization: qA, backend: backend)
        await loader.release(a)
        let b = try await loader.acquire(manifest: mB, quantization: qB, backend: backend)
        await loader.release(b)

        // Third model exceeds the 2500 MB budget → LRU (a) must go.
        _ = try await loader.acquire(manifest: mC, quantization: qC, backend: backend)

        let aResident = await loader.isResident(a.key)
        let bResident = await loader.isResident(b.key)
        XCTAssertFalse(aResident)
        XCTAssertTrue(bResident)
        let unloads = await backend.unloadCallCount
        XCTAssertEqual(unloads, 1)
        let count = await loader.loadedCount
        XCTAssertEqual(count, 2)
    }

    func testRecentUseProtectsFromEviction() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 2500 * MB)
        let (mA, qA) = makeManifest(id: "a")
        let (mB, qB) = makeManifest(id: "b")
        let (mC, qC) = makeManifest(id: "c")

        let a = try await loader.acquire(manifest: mA, quantization: qA, backend: backend)
        await loader.release(a)
        let b = try await loader.acquire(manifest: mB, quantization: qB, backend: backend)
        await loader.release(b)

        // Touch "a" — it becomes most recently used.
        let aAgain = try await loader.acquire(manifest: mA, quantization: qA, backend: backend)
        await loader.release(aAgain)

        _ = try await loader.acquire(manifest: mC, quantization: qC, backend: backend)

        let aResident = await loader.isResident(a.key)
        let bResident = await loader.isResident(b.key)
        XCTAssertTrue(aResident, "recently used model must survive")
        XCTAssertFalse(bResident, "LRU model must be the one evicted")
    }

    func testPinnedModelsAreNeverEvicted() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 2500 * MB)
        let (mA, qA) = makeManifest(id: "a")
        let (mB, qB) = makeManifest(id: "b")
        let (mC, qC) = makeManifest(id: "c")

        let a = try await loader.acquire(manifest: mA, quantization: qA, backend: backend) // stays pinned
        let b = try await loader.acquire(manifest: mB, quantization: qB, backend: backend)
        await loader.release(b)

        _ = try await loader.acquire(manifest: mC, quantization: qC, backend: backend)

        let aResident = await loader.isResident(a.key)
        let bResident = await loader.isResident(b.key)
        XCTAssertTrue(aResident, "pinned model must never be evicted")
        XCTAssertFalse(bResident)
    }

    func testAllPinnedOverBudgetThrowsWithZeroReclaimable() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 2500 * MB)
        let (mA, qA) = makeManifest(id: "a")
        let (mB, qB) = makeManifest(id: "b")
        let (mC, qC) = makeManifest(id: "c")

        _ = try await loader.acquire(manifest: mA, quantization: qA, backend: backend)
        _ = try await loader.acquire(manifest: mB, quantization: qB, backend: backend)

        do {
            _ = try await loader.acquire(manifest: mC, quantization: qC, backend: backend)
            XCTFail("expected budgetExceeded")
        } catch let error as LoaderError {
            guard case .budgetExceeded(let required, let budget, let reclaimable) = error else {
                return XCTFail("unexpected loader error \(error)")
            }
            XCTAssertEqual(required, 1000 * MB)
            XCTAssertEqual(budget, 2500 * MB)
            XCTAssertEqual(reclaimable, 0, "everything resident is pinned")
        }

        // The freshly loaded instance must have been unloaded — no leak.
        let active = await backend.activeInstanceIDs
        XCTAssertEqual(active.count, 2)
    }

    func testModelLargerThanBudgetIsRefusedBeforeLoading() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 2500 * MB)
        let (manifest, quant) = makeManifest(id: "huge", memoryMB: 5000)

        do {
            _ = try await loader.acquire(manifest: manifest, quantization: quant, backend: backend)
            XCTFail("expected budgetExceeded")
        } catch let error as LoaderError {
            guard case .budgetExceeded(let required, let budget, _) = error else {
                return XCTFail("unexpected loader error \(error)")
            }
            XCTAssertEqual(required, 5000 * MB)
            XCTAssertEqual(budget, 2500 * MB)
        }
        let loadCalls = await backend.loadCallCount
        XCTAssertEqual(loadCalls, 0, "an impossible load must be refused before touching the backend")
    }

    // MARK: Release semantics

    func testOverReleaseAndUnknownReleaseAreSafeNoops() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 10_000 * MB)
        let (manifest, quant) = makeManifest(id: "a")

        let model = try await loader.acquire(manifest: manifest, quantization: quant, backend: backend)
        await loader.release(model)
        await loader.release(model) // over-release: clamped, no trap
        let pins = await loader.pinCount(for: model.key)
        XCTAssertEqual(pins, 0)

        let phantom = LoadedModel(
            key: ModelKey(manifestID: "ghost", quantizationName: "q4", runtimeID: "sim"),
            instance: BackendModelInstance(id: "ghost"),
            manifest: manifest, quantization: quant
        )
        await loader.release(phantom) // unknown handle: documented no-op
        let count = await loader.loadedCount
        XCTAssertEqual(count, 1)
    }

    // MARK: Trim

    func testTrimToZeroEvictsAllUnpinned() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 10_000 * MB)
        let (mA, qA) = makeManifest(id: "a")
        let (mB, qB) = makeManifest(id: "b")

        let a = try await loader.acquire(manifest: mA, quantization: qA, backend: backend)
        await loader.release(a)
        let b = try await loader.acquire(manifest: mB, quantization: qB, backend: backend)
        await loader.release(b)

        let freed = await loader.trim(toFraction: 0)

        XCTAssertEqual(freed, 2000 * MB)
        let count = await loader.loadedCount
        XCTAssertEqual(count, 0)
        let unloads = await backend.unloadCallCount
        XCTAssertEqual(unloads, 2)
    }

    func testTrimRespectsPins() async throws {
        let backend = makeBackend()
        let loader = ModelLoader(budgetBytes: 10_000 * MB)
        let (mA, qA) = makeManifest(id: "a")
        let (mB, qB) = makeManifest(id: "b")

        _ = try await loader.acquire(manifest: mA, quantization: qA, backend: backend) // pinned
        let b = try await loader.acquire(manifest: mB, quantization: qB, backend: backend)
        await loader.release(b)

        let freed = await loader.trim(toFraction: 0)

        XCTAssertEqual(freed, 1000 * MB, "only the unpinned model may be freed")
        let count = await loader.loadedCount
        XCTAssertEqual(count, 1)
    }
}
