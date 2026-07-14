import XCTest
@testable import LLMRuntimeKit

final class RuntimeSelectorTests: XCTestCase {
    private let selector = RuntimeSelector()

    // MARK: Degenerate inputs

    func testEmptyQuantizationsRejectsEveryRuntime() {
        let model = Fixtures.manifest(quants: [])
        let runtimes = [Fixtures.runtime(id: "aaa"), Fixtures.runtime(id: "bbb")]
        let decision = selector.select(
            model: model, device: Fixtures.device(),
            runtimes: runtimes, policy: SelectionPolicy()
        )
        XCTAssertNil(decision.selected)
        XCTAssertEqual(decision.rejected.count, 2)
        XCTAssertTrue(decision.rejected.allSatisfy { $0.reason == .noQuantizationsDeclared })
    }

    func testEmptyRuntimeListYieldsNoSelection() {
        let model = Fixtures.manifest(quants: [Fixtures.quant(name: "q4", memoryMB: 100, quality: 0.8)])
        let decision = selector.select(
            model: model, device: Fixtures.device(),
            runtimes: [], policy: SelectionPolicy()
        )
        XCTAssertNil(decision.selected)
        XCTAssertTrue(decision.rejected.isEmpty)
    }

    func testZeroUsableMemoryRejectsAllCandidates() {
        let model = Fixtures.manifest(quants: [Fixtures.quant(name: "q4", memoryMB: 100, quality: 0.8)])
        let decision = selector.select(
            model: model, device: Fixtures.device(usableMB: 0),
            runtimes: [Fixtures.runtime(id: "aaa")], policy: SelectionPolicy()
        )
        XCTAssertNil(decision.selected)
        XCTAssertEqual(decision.rejected.count, 1)
        if case .insufficientMemory = decision.rejected[0].reason {} else {
            XCTFail("expected insufficientMemory, got \(decision.rejected[0].reason)")
        }
    }

    // MARK: Runtime-level gates

    func testUnsupportedFormatIsRejected() {
        let model = Fixtures.manifest(format: .mlxSafetensors,
                                      quants: [Fixtures.quant(name: "q4", memoryMB: 100, quality: 0.8)])
        let decision = selector.select(
            model: model, device: Fixtures.device(),
            runtimes: [Fixtures.runtime(id: "coreml", formats: [.coreMLPackage])],
            policy: SelectionPolicy()
        )
        XCTAssertNil(decision.selected)
        XCTAssertEqual(decision.rejected, [
            RejectedCandidate(runtimeID: "coreml", quantizationName: nil,
                              reason: .unsupportedFormat(.mlxSafetensors))
        ])
    }

    func testOSGate() {
        let model = Fixtures.manifest(quants: [Fixtures.quant(name: "q4", memoryMB: 100, quality: 0.8)])
        let decision = selector.select(
            model: model, device: Fixtures.device(os: 17),
            runtimes: [Fixtures.runtime(id: "modern", minOS: 26)],
            policy: SelectionPolicy()
        )
        XCTAssertEqual(decision.rejected.first?.reason, .osTooOld(required: 26, actual: 17))
    }

    func testNeuralEngineGate() {
        let model = Fixtures.manifest(quants: [Fixtures.quant(name: "q4", memoryMB: 100, quality: 0.8)])
        let decision = selector.select(
            model: model, device: Fixtures.device(ane: false),
            runtimes: [Fixtures.runtime(id: "ane-only", requiresANE: true)],
            policy: SelectionPolicy()
        )
        XCTAssertEqual(decision.rejected.first?.reason, .neuralEngineUnavailable)
    }

    func testStreamingGate() {
        let model = Fixtures.manifest(quants: [Fixtures.quant(name: "q4", memoryMB: 100, quality: 0.8)])
        let decision = selector.select(
            model: model, device: Fixtures.device(),
            runtimes: [Fixtures.runtime(id: "batchy", streaming: false)],
            policy: SelectionPolicy(requireStreaming: true)
        )
        XCTAssertEqual(decision.rejected.first?.reason, .streamingUnsupported)
    }

    // MARK: Candidate-level gates

    func testInsufficientMemoryCarriesExactNumbers() {
        // usable 4096 MB, headroom 0.2 → allowed = floor(4096 MB × 0.8).
        // quant 3300 MB × overhead 1.25 → projected = ceil(4125 MB) > allowed.
        let quant = Fixtures.quant(name: "q8", memoryMB: 3300, quality: 0.95)
        let model = Fixtures.manifest(quants: [quant])
        let runtime = Fixtures.runtime(id: "aaa", overhead: 1.25)
        let device = Fixtures.device(usableMB: 4096)
        let decision = selector.select(
            model: model, device: device, runtimes: [runtime],
            policy: SelectionPolicy(requiredMemoryHeadroomFraction: 0.2)
        )
        let expectedAllowed = Int64((Double(4096 * MB) * 0.8).rounded(.down))
        let expectedProjected = Int64((Double(3300 * MB) * 1.25).rounded(.up))
        XCTAssertNil(decision.selected)
        XCTAssertEqual(decision.rejected, [
            RejectedCandidate(runtimeID: "aaa", quantizationName: "q8",
                              reason: .insufficientMemory(projectedBytes: expectedProjected,
                                                          allowedBytes: expectedAllowed))
        ])
    }

    func testInsufficientDiskIsRejected() {
        let quant = Fixtures.quant(name: "q4", memoryMB: 100, diskMB: 5000, quality: 0.8)
        let model = Fixtures.manifest(quants: [quant])
        let decision = selector.select(
            model: model, device: Fixtures.device(freeDiskMB: 1000),
            runtimes: [Fixtures.runtime(id: "aaa")], policy: SelectionPolicy()
        )
        XCTAssertEqual(decision.rejected.first?.reason,
                       .insufficientDisk(requiredBytes: 5000 * MB, freeBytes: 1000 * MB))
    }

    // MARK: Objectives

    func testQualityObjectivePrefersHigherQuality() {
        let q4 = Fixtures.quant(name: "q4", memoryMB: 500, quality: 0.7)
        let q8 = Fixtures.quant(name: "q8", memoryMB: 1000, quality: 0.95)
        let model = Fixtures.manifest(quants: [q4, q8])
        let decision = selector.select(
            model: model, device: Fixtures.device(),
            runtimes: [Fixtures.runtime(id: "aaa")],
            policy: SelectionPolicy(objective: .maximizeQuality)
        )
        XCTAssertEqual(decision.selected?.quantization.name, "q8")
    }

    func testThroughputObjectivePrefersFasterRuntime() {
        let quant = Fixtures.quant(name: "q4", memoryMB: 500, quality: 0.8)
        let model = Fixtures.manifest(quants: [quant])
        let slow = Fixtures.runtime(id: "slow", throughput: 0.3)
        let fast = Fixtures.runtime(id: "fast", throughput: 0.9)
        let decision = selector.select(
            model: model, device: Fixtures.device(),
            runtimes: [slow, fast],
            policy: SelectionPolicy(objective: .maximizeThroughput)
        )
        XCTAssertEqual(decision.selected?.runtime.id, "fast")
    }

    func testHeadroomObjectivePrefersSmallestFootprint() {
        let q4 = Fixtures.quant(name: "q4", memoryMB: 500, quality: 0.7)
        let q8 = Fixtures.quant(name: "q8", memoryMB: 1000, quality: 0.95)
        let model = Fixtures.manifest(quants: [q4, q8])
        let decision = selector.select(
            model: model, device: Fixtures.device(),
            runtimes: [Fixtures.runtime(id: "aaa")],
            policy: SelectionPolicy(objective: .maximizeMemoryHeadroom)
        )
        XCTAssertEqual(decision.selected?.quantization.name, "q4")
    }

    // MARK: Thermal behavior

    func testThermalRefusalWhenDegradationDisabled() {
        let model = Fixtures.manifest(quants: [Fixtures.quant(name: "q4", memoryMB: 100, quality: 0.8)])
        let decision = selector.select(
            model: model, device: Fixtures.device(thermal: .serious),
            runtimes: [Fixtures.runtime(id: "aaa"), Fixtures.runtime(id: "bbb")],
            policy: SelectionPolicy(maxThermalState: .fair, degradeUnderThermalPressure: false)
        )
        XCTAssertNil(decision.selected)
        XCTAssertEqual(decision.rejected.count, 2)
        XCTAssertTrue(decision.rejected.allSatisfy {
            $0.reason == .thermalStateExceeded(deviceState: .serious, maximumAllowed: .fair)
        })
    }

    func testThermalDegradationFlipsObjectiveToHeadroom() {
        let q4 = Fixtures.quant(name: "q4", memoryMB: 500, quality: 0.7)
        let q8 = Fixtures.quant(name: "q8", memoryMB: 1000, quality: 0.95)
        let model = Fixtures.manifest(quants: [q4, q8])
        let decision = selector.select(
            model: model, device: Fixtures.device(thermal: .serious),
            runtimes: [Fixtures.runtime(id: "aaa")],
            policy: SelectionPolicy(objective: .maximizeQuality,
                                    maxThermalState: .fair,
                                    degradeUnderThermalPressure: true)
        )
        XCTAssertEqual(decision.effectiveObjective, .maximizeMemoryHeadroom)
        XCTAssertEqual(decision.selected?.quantization.name, "q4")
    }

    // MARK: Joint selection and determinism

    func testRuntimeAndQuantizationAreChosenJointly() {
        // "heavy" is the faster runtime, but its overhead factor prices every
        // quantization out of the budget, and q8 is too big for either
        // runtime (allowed = 4096 MB × 0.8 = 3276.8 MB). A runtime-first
        // strategy chasing throughput would commit to "heavy" and fail; the
        // joint search must land on "light" + q4.
        let q8 = Fixtures.quant(name: "q8", memoryMB: 3400, quality: 0.95)
        let q4 = Fixtures.quant(name: "q4", memoryMB: 1800, quality: 0.8)
        let model = Fixtures.manifest(quants: [q8, q4])
        let heavy = Fixtures.runtime(id: "heavy", overhead: 2.0, throughput: 0.9)
        let light = Fixtures.runtime(id: "light", overhead: 1.0, throughput: 0.4)
        let decision = selector.select(
            model: model, device: Fixtures.device(usableMB: 4096),
            runtimes: [heavy, light],
            policy: SelectionPolicy(objective: .maximizeThroughput,
                                    requiredMemoryHeadroomFraction: 0.2)
        )
        XCTAssertEqual(decision.selected?.runtime.id, "light")
        XCTAssertEqual(decision.selected?.quantization.name, "q4")
        // Both of heavy's quantizations must appear in the decision log.
        let heavyRejections = decision.rejected.filter { $0.runtimeID == "heavy" }
        XCTAssertEqual(heavyRejections.count, 2)
    }

    func testTieBreakIsDeterministicByRuntimeID() {
        let quant = Fixtures.quant(name: "q4", memoryMB: 500, quality: 0.8)
        let model = Fixtures.manifest(quants: [quant])
        let a = Fixtures.runtime(id: "aaa", throughput: 0.5)
        let b = Fixtures.runtime(id: "bbb", throughput: 0.5)
        let policy = SelectionPolicy()
        // Same scores on every axis; runtime id ascending must win — and the
        // result must not depend on input order.
        let first = selector.select(model: model, device: Fixtures.device(),
                                    runtimes: [a, b], policy: policy)
        let second = selector.select(model: model, device: Fixtures.device(),
                                     runtimes: [b, a], policy: policy)
        XCTAssertEqual(first.selected?.runtime.id, "aaa")
        XCTAssertEqual(second.selected?.runtime.id, "aaa")
    }

    func testEligibleButOutrankedIsNotListedAsRejected() {
        let q4 = Fixtures.quant(name: "q4", memoryMB: 500, quality: 0.7)
        let q8 = Fixtures.quant(name: "q8", memoryMB: 1000, quality: 0.95)
        let model = Fixtures.manifest(quants: [q4, q8])
        let decision = selector.select(
            model: model, device: Fixtures.device(),
            runtimes: [Fixtures.runtime(id: "aaa")],
            policy: SelectionPolicy(objective: .maximizeQuality)
        )
        XCTAssertEqual(decision.selected?.quantization.name, "q8")
        XCTAssertTrue(decision.rejected.isEmpty,
                      "q4 lost the ranking but was never hard-disqualified")
    }
}
