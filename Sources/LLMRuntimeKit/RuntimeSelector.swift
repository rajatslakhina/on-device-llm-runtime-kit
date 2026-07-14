import Foundation

// MARK: - Selection policy

/// What the app is optimizing for when several (runtime × quantization)
/// candidates are viable.
public struct SelectionPolicy: Sendable, Hashable {
    public enum Objective: String, Sendable, Hashable, CaseIterable {
        /// Best answer quality that fits (default).
        case maximizeQuality
        /// Fastest decode wins ties on the runtime axis.
        case maximizeThroughput
        /// Smallest resident footprint wins — the "we are one memory warning
        /// away from jetsam" posture.
        case maximizeMemoryHeadroom
    }

    public let objective: Objective
    /// Fraction of `usableMemoryBytes` that must remain free *after* the
    /// model is resident. Loading a model that fits with 0 headroom is how
    /// the very first real allocation after load gets the app jetsammed.
    public let requiredMemoryHeadroomFraction: Double
    /// Above this thermal state, selection either refuses or degrades
    /// (see `degradeUnderThermalPressure`).
    public let maxThermalState: ThermalState
    /// If `true`, exceeding `maxThermalState` flips the effective objective
    /// to `.maximizeMemoryHeadroom` (smallest viable footprint) instead of
    /// refusing outright. Rationale: a smaller working set means less memory
    /// traffic per decode step, which is the lever the client actually has
    /// over sustained power draw.
    public let degradeUnderThermalPressure: Bool
    /// Reject runtimes that cannot stream tokens. Non-streaming generation
    /// on device means seconds of dead air — almost always unacceptable UX.
    public let requireStreaming: Bool

    public init(
        objective: Objective = .maximizeQuality,
        requiredMemoryHeadroomFraction: Double = 0.2,
        maxThermalState: ThermalState = .fair,
        degradeUnderThermalPressure: Bool = true,
        requireStreaming: Bool = true
    ) {
        self.objective = objective
        self.requiredMemoryHeadroomFraction = min(max(requiredMemoryHeadroomFraction, 0), 0.95)
        self.maxThermalState = maxThermalState
        self.degradeUnderThermalPressure = degradeUnderThermalPressure
        self.requireStreaming = requireStreaming
    }
}

// MARK: - Decision output

/// Why a runtime (or a specific runtime × quantization pair) was rejected.
/// Every rejection carries the numbers behind it — a selection decision that
/// cannot be explained in a bug report is a selection decision that cannot
/// be defended in review.
public enum RejectionReason: Sendable, Hashable, CustomStringConvertible {
    case noQuantizationsDeclared
    case unsupportedFormat(ModelFormat)
    case osTooOld(required: Int, actual: Int)
    case neuralEngineUnavailable
    case streamingUnsupported
    case insufficientMemory(projectedBytes: Int64, allowedBytes: Int64)
    case insufficientDisk(requiredBytes: Int64, freeBytes: Int64)
    case thermalStateExceeded(deviceState: ThermalState, maximumAllowed: ThermalState)

    public var description: String {
        switch self {
        case .noQuantizationsDeclared:
            return "manifest declares no quantizations"
        case .unsupportedFormat(let format):
            return "runtime does not execute \(format.rawValue)"
        case .osTooOld(let required, let actual):
            return "requires OS \(required)+, device is on \(actual)"
        case .neuralEngineUnavailable:
            return "requires the Neural Engine, which this device lacks"
        case .streamingUnsupported:
            return "cannot stream tokens and policy requires streaming"
        case .insufficientMemory(let projected, let allowed):
            return "needs \(projected) B resident, budget allows \(allowed) B"
        case .insufficientDisk(let required, let free):
            return "needs \(required) B on disk, only \(free) B free"
        case .thermalStateExceeded(let device, let maximum):
            return "device thermal state \(device) exceeds allowed \(maximum)"
        }
    }
}

public struct RejectedCandidate: Sendable, Hashable {
    public let runtimeID: String
    /// `nil` means the runtime was rejected wholesale, before any
    /// quantization was considered.
    public let quantizationName: String?
    public let reason: RejectionReason

    public init(runtimeID: String, quantizationName: String?, reason: RejectionReason) {
        self.runtimeID = runtimeID
        self.quantizationName = quantizationName
        self.reason = reason
    }
}

public struct SelectedCandidate: Sendable, Hashable {
    public let runtime: RuntimeDescriptor
    public let quantization: QuantizationOption
    /// Quantization memory × runtime overhead factor, rounded up.
    public let projectedMemoryBytes: Int64
    /// Fraction of usable memory still free after load (`0...1`).
    public let projectedHeadroomFraction: Double
}

/// The full, auditable outcome: what won, what was rejected and exactly why,
/// and which objective was actually in force (it can differ from the policy's
/// objective under thermal degradation).
public struct SelectionDecision: Sendable {
    public let selected: SelectedCandidate?
    public let rejected: [RejectedCandidate]
    public let effectiveObjective: SelectionPolicy.Objective
}

// MARK: - Selector

/// Pure decision function choosing the (runtime × quantization) pair that
/// should serve a model on a given device.
///
/// Design notes a reviewer should challenge:
///
/// * **Runtime and quantization are chosen jointly**, not runtime-first.
///   A runtime-first pass would happily commit to MLX and then discover the
///   only quantization MLX can host is one the memory budget rejects, while
///   Core ML with int8 would have fit. The joint search space here is tiny
///   (runtimes × quantizations, both single digits), so there is no
///   combinatorial excuse for getting this wrong.
/// * **Determinism is a feature.** Given equal scores, ties break by runtime
///   id then quantization name, ascending. Two runs on identical input must
///   produce identical decisions or A/B comparisons and bug reports are
///   meaningless.
/// * **Eligible-but-outranked candidates are not "rejected".** The rejected
///   list is a decision log of *hard* disqualifications only; ranking among
///   the eligible is expressed by what won.
public struct RuntimeSelector: Sendable {
    public init() {}

    public func select(
        model: ModelManifest,
        device: DeviceProfile,
        runtimes: [RuntimeDescriptor],
        policy: SelectionPolicy
    ) -> SelectionDecision {
        var rejected: [RejectedCandidate] = []

        guard !model.quantizations.isEmpty else {
            // Nothing can host a model with no declared variants; record one
            // rejection per runtime so the report stays per-runtime shaped.
            for runtime in runtimes {
                rejected.append(RejectedCandidate(
                    runtimeID: runtime.id,
                    quantizationName: nil,
                    reason: .noQuantizationsDeclared
                ))
            }
            return SelectionDecision(selected: nil, rejected: rejected, effectiveObjective: policy.objective)
        }

        // Thermal gate: refuse or degrade.
        var effectiveObjective = policy.objective
        if device.thermalState > policy.maxThermalState {
            if policy.degradeUnderThermalPressure {
                effectiveObjective = .maximizeMemoryHeadroom
            } else {
                for runtime in runtimes {
                    rejected.append(RejectedCandidate(
                        runtimeID: runtime.id,
                        quantizationName: nil,
                        reason: .thermalStateExceeded(
                            deviceState: device.thermalState,
                            maximumAllowed: policy.maxThermalState
                        )
                    ))
                }
                return SelectionDecision(selected: nil, rejected: rejected, effectiveObjective: effectiveObjective)
            }
        }

        // Memory allowance: what may be resident while preserving headroom.
        let usable = Double(device.usableMemoryBytes)
        let allowedBytes = Int64((usable * (1.0 - policy.requiredMemoryHeadroomFraction)).rounded(.down))

        var eligible: [SelectedCandidate] = []

        for runtime in runtimes {
            // Runtime-level gates, cheapest first; one reason per runtime so
            // the decision log reads like a log, not spam.
            guard runtime.supportedFormats.contains(model.format) else {
                rejected.append(RejectedCandidate(
                    runtimeID: runtime.id, quantizationName: nil,
                    reason: .unsupportedFormat(model.format)))
                continue
            }
            guard device.osMajorVersion >= runtime.minimumOSMajorVersion else {
                rejected.append(RejectedCandidate(
                    runtimeID: runtime.id, quantizationName: nil,
                    reason: .osTooOld(required: runtime.minimumOSMajorVersion, actual: device.osMajorVersion)))
                continue
            }
            guard !runtime.requiresNeuralEngine || device.hasNeuralEngine else {
                rejected.append(RejectedCandidate(
                    runtimeID: runtime.id, quantizationName: nil,
                    reason: .neuralEngineUnavailable))
                continue
            }
            guard !policy.requireStreaming || runtime.supportsStreaming else {
                rejected.append(RejectedCandidate(
                    runtimeID: runtime.id, quantizationName: nil,
                    reason: .streamingUnsupported))
                continue
            }

            // Candidate-level gates: each quantization judged on projected
            // resident memory (with runtime overhead) and disk footprint.
            for quantization in model.quantizations {
                let projected = Int64(
                    (Double(quantization.estimatedMemoryBytes) * runtime.memoryOverheadFactor).rounded(.up)
                )
                guard projected <= allowedBytes else {
                    rejected.append(RejectedCandidate(
                        runtimeID: runtime.id, quantizationName: quantization.name,
                        reason: .insufficientMemory(projectedBytes: projected, allowedBytes: allowedBytes)))
                    continue
                }
                guard quantization.estimatedDiskBytes <= device.freeDiskBytes else {
                    rejected.append(RejectedCandidate(
                        runtimeID: runtime.id, quantizationName: quantization.name,
                        reason: .insufficientDisk(
                            requiredBytes: quantization.estimatedDiskBytes,
                            freeBytes: device.freeDiskBytes)))
                    continue
                }

                let headroom: Double
                if device.usableMemoryBytes > 0 {
                    headroom = max(0, 1.0 - (Double(projected) / usable))
                } else {
                    headroom = 0
                }
                eligible.append(SelectedCandidate(
                    runtime: runtime,
                    quantization: quantization,
                    projectedMemoryBytes: projected,
                    projectedHeadroomFraction: headroom
                ))
            }
        }

        let winner = eligible.max { lhs, rhs in
            Self.isOrderedBefore(lhs, rhs, objective: effectiveObjective)
        }

        return SelectionDecision(selected: winner, rejected: rejected, effectiveObjective: effectiveObjective)
    }

    /// Strict-weak ordering: `true` when `lhs` ranks *below* `rhs`.
    /// Primary/secondary/tertiary axes depend on the objective; the final
    /// tie-break (runtime id, then quantization name, ascending) makes the
    /// whole ordering total and therefore the decision deterministic.
    private static func isOrderedBefore(
        _ lhs: SelectedCandidate,
        _ rhs: SelectedCandidate,
        objective: SelectionPolicy.Objective
    ) -> Bool {
        let lhsAxes = axes(for: lhs, objective: objective)
        let rhsAxes = axes(for: rhs, objective: objective)
        for (l, r) in zip(lhsAxes, rhsAxes) where l != r {
            return l < r
        }
        // Deterministic final tie-break. Note the inversion: for `max`,
        // "ordered before" must mean "loses", and the *lexicographically
        // smaller* id should win, so the smaller id must sort as "greater".
        if lhs.runtime.id != rhs.runtime.id {
            return lhs.runtime.id > rhs.runtime.id
        }
        return lhs.quantization.name > rhs.quantization.name
    }

    private static func axes(
        for candidate: SelectedCandidate,
        objective: SelectionPolicy.Objective
    ) -> [Double] {
        let quality = candidate.quantization.qualityScore
        let throughput = candidate.runtime.throughputScore
        let headroom = candidate.projectedHeadroomFraction
        switch objective {
        case .maximizeQuality:
            return [quality, throughput, headroom]
        case .maximizeThroughput:
            return [throughput, quality, headroom]
        case .maximizeMemoryHeadroom:
            return [headroom, quality, throughput]
        }
    }
}
