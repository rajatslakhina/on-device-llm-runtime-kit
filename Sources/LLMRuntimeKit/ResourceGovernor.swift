import Foundation

// MARK: - Signals

public enum MemoryPressure: Int, Sendable, Hashable, Comparable, CaseIterable {
    case warning = 0
    case critical = 1

    public static func < (lhs: MemoryPressure, rhs: MemoryPressure) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Platform-agnostic resource signals. On device these are fed from
/// `DispatchSource.makeMemoryPressureSource` and thermal-state notifications;
/// under test they are injected directly — which is the point of defining
/// them here instead of coupling the governor to platform APIs.
public enum ResourceSignal: Sendable, Hashable {
    case memoryPressure(MemoryPressure)
    case thermal(ThermalState)
}

// MARK: - Policy and actions

public struct GovernorPolicy: Sendable, Hashable {
    /// KV budget fraction to trim to on a memory *warning*.
    public let warningKVTrimFraction: Double
    /// KV budget fraction to trim to on *critical* memory pressure.
    public let criticalKVTrimFraction: Double
    /// Whether critical memory pressure also evicts idle (unpinned) models.
    public let criticalUnloadsIdleModels: Bool
    /// KV budget fraction to trim to at critical thermal state. Models are
    /// deliberately *not* unloaded for thermal reasons alone: reloading a
    /// model later costs more energy than keeping it resident, so shrinking
    /// active context (less memory traffic per decode step) is the honest
    /// thermal lever, and jetsam — the reason to unload — is a memory
    /// phenomenon, not a thermal one.
    public let thermalCriticalKVTrimFraction: Double

    public init(
        warningKVTrimFraction: Double = 0.5,
        criticalKVTrimFraction: Double = 0.0,
        criticalUnloadsIdleModels: Bool = true,
        thermalCriticalKVTrimFraction: Double = 0.5
    ) {
        self.warningKVTrimFraction = min(max(warningKVTrimFraction, 0), 1)
        self.criticalKVTrimFraction = min(max(criticalKVTrimFraction, 0), 1)
        self.criticalUnloadsIdleModels = criticalUnloadsIdleModels
        self.thermalCriticalKVTrimFraction = min(max(thermalCriticalKVTrimFraction, 0), 1)
    }
}

/// Every action the governor takes is recorded, so tests can assert on the
/// exact response to a signal and apps can surface "why did my context just
/// shrink?" to their own diagnostics.
public enum GovernorAction: Sendable, Hashable {
    case trimmedKVCache(toFraction: Double, freedBytes: Int64)
    case unloadedIdleModels(freedBytes: Int64)
    case noAction(ResourceSignal)
}

// MARK: - Governor

/// Maps resource-pressure signals to concrete recovery actions on the KV
/// store and the model loader.
///
/// Response ladder, and the reasoning behind it:
///
/// * **Memory warning** → trim KV caches. Cheapest recovery: evicted
///   sessions re-prefill later; nothing is torn down.
/// * **Memory critical** → flush KV caches *and* evict idle models. Pinned
///   models stay — evicting a model mid-generation guarantees a broken turn,
///   whereas jetsam is only a probability.
/// * **Thermal critical** → trim KV caches only (see `GovernorPolicy`).
/// * **Thermal below critical** → recorded, no action. Sub-critical thermal
///   response belongs in *selection* policy (pick smaller quantizations for
///   new work), not in tearing down running state.
public actor ResourceGovernor {
    private let kvStore: KVCacheStore
    private let loader: ModelLoader
    private let policy: GovernorPolicy
    private var actionLog: [GovernorAction] = []
    private var pumpTask: Task<Void, Never>?

    public init(kvStore: KVCacheStore, loader: ModelLoader, policy: GovernorPolicy = GovernorPolicy()) {
        self.kvStore = kvStore
        self.loader = loader
        self.policy = policy
    }

    /// Handles one signal synchronously with respect to this actor.
    /// Exposed publicly so tests (and platforms with their own delivery
    /// mechanisms) can bypass the stream pump.
    public func handle(_ signal: ResourceSignal) async {
        switch signal {
        case .memoryPressure(.warning):
            let freed = await kvStore.trim(toFraction: policy.warningKVTrimFraction)
            actionLog.append(.trimmedKVCache(toFraction: policy.warningKVTrimFraction, freedBytes: freed))

        case .memoryPressure(.critical):
            let freedKV = await kvStore.trim(toFraction: policy.criticalKVTrimFraction)
            actionLog.append(.trimmedKVCache(toFraction: policy.criticalKVTrimFraction, freedBytes: freedKV))
            if policy.criticalUnloadsIdleModels {
                let freedModels = await loader.trim(toFraction: 0)
                actionLog.append(.unloadedIdleModels(freedBytes: freedModels))
            }

        case .thermal(let state) where state >= .critical:
            let freed = await kvStore.trim(toFraction: policy.thermalCriticalKVTrimFraction)
            actionLog.append(.trimmedKVCache(toFraction: policy.thermalCriticalKVTrimFraction, freedBytes: freed))

        case .thermal:
            actionLog.append(.noAction(signal))
        }
    }

    /// Starts consuming a signal stream. Restarting replaces the previous
    /// pump. Callers own the stream's lifetime; the governor only reads.
    public func start(signals: AsyncStream<ResourceSignal>) {
        pumpTask?.cancel()
        pumpTask = Task { [weak self] in
            for await signal in signals {
                guard let self else { return }
                await self.handle(signal)
            }
        }
    }

    public func stop() {
        pumpTask?.cancel()
        pumpTask = nil
    }

    /// Returns and clears the recorded actions.
    public func drainActionLog() -> [GovernorAction] {
        let log = actionLog
        actionLog.removeAll()
        return log
    }
}
