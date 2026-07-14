import Foundation

// MARK: - Keys and handles

/// Identity of one loadable unit: a manifest in a specific quantization on a
/// specific runtime. Two quantizations of the same model are two entries —
/// they have different working sets and must be budgeted separately.
public struct ModelKey: Sendable, Hashable {
    public let manifestID: String
    public let quantizationName: String
    public let runtimeID: String

    public init(manifestID: String, quantizationName: String, runtimeID: String) {
        self.manifestID = manifestID
        self.quantizationName = quantizationName
        self.runtimeID = runtimeID
    }
}

/// A resident, usable model. Holding one of these does **not** pin it —
/// pinning is the loader's job via acquire/release, so sessions must treat a
/// `LoadedModel` obtained without `acquire` as potentially evictable.
public struct LoadedModel: Sendable, Hashable {
    public let key: ModelKey
    public let instance: BackendModelInstance
    public let manifest: ModelManifest
    public let quantization: QuantizationOption
}

public enum LoaderError: Error, Sendable, Hashable {
    /// The model cannot be made resident: even after evicting every unpinned
    /// entry, `requiredBytes` would exceed `budgetBytes`. `reclaimableBytes`
    /// says how much eviction could have freed — 0 means everything resident
    /// is pinned by live sessions, which is an app-logic problem (too many
    /// concurrent sessions), not a loader problem.
    case budgetExceeded(requiredBytes: Int64, budgetBytes: Int64, reclaimableBytes: Int64)
}

// MARK: - Loader

/// Owns which models are resident. One instance per process.
///
/// Guarantees, stated the way they would be defended in review:
///
/// * **Single-flight:** N concurrent `acquire` calls for the same key perform
///   exactly one backend load; all callers share the outcome, success or
///   failure. A failed load is forgotten immediately — the *next* acquire
///   after a failure starts a fresh attempt (no negative caching, because on
///   device the failure cause is usually transient memory pressure).
/// * **Pin counting:** sessions `acquire` (pin) and `release` (unpin).
///   Eviction only ever touches unpinned entries. This is what makes a
///   memory-pressure trim safe to run at any moment.
/// * **LRU by use, byte-cost budgeted:** eviction is least-recently-used
///   ordered and frees whole models until the budget fits. Cost is the
///   quantization's projected resident bytes — evicting by *count* would let
///   two 3B models starve five 0.5B ones.
/// * **No half-registered state:** if capacity cannot be found for a freshly
///   loaded instance, the instance is unloaded from the backend before the
///   error is thrown. The dictionary never holds a model the budget math
///   does not account for.
public actor ModelLoader {
    private struct Entry {
        var model: LoadedModel
        var backend: any InferenceBackend
        var memoryBytes: Int64
        var lastUsed: UInt64
        var pinCount: Int
    }

    private struct InFlight {
        let token: UUID
        let task: Task<LoadedModel, Error>
    }

    private let budgetBytes: Int64
    private var entries: [ModelKey: Entry] = [:]
    private var inFlight: [ModelKey: InFlight] = [:]
    private var useCounter: UInt64 = 0

    public init(budgetBytes: Int64) {
        self.budgetBytes = max(0, budgetBytes)
    }

    // MARK: Introspection

    public var loadedBytes: Int64 {
        entries.values.reduce(0) { $0 + $1.memoryBytes }
    }

    public var loadedCount: Int {
        entries.count
    }

    public func isResident(_ key: ModelKey) -> Bool {
        entries[key] != nil
    }

    public func pinCount(for key: ModelKey) -> Int {
        entries[key]?.pinCount ?? 0
    }

    // MARK: Acquire / release

    /// Returns a pinned, resident model — loading it if necessary, joining an
    /// in-flight load if one exists. Every successful `acquire` must be
    /// balanced by exactly one `release`.
    public func acquire(
        manifest: ModelManifest,
        quantization: QuantizationOption,
        backend: any InferenceBackend
    ) async throws -> LoadedModel {
        let key = ModelKey(
            manifestID: manifest.id,
            quantizationName: quantization.name,
            runtimeID: backend.descriptor.id
        )

        // A bounded number of retries covers the (rare, but real) race where
        // a freshly registered, still-unpinned entry is evicted by a
        // concurrent load between our await resuming and us pinning it.
        for _ in 0..<3 {
            // Fast path: already resident — pin and go.
            if var entry = entries[key] {
                entry.pinCount += 1
                entry.lastUsed = nextUse()
                entries[key] = entry
                return entry.model
            }

            // Join an in-flight load, or start one.
            let flight: InFlight
            if let existing = inFlight[key] {
                flight = existing
            } else {
                let token = UUID()
                let task = Task { [weak self] () throws -> LoadedModel in
                    guard let self else { throw CancellationError() }
                    return try await self.performLoad(
                        key: key, manifest: manifest,
                        quantization: quantization, backend: backend,
                        token: token
                    )
                }
                flight = InFlight(token: token, task: task)
                inFlight[key] = flight
            }

            _ = try await flight.task.value
            // Loop re-checks `entries[key]`: the registration performed by
            // the flight is pinned on the next iteration's fast path.
        }
        throw LoaderError.budgetExceeded(
            requiredBytes: projectedBytes(for: quantization, backend: backend),
            budgetBytes: budgetBytes,
            reclaimableBytes: unpinnedBytes()
        )
    }

    /// Unpins a previously acquired model. Releasing a handle that is no
    /// longer resident (or over-releasing) is a documented no-op rather than
    /// a crash: the loader is the last place an unattended process should be
    /// able to trap.
    public func release(_ model: LoadedModel) {
        guard var entry = entries[model.key] else { return }
        entry.pinCount = max(0, entry.pinCount - 1)
        entries[model.key] = entry
    }

    // MARK: Trimming

    /// Evicts unpinned models (LRU first) until resident bytes are at or
    /// below `fraction` of the budget. Returns bytes freed. `fraction` is
    /// clamped to `0...1`; 0 means "evict every unpinned model".
    @discardableResult
    public func trim(toFraction fraction: Double) async -> Int64 {
        let clamped = min(max(fraction, 0), 1)
        let target = Int64((Double(budgetBytes) * clamped).rounded(.down))
        var victims: [Entry] = []
        while loadedBytes > target {
            guard let victimKey = lruUnpinnedKey() else { break }
            if let entry = entries.removeValue(forKey: victimKey) {
                victims.append(entry)
            }
        }
        // State is already consistent; backend unloads happen after the
        // mutation so actor reentrancy during `await` cannot double-evict.
        var freed: Int64 = 0
        for victim in victims {
            freed += victim.memoryBytes
            await victim.backend.unloadModel(victim.model.instance)
        }
        return freed
    }

    // MARK: Internals

    private func performLoad(
        key: ModelKey,
        manifest: ModelManifest,
        quantization: QuantizationOption,
        backend: any InferenceBackend,
        token: UUID
    ) async throws -> LoadedModel {
        defer { clearInFlight(key: key, token: token) }

        let needed = projectedBytes(for: quantization, backend: backend)

        // Refuse before doing expensive work if the model can never fit.
        guard needed <= budgetBytes else {
            throw LoaderError.budgetExceeded(
                requiredBytes: needed,
                budgetBytes: budgetBytes,
                reclaimableBytes: unpinnedBytes()
            )
        }

        let instance = try await backend.loadModel(manifest: manifest, quantization: quantization)

        // Make room. If that fails, unload the fresh instance before
        // throwing — never leak a loaded-but-untracked model.
        do {
            try await ensureCapacity(for: needed)
        } catch {
            await backend.unloadModel(instance)
            throw error
        }

        let model = LoadedModel(
            key: key, instance: instance,
            manifest: manifest, quantization: quantization
        )
        entries[key] = Entry(
            model: model, backend: backend,
            memoryBytes: needed, lastUsed: nextUse(), pinCount: 0
        )
        return model
    }

    private func ensureCapacity(for needed: Int64) async throws {
        var victims: [Entry] = []
        while loadedBytes + needed > budgetBytes {
            guard let victimKey = lruUnpinnedKey() else {
                // Roll back nothing — victims already removed stay removed
                // (they were evictable anyway); report what remains.
                for victim in victims {
                    await victim.backend.unloadModel(victim.model.instance)
                }
                throw LoaderError.budgetExceeded(
                    requiredBytes: needed,
                    budgetBytes: budgetBytes,
                    reclaimableBytes: unpinnedBytes()
                )
            }
            if let entry = entries.removeValue(forKey: victimKey) {
                victims.append(entry)
            }
        }
        for victim in victims {
            await victim.backend.unloadModel(victim.model.instance)
        }
    }

    private func lruUnpinnedKey() -> ModelKey? {
        entries
            .filter { $0.value.pinCount == 0 }
            .min { $0.value.lastUsed < $1.value.lastUsed }?
            .key
    }

    private func unpinnedBytes() -> Int64 {
        entries.values
            .filter { $0.pinCount == 0 }
            .reduce(0) { $0 + $1.memoryBytes }
    }

    private func projectedBytes(for quantization: QuantizationOption, backend: any InferenceBackend) -> Int64 {
        Int64((Double(quantization.estimatedMemoryBytes) * backend.descriptor.memoryOverheadFactor).rounded(.up))
    }

    private func nextUse() -> UInt64 {
        useCounter += 1
        return useCounter
    }

    private func clearInFlight(key: ModelKey, token: UUID) {
        // Identity check: only the flight that owns this token may clear the
        // slot. Without it, a slow flight finishing late could clear a
        // *newer* flight for the same key, breaking single-flight.
        if inFlight[key]?.token == token {
            inFlight[key] = nil
        }
    }
}
