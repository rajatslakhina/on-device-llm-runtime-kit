# LLMRuntimeKit

**The hard part of on-device LLM inference isn't calling the model. It's deciding which model, in which quantization, on which runtime, with how much memory — and surviving what the device does to you mid-generation.**

Every team shipping on-device AI in 2026 faces the same fork: MLX, Core ML, llama.cpp, or Apple's Foundation Models? The honest answer is *"it depends on the device in your user's hand right now"* — and that answer changes with thermal state, free memory, and OS version, per request. `LLMRuntimeKit` is that answer expressed as tested, deterministic code: a runtime/quantization decision engine, a pin-counted model lifecycle manager, a byte-budgeted KV-cache store, transactional streaming sessions, and a resource governor that reacts to memory and thermal pressure — everything *above* the inference call, which is exactly the layer teams keep rewriting badly.

## Why this matters

An Engineering Lead's runtime-selection call is usually made once, in a design doc, for the average device. The devices that crash are never average:

- The 4 GB iPhone where the q8 model "fits" until the first real allocation after load gets the app jetsammed.
- The KV cache that grows `kvBytesPerToken × context` *while the user watches*, until iOS kills the process instead of the feature degrading.
- The generation abandoned mid-stream that leaks phantom tokens into the cache budget until innocent sessions get evicted for someone else's crime.
- The thermal throttle that turns "our fastest runtime" into "our battery-hungriest mistake."

This kit turns each of those from a production incident into a unit-tested code path. **56 XCTest cases** cover eviction-under-pressure, retry-after-failed-load, mid-stream failure rollback, cancellation rollback, all-pinned-over-budget, zero-budget, zero-memory, and thermal-degradation scenarios — the failure modes a staff engineer gets asked about in review, exercised for real.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Your app / demo                       │
└──────────────┬───────────────────────────────┬───────────────┘
               │ pick                          │ chat
┌──────────────▼──────────────┐  ┌─────────────▼───────────────┐
│       RuntimeSelector       │  │      InferenceSession       │
│  (pure decision function)   │  │ (actor: streaming, stats,   │
│  model × device × policy →  │  │  transactional KV turns)    │
│  SelectionDecision w/ full  │  └──────┬───────────────┬──────┘
│  rejection audit trail      │         │               │
└─────────────────────────────┘  ┌──────▼─────┐  ┌──────▼──────┐
┌─────────────────────────────┐  │ ModelLoader│  │ KVCacheStore│
│      ResourceGovernor       │─▶│ (actor:    │  │ (actor: byte│
│ (actor: memory/thermal      │  │ single-    │  │ -budgeted,  │
│  signals → trim/unload,     │─▶│ flight, pin│  │ LRU + window│
│  auditable action log)      │  │ count, LRU)│  │ eviction)   │
└─────────────────────────────┘  └──────┬─────┘  └─────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │ InferenceBackend  │  ← the seam
                              │ (protocol)        │
                              └───┬────┬────┬─────┘
                                MLX  CoreML llama.cpp …
```

Everything above the `InferenceBackend` seam is runtime-agnostic, platform-agnostic (builds and tests on Linux), and deterministic under test. Real MLX / Core ML / llama.cpp adapters implement one protocol and slot in unchanged. A `SimulatedInferenceBackend` ships in the library — clearly documented as a simulation — so the full pipeline runs on a Simulator with zero model weights, and so every failure mode above the seam is deterministically injectable in tests.

## Design decisions (and rejected alternatives)

**Runtime and quantization are chosen jointly, not runtime-first.**
*Rejected: pick the "best" runtime, then find a quantization that fits.* A runtime-first pass happily commits to MLX for its throughput, then discovers the only quantization MLX can host is one the memory budget rejects — while Core ML with int8 would have fit. The joint search space is single-digits × single-digits; there is no combinatorial excuse for the wrong answer. A test (`testRuntimeAndQuantizationAreChosenJointly`) pins this behavior.

**Every rejection carries the numbers behind it.**
*Rejected: return the winner, log the rest.* A selection decision you can't explain in a bug report ("why did this user get q4 on llama.cpp?") is a decision you can't defend. `SelectionDecision` is an audit trail: each rejected candidate names its reason with the exact projected/allowed bytes.

**Selection is deterministic, by construction.**
*Rejected: leave equal-score ordering to dictionary iteration.* Ties break by runtime id, then quantization name. Two runs on identical input produce identical decisions, or A/B tests and crash triage are meaningless. Tested with permuted input order.

**Model eviction is pin-counted; the KV cache is not.**
*Rejected: one unified eviction domain.* Evicting a *model* mid-generation guarantees a broken turn, so sessions pin models (`acquire`/`release`) and eviction only touches unpinned entries — if everything is pinned and the budget is exceeded, the loader throws `budgetExceeded(reclaimable: 0)`, which is an app-concurrency bug surfaced honestly rather than papered over. Evicting a *KV cache* merely costs a re-prefill, so KV eviction stays cheap and pin-free: whole-session, LRU-first — never partial, because attention state with holes in it is garbage that still costs memory.

**Turns are transactional with respect to the KV cache.**
*Rejected: append-and-hope.* A turn that fails or is cancelled rolls back exactly the tokens it appended (prompt included). Without this, every mid-stream failure leaks phantom tokens into the budget and the accounting drifts until innocent sessions get evicted. Both failure and cancellation paths are tested.

**Single-flight loads with failure amnesia.**
*Rejected: negative caching of failed loads.* N concurrent requests for the same model perform one backend load and share the outcome — including shared failure. But the failure is forgotten immediately: on a phone, the usual cause is transient memory pressure, so the *next* acquire deserves a fresh attempt. In-flight identity is token-checked so a slow, stale flight can never clobber a newer one.

**Thermal pressure trims context; it never unloads models.**
*Rejected: unload everything when hot.* Reloading a model costs more energy than keeping it resident — unloading *because of heat* is self-defeating. The governor's thermal lever is KV trimming (smaller context → less memory traffic per decode step); model unloading is reserved for critical *memory* pressure, where the alternative is jetsam. Sub-critical thermal response belongs in *selection* policy (`degradeUnderThermalPressure` flips the objective to smallest-footprint for new work) — not in tearing down running state.

## What real adapters would add

The bundled backend is a simulation; this is disclosed everywhere it appears. A real `MLXBackend` maps `loadModel` to MLX weight loading, `generate` to its token loop, and reports honest `memoryOverheadFactor`/`throughputScore` numbers measured offline. The kit's value is that those adapters stay ~100 lines each, because coalescing, budgeting, rollback, and governance already live above the seam — written once, tested 56 times.

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/rajatslakhina/on-device-llm-runtime-kit.git", branch: "main")
]
// target dependencies:
.product(name: "LLMRuntimeKit", package: "on-device-llm-runtime-kit")
```

## Usage

```swift
import LLMRuntimeKit

// 1. Decide — pure function, fully auditable.
let decision = RuntimeSelector().select(
    model: manifest, device: .current(), runtimes: registry,
    policy: SelectionPolicy(objective: .maximizeQuality)
)
guard let choice = decision.selected else { /* show decision.rejected */ return }

// 2. Load — single-flight, pin-counted, LRU-budgeted.
let loader = ModelLoader(budgetBytes: 3_000_000_000)
let model = try await loader.acquire(
    manifest: manifest, quantization: choice.quantization, backend: backend
)

// 3. Chat — streaming, transactional, instrumented.
let kv = KVCacheStore(budgetBytes: 256_000_000)
let session = await InferenceSession(model: model, backend: backend, kvStore: kv)
for try await event in try await session.respond(to: "Why is the sky blue?") {
    switch event {
    case .started(let p): print("prefill \(p) tokens")
    case .token(let t): print(t, terminator: "")
    case .finished(let stats): print("\n\(stats.tokensPerSecond) tok/s")
    }
}

// 4. Govern — wire memory/thermal signals once, forget about it.
let governor = ResourceGovernor(kvStore: kv, loader: loader)
await governor.start(signals: platformSignalStream)
```

## Running the tests

```bash
swift test
```

No Xcode required — the package is platform-agnostic and the full suite runs on macOS or Linux.

## Demo app

**[Runtime Lab →](https://github.com/rajatslakhina/on-device-llm-runtime-kit-demo-app)** — a separate repo containing `Demo.xcodeproj`, which consumes this package as a **remote** Swift Package dependency (by this repo's public GitHub URL, branch `main`), exactly the way an external consumer would. It puts the whole pipeline under your thumbs: device-profile knobs feeding live selection decisions with full rejection logs, streaming chat with KV-cache gauges, and buttons that inject memory/thermal pressure to watch the governor respond.

## Verification status (honest)

- `swift build` and `swift test` were run for real on Swift 6.0.3 (Linux, aarch64): **56/56 tests passing, zero warnings, Swift 6 strict-concurrency language mode.**
- Two test-fixture bugs were caught and fixed by running the suite (a quantization sized so it accidentally fit the budget it was meant to exceed, and a cancellation test that `break`-ed out of stream iteration — which does *not* terminate an `AsyncThrowingStream`; explicit task cancellation does). The library logic needed no changes in either case.
- This library repo contains **no app target**; the runnable demo lives in the companion repo above and consumes this package by its public GitHub URL, exactly as an external consumer would.
