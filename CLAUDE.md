# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Swift Package that runs LLMs on Apple's Neural Engine (ANE) — ANE-first, battery-friendly, no server. Targets iOS 18+ / macOS 15+. The key constraint throughout is **ANE placement**: operations that fall back to GPU or CPU are bugs or regressions, not just inefficiencies.

## Build & run

```bash
# Build the library and all executables
swift build

# Run tests (macOS, no device required)
swift test

# Run a single test class
swift test --filter ChunkedEngineKVParityTests

# Open the iOS chat sample in Xcode
open Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj

# Smoke test CLI (macOS, requires downloaded model)
swift run coreml-llm-smoke

# Determinism oracle (quality gate — runs a fixed corpus at argmax, compares to committed oracle)
swift run determinism-oracle

# Accept-rate benchmark
swift run accept-rate-bench
```

**Conversion pipeline** (Python, separate venv):
```bash
cd conversion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python build_gemma4_bundle.py --model gemma4-e2b --ctx 2048
python build_gemma4_3way.py --model gemma4-e2b --ctx 2048   # 3-chunk decode (default since v1.7)
python convert.py --model qwen2.5-0.5b --output ./output/qwen2.5-0.5b
```

**Environment variable gates** (set in Xcode scheme → Environment Variables, or shell):
- `LLM_3CHUNK=1` — opt into 3-chunk decode (+8.2 % tok/s, bit-equivalent to 4-chunk)
- `LLM_EAGLE3_ENABLE=1` — EAGLE-3 speculative drafter
- `SPECULATIVE_PROFILE=1` — emit per-step acceptance-rate metrics

## Architecture

### Inference engines

**`ChunkedEngine.swift`** — the primary engine for Gemma 4 and Qwen3.5. Sliding-window-attention (SWA) decode with a ring-buffer KV cache.
- Prefill: batched calls at T ∈ [32, 512] tokens per ANE forward pass
- Decode: 1 token/step, 3-chunk (`c1+c2_3way+c3_3way`) or 4-chunk (`c1+c2+c3+c4`) ANE dispatch
- KV cache is explicit tensor I/O (not `MLState`) — required for ANE placement; `MLState` breaks it for Gemma 4 due to int64 state indices

**`Gemma4StatefulEngine.swift`** — engine for Qwen3-VL 2B stateful. Uses `MLState` (safe here because Qwen3-VL avoids int64 indices). Supports cross-turn KV reuse: LCP-matched resume makes 2nd-turn TTFT ~32× faster (4 s → 125 ms). Auto-detects 1/3/4-chunk modes.

**Speculative decoding** (all opt-in, env-gated, in `Sources/CoreMLLLM/`):
- `SpeculativeLoop.swift` / `CrossVocabSpeculativeEngine.swift` — EAGLE-3 drafter
- `MtpSpeculativeEngine.swift` — multi-token prediction (MTP, 4-layer shallow)
- `SuffixSpeculativeEngine.swift` / `SuffixTree.swift` — prompt-lookup n-gram
- `LookaheadEngine.swift` — Jacobi warm-start linear drafter
- `DrafterUnion.swift` — router selecting the best drafter per prompt

### ANE optimization patterns

These patterns appear throughout the conversion scripts and are load-bearing for ANE placement:
- **Conv2d-Linear**: `nn.Linear` → `nn.Conv2d(kernel_size=1)` (~3× faster on ANE than matmul)
- **ANERMSNorm**: `cat([x, -x])` → LayerNorm → slice (ANE has optimized LayerNorm; bare RMSNorm is slow)
- **In-graph argmax**: keeps 256K logits on ANE, avoids CPU round-trip
- **Pre-computed RoPE**: cos/sin as model inputs looked up in Swift (eliminates `gather`/int ops → CPU)
- **Explicit KV I/O**: plain tensor inputs/outputs, no `MLState` for Gemma 4 (avoids int64 state indices)

### Public API (`CoreMLLLM.swift`)

```swift
let llm = try await CoreMLLLM.load(model: .gemma4e2b) { progress in ... }
let text = try await llm.generate("prompt")
for await tok in try await llm.stream("prompt") { ... }
// Multimodal (Gemma 4 only)
let caption = try await llm.generate("Describe this", image: cgImage)
let analysis = try await llm.generate("Describe video", videoURL: url, videoOptions: .init(fps: 1.0, maxFrames: 6))
```

Specialist models have separate narrow APIs: `FunctionGemma` (`Gemma3FunctionGemma.swift`) and `EmbeddingGemma` (`Gemma3EmbeddingGemma.swift`).

### Multimodal pipelines

- **Vision** (`ImageProcessor.swift`): aspect-ratio-preserving resize → 16×16 patch extraction → `(1, 2520, 768)` `MLMultiArray`
- **Video** (`VideoProcessor.swift`): per-frame vision encoder + variable FPS extraction
- **Audio** (`AudioProcessor.swift`): Mel-spectrogram (200 frames × 128 bins) + 12-layer Conformer encoder

Multimodal encoders are lazy-loaded (~1 GB). The "Download Options → Include multimodal" toggle in the picker skips them for text-only installs.

### Download (`ModelDownloader.swift`)

`URLSessionConfiguration.background` with 4 concurrent connections. `finishDownload` hardlinks shared decode↔prefill weights (e.g. `chunk1↔prefill_chunk1`) instead of copying to save ~682 MB on disk.

## Key docs

| Topic | File |
|---|---|
| Decision log (rejected: WFA, Flash, W8A8, Medusa, SDPA fusion, KV alias; accepted: EAGLE-3, 3-chunk) | `docs/EXPERIMENTS.md` |
| Adding a new model architecture | `docs/ADDING_MODELS.md` |
| INT4/INT8/W8A8 rationale, ANE tricks, format gotchas | `docs/CONVERSION.md` |
| Benchmark methodology | `docs/BENCHMARKING.md` |
| 3-chunk decode analysis | `docs/THREE_CHUNK_MAC_BENCH.md` |
| `.mlpackage` vs `.mlmodelc` | `docs/DEPLOYMENT.md` |

## Testing conventions

Tests live in `Tests/CoreMLLLMTests/`. All four test files exercise correctness, not performance:
- `ChunkedEngineKVParityTests` — verify-commit KV state matches serial decode (the "11c protocol")
- `CrossVocabDraftTests` / `CrossVocabMapTests` — cross-vocab drafter parity
- `SuffixTreeTests` — n-gram trie correctness

When adding a new decode path, add a KV-parity test using the `_testChunkedEngine` internal accessor before any iPhone benchmarking.

## Dependency

Single SPM dependency: `huggingface/swift-transformers` (version range `1.0.0..<1.1.0`, pinned to avoid deadlock with consumers that also pull `mlx-swift-examples`). Provides the `Tokenizers` product used for `AutoTokenizer`.
