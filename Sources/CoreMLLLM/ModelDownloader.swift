import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Downloads and caches CoreML models with background URLSession and pause/resume support.
/// Large files are fetched as parallel HTTP Range segments (see `rangeSegmentSize`).
@Observable
public final class ModelDownloader: NSObject {
    public static let shared = ModelDownloader()

    // MARK: - Observable State

    public var isDownloading = false
    public var isPaused = false
    public var progress: Double = 0
    public var status = ""
    /// UserDefaults key for the multimodal opt-in toggle. Default true
    /// (multimodal encoders + sidecars included). Toggling this only
    /// affects models that ship vision/audio encoders (currently
    /// gemma4-e2b and gemma4-e2b-3way). Engine load() detects encoder
    /// absence at runtime and disables vision/audio features cleanly,
    /// so a text-only install just looks like a regular text decoder.
    public static let includeMultimodalKey = "gemma4DownloadMultimodal"

    public var availableModels: [ModelInfo] = ModelInfo.defaults
    public var refreshTrigger = 0
    public var downloadingModelId: String?

    // MARK: - Private

    private let fileManager = FileManager.default
    /// One background session per lane — see `laneCount`.
    private var sessions: [URLSession] = []
    /// In-process sessions used while the app is in the foreground — see
    /// `foregroundSessionCount`.
    private var foregroundSessions: [URLSession] = []
    private var currentModel: ModelInfo?
    private var destDir: URL?
    private var pendingFiles: [DownloadFile] = []
    private var totalBytesForAllFiles: Int64 = 0
    private var downloadContinuation: CheckedContinuation<URL, Error>?

    /// System completion handlers from `handleEventsForBackgroundURLSession`,
    /// keyed by session identifier (one per lane). Main queue only.
    private var backgroundCompletionHandlers: [String: () -> Void] = [:]
    private static let sessionIdentifier = "com.coreml-llm.model-download"

    /// Parallel background sessions ("lanes"). URLSession multiplexes every
    /// task to a host over ONE HTTP/2 connection per session — however many
    /// tasks, whatever `httpMaximumConnectionsPerHost` says — and HF's Xet
    /// CDN caps each connection (1–9 MB/s from Sydney on a 250 Mbps line;
    /// every request is an uncached pull from its US origin). Separate
    /// sessions get separate connections: 1 session × 8 tasks 4.7 MB/s vs
    /// 8 sessions 18.2 MB/s, measured back to back. Six, not more: on an
    /// iPhone 16e nsurlsessiond ran at most 6 lanes at once — with 8, two
    /// sat idle for 4 minutes and only started once others drained, which
    /// just stretched the tail. Lane 0 keeps the pre-lane identifier so
    /// tasks from older builds are still adopted.
    private static let laneCount = 6
    nonisolated static func laneIdentifier(_ lane: Int) -> String {
        lane == 0 ? sessionIdentifier : "\(sessionIdentifier).lane\(lane)"
    }

    /// Foreground-first, hand off once (iOS). Background sessions run in
    /// nsurlsessiond, and on device that costs a lot: a slow start of 75 s
    /// to 3 min before they reach speed (varies run to run; HF measured fast
    /// from a Mac at the same moment), and they're starved to ~0.07 MB/s
    /// while the app has its own foreground traffic. In-process sessions
    /// reached ~30 MB/s within 5 s. So while the app is in the foreground,
    /// files go to `foregroundSessions` a few at a time (refilled as each
    /// lands — no tail of idle lanes); on backgrounding, everything left
    /// moves to the background lanes for good (`handOffToBackgroundLanes`).
    private static let foregroundSessionCount = 8
    private static let foregroundTasksPerSession = 2
    #if os(iOS)
    /// Set while this download is parked on the background lanes (app in
    /// the background); cleared when `reclaimUnstartedLaneTasks` pulls
    /// work back to the foreground sessions.
    private var handedOffToBackground = false
    #endif

    /// `taskIdentifier` is only unique within one session. Foreground
    /// sessions have no identifier, so they're named by `sessionDescription`.
    private struct TaskKey: Hashable {
        let session: String
        let id: Int
        var isForeground: Bool { session.hasPrefix("fg") }
    }
    nonisolated private static func key(_ session: URLSession, _ task: URLSessionTask) -> TaskKey {
        TaskKey(session: session.sessionDescription ?? session.configuration.identifier ?? "",
                id: task.taskIdentifier)
    }

    // Parallel download state. All pending files are enqueued with the
    // sessions at once (see fillDownloadSlots).
    private var nextFileIndex = 0
    private var completedBytes: Int64 = 0
    /// Bytes each file contributes to `completedBytes`, by `pendingFiles`
    /// index. Counting goes through `count(_:bytes:)` so a file can never be
    /// counted twice: a relaunch mid-download once summed the on-disk
    /// segments at restore and again in the first dispatch walk (~3 GB
    /// counted twice) and tripped the 1.5× oversize abort.
    private var countedBytes: [Int: Int64] = [:]
    /// Highest byte count shown so far. Moving tasks between session kinds
    /// drops their partial bytes, and a restore re-walks from zero; the
    /// published progress holds still until the real count catches up
    /// instead of running backwards.
    private var shownBytesHighWater: Int64 = 0
    private var activeDownloadTasks: [TaskKey: URLSessionDownloadTask] = [:]
    private var activeTaskFileIndex: [TaskKey: Int] = [:]
    private var activeTaskBytes: [TaskKey: Int64] = [:]

    // Failed-file retry state. A task that dies with a transient error
    // (network handoff, system cancelling a background task) must put its
    // file back in the queue — `nextFileIndex` has already walked past it,
    // so dropping it would let the download "finish" without the file and
    // the model fail at load time with a missing-file error.
    private var retryFileIndices: [Int] = []
    private var fileRetryCounts: [Int: Int] = [:]  // pendingFiles index → attempts
    private var completionSweepsDone = 0
    private let maxRetriesPerFile = 3

    // A background URLSession can carry over tasks from a prior process.
    // We adopt those (or cancel orphans) once on init via getAllTasks; until
    // adoption completes we defer download/resume so we don't spawn fresh
    // tasks that would race with — and double-download — the survivors.
    private var tasksAdopted = false
    private var lanesAwaitingAdoption = 0
    private var fillDeferredToAdoption = false

    // Console timing breadcrumbs (`logProgressIfDue`).
    private var progressLogStart: Date?
    private var lastProgressLog: Date?
    private var loggedFirstBytes = false
    private var pendingAdoptionActions: [() -> Void] = []

    // MARK: - Types

    public struct ModelInfo: Identifiable, Sendable {
        public let id: String
        public let name: String
        public let size: String
        public let downloadURL: String
        public let folderName: String

        public init(id: String, name: String, size: String, downloadURL: String, folderName: String) {
            self.id = id
            self.name = name
            self.size = size
            self.downloadURL = downloadURL
            self.folderName = folderName
        }

        /// Gemma 4 E2B — multimodal (image + audio + video + text), 3.1 GB,
        /// ANE-optimized. Includes a native video vision encoder
        /// (`vision_video.mlmodelc`, 64 tokens/frame) so the Swift 2×2 pool
        /// no longer sits in the video path.
        public static let gemma4e2b = ModelInfo(
            id: "gemma4-e2b", name: "Gemma 4 E2B (4-chunk legacy)", size: "5.4 GB",
            // n1024 branch ships the N=1024 batched prefill that pairs with
            // the Swift SWA write fix (a878c44). Old clones still point at
            // `main` and keep downloading N=512, which is safe with the
            // unfixed Swift binary.
            downloadURL: "https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml/resolve/n1024",
            folderName: "gemma4-e2b")

        /// Qwen2.5 0.5B — text only, 309 MB.
        public static let qwen25_05b = ModelInfo(
            id: "qwen2.5-0.5b", name: "Qwen2.5 0.5B (Text)", size: "309 MB",
            downloadURL: "https://github.com/john-rocky/CoreML-LLM/releases/download/v0.1.0/qwen2.5-0.5b-coreml.zip",
            folderName: "qwen2.5-0.5b")

        /// Qwen3.5 0.8B — hybrid Gated-DeltaNet SSM + attention, text-only.
        /// Ships the INT8 palettized decode mlpackage (754 MB) — same
        /// semantic precision as fp16 (top-3 = 100% parity vs fp32 oracle),
        /// half the bundle size. Prefill is performed via the same model
        /// recurrently. Runs on CPU / GPU / ANE.
        public static let qwen35_08b = ModelInfo(
            id: "qwen3.5-0.8b", name: "Qwen3.5 0.8B (ANE)", size: "754 MB",
            downloadURL: "https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreML/resolve/main",
            folderName: "qwen3.5-0.8b")

        /// Qwen3.5 2B — same hybrid SSM/attention architecture as 0.8B,
        /// just hidden_size doubled (1024→2048) and intermediate
        /// (3072→6144). Shipped as 4 INT8 chunks (6 layers each, ~1.7 GB
        /// fp16-dequantized per chunk) matching the Gemma 4 E4B pattern
        /// that fits iPhone's single-mlprogram ANE compile budget.
        /// 2-chunk at 2 GB fp16/chunk failed ANE and fell to GPU; 4-chunk
        /// stays ANE-resident. Bigger = higher quality, slower vs 0.8B.
        public static let qwen35_2b = ModelInfo(
            id: "qwen3.5-2b", name: "Qwen3.5 2B (ANE)", size: "2.4 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreML/resolve/main",
            folderName: "qwen3.5-2b")

        /// Qwen3-VL 2B — text backbone of the multimodal Qwen3-VL
        /// model. 36-layer plain GQA (head_dim=128, 8 KV heads,
        /// hidden=2560), shipped as 6 INT8 body chunks (6 layers each)
        /// + a tail (final_norm + lm_head) + raw fp16 embed sidecar
        /// that Swift mmaps. Vision tower is dropped in this Phase 1
        /// release; Phase 2 will add it via a separate vision_video
        /// mlpackage + DeepStack layer-tap injection.
        public static let qwen3vl_2b = ModelInfo(
            id: "qwen3-vl-2b", name: "Qwen3-VL 2B (text + vision, ANE)", size: "4.7 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/qwen3-vl-2b-coreml/resolve/main",
            folderName: "qwen3-vl-2b")

        /// Qwen3-VL 2B stateful (Phase 1) — MLState + slice_update KV,
        /// 4-chunk INT8 + fp16 embed sidecar. iPhone 17 Pro bench
        /// 24.4 tok/s decode / 264 MB phys_footprint — 6.4× memory drop
        /// vs v1.4.0's 1.7 GB recurrent build. Text-only first ship;
        /// sideload-only under Documents/Models/qwen3-vl-2b-stateful/
        /// via scripts/qwen3vl_stateful_push.sh.
        public static let qwen3vl_2b_stateful = ModelInfo(
            id: "qwen3-vl-2b-stateful", name: "Qwen3-VL 2B (stateful, Phase 1)",
            size: "2.3 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/qwen3-vl-2b-stateful-coreml/resolve/main",
            folderName: "qwen3-vl-2b-stateful")

        /// Gemma 4 E4B — 42 layers, hidden=2560, 2 KV heads, text-only decoder.
        /// INT4 palettized, ctx=2048. Baseline ~14 tok/s on iPhone 17 Pro.
        /// A local build (`conversion/build_gemma4_bundle.py --model gemma4-e4b`)
        /// + USB sideload to `Documents/Models/gemma4-e4b/` is also supported —
        /// the app treats the folder as "downloaded" once present.
        public static let gemma4e4b = ModelInfo(
            id: "gemma4-e4b", name: "Gemma 4 E4B", size: "5.5 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/gemma-4-E4B-coreml/resolve/main",
            folderName: "gemma4-e4b")

        /// Gemma 4 E2B Fashion — MB dress/casual theory vision advisor.
        /// Local PEFT LoRA (rank=16, alpha=32) fine-tune on 598 Unsplash/Pexels
        /// outfit photos labelled by Claude Vision, merged into the E2B base
        /// and rebuilt via `build_gemma4_bundle.py --hf-dir <merged>`. Vision
        /// tower is the stock `vision_video.mlmodelc` (64 tok/frame) grafted
        /// from the production gemma4-e2b bundle — LoRA targets language_model
        /// only, so vision weights are bit-identical to the base. Sideload-only
        /// under `Documents/Models/gemma4-e2b-fashion/`; outputs a fixed JSON
        /// schema (items, overall_dress_ratio, tpo_assumption, verdict, advice).
        public static let gemma4e2bFashion = ModelInfo(
            id: "gemma4-e2b-fashion", name: "Gemma 4 E2B Fashion (MB)",
            size: "4.0 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-fashion")

        /// Gemma 4 E2B + EAGLE-3 speculative — same 4.6B E2B base but with
        /// decode chunks that emit `hidden_at_L{8,17,34}` taps plus three
        /// extra mlmodelc bundles (`eagle3_draft`, `eagle3_fusion`,
        /// `verify_chunk{1..4}`). CoreMLLLM auto-loads SpeculativeLoop when
        /// all three are present and the decode stream uses K=3 speculative
        /// bursts with T=1 fallback on any burst error. Sideload-only (no HF
        /// distribution yet); the app treats the folder as "downloaded"
        /// once present under `Documents/Models/gemma4-e2b-eagle3/`.
        public static let gemma4e2bEagle3 = ModelInfo(
            id: "gemma4-e2b-eagle3", name: "Gemma 4 E2B + EAGLE-3 (K=3)", size: "5.0 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-eagle3")

        /// Gemma 4 E2B + LookAhead K=8 probe bundle. Same base as gemma4e2b
        /// but the four chunks are multifunction (decode_q1 + verify_qK=8)
        /// and a `probe.marker` file asks LLMRunner to auto-enable
        /// `SPECULATIVE_PROFILE` so verify chunks actually load. Sideloaded
        /// to `Documents/Models/gemma4-e2b-lookahead-probe/` — keeps the
        /// production `gemma4-e2b/` bundle untouched so users can flip
        /// between the two from the model picker. See
        /// `docs/LOOKAHEAD_PROBE_RESULTS.md` for the workflow.
        public static let gemma4e2bLookaheadProbe = ModelInfo(
            id: "gemma4-e2b-lookahead-probe", name: "Gemma 4 E2B + LookAhead (K=8, probe)",
            size: "5.7 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-lookahead-probe")

        /// Gemma 4 E2B stateful — MLState + slice_update KV cache, mirrors
        /// the Qwen3-VL 2B v1.5.0 stateful pattern. Routed through
        /// `Gemma4StatefulGenerator` (Examples/CoreMLLLMChat). Lets us drop
        /// the explicit kv13/kv14 passthrough in chunk_2 → 3/4 in favor of
        /// CoreML-managed state buffers, plus enables cross-turn KV reuse
        /// for ~zero TTFT on prefix-extending prompts. Built by
        /// `conversion/build_gemma4_e2b_stateful_chunks.py`. Sideload-only
        /// to `Documents/Models/gemma4-e2b-stateful/gemma4_e2b_stateful_chunks/`.
        public static let gemma4e2bStateful = ModelInfo(
            id: "gemma4-e2b-stateful",
            name: "Gemma 4 E2B (stateful, MLState)", size: "4.0 GB",
            downloadURL: "",
            folderName: "gemma4-e2b-stateful")

        /// Gemma 4 E2B (3-chunk decode) — Stage 7 ship default.
        ///
        /// Same multimodal bundle as legacy `gemma4e2b` but the decode path
        /// is the 3-chunk variant: `chunk1` (L0-7, identical binary) +
        /// `chunk2_3way` (L8-24 merged, 17 layers, owns + KV-shared) +
        /// `chunk3_3way` (L25-34 + lm_head). Saves one ANE dispatch per
        /// decode step (~+10% tok/s vs legacy 4-chunk on iPhone). Prefill
        /// stays on the legacy 4-chunk graphs (T=1024 with vision-aware
        /// bidirectional mask) so multimodal works unchanged.
        ///
        /// Bundle sharing: `folderName` matches legacy `gemma4e2b` so users
        /// switching between the two reuse on-disk chunk1 / prefill /
        /// sidecars / encoders. Only chunk2_3way + chunk3_3way are
        /// 3way-specific. The 4-chunk decode files (chunk2/3/4) are NOT
        /// downloaded by this entry — pick legacy `gemma4e2b` if the
        /// 4-chunk fallback is needed.
        public static let gemma4e2b3way = ModelInfo(
            id: "gemma4-e2b-3way",
            name: "Gemma 4 E2B", size: "5.4 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml/resolve/main",
            folderName: "gemma4-e2b")

        /// Gemma 4 E2B stateful (3-chunk merged + Linear) — Stage 3 ship.
        /// MLState + slice_update KV, 3-chunk layout (chunk_1 L0-7 +
        /// chunk_2 merged L8-24 + chunk_3 L25-34+lm_head). Linear
        /// projections (cml9 PR #2577 native `linear` op). Mac decode
        /// 34.6 tok/s + multifunction prefill_b8 7.77×; iPhone 17 Pro
        /// decode 33.4 tok/s with T=1 prefill fallback (multifunction
        /// T>1 unsupported on iPhone ANE 18). Cross-turn KV reuse via
        /// LCP-match — multi-turn TTFT -95%. Built by
        /// `conversion/build_gemma4_e2b_stateful_3chunks.py`. Text-only
        /// (vision/audio stay on the legacy gemma4e2b multimodal bundle).
        public static let gemma4e2bStatefulLinear = ModelInfo(
            id: "gemma4-e2b-stateful-linear",
            name: "Gemma 4 E2B (stateful research, text-only)", size: "3.7 GB",
            downloadURL: "https://huggingface.co/mlboydaisuke/gemma-4-E2B-stateful-coreml/resolve/main",
            folderName: "gemma4-e2b-stateful-linear")

        /// Visible in the UI picker. EAGLE-3 / LookAhead probe variants are
        /// hidden unless `LLM_SHOW_EXPERIMENTAL=1` is set (or the
        /// UserDefaults key `showExperimentalModels` is true). Keeps the
        /// production picker clean while letting devs flip the flag for
        /// sideload testing.
        public static var defaults: [ModelInfo] {
            let experimental =
                ProcessInfo.processInfo.environment["LLM_SHOW_EXPERIMENTAL"] == "1"
                || UserDefaults.standard.bool(forKey: "showExperimentalModels")
            // Stage 7 picker order:
            //   1. gemma4e2b3way   — 3-chunk decode + 4-chunk prefill,
            //                         multimodal default (image+audio+video,
            //                         +10% decode tok/s vs legacy 4-chunk).
            //   2. gemma4e2b       — legacy 4-chunk decode, kept for back-
            //                         compat with users who downloaded the
            //                         pre-Stage-7 bundle. Same multimodal
            //                         capability, slightly slower decode.
            //   3. gemma4e2bStatefulLinear — text-only research entry
            //                         (MLState + multifunction prefill_b8).
            //                         Multimodal not supported; the prefill
            //                         graph can't span the full image span
            //                         in one batch (see
            //                         docs/SESSION_2026_04_27_STAGE6_MULTIMODAL.md).
            var list: [ModelInfo] = [
                gemma4e2b3way, gemma4e2b, gemma4e2bStatefulLinear,
                gemma4e4b, gemma4e2bFashion,
                qwen25_05b, qwen35_08b, qwen35_2b,
                qwen3vl_2b, qwen3vl_2b_stateful,
            ]
            if experimental {
                list.insert(gemma4e2bEagle3, at: 3)
                list.insert(gemma4e2bLookaheadProbe, at: 4)
                list.insert(gemma4e2bStateful, at: 5)        // Conv2d variant
            }
            return list
        }
    }

    // Internal (not private) for the unit tests, like the three helpers
    // `rangeSegmented` / `isValidRangeResponse` / `joinParts`.
    struct DownloadFile: Codable, Equatable {
        let remotePath: String
        let localPath: String
        let estimatedSize: Int64
        // Range segment of a larger file (see `rangeSegmented`). Optional so
        // states persisted before segmentation still decode. `rangeEnd` is
        // inclusive; nil = to end of file (the last segment absorbs any
        // estimate error). `joinTarget` is the whole file's localPath —
        // `finishDownload` concatenates its segments into it.
        var rangeStart: Int64? = nil
        var rangeEnd: Int64? = nil
        var joinTarget: String? = nil
    }

    private struct PersistedState: Codable {
        let modelId: String
        let totalBytes: Int64
        let files: [DownloadFile]
        // Optional so states persisted before these fields still decode.
        // Apps can download a ModelInfo that isn't in `availableModels` (or
        // shadows an entry's id with a different mirror URL — Evie does);
        // resolving the restored id against `availableModels` would silently
        // re-point gap re-fetches at the wrong host.
        var downloadURL: String?
        var folderName: String?
        var modelName: String?
    }

    // MARK: - Range segments + joins

    /// Files of at least twice this size download as parallel HTTP Range
    /// segments of this size (`<localPath>.segNN`), and `finishDownload`
    /// concatenates them. A whole file rides one connection at the CDN's
    /// per-connection cap and pins the install tail (a lone 2.35 GB embed
    /// did), and a transient failure costs one segment instead of the file.
    static let rangeSegmentSize: Int64 = 64 * 1024 * 1024

    /// Expand large files into Range segments. The count rounds the estimate
    /// down and the last segment is open-ended, so an estimate that's off by
    /// less than one segment still covers the file exactly.
    static func rangeSegmented(_ files: [DownloadFile]) -> [DownloadFile] {
        files.flatMap { file -> [DownloadFile] in
            let count = file.estimatedSize / rangeSegmentSize
            guard count >= 2 else { return [file] }
            return (0..<count).map { i in
                let start = i * rangeSegmentSize
                let isLast = i == count - 1
                return DownloadFile(
                    remotePath: file.remotePath,
                    localPath: file.localPath + String(format: ".seg%02ld", Int(i)),
                    estimatedSize: isLast ? file.estimatedSize - start : rangeSegmentSize,
                    rangeStart: start,
                    rangeEnd: isLast ? nil : start + rangeSegmentSize - 1,
                    joinTarget: file.localPath)
            }
        }
    }

    /// Legacy split-embed parts: builds before Range segments fetched the
    /// 2.35 GB per-layer embed as 4 mirror-hosted `.partN` files. Fresh
    /// downloads no longer list them, but a state persisted mid-download by
    /// such a build still does, and they must still join.
    private static let splitEmbedJoinedName = "embed_tokens_per_layer_q8.bin"

    nonisolated private static func isSplitEmbedPart(_ localPath: String) -> Bool {
        localPath.hasPrefix(splitEmbedJoinedName + ".part")
    }

    /// The whole file a piece concatenates into; nil for a plain file.
    private static func joinTarget(of file: DownloadFile) -> String? {
        if let target = file.joinTarget { return target }
        return isSplitEmbedPart(file.localPath) ? splitEmbedJoinedName : nil
    }

    /// Guards against a second `finishDownload` (e.g. a re-attached caller
    /// triggering `resumeDownload`) dispatching a concurrent join over the
    /// same temp files while one is already in flight.
    private var isJoiningParts = false

    // MARK: - Init

    override init() {
        super.init()
        cleanGraveyard()
        restorePendingDownload()
        sessions = (0..<Self.laneCount).map { lane in
            let config = URLSessionConfiguration.background(withIdentifier: Self.laneIdentifier(lane))
            config.isDiscretionary = false
            config.sessionSendsLaunchEvents = true
            config.timeoutIntervalForResource = 7200
            // HTTP/2 ignores this; it bounds an HTTP/1.1 fallback.
            config.httpMaximumConnectionsPerHost = 2
            return URLSession(configuration: config, delegate: self, delegateQueue: nil)
        }
        foregroundSessions = (0..<Self.foregroundSessionCount).map { i in
            let config = URLSessionConfiguration.ephemeral
            config.urlCache = nil
            config.requestCachePolicy = .reloadIgnoringLocalCacheData
            config.timeoutIntervalForResource = 7200
            let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
            session.sessionDescription = "fg\(i)"
            return session
        }
        #if os(iOS)
        NotificationCenter.default.addObserver(
            self, selector: #selector(appDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification, object: nil)
        NotificationCenter.default.addObserver(
            self, selector: #selector(appDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification, object: nil)
        #endif
        lanesAwaitingAdoption = sessions.count
        for session in sessions {
            session.getAllTasks { [weak self] tasks in
                DispatchQueue.main.async { self?.adoptExistingTasks(tasks, in: session) }
            }
        }
    }

    /// The app delegate's `handleEventsForBackgroundURLSession` hands its
    /// completion handler here; it runs once that session's queued events
    /// are delivered. Each lane is its own session, so a wake can deliver
    /// several — keep one handler per identifier.
    public func setBackgroundCompletionHandler(_ handler: @escaping () -> Void,
                                               forSession identifier: String) {
        backgroundCompletionHandlers[identifier] = handler
    }

    /// Claim background tasks that survived a prior app process. Tasks whose
    /// `taskDescription` matches a file in the restored `pendingFiles` are
    /// reattached to the in-memory state; everything else is an orphan
    /// (different model, stale state) and gets cancelled. Without this,
    /// `resumeDownload` would create a second task for the same file and
    /// `completedBytes` would be counted twice.
    private func adoptExistingTasks(_ tasks: [URLSessionTask], in session: URLSession) {
        var pathToIndex: [String: Int] = [:]
        for (i, f) in pendingFiles.enumerated() { pathToIndex[f.localPath] = i }
        for t in tasks {
            if let dl = t as? URLSessionDownloadTask,
               let desc = t.taskDescription,
               let idx = pathToIndex[desc],
               !activeTaskFileIndex.values.contains(idx) {
                let key = Self.key(session, t)
                activeDownloadTasks[key] = dl
                activeTaskFileIndex[key] = idx
            } else {
                t.cancel()
            }
        }
        lanesAwaitingAdoption -= 1
        guard lanesAwaitingAdoption == 0 else { return }
        tasksAdopted = true
        #if os(iOS)
        // A relaunch mid-download: survivors sit on the lanes. In the
        // background, keep new work there too; in the foreground, reclaim
        // the unstarted ones (an `.inactive` launch gets it from
        // `appDidBecomeActive`).
        if !activeDownloadTasks.isEmpty {
            switch UIApplication.shared.applicationState {
            case .background: handedOffToBackground = true
            case .active: scheduleReclaim()
            default: break
            }
        }
        #endif
        let actions = pendingAdoptionActions
        pendingAdoptionActions.removeAll()
        for a in actions { a() }
    }

    private func runAfterAdoption(_ action: @escaping () -> Void) {
        if tasksAdopted { action() } else { pendingAdoptionActions.append(action) }
    }

    // MARK: - Public

    public func isDownloaded(_ model: ModelInfo) -> Bool {
        if isDownloading && downloadingModelId == model.id { return false }
        return localModelURL(for: model) != nil
    }

    public func hasFiles(_ model: ModelInfo) -> Bool {
        fileManager.fileExists(atPath: modelsDirectory.appendingPathComponent(model.folderName).path)
    }

    public func localModelURL(for model: ModelInfo) -> URL? {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        // Qwen3.5 has its own mlpackage names (no `model.mlpackage`).
        // Check shipping variants in order. Any one of these marks the
        // folder as a Qwen3.5 model folder.
        //   1. 2B chunked (chunk_a + chunk_b must both exist)
        //   2. 0.8B INT8 (default shipping)
        //   3. 0.8B fp16 (ground-truth)
        //   4. 2B monolithic INT8 (Mac fallback; fails ANE budget on iPhone)
        let chunksDir = dir.appendingPathComponent("qwen3_5_2b_decode_chunks")
        func chunkExists(_ base: String) -> Bool {
            // HF-downloaded layout: chunk_X.mlpackage/Data/.../weight.bin
            let pkgWeights = chunksDir
                .appendingPathComponent("\(base).mlpackage")
                .appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
            if fileManager.fileExists(atPath: pkgWeights.path) { return true }
            // Sideload layout: chunk_X.mlmodelc/weights/weight.bin
            let mlcWeights = chunksDir
                .appendingPathComponent("\(base).mlmodelc")
                .appendingPathComponent("weights/weight.bin")
            return fileManager.fileExists(atPath: mlcWeights.path)
        }
        // 4-chunk + embed-bin layout: all must be present to count as
        // downloaded. Missing any piece → fall through to monolithic /
        // 0.8B layouts below. Embed is a raw .bin, not an mlpackage.
        let embedBinURL = chunksDir.appendingPathComponent("embed_weight.bin")
        let embedPresent = fileManager.fileExists(atPath: embedBinURL.path)
        let requiredChunks = ["chunk_a", "chunk_b", "chunk_c", "chunk_d"]
        if embedPresent && requiredChunks.allSatisfy(chunkExists) {
            // Return chunksDir itself (a directory under the model folder)
            // so callers that do `url.deletingLastPathComponent()` land on
            // the model folder root — same convention 0.8B follows when
            // its mlpackage is returned directly. Qwen35Generator resolves
            // the actual chunk_*.{mlpackage,mlmodelc} + embed_weight.bin
            // from the folder.
            return chunksDir
        }
        for name in ["qwen3_5_0_8b_decode_int8_mseq128.mlpackage",
                     "qwen3_5_0_8b_decode_fp16_mseq128.mlpackage",
                     "qwen3_5_2b_decode_int8_mseq128.mlpackage"] {
            let pkg = dir.appendingPathComponent(name)
            if fileManager.fileExists(atPath: pkg.appendingPathComponent(
                "Data/com.apple.CoreML/weights/weight.bin").path) {
                return pkg
            }
        }

        // Qwen3-VL 2B stateful (Phase 1): chunk_0..N + chunk_head +
        // embed_weight.bin under qwen3_vl_2b_stateful_chunks/. N ≥ 2
        // (we ship 4, but any count ≥ 2 loads fine). All sideloaded.
        let vl2bStatefulDir = dir.appendingPathComponent("qwen3_vl_2b_stateful_chunks")
        func vl2bStatefulChunkExists(_ base: String) -> Bool {
            let pkgWeights = vl2bStatefulDir
                .appendingPathComponent("\(base).mlpackage")
                .appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
            if fileManager.fileExists(atPath: pkgWeights.path) { return true }
            let mlcWeights = vl2bStatefulDir
                .appendingPathComponent("\(base).mlmodelc")
                .appendingPathComponent("weights/weight.bin")
            return fileManager.fileExists(atPath: mlcWeights.path)
        }
        let vl2bStatefulEmbed = vl2bStatefulDir.appendingPathComponent("embed_weight.bin")
        if fileManager.fileExists(atPath: vl2bStatefulEmbed.path)
            && vl2bStatefulChunkExists("chunk_0")
            && vl2bStatefulChunkExists("chunk_1")
            && vl2bStatefulChunkExists("chunk_head") {
            // Return the INNER chunks dir. ChatView strips one level
            // via .deletingLastPathComponent() before passing to its
            // local loadModel(), and LLMRunner does the same again, so
            // we need an extra layer of nesting in the URL we return.
            // Mirror the Qwen3.5 chunked + Qwen3-VL v1.4.0 convention.
            return vl2bStatefulDir
        }

        // Gemma 4 E2B stateful (MLState + slice_update): chunk_{1..4} +
        // embed_tokens_q8.bin + RoPE tables + tokenizer under
        // gemma4_e2b_stateful_chunks/. Two folder names share this
        // layout: gemma4-e2b-stateful (Conv2d) and
        // gemma4-e2b-stateful-linear (Plan 3 Linear A/B partner).
        let g4StatefulDir = dir.appendingPathComponent("gemma4_e2b_stateful_chunks")
        func g4StatefulChunkExists(_ base: String) -> Bool {
            let pkg = g4StatefulDir
                .appendingPathComponent("\(base).mlpackage")
                .appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
            if fileManager.fileExists(atPath: pkg.path) { return true }
            let mlc = g4StatefulDir
                .appendingPathComponent("\(base).mlmodelc")
                .appendingPathComponent("weights/weight.bin")
            return fileManager.fileExists(atPath: mlc.path)
        }
        let g4StatefulEmbed = g4StatefulDir
            .appendingPathComponent("embed_tokens_q8.bin")
        // Accept either the 3-chunk merged layout (chunks 1-3, ship from
        // Stage 3 e8fb25c) or the legacy 4-chunk layout (chunks 1-4) —
        // Gemma4StatefulEngine auto-detects via chunk_3 signature.
        let g4Has3Chunks = (1...3).allSatisfy {
            g4StatefulChunkExists("chunk_\($0)") }
        let g4Has4Chunks = g4Has3Chunks && g4StatefulChunkExists("chunk_4")
        if fileManager.fileExists(atPath: g4StatefulEmbed.path)
            && (g4Has3Chunks || g4Has4Chunks)
        {
            // Same nesting convention as Qwen3-VL stateful — return the
            // INNER chunks dir; LLMRunner strips one level and adds the
            // subdir back via its own resolver.
            return g4StatefulDir
        }

        // Qwen3-VL 2B: 4 body chunks + chunk_head + embed_weight.bin
        // under qwen3_vl_2b_decode_chunks/. All-or-nothing.
        let vl2bDir = dir.appendingPathComponent("qwen3_vl_2b_decode_chunks")
        func vl2bChunkExists(_ base: String) -> Bool {
            let pkgWeights = vl2bDir
                .appendingPathComponent("\(base).mlpackage")
                .appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
            if fileManager.fileExists(atPath: pkgWeights.path) { return true }
            let mlcWeights = vl2bDir
                .appendingPathComponent("\(base).mlmodelc")
                .appendingPathComponent("weights/weight.bin")
            return fileManager.fileExists(atPath: mlcWeights.path)
        }
        let vl2bEmbedURL = vl2bDir.appendingPathComponent("embed_weight.bin")
        let vl2bEmbedPresent = fileManager.fileExists(atPath: vl2bEmbedURL.path)
        let vl2bAll = (0..<4).allSatisfy { vl2bChunkExists("chunk_\($0)") }
            && vl2bChunkExists("chunk_head")
        if vl2bEmbedPresent && vl2bAll {
            return vl2bDir
        }
        let chunk1 = dir.appendingPathComponent("chunk1.mlmodelc")
        if fileManager.fileExists(atPath: chunk1.appendingPathComponent("weights/weight.bin").path) {
            // chunk1 (a big, download-first weight) is present, but that alone
            // does NOT mean the bundle is loadable. The tiny core sidecars —
            // `model_config.json` above all — download last, so an install
            // interrupted between the weights and the small-file batch leaves
            // chunk1 on disk without the config `load(from:)` needs. Reporting
            // such a folder as "present" makes the caller skip the resumable
            // repair download and loop forever on "model_config.json not
            // found". Treat a missing config as not-downloaded so the next
            // download() rebuilds the file list and re-fetches only the gap.
            guard fileManager.fileExists(
                atPath: dir.appendingPathComponent("model_config.json").path) else {
                return nil
            }
            // Prefill weights aren't downloaded — finishDownload hardlinks
            // them from the decode chunks as a post-processing step. A kill
            // between the last file landing and that step leaves prefill
            // metadata (coremldata.bin) without weights, which CoreML rejects
            // at load time ("Could not open …/prefill_chunkN/weights/
            // weight.bin"). Report such a bundle as not-downloaded so the
            // caller re-runs download(), which skips every existing file and
            // just finishes the link step. Prefill-less layouts (e.g. E4B)
            // have no prefill dirs and fall through untouched.
            for i in 1...4 {
                let prefill = dir.appendingPathComponent("prefill_chunk\(i).mlmodelc")
                if fileManager.fileExists(atPath: prefill.appendingPathComponent("coremldata.bin").path)
                    && !fileManager.fileExists(atPath: prefill.appendingPathComponent("weights/weight.bin").path) {
                    return nil
                }
            }
            if isChunkCtxMismatched(modelDir: dir, chunk1Dir: chunk1) {
                try? fileManager.removeItem(at: dir)
                return nil
            }
            return chunk1
        }
        let modelc = dir.appendingPathComponent("model.mlmodelc")
        if fileManager.fileExists(atPath: modelc.appendingPathComponent("weights/weight.bin").path) {
            return modelc
        }
        let pkg = dir.appendingPathComponent("model.mlpackage")
        if fileManager.fileExists(atPath: pkg.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin").path) {
            return pkg
        }
        return nil
    }

    /// Compare chunk1's declared `causal_mask_full` ctx (from model.mil) against
    /// model_config.json's context_length. Returns true on mismatch so callers
    /// can invalidate the cache and force a fresh download.
    private func isChunkCtxMismatched(modelDir: URL, chunk1Dir: URL) -> Bool {
        guard let configData = try? Data(contentsOf: modelDir.appendingPathComponent("model_config.json")),
              let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
              let expected = json["context_length"] as? Int else {
            return false  // no config to compare against; leave cache alone
        }
        let milURL = chunk1Dir.appendingPathComponent("model.mil")
        guard let mil = try? String(contentsOf: milURL, encoding: .utf8) else {
            return false
        }
        // Look for the causal_mask_full tensor declaration. The shape's last
        // dimension is ctx: e.g. `tensor<fp16, [1, 1, 1, 2048]> causal_mask_full`.
        let pattern = #"tensor<fp16,\s*\[\s*1,\s*1,\s*1,\s*(\d+)\s*\]>\s*causal_mask_full"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: mil, range: NSRange(mil.startIndex..., in: mil)),
              match.numberOfRanges >= 2,
              let range = Range(match.range(at: 1), in: mil),
              let actual = Int(mil[range]) else {
            return false
        }
        return actual != expected
    }

    /// Download a model, skipping files that already exist on disk.
    /// Set `repair: true` to re-check and download any missing files.
    public func download(_ model: ModelInfo, repair: Bool = false) async throws -> URL {
        if !repair, let existing = localModelURL(for: model) { return existing }

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.main.async { [weak self] in
                self?.runAfterAdoption {
                    guard let self else { return }
                    // Same model already in flight (paused-restored after a
                    // relaunch, or unparked and still running) — attach to it
                    // instead of cancel-restarting, which would throw away the
                    // daemon's in-flight transfers. A second concurrent caller
                    // would orphan the first await; cancel it rather than leak.
                    if self.isDownloading && self.currentModel?.id == model.id {
                        self.downloadContinuation?.resume(throwing: CancellationError())
                        self.downloadContinuation = continuation
                        if self.progressLogStart == nil { self.progressLogStart = Date() }
                        if self.isPaused {
                            self.resumeDownload()
                        }
                        return
                    }

                    // Cancel any in-progress download for a different model
                    if self.isDownloading {
                        self.cancelDownload()
                    }

                    self.downloadContinuation = continuation
                    self.currentModel = model
                    self.downloadingModelId = model.id
                    self.isDownloading = true
                    self.isPaused = false
                    self.progress = 0
                    self.status = "Starting..."
                    self.shownBytesHighWater = 0
                    self.progressLogStart = Date()
                    self.lastProgressLog = nil
                    self.loggedFirstBytes = false
                    #if os(iOS)
                    self.handedOffToBackground = false
                    #endif
                    self.resetRetryState()

                    let dest = self.modelsDirectory.appendingPathComponent(model.folderName)
                    self.destDir = dest
                    try? self.fileManager.createDirectory(at: dest, withIntermediateDirectories: true)

                    if model.downloadURL.contains("huggingface.co") {
                        self.buildHuggingFaceFileList(model)
                        self.fillDownloadSlots()
                    } else {
                        try? self.fileManager.removeItem(at: dest)
                        try? self.fileManager.createDirectory(at: dest, withIntermediateDirectories: true)
                        self.pendingFiles = [DownloadFile(
                            remotePath: model.downloadURL,
                            localPath: "__archive.zip",
                            estimatedSize: 350_000_000
                        )]
                        self.totalBytesForAllFiles = 350_000_000
                        self.completedBytes = 0
                        self.countedBytes = [:]
                        self.nextFileIndex = 0
                        self.fillDownloadSlots()
                    }
                }
            }
        }
    }

    public func pause() {
        guard isDownloading, !isPaused else { return }
        isPaused = true
        status = "Paused"

        // Cancel all active tasks. On resume, incomplete files are re-downloaded.
        for task in activeDownloadTasks.values {
            task.cancel()
        }
        activeDownloadTasks.removeAll()
        activeTaskFileIndex.removeAll()
        activeTaskBytes.removeAll()
        saveState()
    }

    public func resumeDownload() {
        guard isPaused else { return }
        runAfterAdoption { [weak self] in
            guard let self, self.isPaused else { return }
            self.isPaused = false
            self.status = "Resuming..."

            // Re-scan from beginning — fillDownloadSlots skips completed files
            // on disk and skips files an adopted task is already fetching.
            self.nextFileIndex = 0
            self.completedBytes = 0
            self.countedBytes = [:]
            self.resetRetryState()
            self.fillDownloadSlots()
        }
    }

    public func cancelDownload() {
        shownBytesHighWater = 0
        for task in activeDownloadTasks.values {
            task.cancel()
        }
        activeDownloadTasks.removeAll()
        activeTaskFileIndex.removeAll()
        activeTaskBytes.removeAll()
        nextFileIndex = 0
        completedBytes = 0
        countedBytes = [:]
        resetRetryState()
        isDownloading = false
        isPaused = false
        progress = 0
        status = ""
        downloadingModelId = nil
        pendingFiles = []
        cleanupPersistedState()

        downloadContinuation?.resume(throwing: CancellationError())
        downloadContinuation = nil
    }

    public func delete(_ model: ModelInfo) throws {
        if isDownloading && currentModel?.id == model.id {
            cancelDownload()
        }
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        defer { refreshTrigger += 1 }
        guard fileManager.fileExists(atPath: dir.path) else { return }
        try evictToGraveyard(dir)
    }

    /// Remove every model folder under `modelsDirectory`. Used as an escape
    /// hatch when a stale/incompatible artifact from a prior app version
    /// can't be deleted via the per-model trash button.
    public func resetAllModels() throws {
        cancelDownload()
        defer { refreshTrigger += 1 }
        guard fileManager.fileExists(atPath: modelsDirectory.path) else { return }
        let children = (try? fileManager.contentsOfDirectory(
            at: modelsDirectory, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])) ?? []
        var firstError: Error?
        for child in children {
            // Skip the graveyard itself — it gets cleaned asynchronously.
            if child.lastPathComponent.hasPrefix(".graveyard-") { continue }
            do { try evictToGraveyard(child) }
            catch { if firstError == nil { firstError = error } }
        }
        if let e = firstError { throw e }
    }

    /// Move a file / directory out of sight by renaming it into a hidden
    /// graveyard folder, then best-effort delete. Rename succeeds on APFS
    /// even when URLSession background tasks still hold open handles to
    /// files inside — whereas `removeItem` on the same path fails with
    /// "no permission to access" because of those handles.
    ///
    /// The visible model folder disappears immediately. Remaining graveyard
    /// bytes are swept up by `cleanGraveyard` at the next init.
    private func evictToGraveyard(_ url: URL) throws {
        let graveRoot = modelsDirectory.appendingPathComponent(".graveyard", isDirectory: true)
        try? fileManager.createDirectory(at: graveRoot, withIntermediateDirectories: true)
        let grave = graveRoot.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fileManager.moveItem(at: url, to: grave)
        try? fileManager.removeItem(at: grave)  // best-effort; leftovers cleaned next launch
    }

    /// Best-effort removal of any graveyard residue from prior sessions.
    /// Called on init — by then the URLSession daemon from the prior app
    /// run has released its file handles, so removeItem usually succeeds.
    private func cleanGraveyard() {
        let graveRoot = modelsDirectory.appendingPathComponent(".graveyard", isDirectory: true)
        guard fileManager.fileExists(atPath: graveRoot.path) else { return }
        if let children = try? fileManager.contentsOfDirectory(
            at: graveRoot, includingPropertiesForKeys: nil, options: []) {
            for c in children { try? fileManager.removeItem(at: c) }
        }
        try? fileManager.removeItem(at: graveRoot)
    }

    // MARK: - Parallel Download Scheduling

    /// Dispatch every not-yet-active file to the background session.
    /// Skips files that already exist on disk.
    /// Clears failed-file retry bookkeeping. Call wherever the download
    /// queue itself is (re)built or torn down — stale indices from a prior
    /// `pendingFiles` array would point at the wrong files.
    private func resetRetryState() {
        retryFileIndices = []
        fileRetryCounts = [:]
        completionSweepsDone = 0
    }

    private func fillDownloadSlots() {
        guard !isPaused else {
            saveState()
            return
        }
        // Before a relaunch's adoption lands, every surviving background
        // task looks absent and would be dispatched a second time.
        guard tasksAdopted else {
            if !fillDeferredToAdoption {
                fillDeferredToAdoption = true
                pendingAdoptionActions.append { [weak self] in
                    self?.fillDeferredToAdoption = false
                    self?.fillDownloadSlots()
                }
            }
            return
        }

        // Files currently being fetched by an adopted or in-flight task.
        // Without this guard, restarting the loop after a pause/relaunch could
        // hand the same file to a second task and double-count its bytes.
        var activeIndices = Set(activeTaskFileIndex.values)

        // Foreground: a bounded few per in-process session, refilled from
        // the delegate as each lands. Background: hand EVERY remaining file
        // to the background lanes up front — nsurlsessiond runs them while
        // the app is suspended, whereas a bounded queue would need an app
        // wake per refill, and iOS rate-limits those (a ~50-file bundle once
        // crawled for hours in the background and the download-complete
        // notification never fired).
        let foreground = usesForegroundSessions
        var foregroundLoad = [Int](repeating: 0, count: foregroundSessions.count)
        for key in activeDownloadTasks.keys where key.isForeground {
            if let i = Int(key.session.dropFirst(2)), i < foregroundLoad.count { foregroundLoad[i] += 1 }
        }
        let foregroundCapacity = foregroundSessions.count * Self.foregroundTasksPerSession
        while true {
            if foreground && foregroundLoad.reduce(0, +) >= foregroundCapacity { break }
            // Re-queued failures take priority over the sequential walk.
            let idx: Int
            if !retryFileIndices.isEmpty {
                idx = retryFileIndices.removeFirst()
            } else if nextFileIndex < pendingFiles.count {
                idx = nextFileIndex
                nextFileIndex += 1
            } else {
                break
            }

            if activeIndices.contains(idx) { continue }

            let file = pendingFiles[idx]
            guard let dest = destDir else { continue }
            let destFile = dest.appendingPathComponent(file.localPath)

            // Skip already-downloaded files
            if file.localPath != "__archive.zip" && fileManager.fileExists(atPath: destFile.path) {
                let existingSize = (try? fileManager.attributesOfItem(atPath: destFile.path))?[.size] as? Int64 ?? 0
                if existingSize > 0 {
                    count(idx, bytes: existingSize)
                    updateProgress()
                    continue
                }
            }

            // Segment whose joined file already exists (repair sweep over a
            // completed install): the join consumed it, so its absence is
            // expected — satisfied, no network.
            if let target = Self.joinTarget(of: file),
               fileManager.fileExists(atPath: dest.appendingPathComponent(target).path) {
                count(idx, bytes: file.estimatedSize)
                updateProgress()
                continue
            }

            // Build URL
            let urlString: String
            if file.remotePath.hasPrefix("http") {
                urlString = file.remotePath
            } else if let model = currentModel {
                urlString = "\(model.downloadURL)/\(file.remotePath)"
            } else {
                continue
            }

            guard let url = URL(string: urlString) else { continue }

            try? fileManager.createDirectory(at: destFile.deletingLastPathComponent(),
                                              withIntermediateDirectories: true)

            var request = URLRequest(url: url)
            if let start = file.rangeStart {
                let end = file.rangeEnd.map { String($0) } ?? ""
                request.setValue("bytes=\(start)-\(end)", forHTTPHeaderField: "Range")
                // Offsets index the raw file, never a compressed body.
                request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
            }
            // Foreground: the least-loaded session. Background: consecutive
            // indices (a file's segments) land on different lanes, i.e.
            // different connections.
            let lane: URLSession
            if foreground, let i = foregroundLoad.indices.min(by: { foregroundLoad[$0] < foregroundLoad[$1] }) {
                lane = foregroundSessions[i]
                foregroundLoad[i] += 1
            } else {
                lane = sessions[idx % sessions.count]
            }
            let task = lane.downloadTask(with: request)
            task.taskDescription = file.localPath
            task.resume()

            let key = Self.key(lane, task)
            activeDownloadTasks[key] = task
            activeTaskFileIndex[key] = idx
            activeIndices.insert(idx)
        }

        // All files dispatched and all tasks completed → finish
        if activeDownloadTasks.isEmpty && nextFileIndex >= pendingFiles.count
            && retryFileIndices.isEmpty {
            finishDownload()
            return
        }

        saveState()
    }

    /// Downloads restored after a process kill are parked paused (no caller
    /// owns a continuation yet — see `restorePendingDownload`). Session events
    /// arriving means nsurlsessiond already ran the transfers, e.g. the system
    /// relaunched a terminated app in the background to deliver them. Un-park
    /// so `fillDownloadSlots` keeps dispatching and `finishDownload` (sweep +
    /// prefill hardlinks) runs inside that background wake — otherwise the
    /// bundle stays one post-processing step short of complete and the app's
    /// download-finished notification is suppressed. A user-initiated `pause()`
    /// keeps its continuation alive and cancels its tasks, so it never matches.
    private func unparkRestoredDownloadIfNeeded() {
        if isPaused && downloadContinuation == nil {
            isPaused = false
        }
    }

    /// Dispatch to in-process sessions? `.inactive` counts as foreground: a
    /// cold launch starts the install before the app turns `.active`.
    private var usesForegroundSessions: Bool {
        #if os(iOS)
        return !handedOffToBackground && UIApplication.shared.applicationState != .background
        #else
        return false
        #endif
    }

    #if os(iOS)
    /// Backgrounding mid-download: in-process sessions stall once the app is
    /// backgrounded (device: 16 foreground segments moved ~34 MB in 20 s), so
    /// cancel them right away and queue them — with everything not yet
    /// started — on the background lanes, which nsurlsessiond keeps running.
    @objc private func appDidEnterBackground() {
        guard isDownloading, !isPaused, tasksAdopted, !handedOffToBackground else { return }
        handedOffToBackground = true
        let moved = requeue { $0.isForeground }
        fillDownloadSlots()
        print("[Download] backgrounded — \(moved) foreground segments and all queued files moved to background lanes")
    }

    @objc private func appDidBecomeActive() {
        guard isDownloading, activeDownloadTasks.keys.contains(where: { !$0.isForeground }) else { return }
        scheduleReclaim()
    }

    /// Back in the foreground: pull every lane task back to the foreground
    /// sessions, started or not. Lanes start slowly, and ones that did start
    /// while backgrounded then crawled (device: 22 of them at ~2.4 MB/s total
    /// for 7+ min with the app open) — re-fetching their partial segments in
    /// the foreground (~40 MB/s) is far cheaper. Waits a moment so a quick
    /// app switch doesn't churn.
    private func scheduleReclaim() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) { [weak self] in
            guard let self, self.isDownloading, !self.isPaused, self.tasksAdopted,
                  UIApplication.shared.applicationState == .active else { return }
            self.handedOffToBackground = false
            let moved = self.requeue { !$0.isForeground }
            self.fillDownloadSlots()
            if moved > 0 {
                print("[Download] foreground — \(moved) lane tasks moved to foreground sessions")
            }
        }
    }

    /// Cancel matching active tasks and put their files at the front of the
    /// queue (not counted as retry attempts). Returns how many moved.
    private func requeue(where match: (TaskKey) -> Bool) -> Int {
        let keys = activeDownloadTasks.keys.filter(match)
        for key in keys {
            activeDownloadTasks[key]?.cancel()
            if let idx = activeTaskFileIndex[key] { retryFileIndices.append(idx) }
            activeDownloadTasks.removeValue(forKey: key)
            activeTaskFileIndex.removeValue(forKey: key)
            activeTaskBytes.removeValue(forKey: key)
        }
        if !keys.isEmpty { updateProgress() }
        return keys.count
    }
    #endif

    /// A task failed transiently (network error, bad range response, 429 /
    /// 5xx). Runs on the main queue.
    private func handleTaskFailure(taskId: TaskKey, error: Error) {
        unparkRestoredDownloadIfNeeded()
        let fileIndex = activeTaskFileIndex[taskId]
        activeDownloadTasks.removeValue(forKey: taskId)
        activeTaskFileIndex.removeValue(forKey: taskId)
        activeTaskBytes.removeValue(forKey: taskId)

        // Re-queue the failed file (bounded). Without this the file is
        // lost — nextFileIndex already advanced past it — and the
        // download "succeeds" minus one file, which the model only
        // notices at load time ("…couldn't be opened because there is
        // no such file"). Background sessions hit transient task
        // failures routinely, so this is the common path, not the edge.
        if let idx = fileIndex, idx < pendingFiles.count {
            let attempts = (fileRetryCounts[idx] ?? 0) + 1
            fileRetryCounts[idx] = attempts
            if attempts <= maxRetriesPerFile {
                print("[Download] Retry \(attempts)/\(maxRetriesPerFile) for \(pendingFiles[idx].localPath): \(error.localizedDescription)")
                retryFileIndices.append(idx)
                fillDownloadSlots()
                return
            }
        }

        // Retries exhausted for this file. If other work remains, keep
        // going — the completeness sweep in finishDownload will surface
        // the gap; otherwise fail the download with the real error.
        if !activeDownloadTasks.isEmpty
            || nextFileIndex < pendingFiles.count
            || !retryFileIndices.isEmpty {
            fillDownloadSlots()
            return
        }

        status = "Error: \(error.localizedDescription)"
        isDownloading = false
        isPaused = false
        downloadingModelId = nil
        cleanupPersistedState()
        downloadContinuation?.resume(throwing: error)
        downloadContinuation = nil
    }

    /// `requested` is `bytes=START-END` or `bytes=START-`. Valid = a 206
    /// whose `Content-Range: bytes A-B/TOTAL` ends at END (or TOTAL-1 when
    /// open-ended) and a body of exactly the requested span. A may exceed
    /// START: after a dropped connection nsurlsessiond resumes the task by
    /// itself with a narrower Range, stitches the bytes, and reports only
    /// that last response (seen on device: asked 1879048192-, got 206
    /// "1929975452-…", full 64 MiB body) — the body length is the check.
    nonisolated static func isValidRangeResponse(
        _ http: HTTPURLResponse?, requested: String, bodySize: Int64
    ) -> Bool {
        guard let http, http.statusCode == 206,
              let contentRange = http.value(forHTTPHeaderField: "Content-Range"),
              requested.hasPrefix("bytes="), contentRange.hasPrefix("bytes ") else { return false }
        let req = requested.dropFirst("bytes=".count)
            .split(separator: "-", omittingEmptySubsequences: false)
        let span = contentRange.dropFirst("bytes ".count).split(separator: "/")
        guard req.count == 2, let reqStart = Int64(req[0]),
              span.count == 2, let total = Int64(span[1]) else { return false }
        let got = span[0].split(separator: "-")
        guard got.count == 2, let start = Int64(got[0]), let end = Int64(got[1]) else { return false }
        guard let expectedEnd = req[1].isEmpty ? total - 1 : Int64(req[1]) else { return false }
        return start >= reqStart && start <= end && end == expectedEnd
            && bodySize == expectedEnd - reqStart + 1
    }

    /// A console line at first byte and every 5 s after. nsurlsessiond can
    /// sit on queued tasks before any byte moves, and a UI stuck at 0% can't
    /// tell "slow" from "not started" — this is the record that can.
    private func logProgressIfDue(inFlight: Int64, bytes: Int64) {
        guard let start = progressLogStart else { return }
        let now = Date()
        if !loggedFirstBytes, inFlight > 0 {
            loggedFirstBytes = true
            print(String(format: "[Download] first bytes after %.1f s", now.timeIntervalSince(start)))
        }
        if let last = lastProgressLog, now.timeIntervalSince(last) < 5 { return }
        lastProgressLog = now
        let receiving = activeTaskBytes.filter { $0.value > 0 }
        let lanes = Set(receiving.keys.map(\.session)).count
        print(String(format: "[Download] t+%.0fs %.0f / %.0f MB · %d of %d tasks receiving on %d sessions",
                     now.timeIntervalSince(start), Double(bytes) / 1e6,
                     Double(totalBytesForAllFiles) / 1e6, receiving.count,
                     activeDownloadTasks.count, lanes))
    }

    private func count(_ index: Int, bytes: Int64) {
        completedBytes += bytes - (countedBytes[index] ?? 0)
        countedBytes[index] = bytes
    }

    private func updateProgress() {
        let inFlight = activeTaskBytes.values.reduce(0 as Int64, +)
        let bytes = completedBytes + inFlight
        logProgressIfDue(inFlight: inFlight, bytes: bytes)
        let total = Double(max(totalBytesForAllFiles, 1))
        shownBytesHighWater = max(shownBytesHighWater, bytes)
        progress = min(Double(shownBytesHighWater) / total, 0.99)
        let mbDone = Double(shownBytesHighWater) / 1_000_000
        let mbTotal = Double(totalBytesForAllFiles) / 1_000_000
        status = String(format: "%.0f / %.0f MB", mbDone, mbTotal)

        // Safety stop: estimates and actuals agree to within a few percent on
        // this repo. If we cross 1.5x the estimate, the most likely cause is
        // a duplicate-download bug (e.g. a leftover background task fetching
        // the same 2.35 GB file as the new one). Abort so we don't burn the
        // user's data plan or fill the disk.
        if totalBytesForAllFiles > 0,
           bytes > Int64(Double(totalBytesForAllFiles) * 1.5) {
            abortOversizeDownload(bytes: bytes)
        }
    }

    private func abortOversizeDownload(bytes: Int64) {
        for task in activeDownloadTasks.values { task.cancel() }
        activeDownloadTasks.removeAll()
        activeTaskFileIndex.removeAll()
        activeTaskBytes.removeAll()
        pendingFiles = []
        nextFileIndex = 0
        resetRetryState()
        isDownloading = false
        isPaused = false
        downloadingModelId = nil
        cleanupPersistedState()
        let mbDone = bytes / 1_000_000
        let mbTotal = totalBytesForAllFiles / 1_000_000
        status = "Stopped: \(mbDone) MB exceeds expected \(mbTotal) MB"
        let err = NSError(
            domain: "CoreMLLLM.ModelDownloader", code: -2,
            userInfo: [NSLocalizedDescriptionKey:
                "Download aborted: \(mbDone) MB exceeds expected \(mbTotal) MB by 50%+. " +
                "Likely a duplicate-download bug — restart the app and retry."])
        downloadContinuation?.resume(throwing: err)
        downloadContinuation = nil
    }

    private func finishDownload() {
        guard let model = currentModel, let dest = destDir else { return }

        // Completeness sweep: never resume success with files missing.
        // Catches anything that slipped past the per-task retry (e.g. a
        // failed temp-file move in didFinishDownloadingTo). Missing files
        // get re-queued for up to two extra passes; after that, throw so
        // the app's error → retry(repair:) path takes over instead of the
        // model failing at load time with a confusing missing-file error.
        let missing = pendingFiles.indices.filter { idx in
            let f = pendingFiles[idx]
            if f.localPath == "__archive.zip" { return false }  // deleted after extraction
            if isOptionalMlmodelcFile(f.localPath) { return false }  // legitimately 404s
            // A consumed segment is satisfied by its joined output.
            if let target = Self.joinTarget(of: f),
               fileManager.fileExists(atPath: dest.appendingPathComponent(target).path) {
                return false
            }
            return !fileManager.fileExists(atPath: dest.appendingPathComponent(f.localPath).path)
        }
        if !missing.isEmpty {
            if completionSweepsDone < 2 {
                completionSweepsDone += 1
                print("[Download] Completeness sweep \(completionSweepsDone): re-fetching \(missing.count) missing file(s)")
                retryFileIndices.append(contentsOf: missing)
                fillDownloadSlots()
                return
            }
            let names = missing.prefix(3).map { pendingFiles[$0].localPath }.joined(separator: ", ")
            status = "Error: download incomplete"
            isDownloading = false
            isPaused = false
            downloadingModelId = nil
            cleanupPersistedState()
            let err = NSError(domain: "CoreMLLLM.ModelDownloader", code: -3, userInfo: [
                NSLocalizedDescriptionKey:
                    "Download incomplete: \(missing.count) file(s) couldn't be fetched (\(names)). Please try again.",
            ])
            downloadContinuation?.resume(throwing: err)
            downloadContinuation = nil
            return
        }

        // Segment joins: multi-GB sequential file I/O, so hop off the main
        // queue (same pattern as the zip-extraction path) and re-enter the
        // completion tail on main when done. `isJoiningParts` guards a
        // second finishDownload (e.g. a re-attached caller's resumeDownload)
        // from racing a concurrent join over the same temps.
        let joins = pendingJoins(in: dest)
        if !joins.isEmpty {
            guard !isJoiningParts else { return }
            isJoiningParts = true
            status = "Assembling..."
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                var joinError: Error?
                for job in joins {
                    joinError = Self.joinParts(job.parts, into: job.target)
                    if joinError != nil { break }
                }
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.isJoiningParts = false
                    if let joinError {
                        self.status = "Error: couldn't assemble model files"
                        self.isDownloading = false
                        self.isPaused = false
                        self.downloadingModelId = nil
                        self.cleanupPersistedState()
                        self.downloadContinuation?.resume(throwing: joinError)
                        self.downloadContinuation = nil
                    } else {
                        self.completeDownload(model: model, dest: dest)
                    }
                }
            }
            return
        }

        completeDownload(model: model, dest: dest)
    }

    /// Join targets not yet assembled, each with its pieces in file order
    /// (`rangeSegmented` emits a file's segments contiguously, and legacy
    /// parts were listed part1…part4). chunk1's weight — the file
    /// `localModelURL` keys "bundle present" on — joins last, so a kill
    /// mid-join can't leave the folder looking complete. Pieces lingering
    /// beside an already-joined target (e.g. a task adopted from a prior
    /// process landing after the join) are deleted here.
    private func pendingJoins(in dest: URL) -> [(target: URL, parts: [URL])] {
        var order: [String] = []
        var parts: [String: [URL]] = [:]
        for f in pendingFiles {
            guard let target = Self.joinTarget(of: f) else { continue }
            if parts[target] == nil { order.append(target) }
            parts[target, default: []].append(dest.appendingPathComponent(f.localPath))
        }
        if let i = order.firstIndex(of: "chunk1.mlmodelc/weights/weight.bin") {
            order.append(order.remove(at: i))
        }
        return order.compactMap { target in
            let targetURL = dest.appendingPathComponent(target)
            let pieces = parts[target] ?? []
            if fileManager.fileExists(atPath: targetURL.path) {
                for p in pieces where fileManager.fileExists(atPath: p.path) {
                    try? fileManager.removeItem(at: p)
                }
                return nil
            }
            return (targetURL, pieces)
        }
    }

    /// Everything after the completeness sweep + segment joins: prefill
    /// weight sharing, stray-directory cleanup, and resuming the caller.
    /// Runs on the main queue.
    private func completeDownload(model: ModelInfo, dest: URL) {

        // Share decode weights with prefill chunks ONLY if prefill metadata
        // (coremldata.bin) was downloaded for that chunk. Models that don't
        // ship prefill (e.g. gemma4-e4b) would otherwise get half-populated
        // prefill_chunk{i}.mlmodelc directories — just weights, no
        // coremldata.bin — which CoreML rejects at load time.
        //
        // Stage 7: hardlink instead of copy. Decode and prefill weights are
        // bit-identical (md5-verified) for chunk1↔prefill_chunk1 and
        // chunk3_3way↔prefill_chunk4 — a hardlink shares the inode so the
        // 155 + 527 = 682 MB doesn't get duplicated on disk.
        func shareWeight(src: URL, dst: URL, coreML: URL) {
            guard fileManager.fileExists(atPath: coreML.path),
                  fileManager.fileExists(atPath: src.path),
                  !fileManager.fileExists(atPath: dst.path) else { return }
            try? fileManager.createDirectory(at: dst.deletingLastPathComponent(),
                                              withIntermediateDirectories: true)
            // linkItem creates a hardlink; falls back to copy if the FS
            // doesn't support links (uncommon on iOS APFS, but defensive).
            do {
                try fileManager.linkItem(at: src, to: dst)
            } catch {
                try? fileManager.copyItem(at: src, to: dst)
            }
        }
        for i in 1...4 {
            shareWeight(
                src: dest.appendingPathComponent("chunk\(i).mlmodelc/weights/weight.bin"),
                dst: dest.appendingPathComponent("prefill_chunk\(i).mlmodelc/weights/weight.bin"),
                coreML: dest.appendingPathComponent("prefill_chunk\(i).mlmodelc/coremldata.bin"))
        }
        // 3way variant: chunk3_3way (L25-34 + lm_head) shares weights with
        // prefill_chunk4 (same SWAChunk4 graph, T=N prefill flavor). The
        // 1...4 loop above missed it because the source filename is
        // chunk3_3way, not chunk4.
        shareWeight(
            src: dest.appendingPathComponent("chunk3_3way.mlmodelc/weights/weight.bin"),
            dst: dest.appendingPathComponent("prefill_chunk4.mlmodelc/weights/weight.bin"),
            coreML: dest.appendingPathComponent("prefill_chunk4.mlmodelc/coremldata.bin"))

        // Clean up any stray prefill directories that lack the required
        // metadata. These happen when an older build of the app pulled prefill
        // paths that 404'd on a prefill-less repo — the shared-weight copy
        // above then seeded zero-metadata subdirectories, which CoreML can't
        // open. Removing them here makes the device self-heal on next launch.
        for i in 1...4 {
            let prefillDir = dest.appendingPathComponent("prefill_chunk\(i).mlmodelc")
            let coreML = prefillDir.appendingPathComponent("coremldata.bin")
            if fileManager.fileExists(atPath: prefillDir.path)
                && !fileManager.fileExists(atPath: coreML.path) {
                try? fileManager.removeItem(at: prefillDir)
            }
        }

        cleanupPersistedState()
        isDownloading = false
        isPaused = false
        downloadingModelId = nil
        progress = 1.0
        status = "Ready"

        if let url = localModelURL(for: model) {
            downloadContinuation?.resume(returning: url)
        } else {
            downloadContinuation?.resume(throwing: DownloadError.extractionFailed)
        }
        downloadContinuation = nil
    }

    /// Concatenate `parts` (in order) into `target`. Crash-safe: the first
    /// part is renamed to a `.joining` temp (no copy) and the rest appended,
    /// each deleted right after it's consumed (transient disk ≈ one part);
    /// the temp takes the final name only after every part is in. A crash
    /// mid-join leaves a stale temp plus missing consumed parts — the
    /// caller's completeness sweep re-fetches exactly those parts on the
    /// next repair pass and the join restarts from scratch. Pure file I/O;
    /// runs off the main queue.
    nonisolated static func joinParts(_ parts: [URL], into target: URL) -> Error? {
        let fm = FileManager.default
        guard let first = parts.first, parts.allSatisfy({ fm.fileExists(atPath: $0.path) }) else {
            // Caller only dispatches after the completeness sweep passed, so
            // this is defensive.
            return NSError(domain: "CoreMLLLM.ModelDownloader", code: -4, userInfo: [
                NSLocalizedDescriptionKey: "Model file parts are incomplete. Please retry the download.",
            ])
        }
        let temp = target.appendingPathExtension("joining")
        try? fm.removeItem(at: temp)  // stale temp from a crashed join
        do {
            try fm.moveItem(at: first, to: temp)
            let out = try FileHandle(forWritingTo: temp)
            defer { try? out.close() }
            try out.seekToEnd()
            for part in parts.dropFirst() {
                let input = try FileHandle(forReadingFrom: part)
                defer { try? input.close() }
                while true {
                    // Bounded reads keep peak memory at one buffer; the
                    // autoreleasepool drops the bridged NSData each pass.
                    let chunk = try autoreleasepool {
                        try input.read(upToCount: 16 * 1024 * 1024)
                    }
                    guard let chunk, !chunk.isEmpty else { break }
                    try out.write(contentsOf: chunk)
                }
                try? input.close()
                try fm.removeItem(at: part)
            }
            try out.close()
            try fm.moveItem(at: temp, to: target)
            print("[Download] Joined \(parts.count) parts → \(target.lastPathComponent)")
            return nil
        } catch {
            try? fm.removeItem(at: temp)
            return NSError(domain: "CoreMLLLM.ModelDownloader", code: -4, userInfo: [
                NSLocalizedDescriptionKey:
                    "Couldn't assemble \(target.lastPathComponent) from its parts: "
                    + "\(error.localizedDescription). Free up storage and retry.",
            ])
        }
    }

    // MARK: - Persistence

    private var stateURL: URL {
        modelsDirectory.appendingPathComponent(".download_state.json")
    }

    private func saveState() {
        guard let model = currentModel else { return }
        try? fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        let state = PersistedState(
            modelId: model.id,
            totalBytes: totalBytesForAllFiles,
            files: pendingFiles,
            downloadURL: model.downloadURL,
            folderName: model.folderName,
            modelName: model.name
        )
        try? JSONEncoder().encode(state).write(to: stateURL)
    }

    private func cleanupPersistedState() {
        try? fileManager.removeItem(at: stateURL)
    }

    private func restorePendingDownload() {
        guard let data = try? Data(contentsOf: stateURL),
              let state = try? JSONDecoder().decode(PersistedState.self, from: data) else { return }
        let model: ModelInfo
        if let url = state.downloadURL, let folder = state.folderName {
            model = ModelInfo(id: state.modelId, name: state.modelName ?? state.modelId,
                              size: "", downloadURL: url, folderName: folder)
        } else if let known = availableModels.first(where: { $0.id == state.modelId }) {
            model = known
        } else {
            return
        }

        if localModelURL(for: model) != nil {
            cleanupPersistedState()
            return
        }

        currentModel = model
        destDir = modelsDirectory.appendingPathComponent(model.folderName)
        pendingFiles = state.files
        totalBytesForAllFiles = state.totalBytes
        nextFileIndex = 0
        completedBytes = 0
        countedBytes = [:]
        resetRetryState()
        downloadingModelId = model.id
        isDownloading = true
        isPaused = true
        // Scan completed bytes from disk for accurate progress
        if let dest = destDir {
            for (i, file) in pendingFiles.enumerated() {
                let path = dest.appendingPathComponent(file.localPath).path
                if let attrs = try? fileManager.attributesOfItem(atPath: path),
                   let size = attrs[.size] as? Int64, size > 0 {
                    count(i, bytes: size)
                }
            }
        }
        progress = Double(completedBytes) / Double(max(totalBytesForAllFiles, 1))
        status = "Paused"
    }

    // MARK: - HuggingFace File List

    private func buildHuggingFaceFileList(_ model: ModelInfo) {
        // E4B lives in its own repo with a flat layout (chunks at the root,
        // text-only — no prefill, vision, or audio). E2B lives under swa/ +
        // prefill/ + vision/audio at root.
        if model.id == "gemma4-e4b" {
            buildE4BFileList()
            return
        }
        if model.id == "gemma4-e2b-stateful-linear" {
            buildGemma4StatefulLinearFileList()
            return
        }
        if model.id == "qwen3.5-0.8b" {
            buildQwen35FileList()
            return
        }
        if model.id == "qwen3.5-2b" {
            buildQwen35_2B_FileList()
            return
        }
        if model.id == "qwen3-vl-2b" {
            buildQwen3VL2BFileList()
            return
        }
        // 2K-context shipping model lives at the repo root on HF:
        //   - Decode chunks:  swa/chunk{1-4}.mlmodelc/
        //   - Prefill chunks: prefill/chunk{1-4}.mlmodelc/  (remote name is
        //     chunk1, local is prefill_chunk1 to avoid colliding with decode)
        // NOTE: `sdpa/` on HF is actually 8K — its metadata.json says 2048
        // but model.mil has ctx=8192 (authoritative). Don't use `sdpa/` or
        // `sdpa-8k/` until the 8K decode path lands (see docs/SPEED_8K.md).

        func mlc(_ remoteDir: String, _ remoteName: String, _ localName: String, weightSize: Int64) -> [DownloadFile] {
            [.init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/weights/weight.bin",
                   localPath: "\(localName).mlmodelc/weights/weight.bin", estimatedSize: weightSize),
             .init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/coremldata.bin",
                   localPath: "\(localName).mlmodelc/coremldata.bin", estimatedSize: 1_000),
             .init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/model.mil",
                   localPath: "\(localName).mlmodelc/model.mil", estimatedSize: 450_000),
             .init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/metadata.json",
                   localPath: "\(localName).mlmodelc/metadata.json", estimatedSize: 8_000),
             .init(remotePath: "\(remoteDir)/\(remoteName).mlmodelc/analytics/coremldata.bin",
                   localPath: "\(localName).mlmodelc/analytics/coremldata.bin", estimatedSize: 250)]
        }

        // Prefill metadata-only (weights are shared with decode chunks and
        // copied in finishDownload). Remote name is `chunk1` under prefill/;
        // local name is `prefill_chunk1` so it doesn't collide with decode.
        func prefillMeta(_ remoteName: String, _ localName: String) -> [DownloadFile] {
            [.init(remotePath: "prefill/\(remoteName).mlmodelc/coremldata.bin",
                   localPath: "\(localName).mlmodelc/coremldata.bin", estimatedSize: 1_000),
             .init(remotePath: "prefill/\(remoteName).mlmodelc/model.mil",
                   localPath: "\(localName).mlmodelc/model.mil", estimatedSize: 450_000),
             .init(remotePath: "prefill/\(remoteName).mlmodelc/metadata.json",
                   localPath: "\(localName).mlmodelc/metadata.json", estimatedSize: 8_000),
             .init(remotePath: "prefill/\(remoteName).mlmodelc/analytics/coremldata.bin",
                   localPath: "\(localName).mlmodelc/analytics/coremldata.bin", estimatedSize: 250)]
        }

        // Sort large files first so they start downloading immediately on separate connections,
        // while small files fill in around them.
        var largeFiles: [DownloadFile] = []
        var smallFiles: [DownloadFile] = []

        // Stage 7: 3-chunk decode is the new default. The 3way ModelInfo
        // entry skips chunk2/3/4 (replaced by chunk2_3way + chunk3_3way)
        // for a -45 MB bundle delta. The legacy gemma4e2b entry still
        // pulls chunk2/3/4 for backward-compat with apps that haven't
        // upgraded to the 3-chunk loader path.
        // The -split variant is the 3way layout (Evie's mirror-pinned id).
        // It used to fetch the per-layer embed as 4 mirror-hosted parts;
        // Range segments (`rangeSegmented`) now split every large file, so
        // it downloads exactly like -3way.
        let is3Way = (model.id == "gemma4-e2b-3way" || model.id == "gemma4-e2b-3way-split")
        var chunkFiles = mlc("swa", "chunk1", "chunk1", weightSize: 155_436_864)
        if is3Way {
            chunkFiles += mlc("swa", "chunk2_3way", "chunk2_3way",
                              weightSize: 459_245_120)
            chunkFiles += mlc("swa", "chunk3_3way", "chunk3_3way",
                              weightSize: 526_874_880)
        } else {
            chunkFiles += mlc("swa", "chunk2", "chunk2", weightSize: 133_963_968)
            chunkFiles += mlc("swa", "chunk3", "chunk3", weightSize: 325_282_880)
            chunkFiles += mlc("swa", "chunk4", "chunk4", weightSize: 526_874_880)
        }
        // Prefill chunk weights: legacy variant shares them from decode
        // chunks (`finishDownload` copies chunk{i}.weight → prefill_chunk{i}.weight).
        //
        // 3way variant share map (md5-verified bit-identical):
        //   - prefill_chunk1 weight = chunk1 weight       — hardlink
        //   - prefill_chunk2 weight = unique (L8-14 own)  — direct download
        //   - prefill_chunk3 weight = unique (L15-24)     — direct download
        //   - prefill_chunk4 weight = chunk3_3way weight  — hardlink (Stage 7
        //     extra: same SWAChunk4 weights as the decode head, saves 527 MB
        //     on download AND on disk vs duplicate-copy storage).
        let prefillFiles: [DownloadFile]
        if is3Way {
            prefillFiles = prefillMeta("chunk1", "prefill_chunk1")
                + mlc("prefill", "chunk2", "prefill_chunk2", weightSize: 133_963_968)
                + mlc("prefill", "chunk3", "prefill_chunk3", weightSize: 325_282_880)
                + prefillMeta("chunk4", "prefill_chunk4")
        } else {
            prefillFiles = prefillMeta("chunk1", "prefill_chunk1")
                + prefillMeta("chunk2", "prefill_chunk2")
                + prefillMeta("chunk3", "prefill_chunk3")
                + prefillMeta("chunk4", "prefill_chunk4")
        }
        // Core (text-decoder) sidecars. Always required.
        let coreFiles: [DownloadFile] = [
            .init(remotePath: "model_config.json", localPath: "model_config.json", estimatedSize: 500),
            .init(remotePath: "hf_model/tokenizer.json", localPath: "hf_model/tokenizer.json", estimatedSize: 30_000_000),
            .init(remotePath: "hf_model/tokenizer_config.json", localPath: "hf_model/tokenizer_config.json", estimatedSize: 5_000),
            .init(remotePath: "hf_model/config.json", localPath: "hf_model/config.json", estimatedSize: 5_000),
            .init(remotePath: "embed_tokens_q8.bin", localPath: "embed_tokens_q8.bin", estimatedSize: 402_653_184),
            .init(remotePath: "embed_tokens_scales.bin", localPath: "embed_tokens_scales.bin", estimatedSize: 524_288),
            .init(remotePath: "embed_tokens_per_layer_scales.bin", localPath: "embed_tokens_per_layer_scales.bin", estimatedSize: 524_288),
            .init(remotePath: "per_layer_projection.bin", localPath: "per_layer_projection.bin", estimatedSize: 27_525_120),
            .init(remotePath: "per_layer_norm_weight.bin", localPath: "per_layer_norm_weight.bin", estimatedSize: 1_024),
            .init(remotePath: "swa/cos_sliding.npy", localPath: "cos_sliding.npy", estimatedSize: 4_194_432),
            .init(remotePath: "swa/sin_sliding.npy", localPath: "sin_sliding.npy", estimatedSize: 4_194_432),
            .init(remotePath: "swa/cos_full.npy", localPath: "cos_full.npy", estimatedSize: 8_388_736),
            .init(remotePath: "swa/sin_full.npy", localPath: "sin_full.npy", estimatedSize: 8_388_736),
        ]

        // Multimodal encoders + sidecars (~990 MB). Toggleable from the
        // model picker via UserDefaults `gemma4DownloadMultimodal`. Engine
        // load() detects encoder absence and disables vision/audio cleanly,
        // so a text-only install behaves like a normal text decoder.
        let multimodalFiles: [DownloadFile] = [
            .init(remotePath: "vision.mlmodelc/weights/weight.bin", localPath: "vision.mlmodelc/weights/weight.bin", estimatedSize: 320_000_000),
            .init(remotePath: "vision.mlmodelc/coremldata.bin", localPath: "vision.mlmodelc/coremldata.bin", estimatedSize: 200_000),
            .init(remotePath: "vision.mlmodelc/model.mil", localPath: "vision.mlmodelc/model.mil", estimatedSize: 50_000),
            .init(remotePath: "vision.mlmodelc/metadata.json", localPath: "vision.mlmodelc/metadata.json", estimatedSize: 1_000),
            .init(remotePath: "vision.mlmodelc/analytics/coremldata.bin", localPath: "vision.mlmodelc/analytics/coremldata.bin", estimatedSize: 1_000),
            // Video-grade vision encoder (Gemma 4's `video_processor` path,
            // 64 tokens/frame natively). When absent, the app transparently
            // falls back to Swift-side 2×2 pooling of the still-image encoder.
            .init(remotePath: "vision_video.mlmodelc/weights/weight.bin", localPath: "vision_video.mlmodelc/weights/weight.bin", estimatedSize: 338_081_024),
            .init(remotePath: "vision_video.mlmodelc/coremldata.bin", localPath: "vision_video.mlmodelc/coremldata.bin", estimatedSize: 418),
            .init(remotePath: "vision_video.mlmodelc/model.mil", localPath: "vision_video.mlmodelc/model.mil", estimatedSize: 711_289),
            .init(remotePath: "vision_video.mlmodelc/metadata.json", localPath: "vision_video.mlmodelc/metadata.json", estimatedSize: 2_721),
            .init(remotePath: "vision_video.mlmodelc/analytics/coremldata.bin", localPath: "vision_video.mlmodelc/analytics/coremldata.bin", estimatedSize: 243),
            // Audio encoder (Conformer 12-layer, INT8)
            .init(remotePath: "audio.mlmodelc/weights/weight.bin", localPath: "audio.mlmodelc/weights/weight.bin", estimatedSize: 295_373_248),
            .init(remotePath: "audio.mlmodelc/coremldata.bin", localPath: "audio.mlmodelc/coremldata.bin", estimatedSize: 1_000),
            .init(remotePath: "audio.mlmodelc/model.mil", localPath: "audio.mlmodelc/model.mil", estimatedSize: 759_000),
            .init(remotePath: "audio.mlmodelc/metadata.json", localPath: "audio.mlmodelc/metadata.json", estimatedSize: 3_000),
            .init(remotePath: "audio.mlmodelc/analytics/coremldata.bin", localPath: "audio.mlmodelc/analytics/coremldata.bin", estimatedSize: 250),
            .init(remotePath: "mel_filterbank.bin", localPath: "mel_filterbank.bin", estimatedSize: 131_584),
            .init(remotePath: "audio_config.json", localPath: "audio_config.json", estimatedSize: 500),
            // Audio projection weights (Swift-side float32 computation)
            .init(remotePath: "output_proj_weight.npy", localPath: "output_proj_weight.npy", estimatedSize: 3_145_856),
            .init(remotePath: "output_proj_bias.npy", localPath: "output_proj_bias.npy", estimatedSize: 3_200),
            .init(remotePath: "embed_proj_weight.npy", localPath: "embed_proj_weight.npy", estimatedSize: 4_718_720),
        ]

        // 2.35 GB per-layer embeddings — the biggest single file in the
        // bundle; `rangeSegmented` below fetches it in parallel pieces.
        let perLayerEmbedFiles: [DownloadFile] = [
            .init(remotePath: Self.splitEmbedJoinedName,
                  localPath: Self.splitEmbedJoinedName,
                  estimatedSize: 2_348_810_240)
        ]

        // Default: include multimodal (full bundle). User opts out via
        // ModelPickerView's "Include multimodal" toggle (UserDefaults).
        // Stored value = false means text-only install; default unset = true.
        let includeMM = UserDefaults.standard
            .object(forKey: ModelDownloader.includeMultimodalKey) as? Bool ?? true
        let extraFiles = coreFiles + perLayerEmbedFiles + (includeMM ? multimodalFiles : [])
        if !includeMM {
            print("[Download] gemma4-e2b: multimodal opt-out — encoders skipped (saves ~990 MB)")
        }

        let threshold: Int64 = 10_000_000  // 10 MB
        for file in chunkFiles + prefillFiles + extraFiles {
            if file.estimatedSize >= threshold {
                largeFiles.append(file)
            } else {
                smallFiles.append(file)
            }
        }

        // Large files first (sorted biggest-first), each split into Range
        // segments so every connection stays busy through the tail.
        largeFiles.sort { $0.estimatedSize > $1.estimatedSize }
        pendingFiles = Self.rangeSegmented(largeFiles + smallFiles)
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        countedBytes = [:]
        nextFileIndex = 0
    }

    /// Gemma 4 E4B layout on `mlboydaisuke/gemma-4-E4B-coreml`. Text-only
    /// decoder with a flat directory tree (no `swa/` or `prefill/` prefixes,
    /// no vision/audio towers). Produced by
    /// `conversion/build_gemma4_bundle.py --model gemma4-e4b`.
    /// Qwen3.5-0.8B CoreML layout on `mlboydaisuke/qwen3.5-0.8B-CoreML`.
    /// Default ships the INT8 palettized decode (754 MB, same semantic
    /// precision as fp16 — top-3 parity vs fp32 oracle preserved).
    /// Tokenizer is fetched by swift-transformers at runtime from
    /// `Qwen/Qwen3.5-0.8B` on HF.
    ///
    /// mlpackage structure:
    ///   qwen3_5_0_8b_decode_int8_mseq128.mlpackage/
    ///   ├── Manifest.json
    ///   └── Data/com.apple.CoreML/
    ///       ├── model.mlmodel
    ///       └── weights/weight.bin  (753 MB)
    ///
    /// Local layout after download (under `Models/qwen3.5-0.8b/`):
    ///   qwen3_5_0_8b_decode_int8_mseq128.mlpackage/...  (same structure)
    private func buildQwen35FileList() {
        let pkg = "qwen3_5_0_8b_decode_int8_mseq128.mlpackage"
        pendingFiles = [
            .init(remotePath: "\(pkg)/Manifest.json",
                  localPath: "\(pkg)/Manifest.json",
                  estimatedSize: 700),
            .init(remotePath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                  localPath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                  estimatedSize: 645_000),
            .init(remotePath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                  localPath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                  estimatedSize: 753_000_000),
        ]
        // Sort biggest-first so large weight download starts immediately.
        pendingFiles.sort { $0.estimatedSize > $1.estimatedSize }
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        countedBytes = [:]
        nextFileIndex = 0
    }

    /// Qwen3.5-2B CoreML layout on `mlboydaisuke/qwen3.5-2B-CoreML`.
    /// 4 INT8 transformer chunks + 1 raw fp16 embed sidecar under
    /// `qwen3_5_2b_decode_chunks/`:
    ///   chunk_a..c:       6 layers each (pure transformer body)
    ///   chunk_d:          6 layers + final_norm + lm_head
    ///   embed_weight.bin: raw fp16 embed_tokens, Swift mmaps directly
    /// Embed is NOT an mlpackage so CoreML doesn't dequant its 1 GB
    /// into CPU-resident memory — the mmap'd file stays in clean
    /// virtual pages and only the few rows touched per prompt page in.
    /// Every transformer chunk is ≤ 1 GB fp16, fitting iPhone's ANE
    /// single-mlprogram compile envelope.
    private func buildQwen35_2B_FileList() {
        let root = "qwen3_5_2b_decode_chunks"
        // Per-chunk weight.bin sizes measured from the INT8 palettized
        // output. chunk_d carries 1 GB of lm_head; body chunks are just
        // 6 transformer layers.
        let sizes: [(String, Int64)] = [
            ("chunk_a.mlpackage", 340_000_000),  // 6 layers
            ("chunk_b.mlpackage", 340_000_000),  // 6 layers
            ("chunk_c.mlpackage", 340_000_000),  // 6 layers
            ("chunk_d.mlpackage", 850_000_000),  // 6 layers + lm_head
        ]
        var files: [DownloadFile] = []
        for (chunk, weightSize) in sizes {
            let pkg = "\(root)/\(chunk)"
            files.append(.init(
                remotePath: "\(pkg)/Manifest.json",
                localPath: "\(pkg)/Manifest.json",
                estimatedSize: 700))
            files.append(.init(
                remotePath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                localPath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                estimatedSize: 900_000))
            files.append(.init(
                remotePath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                localPath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                estimatedSize: weightSize))
        }
        // Raw fp16 embed sidecar: 248320 × 2048 × 2 bytes ≈ 1.017 GB.
        files.append(.init(
            remotePath: "\(root)/embed_weight.bin",
            localPath: "\(root)/embed_weight.bin",
            estimatedSize: 1_017_000_000))
        pendingFiles = files
        pendingFiles.sort { $0.estimatedSize > $1.estimatedSize }
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        countedBytes = [:]
        nextFileIndex = 0
    }

    /// Qwen3-VL 2B (text-only) CoreML layout on
    /// `mlboydaisuke/qwen3-vl-2b-coreml`. 4 INT8 body chunks
    /// (6 layers each) + chunk_head (final_norm + lm_head) + raw fp16
    /// embed_weight.bin sidecar under `qwen3_vl_2b_decode_chunks/`.
    /// Same shape contract as Qwen3.5 2B v1.1.0 — Swift mmaps the
    /// embed sidecar to keep its 778 MB out of phys_footprint.
    private func buildQwen3VL2BFileList() {
        let root = "qwen3_vl_2b_decode_chunks"
        // 2B is 28 layers split into 4 body chunks × 7 layers each
        // (vs 4B's 36 layers / 6 chunks / 6 each). Per-chunk weight.bin
        // sizes measured from the INT8 palettized output.
        var sizes: [(String, Int64)] = (0..<4).map { i in
            ("chunk_\(i).mlpackage", Int64(353_000_000))  // 7 layers each
        }
        sizes.append(("chunk_head.mlpackage", 311_000_000))  // final_norm + lm_head
        // DeepStack-aware chunk_0 replacement for the vision path —
        // same weight footprint as chunk_0 (353 MB). Shipped alongside
        // the regular chunk_0 so vision can be toggled on per-prompt.
        sizes.append(("chunk_0_vision.mlpackage", 353_000_000))
        // Batched-prefill chunks (T=32) — optional, enables ~10× TTFT
        // improvement for image prompts. Same per-layer weight budget
        // as the decode chunks (they share backbone params just with a
        // T-axis added to the activations), INT8-palettized.
        for i in 0..<4 {
            sizes.append(("prefill_chunk_\(i).mlpackage", 353_000_000))
        }
        sizes.append(("prefill_chunk_0_vision.mlpackage", 353_000_000))
        var files: [DownloadFile] = []
        for (chunk, weightSize) in sizes {
            let pkg = "\(root)/\(chunk)"
            files.append(.init(
                remotePath: "\(pkg)/Manifest.json",
                localPath: "\(pkg)/Manifest.json",
                estimatedSize: 700))
            files.append(.init(
                remotePath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                localPath: "\(pkg)/Data/com.apple.CoreML/model.mlmodel",
                estimatedSize: 900_000))
            files.append(.init(
                remotePath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                localPath: "\(pkg)/Data/com.apple.CoreML/weights/weight.bin",
                estimatedSize: weightSize))
        }
        // Raw fp16 embed sidecar: 151936 × 2048 × 2 bytes ≈ 622 MB.
        files.append(.init(
            remotePath: "\(root)/embed_weight.bin",
            localPath: "\(root)/embed_weight.bin",
            estimatedSize: 622_000_000))
        // Vision encoder (ships alongside the decode chunks).
        // Input: pixel_values (3, 2, 448, 448) fp16, output: merger
        // hidden + 3 DeepStack slices. ~388 MB INT8 palettized.
        let visionPkg = "qwen3_vl_2b_vision/vision.mlpackage"
        files.append(.init(
            remotePath: "\(visionPkg)/Manifest.json",
            localPath: "\(visionPkg)/Manifest.json",
            estimatedSize: 700))
        files.append(.init(
            remotePath: "\(visionPkg)/Data/com.apple.CoreML/model.mlmodel",
            localPath: "\(visionPkg)/Data/com.apple.CoreML/model.mlmodel",
            estimatedSize: 400_000))
        files.append(.init(
            remotePath: "\(visionPkg)/Data/com.apple.CoreML/weights/weight.bin",
            localPath: "\(visionPkg)/Data/com.apple.CoreML/weights/weight.bin",
            estimatedSize: 406_000_000))
        pendingFiles = files
        pendingFiles.sort { $0.estimatedSize > $1.estimatedSize }
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        countedBytes = [:]
        nextFileIndex = 0
    }

    /// Stage 3 ship: 3-chunk merged stateful Linear bundle. HF repo
    /// `mlboydaisuke/gemma-4-E2B-stateful-coreml`. Files land under the
    /// `gemma4_e2b_stateful_chunks/` subdir of the model folder
    /// (LLMRunner expects this layout for the stateful path).
    private func buildGemma4StatefulLinearFileList() {
        let subdir = "gemma4_e2b_stateful_chunks"
        func mlc(_ name: String, weightSize: Int64,
                 milSize: Int64) -> [DownloadFile] {
            [.init(remotePath: "\(name).mlmodelc/weights/weight.bin",
                   localPath: "\(subdir)/\(name).mlmodelc/weights/weight.bin",
                   estimatedSize: weightSize),
             .init(remotePath: "\(name).mlmodelc/coremldata.bin",
                   localPath: "\(subdir)/\(name).mlmodelc/coremldata.bin",
                   estimatedSize: 1_200),
             .init(remotePath: "\(name).mlmodelc/model.mil",
                   localPath: "\(subdir)/\(name).mlmodelc/model.mil",
                   estimatedSize: milSize),
             .init(remotePath: "\(name).mlmodelc/metadata.json",
                   localPath: "\(subdir)/\(name).mlmodelc/metadata.json",
                   estimatedSize: 22_000),
             .init(remotePath: "\(name).mlmodelc/analytics/coremldata.bin",
                   localPath: "\(subdir)/\(name).mlmodelc/analytics/coremldata.bin",
                   estimatedSize: 250)]
        }

        let chunkFiles =
              mlc("chunk_1", weightSize: 155_484_416, milSize: 716_249)
            + mlc("chunk_2", weightSize: 459_299_328, milSize: 1_088_904)
            + mlc("chunk_3", weightSize: 527_440_000, milSize: 494_464)

        let extraFiles: [DownloadFile] = [
            .init(remotePath: "embed_tokens_q8.bin",
                  localPath: "\(subdir)/embed_tokens_q8.bin",
                  estimatedSize: 402_653_184),
            .init(remotePath: "embed_tokens_scales.bin",
                  localPath: "\(subdir)/embed_tokens_scales.bin",
                  estimatedSize: 524_288),
            .init(remotePath: "embed_tokens_per_layer_q8.bin",
                  localPath: "\(subdir)/embed_tokens_per_layer_q8.bin",
                  estimatedSize: 2_348_810_240),
            .init(remotePath: "embed_tokens_per_layer_scales.bin",
                  localPath: "\(subdir)/embed_tokens_per_layer_scales.bin",
                  estimatedSize: 524_288),
            .init(remotePath: "per_layer_projection.bin",
                  localPath: "\(subdir)/per_layer_projection.bin",
                  estimatedSize: 27_525_120),
            .init(remotePath: "per_layer_norm_weight.bin",
                  localPath: "\(subdir)/per_layer_norm_weight.bin",
                  estimatedSize: 1_024),
            .init(remotePath: "cos_sliding.npy",
                  localPath: "\(subdir)/cos_sliding.npy",
                  estimatedSize: 4_194_432),
            .init(remotePath: "sin_sliding.npy",
                  localPath: "\(subdir)/sin_sliding.npy",
                  estimatedSize: 4_194_432),
            .init(remotePath: "cos_full.npy",
                  localPath: "\(subdir)/cos_full.npy",
                  estimatedSize: 8_388_736),
            .init(remotePath: "sin_full.npy",
                  localPath: "\(subdir)/sin_full.npy",
                  estimatedSize: 8_388_736),
            .init(remotePath: "model_config.json",
                  localPath: "\(subdir)/model_config.json",
                  estimatedSize: 700),
            .init(remotePath: "hf_model/tokenizer.json",
                  localPath: "\(subdir)/hf_model/tokenizer.json",
                  estimatedSize: 32_169_626),
            .init(remotePath: "hf_model/tokenizer_config.json",
                  localPath: "\(subdir)/hf_model/tokenizer_config.json",
                  estimatedSize: 2_100),
            .init(remotePath: "hf_model/config.json",
                  localPath: "\(subdir)/hf_model/config.json",
                  estimatedSize: 5_000),
        ]

        var largeFiles: [DownloadFile] = []
        var smallFiles: [DownloadFile] = []
        let threshold: Int64 = 10_000_000
        for file in chunkFiles + extraFiles {
            if file.estimatedSize >= threshold {
                largeFiles.append(file)
            } else {
                smallFiles.append(file)
            }
        }
        largeFiles.sort { $0.estimatedSize > $1.estimatedSize }
        pendingFiles = largeFiles + smallFiles
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        countedBytes = [:]
        nextFileIndex = 0
    }

    private func buildE4BFileList() {
        func mlc(_ name: String, weightSize: Int64) -> [DownloadFile] {
            [.init(remotePath: "\(name).mlmodelc/weights/weight.bin",
                   localPath: "\(name).mlmodelc/weights/weight.bin", estimatedSize: weightSize),
             .init(remotePath: "\(name).mlmodelc/coremldata.bin",
                   localPath: "\(name).mlmodelc/coremldata.bin", estimatedSize: 1_200),
             .init(remotePath: "\(name).mlmodelc/model.mil",
                   localPath: "\(name).mlmodelc/model.mil", estimatedSize: 1_250_000),
             .init(remotePath: "\(name).mlmodelc/metadata.json",
                   localPath: "\(name).mlmodelc/metadata.json", estimatedSize: 25_000),
             .init(remotePath: "\(name).mlmodelc/analytics/coremldata.bin",
                   localPath: "\(name).mlmodelc/analytics/coremldata.bin", estimatedSize: 250)]
        }

        // Chunk weight sizes (observed from the shipping bundle; larger than E2B
        // because hidden=2560 and intermediate=10240 doubles the MLP wide).
        let chunkFiles: [DownloadFile] =
              mlc("chunk1", weightSize: 586_000_000)   // 558.8 MB
            + mlc("chunk2", weightSize: 572_000_000)   // 545.7 MB
            + mlc("chunk3", weightSize: 413_000_000)   // 393.6 MB
            + mlc("chunk4", weightSize: 754_000_000)   // 718.9 MB (includes LM head)

        let extraFiles: [DownloadFile] = [
            .init(remotePath: "model_config.json", localPath: "model_config.json", estimatedSize: 800),
            .init(remotePath: "hf_model/tokenizer.json", localPath: "hf_model/tokenizer.json", estimatedSize: 32_200_000),
            .init(remotePath: "hf_model/tokenizer_config.json", localPath: "hf_model/tokenizer_config.json", estimatedSize: 2_200),
            .init(remotePath: "hf_model/config.json", localPath: "hf_model/config.json", estimatedSize: 5_200),
            .init(remotePath: "hf_model/generation_config.json", localPath: "hf_model/generation_config.json", estimatedSize: 300),
            .init(remotePath: "embed_tokens_q8.bin", localPath: "embed_tokens_q8.bin", estimatedSize: 671_088_640),
            .init(remotePath: "embed_tokens_scales.bin", localPath: "embed_tokens_scales.bin", estimatedSize: 524_288),
            .init(remotePath: "embed_tokens_per_layer_q8.bin", localPath: "embed_tokens_per_layer_q8.bin", estimatedSize: 2_825_912_320),
            .init(remotePath: "embed_tokens_per_layer_scales.bin", localPath: "embed_tokens_per_layer_scales.bin", estimatedSize: 524_288),
            .init(remotePath: "per_layer_projection.bin", localPath: "per_layer_projection.bin", estimatedSize: 55_050_240),
            .init(remotePath: "per_layer_norm_weight.bin", localPath: "per_layer_norm_weight.bin", estimatedSize: 512),
            .init(remotePath: "cos_sliding.npy", localPath: "cos_sliding.npy", estimatedSize: 2_097_280),
            .init(remotePath: "sin_sliding.npy", localPath: "sin_sliding.npy", estimatedSize: 2_097_280),
            .init(remotePath: "cos_full.npy", localPath: "cos_full.npy", estimatedSize: 4_194_432),
            .init(remotePath: "sin_full.npy", localPath: "sin_full.npy", estimatedSize: 4_194_432),
        ]

        var largeFiles: [DownloadFile] = []
        var smallFiles: [DownloadFile] = []
        let threshold: Int64 = 10_000_000
        for file in chunkFiles + extraFiles {
            if file.estimatedSize >= threshold {
                largeFiles.append(file)
            } else {
                smallFiles.append(file)
            }
        }
        largeFiles.sort { $0.estimatedSize > $1.estimatedSize }
        pendingFiles = largeFiles + smallFiles
        totalBytesForAllFiles = pendingFiles.reduce(0) { $0 + $1.estimatedSize }
        completedBytes = 0
        countedBytes = [:]
        nextFileIndex = 0
    }

    // MARK: - ZIP

    private func unzipFile(_ zipURL: URL, to destDir: URL) throws {
        #if os(macOS)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        proc.arguments = ["-xk", zipURL.path, destDir.path]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()
        #else
        // iOS (device + simulator), visionOS, tvOS, watchOS — Foundation's
        // `Process` is macOS-only, so unzip ourselves via the ZIP central
        // directory. Previously this branch used `#if targetEnvironment(simulator)
        // || os(macOS)` which broke iOS Simulator builds with "cannot find
        // 'Process' in scope".
        try extractZipNative(from: zipURL, to: destDir)
        #endif
    }

    #if !os(macOS)
    private func extractZipNative(from zipURL: URL, to destDir: URL) throws {
        let data = try Data(contentsOf: zipURL)
        guard data.count > 22 else { throw DownloadError.extractionFailed }
        var eocdOffset = data.count - 22
        while eocdOffset >= 0 {
            if data[eocdOffset] == 0x50 && data[eocdOffset+1] == 0x4B &&
               data[eocdOffset+2] == 0x05 && data[eocdOffset+3] == 0x06 { break }
            eocdOffset -= 1
        }
        guard eocdOffset >= 0 else { throw DownloadError.extractionFailed }
        let cdOffset = Int(data[eocdOffset+16..<eocdOffset+20].withUnsafeBytes { $0.load(as: UInt32.self) })
        let cdCount = Int(data[eocdOffset+10..<eocdOffset+12].withUnsafeBytes { $0.load(as: UInt16.self) })
        var pos = cdOffset
        for _ in 0..<cdCount {
            guard data[pos] == 0x50, data[pos+1] == 0x4B else { break }
            let uncompSize = Int(data[pos+24..<pos+28].withUnsafeBytes { $0.load(as: UInt32.self) })
            let nameLen = Int(data[pos+28..<pos+30].withUnsafeBytes { $0.load(as: UInt16.self) })
            let extraLen = Int(data[pos+30..<pos+32].withUnsafeBytes { $0.load(as: UInt16.self) })
            let commentLen = Int(data[pos+32..<pos+34].withUnsafeBytes { $0.load(as: UInt16.self) })
            let localOffset = Int(data[pos+42..<pos+46].withUnsafeBytes { $0.load(as: UInt32.self) })
            let name = String(data: data[pos+46..<pos+46+nameLen], encoding: .utf8) ?? ""
            let destPath = destDir.appendingPathComponent(name)
            if name.hasSuffix("/") {
                try fileManager.createDirectory(at: destPath, withIntermediateDirectories: true)
            } else {
                try fileManager.createDirectory(at: destPath.deletingLastPathComponent(), withIntermediateDirectories: true)
                let lnl = Int(data[localOffset+26..<localOffset+28].withUnsafeBytes { $0.load(as: UInt16.self) })
                let lel = Int(data[localOffset+28..<localOffset+30].withUnsafeBytes { $0.load(as: UInt16.self) })
                let ds = localOffset + 30 + lnl + lel
                try Data(data[ds..<ds+uncompSize]).write(to: destPath)
            }
            pos += 46 + nameLen + extraLen + commentLen
        }
    }
    #endif

    /// Files inside an mlmodelc that CoreML doesn't require to load the model.
    /// A 404 on these shouldn't abort the whole download — the upload process
    /// for W8A8 has historically produced HF repos missing `coremldata.bin`
    /// (since fixed) and `metadata.json` (still missing in some uploads); the
    /// latter is purely descriptive. Keep this list conservative — anything
    /// not listed here is treated as required.
    private func isOptionalMlmodelcFile(_ localPath: String) -> Bool {
        // metadata.json and analytics/coremldata.bin are descriptive and
        // missing from some historical uploads — always optional.
        if localPath.hasSuffix(".mlmodelc/metadata.json")
            || localPath.hasSuffix(".mlmodelc/analytics/coremldata.bin") {
            return true
        }
        // 3-chunk variant files are entirely optional — they enable the
        // LLM_3CHUNK=1 opt-in path but are not needed for the default
        // 4-chunk decoder. If HF doesn't yet have them (older snapshot),
        // skip so existing bundles still install cleanly.
        let optionalMlmodelcPrefixes = [
            "chunk2_3way.mlmodelc/",
            "chunk3_3way.mlmodelc/",
        ]
        return optionalMlmodelcPrefixes.contains { localPath.hasPrefix($0) }
    }

    private var modelsDirectory: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("Models")
    }
}

// MARK: - URLSession Delegate

extension ModelDownloader: URLSessionDownloadDelegate {
    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didFinishDownloadingTo location: URL) {
        guard let localPath = downloadTask.taskDescription,
              let dest = destDir else { return }

        // HTTP status check: URLSessionDownloadTask writes the response body
        // to `location` regardless of status code. A 404 from HuggingFace is
        // a short HTML page ("Entry not found") that would otherwise be
        // saved verbatim and later fail a checksum / signature check with
        // a misleading error. Catch it here with a clear error.
        if let http = downloadTask.response as? HTTPURLResponse, http.statusCode >= 400 {
            // Read a small excerpt of the body for the error message.
            let snippet: String = (try? String(contentsOf: location, encoding: .utf8))?
                .prefix(200).trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            try? fileManager.removeItem(at: location)
            let url = downloadTask.originalRequest?.url?.absoluteString ?? "(unknown)"
            let taskId = Self.key(session, downloadTask)

            // Optional files: metadata.json and analytics/coremldata.bin inside
            // an mlmodelc are descriptive, not functional — CoreML loads fine
            // without them. Treat 404 on these as non-fatal so a slightly
            // incomplete HF upload doesn't abort the entire download.
            if http.statusCode == 404 && isOptionalMlmodelcFile(localPath) {
                print("[Download] Skipping optional missing file: \(localPath) (404)")
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.unparkRestoredDownloadIfNeeded()
                    self.activeDownloadTasks.removeValue(forKey: taskId)
                    self.activeTaskFileIndex.removeValue(forKey: taskId)
                    self.activeTaskBytes.removeValue(forKey: taskId)
                    self.fillDownloadSlots()
                }
                return
            }

            // Rate limiting and server hiccups are transient — re-queue
            // through the bounded retry path instead of failing the install.
            // Range segments multiply the request count, so this matters.
            if http.statusCode == 429 || http.statusCode >= 500 {
                let err = NSError(domain: "CoreMLLLM.ModelDownloader", code: http.statusCode,
                                  userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode) fetching \(url)"])
                DispatchQueue.main.async { [weak self] in
                    self?.handleTaskFailure(taskId: taskId, error: err)
                }
                return
            }

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.activeDownloadTasks.removeValue(forKey: taskId)
                self.activeTaskFileIndex.removeValue(forKey: taskId)
                self.activeTaskBytes.removeValue(forKey: taskId)
                self.status = "Error: HTTP \(http.statusCode) for \(localPath)"
                // Surface the actual server message so the user sees *why* it failed
                // (e.g., "Entry not found. Please check the file URL.").
                let err = NSError(domain: "CoreMLLLM.ModelDownloader", code: http.statusCode,
                                  userInfo: [
                                    NSLocalizedDescriptionKey:
                                        "HTTP \(http.statusCode) fetching \(url). " +
                                        (snippet.isEmpty ? "" : "Server: \(snippet)"),
                                  ])
                self.isDownloading = false
                self.downloadingModelId = nil
                self.downloadContinuation?.resume(throwing: err)
                self.downloadContinuation = nil
            }
            return
        }

        // Range segment: the body must be exactly the bytes asked for. A
        // server or redirect hop that ignores Range answers 200 with the
        // whole file, which would corrupt the join — accept only a 206 whose
        // Content-Range matches the request and the body length, else retry.
        if let requested = downloadTask.originalRequest?.value(forHTTPHeaderField: "Range") {
            let http = downloadTask.response as? HTTPURLResponse
            let bodySize = (try? fileManager.attributesOfItem(atPath: location.path))?[.size] as? Int64 ?? -1
            if !Self.isValidRangeResponse(http, requested: requested, bodySize: bodySize) {
                try? fileManager.removeItem(at: location)
                let taskId = Self.key(session, downloadTask)
                let contentRange = http?.value(forHTTPHeaderField: "Content-Range") ?? "none"
                let err = NSError(domain: "CoreMLLLM.ModelDownloader", code: -5, userInfo: [
                    NSLocalizedDescriptionKey:
                        "Unexpected response for \(localPath) (\(requested)): HTTP "
                        + "\(http?.statusCode ?? 0), Content-Range \(contentRange), \(bodySize) bytes",
                ])
                DispatchQueue.main.async { [weak self] in
                    self?.handleTaskFailure(taskId: taskId, error: err)
                }
                return
            }
        }

        let destFile = dest.appendingPathComponent(localPath)

        // Must move synchronously before this method returns
        try? fileManager.createDirectory(at: destFile.deletingLastPathComponent(),
                                          withIntermediateDirectories: true)
        try? fileManager.removeItem(at: destFile)
        try? fileManager.moveItem(at: location, to: destFile)

        let downloadedSize = (try? fileManager.attributesOfItem(atPath: destFile.path))?[.size] as? Int64 ?? 0
        let isZip = localPath == "__archive.zip"
        let taskId = Self.key(session, downloadTask)

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.unparkRestoredDownloadIfNeeded()
            let fileIndex = self.activeTaskFileIndex[taskId]
                ?? self.pendingFiles.firstIndex { $0.localPath == localPath }
            self.activeDownloadTasks.removeValue(forKey: taskId)
            self.activeTaskFileIndex.removeValue(forKey: taskId)
            self.activeTaskBytes.removeValue(forKey: taskId)
            if let fileIndex { self.count(fileIndex, bytes: downloadedSize) }
            self.updateProgress()

            if isZip {
                self.status = "Extracting..."
                DispatchQueue.global(qos: .userInitiated).async {
                    try? self.unzipFile(destFile, to: dest)
                    try? self.fileManager.removeItem(at: destFile)
                    DispatchQueue.main.async {
                        self.fillDownloadSlots()
                    }
                }
            } else {
                self.fillDownloadSlots()
            }
        }
    }

    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                           didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                           totalBytesExpectedToWrite: Int64) {
        let taskId = Self.key(session, downloadTask)
        DispatchQueue.main.async { [weak self] in
            // Only tasks we track: a late callback from a cancelled or
            // not-yet-adopted task would park phantom bytes in the total.
            guard let self, self.activeDownloadTasks[taskId] != nil else { return }
            self.activeTaskBytes[taskId] = totalBytesWritten
            self.updateProgress()
        }
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }
        let taskId = Self.key(session, task)
        if (error as NSError).code == NSURLErrorCancelled {
            DispatchQueue.main.async { [weak self] in
                self?.activeDownloadTasks.removeValue(forKey: taskId)
                self?.activeTaskFileIndex.removeValue(forKey: taskId)
                self?.activeTaskBytes.removeValue(forKey: taskId)
            }
            return
        }

        DispatchQueue.main.async { [weak self] in
            self?.handleTaskFailure(taskId: taskId, error: error)
        }
    }

    public func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        guard let identifier = session.configuration.identifier else { return }
        DispatchQueue.main.async { [weak self] in
            self?.backgroundCompletionHandlers.removeValue(forKey: identifier)?()
        }
    }
}

public enum DownloadError: LocalizedError {
    case invalidURL, extractionFailed
    public var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid download URL"
        case .extractionFailed: return "Failed to extract model"
        }
    }
}
