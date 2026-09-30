import XCTest
@testable import CoreMLLLM

/// Range-segment planning, 206 validation and the segment join.
/// No network; file I/O stays under `NSTemporaryDirectory()`.
final class ModelDownloaderRangeTests: XCTestCase {

    private typealias File = ModelDownloader.DownloadFile
    private let seg = ModelDownloader.rangeSegmentSize

    // MARK: - rangeSegmented

    func testSmallFilesStayWhole() {
        let files = [File(remotePath: "a", localPath: "a", estimatedSize: 2 * seg - 1)]
        XCTAssertEqual(ModelDownloader.rangeSegmented(files), files)
    }

    /// Contiguous, gap-free, in order; the last segment is open-ended and
    /// carries the remainder.
    func testSegmentsTileTheFile() {
        let size = 5 * seg + 12_345
        let segments = ModelDownloader.rangeSegmented(
            [File(remotePath: "r/w.bin", localPath: "w.bin", estimatedSize: size)])
        XCTAssertEqual(segments.count, 5)
        XCTAssertEqual(segments.map(\.localPath),
                       ["w.bin.seg00", "w.bin.seg01", "w.bin.seg02", "w.bin.seg03", "w.bin.seg04"])
        XCTAssertTrue(segments.allSatisfy { $0.remotePath == "r/w.bin" && $0.joinTarget == "w.bin" })
        for (i, s) in segments.enumerated() {
            XCTAssertEqual(s.rangeStart, Int64(i) * seg)
        }
        for s in segments.dropLast() {
            XCTAssertEqual(s.rangeEnd, s.rangeStart! + seg - 1)
            XCTAssertEqual(s.estimatedSize, seg)
        }
        XCTAssertNil(segments.last!.rangeEnd)
        XCTAssertEqual(segments.reduce(0) { $0 + $1.estimatedSize }, size)
    }

    /// The shipped per-layer embed: 2,348,810,240 bytes = exactly 35 segments.
    func testEmbedSegmentCount() {
        let segments = ModelDownloader.rangeSegmented(
            [File(remotePath: "e", localPath: "e", estimatedSize: 2_348_810_240)])
        XCTAssertEqual(segments.count, 35)
        XCTAssertEqual(segments.last!.rangeStart, 34 * seg)
    }

    /// States persisted before segmentation have no range keys.
    func testLegacyPersistedFileDecodes() throws {
        let json = #"{"remotePath":"x","localPath":"x","estimatedSize":7}"#
        let f = try JSONDecoder().decode(File.self, from: Data(json.utf8))
        XCTAssertNil(f.rangeStart)
        XCTAssertNil(f.rangeEnd)
        XCTAssertNil(f.joinTarget)
    }

    // MARK: - isValidRangeResponse

    private func response(_ status: Int, _ contentRange: String?) -> HTTPURLResponse {
        var headers: [String: String] = [:]
        if let contentRange { headers["Content-Range"] = contentRange }
        return HTTPURLResponse(url: URL(string: "https://example.com/f")!, statusCode: status,
                               httpVersion: "HTTP/2", headerFields: headers)!
    }

    func testClosedRangeAccepted() {
        XCTAssertTrue(ModelDownloader.isValidRangeResponse(
            response(206, "bytes 100-199/1000"), requested: "bytes=100-199", bodySize: 100))
    }

    func testOpenEndedRangeAccepted() {
        XCTAssertTrue(ModelDownloader.isValidRangeResponse(
            response(206, "bytes 900-999/1000"), requested: "bytes=900-", bodySize: 100))
    }

    /// nsurlsessiond resumed the task itself after a drop: the last response
    /// covers only the tail, but the stitched body is the whole span.
    func testDaemonResumedRangeAccepted() {
        XCTAssertTrue(ModelDownloader.isValidRangeResponse(
            response(206, "bytes 150-199/1000"), requested: "bytes=100-199", bodySize: 100))
        XCTAssertTrue(ModelDownloader.isValidRangeResponse(
            response(206, "bytes 950-999/1000"), requested: "bytes=900-", bodySize: 100))
        // Only the tail on disk — not stitched.
        XCTAssertFalse(ModelDownloader.isValidRangeResponse(
            response(206, "bytes 150-199/1000"), requested: "bytes=100-199", bodySize: 50))
    }

    /// A server that ignores Range sends the whole file with a 200.
    func testFullBodyRejected() {
        XCTAssertFalse(ModelDownloader.isValidRangeResponse(
            response(200, nil), requested: "bytes=100-199", bodySize: 1000))
    }

    func testMismatchesRejected() {
        // Wrong start.
        XCTAssertFalse(ModelDownloader.isValidRangeResponse(
            response(206, "bytes 0-99/1000"), requested: "bytes=100-199", bodySize: 100))
        // Short range.
        XCTAssertFalse(ModelDownloader.isValidRangeResponse(
            response(206, "bytes 100-149/1000"), requested: "bytes=100-199", bodySize: 50))
        // Truncated body.
        XCTAssertFalse(ModelDownloader.isValidRangeResponse(
            response(206, "bytes 100-199/1000"), requested: "bytes=100-199", bodySize: 99))
        // Open-ended request that stops before EOF.
        XCTAssertFalse(ModelDownloader.isValidRangeResponse(
            response(206, "bytes 900-949/1000"), requested: "bytes=900-", bodySize: 50))
        // Missing header.
        XCTAssertFalse(ModelDownloader.isValidRangeResponse(
            response(206, nil), requested: "bytes=100-199", bodySize: 100))
    }

    // MARK: - joinParts

    func testJoinConcatenatesInOrderAndConsumesParts() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("range-join-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let chunks: [Data] = (0..<4).map { i in
            Data((0..<(1000 + i * 37)).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ i) })
        }
        let target = dir.appendingPathComponent("w.bin")
        let parts = chunks.indices.map { dir.appendingPathComponent(String(format: "w.bin.seg%02d", $0)) }
        for (url, data) in zip(parts, chunks) { try data.write(to: url) }
        // A stale temp from a crashed join must not leak into the output.
        try Data([9, 9, 9]).write(to: target.appendingPathExtension("joining"))

        XCTAssertNil(ModelDownloader.joinParts(parts, into: target))
        XCTAssertEqual(try Data(contentsOf: target), chunks.reduce(Data(), +))
        XCTAssertFalse(parts.contains { FileManager.default.fileExists(atPath: $0.path) })
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: target.appendingPathExtension("joining").path))
    }

    func testJoinWithMissingPartFails() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("range-join-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let present = dir.appendingPathComponent("w.bin.seg00")
        try Data([1, 2, 3]).write(to: present)
        let missing = dir.appendingPathComponent("w.bin.seg01")

        XCTAssertNotNil(ModelDownloader.joinParts([present, missing],
                                                  into: dir.appendingPathComponent("w.bin")))
        XCTAssertTrue(FileManager.default.fileExists(atPath: present.path),
                      "a refused join must not consume parts")
    }
}
