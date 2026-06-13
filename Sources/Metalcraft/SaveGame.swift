import Foundation

/// A world folder on disk, as shown in the Select World list.
struct SaveSummary {
    var dir: URL
    var name: String
    var lastPlayed: Date
}

struct FurnaceSave: Codable {
    var pos: BlockPos
    var input: ItemStack?
    var fuel: ItemStack?
    var output: ItemStack?
    var burnLeft: Float
    var burnTotal: Float
    var cook: Float
}

/// Everything except chunk blocks, written as level.json in the save folder.
struct LevelData: Codable {
    var name: String
    var seed: UInt64
    var timeOfDay: Float
    var playerPos: [Float]
    var playerVel: [Float]
    var yaw: Float
    var pitch: Float
    var health: Float
    var flying: Bool
    var inventory: [ItemStack?]
    var selectedSlot: Int
    var furnaces: [FurnaceSave]
    var plates: [BlockPos]
    var lastPlayed: Date
}

/// Disk layout: ~/Library/Application Support/Metalcraft/saves/<World N>/
/// holds level.json plus chunks.bin (zlib), which carries only the chunks
/// that diverged from pure generation — everything else regenerates from
/// the seed on load.
enum SaveIO {
    private static let cellsPerChunk = World.chunkSize * World.height * World.chunkSize

    static var savesDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Metalcraft/saves", isDirectory: true)
    }

    static func list() -> [SaveSummary] {
        let dirs = (try? FileManager.default.contentsOfDirectory(
            at: savesDir, includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles)) ?? []
        return dirs
            .compactMap { dir in
                readLevel(dir).map { SaveSummary(dir: dir, name: $0.name, lastPlayed: $0.lastPlayed) }
            }
            .sorted { $0.lastPlayed > $1.lastPlayed }
    }

    /// Make a fresh "World N" folder with the first unused number.
    static func createSave() -> (dir: URL, name: String)? {
        let fm = FileManager.default
        let taken = Set((try? fm.contentsOfDirectory(atPath: savesDir.path)) ?? [])
        var n = 1
        while taken.contains("World \(n)") { n += 1 }
        let name = "World \(n)"
        let dir = savesDir.appendingPathComponent(name, isDirectory: true)
        guard (try? fm.createDirectory(at: dir, withIntermediateDirectories: true)) != nil else {
            return nil
        }
        return (dir, name)
    }

    static func delete(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
    }

    static func writeLevel(_ level: LevelData, to dir: URL) {
        guard let data = try? JSONEncoder().encode(level) else { return }
        try? data.write(to: dir.appendingPathComponent("level.json"), options: .atomic)
    }

    static func readLevel(_ dir: URL) -> LevelData? {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("level.json"))
        else { return nil }
        return try? JSONDecoder().decode(LevelData.self, from: data)
    }

    /// chunks.bin: UInt32 count, then per chunk Int32 x, Int32 z (little
    /// endian), raw block bytes, raw light bytes; the whole stream zlibbed.
    static func writeChunks(_ chunks: [(coord: ChunkCoord, chunk: Chunk)], to dir: URL) {
        var data = Data(capacity: 4 + chunks.count * (8 + cellsPerChunk * 2))
        func append32(_ v: Int32) {
            withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
        }
        append32(Int32(chunks.count))
        for (c, chunk) in chunks {
            append32(Int32(c.x))
            append32(Int32(c.z))
            data.append(contentsOf: chunk.blocks)
            data.append(contentsOf: chunk.light)
        }
        let payload = ((try? (data as NSData).compressed(using: .zlib)) as Data?) ?? data
        try? payload.write(to: dir.appendingPathComponent("chunks.bin"), options: .atomic)
    }

    static func readChunks(_ dir: URL) -> [ChunkCoord: Chunk] {
        guard let raw = try? Data(contentsOf: dir.appendingPathComponent("chunks.bin"))
        else { return [:] }
        let data = ((try? (raw as NSData).decompressed(using: .zlib)) as Data?) ?? raw
        let bytes = [UInt8](data)
        var off = 0
        func read32() -> Int32? {
            guard off + 4 <= bytes.count else { return nil }
            defer { off += 4 }
            return Int32(bytes[off]) | (Int32(bytes[off + 1]) << 8)
                | (Int32(bytes[off + 2]) << 16) | (Int32(bytes[off + 3]) << 24)
        }
        guard let count = read32() else { return [:] }
        var out: [ChunkCoord: Chunk] = [:]
        for _ in 0..<count {
            guard let x = read32(), let z = read32(),
                  off + cellsPerChunk * 2 <= bytes.count else { break }
            let chunk = Chunk()
            chunk.blocks = Array(bytes[off..<(off + cellsPerChunk)])
            off += cellsPerChunk
            chunk.light = Array(bytes[off..<(off + cellsPerChunk)])
            off += cellsPerChunk
            out[ChunkCoord(x: Int(x), z: Int(z))] = chunk
        }
        return out
    }
}
