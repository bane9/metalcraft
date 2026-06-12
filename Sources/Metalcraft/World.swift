import simd

enum Block: UInt8 {
    case air = 0, grass, dirt, stone, sand, wood, leaves, bedrock, snow, cactus
    case planks = 10, cobblestone, coalOre, ironOre, goldOre, diamondOre
    case craftingTable = 20, furnace, furnaceLit
    // water cases stay contiguous at the end: source, then flowing levels 1-7
    case water = 100, flow1, flow2, flow3, flow4, flow5, flow6, flow7
}

extension Block {
    var isWater: Bool { rawValue >= Block.water.rawValue }
    /// 0 = source (full), 1-7 = flowing, weaker as the number rises
    var waterLevel: Int { Int(rawValue) - Int(Block.water.rawValue) }
    static func flowing(_ level: Int) -> Block {
        Block(rawValue: Block.water.rawValue + UInt8(max(0, min(7, level))))!
    }
}

enum Biome {
    case plains, forest, desert, snowy, mountains
}

struct ChunkCoord: Hashable {
    var x: Int
    var z: Int
}

struct BlockPos: Hashable {
    var x: Int
    var y: Int
    var z: Int
}

final class Chunk {
    var blocks = [UInt8](repeating: 0, count: World.chunkSize * World.height * World.chunkSize)

    @inline(__always)
    static func index(_ lx: Int, _ y: Int, _ lz: Int) -> Int {
        (y * World.chunkSize + lz) * World.chunkSize + lx
    }
}

/// Infinite voxel world: chunks are generated on demand from a seed and kept
/// in memory once created, so player edits persist when chunks scroll out of
/// render range.
final class World {
    static let chunkSize = 16
    static let height = 96
    static let waterLevel = 22  // terrain below this is flooded; beaches just above
    static let stoneLine = 44   // mountains turn to bare stone above this
    static let snowLine = 58    // and to snow caps above this

    private(set) var chunks: [ChunkCoord: Chunk] = [:]
    var dirtyChunks = Set<ChunkCoord>()
    private var pendingWater = Set<BlockPos>()
    /// Immutable and self-contained, so background workers can capture it by
    /// value and build chunks off the main thread without any locking.
    private(set) var generator = TerrainGen(seed: 1)

    @inline(__always)
    static func chunkCoord(blockX x: Int, blockZ z: Int) -> ChunkCoord {
        ChunkCoord(x: x >> 4, z: z >> 4) // arithmetic shift = floor division for negatives
    }

    func reset(seed: UInt64) {
        generator = TerrainGen(seed: seed)
        chunks.removeAll()
        dirtyChunks.removeAll()
        pendingWater.removeAll()
    }

    func isGenerated(_ c: ChunkCoord) -> Bool { chunks[c] != nil }

    func generateChunk(_ c: ChunkCoord) {
        guard chunks[c] == nil else { return }
        chunks[c] = generator.buildChunk(c)
    }

    /// Adopt a chunk built off-thread; first writer wins.
    func insertChunk(_ chunk: Chunk, at c: ChunkCoord) {
        guard chunks[c] == nil else { return }
        chunks[c] = chunk
    }

    /// Copy-on-write handle to a chunk's block storage for snapshotting.
    func chunkBlocks(_ c: ChunkCoord) -> [UInt8]? { chunks[c]?.blocks }

    func block(_ x: Int, _ y: Int, _ z: Int) -> Block {
        guard y >= 0 && y < Self.height,
              let chunk = chunks[Self.chunkCoord(blockX: x, blockZ: z)] else { return .air }
        return Block(rawValue: chunk.blocks[Chunk.index(x & 15, y, z & 15)]) ?? .air
    }

    func isSolid(_ x: Int, _ y: Int, _ z: Int) -> Bool {
        if y < 0 { return true }
        if y >= Self.height { return false }
        guard let chunk = chunks[Self.chunkCoord(blockX: x, blockZ: z)] else {
            return true // ungenerated terrain blocks movement until it streams in
        }
        let b = Block(rawValue: chunk.blocks[Chunk.index(x & 15, y, z & 15)]) ?? .air
        return b != .air && !b.isWater
    }

    func setBlock(_ x: Int, _ y: Int, _ z: Int, _ b: Block) {
        guard y >= 0 && y < Self.height else { return }
        let cc = Self.chunkCoord(blockX: x, blockZ: z)
        guard let chunk = chunks[cc] else { return }
        chunk.blocks[Chunk.index(x & 15, y, z & 15)] = b.rawValue
        dirtyChunks.insert(cc)
        let lx = x & 15, lz = z & 15
        if lx == 0 { dirtyChunks.insert(ChunkCoord(x: cc.x - 1, z: cc.z)) }
        if lx == 15 { dirtyChunks.insert(ChunkCoord(x: cc.x + 1, z: cc.z)) }
        if lz == 0 { dirtyChunks.insert(ChunkCoord(x: cc.x, z: cc.z - 1)) }
        if lz == 15 { dirtyChunks.insert(ChunkCoord(x: cc.x, z: cc.z + 1)) }
        // any edit can change how water wants to flow here and around here
        pendingWater.insert(BlockPos(x: x, y: y, z: z))
        pendingWater.insert(BlockPos(x: x + 1, y: y, z: z))
        pendingWater.insert(BlockPos(x: x - 1, y: y, z: z))
        pendingWater.insert(BlockPos(x: x, y: y + 1, z: z))
        pendingWater.insert(BlockPos(x: x, y: y - 1, z: z))
        pendingWater.insert(BlockPos(x: x, y: y, z: z + 1))
        pendingWater.insert(BlockPos(x: x, y: y, z: z - 1))
    }

    func surfaceHeight(_ x: Int, _ z: Int) -> Int {
        for y in stride(from: Self.height - 1, through: 0, by: -1) where block(x, y, z) != .air {
            return y
        }
        return 0
    }

    // MARK: - Water flow simulation
    // Cell-centric, Minecraft-style: each scheduled cell recomputes what it
    // should contain from its neighbors. Convergent — every change schedules
    // the neighbors, and cells stop changing once levels settle.

    func tickWater() {
        guard !pendingWater.isEmpty else { return }
        var queue = Array(pendingWater)
        pendingWater.removeAll()
        if queue.count > 4096 {
            pendingWater = Set(queue[4096...])
            queue = Array(queue[..<4096])
        }
        for p in queue {
            guard p.y >= 0 && p.y < Self.height,
                  isGenerated(Self.chunkCoord(blockX: p.x, blockZ: p.z)) else { continue }
            let current = block(p.x, p.y, p.z)
            guard current == .air || current.isWater else { continue }
            let desired = desiredWaterState(p.x, p.y, p.z, current: current)
            if desired != current {
                setBlock(p.x, p.y, p.z, desired) // marks meshes dirty + reschedules neighbors
            }
        }
    }

    private func desiredWaterState(_ x: Int, _ y: Int, _ z: Int, current: Block) -> Block {
        if current == .water { return .water } // sources persist until mined

        let above = block(x, y + 1, z)
        let belowSolid = isSolid(x, y - 1, z)
        let below = block(x, y - 1, z)

        var inflow = Int.max
        var adjacentSources = 0
        if above.isWater { inflow = 1 } // falling water is nearly full

        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let n = block(x + dx, y, z + dz)
            guard n.isWater else { continue }
            if n == .water { adjacentSources += 1 }
            // a neighbor only pushes sideways if it can't just fall instead
            let nSupported = isSolid(x + dx, y - 1, z + dz) || block(x + dx, y - 1, z + dz).isWater
            if nSupported && n.waterLevel < 7 {
                inflow = min(inflow, n.waterLevel + 1)
            }
        }

        // still ponds heal: two adjacent sources over support form a new source
        if adjacentSources >= 2 && (belowSolid || below == .water) { return .water }
        if inflow > 7 { return .air }
        return Block.flowing(inflow)
    }

}

/// Pure terrain generation: every result is a function of (seed, coords)
/// only, so a captured copy can build chunks on any thread.
struct TerrainGen {
    let seed: UInt64

    /// 3D variant for ore scattering: folds y into the 2D column hash.
    private func hash3(_ x: Int, _ y: Int, _ z: Int, _ salt: UInt64) -> Float {
        hash01(x &+ y &* 7919, z &- y &* 6271, salt)
    }

    /// Single-block ore rolls inside stone, denser ores deeper down.
    private func oreAt(_ x: Int, _ y: Int, _ z: Int) -> Block? {
        if hash3(x, y, z, 200) < 0.011 { return .coalOre }
        if y < 32 && hash3(x, y, z, 201) < 0.008 { return .ironOre }
        if y < 20 && hash3(x, y, z, 202) < 0.004 { return .goldOre }
        if y < 13 && hash3(x, y, z, 203) < 0.003 { return .diamondOre }
        return nil
    }

    private func hash01(_ x: Int, _ z: Int, _ salt: UInt64) -> Float {
        var h = seed &+ salt &* 0xD6E8FEB86659FD93
        h ^= UInt64(bitPattern: Int64(x)) &* 0x9E3779B97F4A7C15
        h ^= UInt64(bitPattern: Int64(z)) &* 0xC2B2AE3D27D4EB4F
        h = (h ^ (h >> 30)) &* 0xBF58476D1CE4E5B9
        h = (h ^ (h >> 27)) &* 0x94D049BB133111EB
        h ^= h >> 31
        return Float(h & 0xFFFFFF) * (1.0 / 16777215.0)
    }

    private func valueNoise(_ fx: Float, _ fz: Float, _ salt: UInt64) -> Float {
        let x0 = Int(fx.rounded(.down)), z0 = Int(fz.rounded(.down))
        let tx = fx - Float(x0), tz = fz - Float(z0)
        let sx = tx * tx * (3 - 2 * tx)
        let sz = tz * tz * (3 - 2 * tz)
        let a = hash01(x0, z0, salt)
        let b = hash01(x0 + 1, z0, salt)
        let c = hash01(x0, z0 + 1, salt)
        let d = hash01(x0 + 1, z0 + 1, salt)
        return simd_mix(simd_mix(a, b, sx), simd_mix(c, d, sx), sz)
    }

    func biomeAt(_ x: Int, _ z: Int) -> Biome {
        let fx = Float(x), fz = Float(z)
        if valueNoise(fx / 280, fz / 280, 12) > 0.64 { return .mountains }
        let temp = valueNoise(fx / 240, fz / 240, 10)
        let moist = valueNoise(fx / 170, fz / 170, 11)
        if temp > 0.62 && moist < 0.55 { return .desert }
        if temp < 0.34 { return .snowy }
        if moist > 0.56 { return .forest }
        return .plains
    }

    private func terrainHeight(_ x: Int, _ z: Int) -> Int {
        let fx = Float(x), fz = Float(z)
        // the mountain mask amplifies the same base noise, so ranges rise
        // smoothly out of the surrounding biomes instead of stepping
        let mountain = valueNoise(fx / 280, fz / 280, 12)
        let amp = 1 + smoothstepf(0.52, 0.72, mountain) * 2.3
        var n: Float = 0
        n += (valueNoise(fx / 42, fz / 42, 1) * 2 - 1) * 14
        n += (valueNoise(fx / 18, fz / 18, 2) * 2 - 1) * 5
        n += (valueNoise(fx / 7, fz / 7, 3) * 2 - 1) * 2
        return max(4, min(World.height - 10, 26 + Int((n * amp).rounded())))
    }

    /// A tree is a pure function of (seed, column), so every chunk that a
    /// tree's canopy touches reproduces it identically.
    private func treeAt(_ x: Int, _ z: Int) -> (base: Int, trunk: Int)? {
        let density: Float
        switch biomeAt(x, z) {
        case .forest: density = 0.028
        case .plains: density = 0.004
        case .snowy: density = 0.0025
        case .mountains: density = 0.002
        case .desert: return nil
        }
        guard hash01(x, z, 123) < density else { return nil }
        let h = terrainHeight(x, z)
        guard h > World.waterLevel + 1, h + 9 < World.height, h < World.stoneLine else { return nil }
        return (base: h, trunk: 4 + Int(hash01(x, z, 124) * 2))
    }

    private func cactusAt(_ x: Int, _ z: Int) -> (base: Int, height: Int)? {
        guard biomeAt(x, z) == .desert, hash01(x, z, 127) < 0.005 else { return nil }
        let h = terrainHeight(x, z)
        guard h > World.waterLevel + 1, h + 4 < World.height else { return nil }
        return (base: h, height: 1 + Int(hash01(x, z, 128) * 2.99))
    }

    func buildChunk(_ c: ChunkCoord) -> Chunk {
        let chunk = Chunk()
        let bx = c.x << 4, bz = c.z << 4

        for lz in 0..<World.chunkSize {
            for lx in 0..<World.chunkSize {
                let x = bx + lx, z = bz + lz
                let h = terrainHeight(x, z)
                let biome = biomeAt(x, z)
                let beach = h <= World.waterLevel + 1
                for y in 0...h {
                    let b: Block
                    if y == 0 {
                        b = .bedrock
                    } else if y < h - 4 {
                        b = oreAt(x, y, z) ?? .stone
                    } else if y < h {
                        switch biome {
                        case .desert: b = .sand
                        case .mountains: b = h >= World.stoneLine ? .stone : .dirt
                        default: b = .dirt
                        }
                    } else if beach {
                        b = .sand
                    } else {
                        switch biome {
                        case .desert: b = .sand
                        case .snowy: b = .snow
                        case .mountains:
                            b = h >= World.snowLine ? .snow : (h >= World.stoneLine ? .stone : .grass)
                        default: b = .grass
                        }
                    }
                    chunk.blocks[Chunk.index(lx, y, lz)] = b.rawValue
                }
                if h < World.waterLevel {
                    for y in (h + 1)...World.waterLevel {
                        chunk.blocks[Chunk.index(lx, y, lz)] = Block.water.rawValue
                    }
                }
                if let cactus = cactusAt(x, z) {
                    for i in 1...cactus.height {
                        chunk.blocks[Chunk.index(lx, cactus.base + i, lz)] = Block.cactus.rawValue
                    }
                }
            }
        }

        // Scan a margin around the chunk so trees rooted in neighbors still
        // drop their canopy blocks into this chunk. Only cells inside this
        // chunk are written, so ownership is unambiguous across borders.
        let margin = 3
        for tz in (bz - margin)..<(bz + World.chunkSize + margin) {
            for tx in (bx - margin)..<(bx + World.chunkSize + margin) {
                guard let tree = treeAt(tx, tz) else { continue }

                func put(_ wx: Int, _ wy: Int, _ wz: Int, _ b: Block, onlyAir: Bool) {
                    let lx = wx - bx, lz = wz - bz
                    guard lx >= 0 && lx < World.chunkSize && lz >= 0 && lz < World.chunkSize,
                          wy >= 0 && wy < World.height else { return }
                    let i = Chunk.index(lx, wy, lz)
                    if onlyAir && chunk.blocks[i] != 0 { return }
                    chunk.blocks[i] = b.rawValue
                }

                for i in 1...tree.trunk {
                    put(tx, tree.base + i, tz, .wood, onlyAir: false)
                }
                for dy in (tree.trunk - 1)...(tree.trunk + 1) {
                    let r = dy == tree.trunk + 1 ? 1 : 2
                    for dz in -r...r {
                        for dx in -r...r {
                            if dx == 0 && dz == 0 && dy <= tree.trunk { continue }
                            if abs(dx) == r && abs(dz) == r && hash01(tx + dx * 31, tz + dz * 17, 125) < 0.5 { continue }
                            put(tx + dx, tree.base + dy, tz + dz, .leaves, onlyAir: true)
                        }
                    }
                }
            }
        }

        return chunk
    }
}
