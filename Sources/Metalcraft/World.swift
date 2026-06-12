import simd

enum Block: UInt8 {
    case air = 0, grass, dirt, stone, sand, wood, leaves, bedrock, snow, cactus
    case planks = 10, cobblestone, coalOre, ironOre, goldOre, diamondOre, redstoneOre, gravel
    case craftingTable = 20, furnace, furnaceLit, torch
    // water cases stay contiguous at the end: source, then flowing levels 1-7
    case water = 100, flow1, flow2, flow3, flow4, flow5, flow6, flow7
}

extension Block {
    var isWater: Bool { rawValue >= Block.water.rawValue }

    /// Light lost per propagation step: 1 = clear, 16 = fully blocks light.
    var lightOpacity: UInt8 {
        switch self {
        case .air, .torch: return 1
        case .leaves: return 2
        default: return isWater ? 3 : 16
        }
    }

    var lightEmission: UInt8 {
        switch self {
        case .torch: return 14
        case .furnaceLit: return 13
        default: return 0
        }
    }
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
    /// Authoritative light values (sky<<4 | block), maintained incrementally
    /// like beta Minecraft: small flood fills on edits, never bulk recompute.
    var light = [UInt8](repeating: 0, count: World.chunkSize * World.height * World.chunkSize)

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
        reconcileLight(c)
    }

    /// Adopt a chunk built off-thread; first writer wins.
    func insertChunk(_ chunk: Chunk, at c: ChunkCoord) {
        guard chunks[c] == nil else { return }
        chunks[c] = chunk
        reconcileLight(c)
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
        return b != .air && !b.isWater && b != .torch
    }

    /// Copy-on-write handle to a chunk's light storage for snapshotting.
    func chunkLight(_ c: ChunkCoord) -> [UInt8]? { chunks[c]?.light }

    /// (sky, block) light 0-15 at a cell; unlit chunks default to daylight.
    func lightAt(_ x: Int, _ y: Int, _ z: Int) -> SIMD2<Float> {
        guard y >= 0 else { return .zero }
        guard y < Self.height else { return SIMD2(15, 0) }
        guard let chunk = chunks[Self.chunkCoord(blockX: x, blockZ: z)] else { return SIMD2(15, 0) }
        let v = chunk.light[Chunk.index(x & 15, y, z & 15)]
        return SIMD2(Float(v >> 4), Float(v & 15))
    }

    // MARK: - Incremental lighting (beta-style)
    // Light lives in the chunks and is repaired locally on every change:
    // an "unlight" flood removes a source's contribution, then an addition
    // flood re-relaxes from the boundary. Work scales with the affected
    // region (a torch is ~10² cells), not with chunk size.

    private static let lightDirs: [(Int, Int, Int)] = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ]

    /// -1 = no data there (ungenerated/out of world): propagation stops.
    @inline(__always)
    private func lightValue(sky isSky: Bool, _ x: Int, _ y: Int, _ z: Int) -> Int {
        if y >= Self.height { return isSky ? 15 : 0 }
        if y < 0 { return -1 }
        guard let ch = chunks[Self.chunkCoord(blockX: x, blockZ: z)] else { return -1 }
        let v = ch.light[Chunk.index(x & 15, y, z & 15)]
        return Int(isSky ? v >> 4 : v & 15)
    }

    @inline(__always)
    private func setLightValue(sky isSky: Bool, _ x: Int, _ y: Int, _ z: Int, _ v: Int) {
        guard y >= 0 && y < Self.height else { return }
        let cc = Self.chunkCoord(blockX: x, blockZ: z)
        guard let ch = chunks[cc] else { return }
        let i = Chunk.index(x & 15, y, z & 15)
        ch.light[i] = isSky ? (UInt8(v) << 4) | (ch.light[i] & 0x0F)
                            : (ch.light[i] & 0xF0) | UInt8(v)
        // remesh every chunk whose smooth lighting samples this cell
        dirtyChunks.insert(cc)
        let lx = x & 15, lz = z & 15
        if lx == 0 { dirtyChunks.insert(ChunkCoord(x: cc.x - 1, z: cc.z)) }
        if lx == 15 { dirtyChunks.insert(ChunkCoord(x: cc.x + 1, z: cc.z)) }
        if lz == 0 { dirtyChunks.insert(ChunkCoord(x: cc.x, z: cc.z - 1)) }
        if lz == 15 { dirtyChunks.insert(ChunkCoord(x: cc.x, z: cc.z + 1)) }
    }

    /// Relax outward from the queued cells until nothing brightens.
    private func propagateLight(sky isSky: Bool, _ queue: inout [BlockPos]) {
        var head = 0
        while head < queue.count {
            let p = queue[head]; head += 1
            let l = lightValue(sky: isSky, p.x, p.y, p.z)
            guard l > 1 else { continue }
            for (d, dir) in Self.lightDirs.enumerated() {
                let nx = p.x + dir.0, ny = p.y + dir.1, nz = p.z + dir.2
                guard ny >= 0 && ny < Self.height else { continue }
                let op = Int(block(nx, ny, nz).lightOpacity)
                guard op < 16 else { continue }
                let cur = lightValue(sky: isSky, nx, ny, nz)
                guard cur >= 0 else { continue }
                // full skylight pours straight down without attenuation
                let nl = (isSky && d == 3 && l == 15) ? 15 - (op - 1) : l - op
                if nl > cur {
                    setLightValue(sky: isSky, nx, ny, nz, nl)
                    queue.append(BlockPos(x: nx, y: ny, z: nz))
                }
            }
        }
        queue.removeAll(keepingCapacity: true)
    }

    /// Spread darkness from a removed source: cells this path lit go to 0 and
    /// keep cascading; brighter boundary cells become re-fill seeds.
    private func unlight(sky isSky: Bool, start: [(BlockPos, Int)], seeds: inout [BlockPos]) {
        var queue = start
        var head = 0
        while head < queue.count {
            let (p, old) = queue[head]; head += 1
            for (d, dir) in Self.lightDirs.enumerated() {
                let nx = p.x + dir.0, ny = p.y + dir.1, nz = p.z + dir.2
                guard ny >= 0 && ny < Self.height else { continue }
                let cur = lightValue(sky: isSky, nx, ny, nz)
                guard cur > 0 else { continue }
                if cur < old || (isSky && d == 3 && old == 15 && cur == 15) {
                    setLightValue(sky: isSky, nx, ny, nz, 0)
                    queue.append((BlockPos(x: nx, y: ny, z: nz), cur))
                } else {
                    seeds.append(BlockPos(x: nx, y: ny, z: nz))
                }
            }
        }
    }

    /// Repair both channels around one changed cell.
    private func relight(_ x: Int, _ y: Int, _ z: Int, old: Block, new: Block) {
        guard old.lightOpacity != new.lightOpacity
            || old.lightEmission != new.lightEmission else { return }
        let pos = BlockPos(x: x, y: y, z: z)
        for isSky in [true, false] {
            var seeds: [BlockPos] = []
            let cur = lightValue(sky: isSky, x, y, z)
            if cur > 0 {
                setLightValue(sky: isSky, x, y, z, 0)
                unlight(sky: isSky, start: [(pos, cur)], seeds: &seeds)
            }
            if !isSky {
                let em = Int(new.lightEmission)
                if em > 0 {
                    setLightValue(sky: false, x, y, z, em)
                    seeds.append(pos)
                }
            }
            // neighbors re-fill the hole (and the cell itself, if transparent)
            for dir in Self.lightDirs {
                seeds.append(BlockPos(x: x + dir.0, y: y + dir.1, z: z + dir.2))
            }
            propagateLight(sky: isSky, &seeds)
        }
    }

    /// Stitch a freshly inserted chunk's light to its generated neighbors:
    /// push whichever side of each border pair can brighten the other.
    private func reconcileLight(_ c: ChunkCoord) {
        var skySeeds: [BlockPos] = []
        var blkSeeds: [BlockPos] = []
        let bx = c.x << 4, bz = c.z << 4
        let borders: [(ChunkCoord, (Int) -> (BlockPos, BlockPos))] = [
            (ChunkCoord(x: c.x - 1, z: c.z), { i in
                (BlockPos(x: bx, y: i >> 4, z: bz + (i & 15)),
                 BlockPos(x: bx - 1, y: i >> 4, z: bz + (i & 15))) }),
            (ChunkCoord(x: c.x + 1, z: c.z), { i in
                (BlockPos(x: bx + 15, y: i >> 4, z: bz + (i & 15)),
                 BlockPos(x: bx + 16, y: i >> 4, z: bz + (i & 15))) }),
            (ChunkCoord(x: c.x, z: c.z - 1), { i in
                (BlockPos(x: bx + (i & 15), y: i >> 4, z: bz),
                 BlockPos(x: bx + (i & 15), y: i >> 4, z: bz - 1)) }),
            (ChunkCoord(x: c.x, z: c.z + 1), { i in
                (BlockPos(x: bx + (i & 15), y: i >> 4, z: bz + 15),
                 BlockPos(x: bx + (i & 15), y: i >> 4, z: bz + 16)) }),
        ]
        for (nc, cellPair) in borders where chunks[nc] != nil {
            for i in 0..<(16 * Self.height) {
                let (inner, outer) = cellPair(i)
                let sa = lightValue(sky: true, inner.x, inner.y, inner.z)
                let sb = lightValue(sky: true, outer.x, outer.y, outer.z)
                if sa > sb + 1 { skySeeds.append(inner) }
                else if sb > sa + 1 { skySeeds.append(outer) }
                let ba = lightValue(sky: false, inner.x, inner.y, inner.z)
                let bb = lightValue(sky: false, outer.x, outer.y, outer.z)
                if ba > bb + 1 { blkSeeds.append(inner) }
                else if bb > ba + 1 { blkSeeds.append(outer) }
            }
        }
        propagateLight(sky: true, &skySeeds)
        propagateLight(sky: false, &blkSeeds)
    }

    func setBlock(_ x: Int, _ y: Int, _ z: Int, _ b: Block) {
        guard y >= 0 && y < Self.height else { return }
        let cc = Self.chunkCoord(blockX: x, blockZ: z)
        guard let chunk = chunks[cc] else { return }
        let old = Block(rawValue: chunk.blocks[Chunk.index(x & 15, y, z & 15)]) ?? .air
        chunk.blocks[Chunk.index(x & 15, y, z & 15)] = b.rawValue
        dirtyChunks.insert(cc)
        let lx = x & 15, lz = z & 15
        if lx == 0 { dirtyChunks.insert(ChunkCoord(x: cc.x - 1, z: cc.z)) }
        if lx == 15 { dirtyChunks.insert(ChunkCoord(x: cc.x + 1, z: cc.z)) }
        if lz == 0 { dirtyChunks.insert(ChunkCoord(x: cc.x, z: cc.z - 1)) }
        if lz == 15 { dirtyChunks.insert(ChunkCoord(x: cc.x, z: cc.z + 1)) }
        // repair light locally; touched chunks are marked dirty as cells change
        relight(x, y, z, old: old, new: b)
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
        if hash3(x, y, z, 204) < 0.012 { return .gravel } // pockets; drops flint
        if y < 32 && hash3(x, y, z, 201) < 0.008 { return .ironOre }
        if y < 20 && hash3(x, y, z, 202) < 0.004 { return .goldOre }
        if y < 16 && hash3(x, y, z, 205) < 0.006 { return .redstoneOre }
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

        computeInitialLight(chunk)
        return chunk
    }

    /// Generation-time skylight: per-column fill from the sky, then a flood
    /// fill bounded to this chunk so overhangs and cave mouths grade smoothly.
    /// Cross-chunk seams are stitched on the main thread when the chunk is
    /// adopted (World.reconcileLight).
    private func computeInitialLight(_ chunk: Chunk) {
        let H = World.height
        // hFull[col] = lowest y from which everything above is full skylight
        var hFull = [Int](repeating: 0, count: 256)
        for lz in 0..<16 {
            for lx in 0..<16 {
                var l = 15
                for y in stride(from: H - 1, through: 0, by: -1) {
                    let i = Chunk.index(lx, y, lz)
                    let o = Int((Block(rawValue: chunk.blocks[i]) ?? .air).lightOpacity)
                    if o >= 16 { l = 0 } else if o > 1 { l = max(0, l - o) }
                    if l < 15 && hFull[lz * 16 + lx] == 0 { hFull[lz * 16 + lx] = y + 1 }
                    if l == 0 { break } // rest of the column stays dark
                    chunk.light[i] = UInt8(l) << 4
                }
            }
        }

        // seed sideways spread where a column's full-lit cells border a
        // shadowed stretch of an adjacent column
        var queue: [Int] = [] // packed (lx, y, lz)
        for lz in 0..<16 {
            for lx in 0..<16 {
                let own = hFull[lz * 16 + lx]
                var hMax = own
                if lx > 0 { hMax = max(hMax, hFull[lz * 16 + lx - 1]) }
                if lx < 15 { hMax = max(hMax, hFull[lz * 16 + lx + 1]) }
                if lz > 0 { hMax = max(hMax, hFull[(lz - 1) * 16 + lx]) }
                if lz < 15 { hMax = max(hMax, hFull[(lz + 1) * 16 + lx]) }
                for y in own..<hMax {
                    queue.append((y << 8) | (lz << 4) | lx)
                }
            }
        }

        var head = 0
        while head < queue.count {
            let p = queue[head]; head += 1
            let lx = p & 15, lz = (p >> 4) & 15, y = p >> 8
            let l = Int(chunk.light[Chunk.index(lx, y, lz)] >> 4)
            guard l > 1 else { continue }
            for (d, dir) in [(0, (1, 0, 0)), (1, (-1, 0, 0)), (2, (0, 1, 0)),
                             (3, (0, -1, 0)), (4, (0, 0, 1)), (5, (0, 0, -1))] {
                let nx = lx + dir.0, ny = y + dir.1, nz = lz + dir.2
                guard nx >= 0, nx < 16, nz >= 0, nz < 16, ny >= 0, ny < H else { continue }
                let ni = Chunk.index(nx, ny, nz)
                let o = Int((Block(rawValue: chunk.blocks[ni]) ?? .air).lightOpacity)
                guard o < 16 else { continue }
                let cur = Int(chunk.light[ni] >> 4)
                let nl = (d == 3 && l == 15) ? 15 - (o - 1) : l - o
                if nl > cur {
                    chunk.light[ni] = UInt8(nl) << 4
                    queue.append((ny << 8) | (nz << 4) | nx)
                }
            }
        }
    }
}
