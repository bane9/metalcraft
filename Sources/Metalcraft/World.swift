import simd

enum Block: UInt8, Codable {
    case air = 0, grass, dirt, stone, sand, wood, leaves, bedrock, snow, cactus
    case planks = 10, cobblestone, coalOre, ironOre, goldOre, diamondOre, redstoneOre, gravel
    case craftingTable = 20, furnace, furnaceLit, torch, wool
    // doors encode their state in the raw value: base + facing(bits 0-1)
    // + open(bit 2) + top half(bit 3); wood at 32-47, iron at 48-63
    case doorWood0 = 32, doorWood1, doorWood2, doorWood3, doorWood4, doorWood5
    case doorWood6, doorWood7, doorWood8, doorWood9, doorWood10, doorWood11
    case doorWood12, doorWood13, doorWood14, doorWood15
    case doorIron0 = 48, doorIron1, doorIron2, doorIron3, doorIron4, doorIron5
    case doorIron6, doorIron7, doorIron8, doorIron9, doorIron10, doorIron11
    case doorIron12, doorIron13, doorIron14, doorIron15
    // beds: 64 + facing(bits 0-1, foot→head direction) + head half(bit 2)
    case bed0 = 64, bed1, bed2, bed3, bed4, bed5, bed6, bed7
    // redstone: wire carries its 0-15 power level in the raw value;
    // torches, levers and plates are on/off pairs
    case wire0 = 72, wire1, wire2, wire3, wire4, wire5, wire6, wire7
    case wire8, wire9, wire10, wire11, wire12, wire13, wire14, wire15
    case redstoneTorch = 88, redstoneTorchOff = 89
    case leverOff = 90, leverOn = 91
    case plateOff = 92, plateOn = 93
    case tnt = 94
    // water cases stay contiguous at the end: source, then flowing levels 1-7
    case water = 100, flow1, flow2, flow3, flow4, flow5, flow6, flow7
}

extension Block {
    var isWater: Bool { rawValue >= Block.water.rawValue }

    // MARK: Doors and beds (state packed into the raw value)

    /// The four cardinal directions doors and beds face: 0:-Z 1:+X 2:+Z 3:-X.
    static let cardinal: [SIMD3<Int32>] = [
        SIMD3(0, 0, -1), SIMD3(1, 0, 0), SIMD3(0, 0, 1), SIMD3(-1, 0, 0),
    ]

    var isDoor: Bool { rawValue >= 32 && rawValue < 64 }
    var isIronDoor: Bool { rawValue >= 48 && rawValue < 64 }
    var doorFacing: Int { Int(rawValue & 3) }
    var doorOpen: Bool { rawValue & 4 != 0 }
    var doorTop: Bool { rawValue & 8 != 0 }
    static func door(iron: Bool, facing: Int, open: Bool, top: Bool) -> Block {
        Block(rawValue: (iron ? 48 : 32) | UInt8(facing & 3)
            | (open ? 4 : 0) | (top ? 8 : 0))!
    }

    var isBed: Bool { rawValue >= 64 && rawValue < 72 }
    var bedFacing: Int { Int(rawValue & 3) } // foot → head direction
    var bedHead: Bool { rawValue & 4 != 0 }
    static func bed(facing: Int, head: Bool) -> Block {
        Block(rawValue: 64 | UInt8(facing & 3) | (head ? 4 : 0))!
    }

    /// The edge a door's panel currently stands against (swung when open).
    var doorEdge: Int { doorOpen ? (doorFacing + 1) & 3 : doorFacing }

    // MARK: Redstone

    var isWire: Bool { rawValue >= 72 && rawValue < 88 }
    var wireLevel: Int { Int(rawValue) - 72 }
    static func wire(_ level: Int) -> Block {
        Block(rawValue: UInt8(72 + max(0, min(15, level))))!
    }
    /// Components that emit full power while active.
    var isRedstoneSource: Bool {
        self == .redstoneTorch || self == .leverOn || self == .plateOn
    }

    /// Tiny attachments that entities pass straight through.
    var noCollision: Bool {
        self == .air || isWater || self == .torch || isWire
            || self == .redstoneTorch || self == .redstoneTorchOff
            || self == .leverOff || self == .leverOn
            || self == .plateOff || self == .plateOn
    }

    /// Box the targeting wireframe hugs — the visual shape of partial
    /// blocks (wire, torches, levers, plates) rather than the whole cell.
    var outlineBox: (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        if let box = collisionBox { return box }
        if isWire { return (SIMD3(0, 0, 0), SIMD3(1, 1.0 / 16, 1)) }
        switch self {
        case .torch, .redstoneTorch, .redstoneTorchOff:
            return (SIMD3(6.0 / 16, 0, 6.0 / 16), SIMD3(10.0 / 16, 10.0 / 16, 10.0 / 16))
        case .leverOff, .leverOn:
            return (SIMD3(4.0 / 16, 0, 3.0 / 16), SIMD3(12.0 / 16, 11.0 / 16, 13.0 / 16))
        case .plateOff, .plateOn:
            return (SIMD3(1.0 / 16, 0, 1.0 / 16), SIMD3(15.0 / 16, 1.0 / 16, 15.0 / 16))
        default:
            return (SIMD3(0, 0, 0), SIMD3(1, 1, 1))
        }
    }

    /// Full opaque cube, for face culling and light blocking.
    var occludes: Bool {
        !(self == .leaves || isDoor || isBed || noCollision)
    }

    /// Collision box in block-local 0-1 coordinates; nil = walk-through.
    /// Doors collide as their 3px panel, beds as a 9px slab.
    var collisionBox: (lo: SIMD3<Float>, hi: SIMD3<Float>)? {
        if noCollision { return nil }
        if isDoor {
            let t: Float = 3.0 / 16
            switch doorEdge {
            case 0: return (SIMD3(0, 0, 0), SIMD3(1, 1, t))
            case 1: return (SIMD3(1 - t, 0, 0), SIMD3(1, 1, 1))
            case 2: return (SIMD3(0, 0, 1 - t), SIMD3(1, 1, 1))
            default: return (SIMD3(0, 0, 0), SIMD3(t, 1, 1))
            }
        }
        if isBed { return (SIMD3(0, 0, 0), SIMD3(1, 9.0 / 16, 1)) }
        return (SIMD3(0, 0, 0), SIMD3(1, 1, 1))
    }

    /// Light lost per propagation step: 1 = clear, 16 = fully blocks light.
    var lightOpacity: UInt8 {
        if isDoor || isBed || noCollision { return 1 }
        switch self {
        case .leaves: return 2
        default: return isWater ? 3 : 16
        }
    }

    var lightEmission: UInt8 {
        switch self {
        case .torch: return 14
        case .furnaceLit: return 13
        case .redstoneTorch: return 7
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

struct BlockPos: Hashable, Codable {
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
    /// Chunks whose blocks diverge from pure generation (player edits, water
    /// flow, redstone). Only these need to be written to a save file.
    private(set) var editedChunks = Set<ChunkCoord>()
    private var pendingWater = Set<BlockPos>()
    private var pendingRedstone = Set<BlockPos>()
    private var redstoneOpenDoors = Set<BlockPos>() // doors held open by power
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
        editedChunks.removeAll()
        pendingWater.removeAll()
        pendingRedstone.removeAll()
        redstoneOpenDoors.removeAll()
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

    /// Adopt a chunk restored from a save: blocks and light come back exactly
    /// as written (light was globally consistent at save time), so no
    /// reconciliation against the generator is needed. Restored chunks stay
    /// marked as edited so later saves keep carrying them.
    func restoreChunk(_ chunk: Chunk, at c: ChunkCoord) {
        chunks[c] = chunk
        editedChunks.insert(c)
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
        if b.isDoor { return !b.doorOpen } // open doors don't block the cell
        return !b.noCollision
    }

    /// World-space collision box of the cell's block, nil if passable.
    /// Out-of-world and ungenerated cells block as full cubes, like isSolid.
    func collisionBox(_ x: Int, _ y: Int, _ z: Int) -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        let cell = SIMD3<Float>(Float(x), Float(y), Float(z))
        let full = (cell, cell + SIMD3<Float>(1, 1, 1))
        if y < 0 { return full }
        if y >= Self.height { return nil }
        guard let chunk = chunks[Self.chunkCoord(blockX: x, blockZ: z)] else { return full }
        let b = Block(rawValue: chunk.blocks[Chunk.index(x & 15, y, z & 15)]) ?? .air
        guard let box = b.collisionBox else { return nil }
        return (cell + box.lo, cell + box.hi)
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
        editedChunks.insert(cc)
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
        // and any edit can affect circuits — wires connect diagonally, so
        // schedule the whole 3×3×3 neighborhood
        for dy in -1...1 {
            for dz in -1...1 {
                for dx in -1...1 {
                    pendingRedstone.insert(BlockPos(x: x + dx, y: y + dy, z: z + dz))
                }
            }
        }
    }

    func surfaceHeight(_ x: Int, _ z: Int) -> Int {
        for y in stride(from: Self.height - 1, through: 0, by: -1) where block(x, y, z) != .air {
            return y
        }
        return 0
    }

    // MARK: - Redstone simulation
    // Cell-centric like the water: every scheduled cell recomputes what it
    // should be from its neighbors, and changes reschedule the neighborhood.
    // Wire levels strictly relax toward the fixed point, so it converges.

    /// Whether a cell receives redstone power: an adjacent live source or a
    /// charged wire. A torch never powers its own support block (the torch
    /// sitting above doesn't count), which is what makes inversion stable.
    func isPowered(_ x: Int, _ y: Int, _ z: Int) -> Bool {
        for (i, d) in Self.lightDirs.enumerated() {
            let b = block(x + d.0, y + d.1, z + d.2)
            if b == .leverOn || b == .plateOn { return true }
            if b == .redstoneTorch && i != 2 { return true } // 2 = above
            if b.isWire && b.wireLevel > 0 { return true }
        }
        return false
    }

    /// Wire relaxation: full power beside a live source, else the strongest
    /// neighboring wire minus one. Wires connect across one-block steps.
    private func desiredWireLevel(_ x: Int, _ y: Int, _ z: Int) -> Int {
        for d in Self.lightDirs where block(x + d.0, y + d.1, z + d.2).isRedstoneSource {
            return 15
        }
        var p = 0
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            for dy in -1...1 {
                let n = block(x + dx, y + dy, z + dz)
                if n.isWire { p = max(p, n.wireLevel - 1) }
            }
        }
        return p
    }

    private func setDoor(_ p: BlockPos, open: Bool) {
        let b = block(p.x, p.y, p.z)
        guard b.isDoor else { return }
        setBlock(p.x, p.y, p.z, .door(iron: b.isIronDoor, facing: b.doorFacing,
                                      open: open, top: b.doorTop))
        let t = block(p.x, p.y + 1, p.z)
        if t.isDoor {
            setBlock(p.x, p.y + 1, p.z, .door(iron: t.isIronDoor, facing: t.doorFacing,
                                              open: open, top: t.doorTop))
        }
    }

    /// One redstone tick. Wires relax, torches invert (one flip per tick, so
    /// torch loops make clocks), unsupported parts pop off, powered doors
    /// open. Returns popped blocks (for drops) and TNT cells to arm.
    func tickRedstone() -> (pops: [(BlockPos, Block)], primed: [BlockPos]) {
        var pops: [(BlockPos, Block)] = []
        var primed: [BlockPos] = []
        guard !pendingRedstone.isEmpty else { return (pops, primed) }
        var queue = Array(pendingRedstone)
        pendingRedstone.removeAll()
        if queue.count > 4096 {
            pendingRedstone = Set(queue[4096...])
            queue = Array(queue[..<4096])
        }
        for p in queue {
            guard p.y >= 0 && p.y < Self.height,
                  isGenerated(Self.chunkCoord(blockX: p.x, blockZ: p.z)) else { continue }
            let b = block(p.x, p.y, p.z)
            switch b {
            case let w where w.isWire:
                guard isSolid(p.x, p.y - 1, p.z) else {
                    setBlock(p.x, p.y, p.z, .air)
                    pops.append((p, b))
                    continue
                }
                let want = desiredWireLevel(p.x, p.y, p.z)
                if want != w.wireLevel { setBlock(p.x, p.y, p.z, .wire(want)) }
            case .redstoneTorch, .redstoneTorchOff:
                guard isSolid(p.x, p.y - 1, p.z) else {
                    setBlock(p.x, p.y, p.z, .air)
                    pops.append((p, b))
                    continue
                }
                let wantOn = !isPowered(p.x, p.y - 1, p.z)
                if (b == .redstoneTorch) != wantOn {
                    setBlock(p.x, p.y, p.z, wantOn ? .redstoneTorch : .redstoneTorchOff)
                }
            case .leverOff, .leverOn, .plateOff, .plateOn:
                guard isSolid(p.x, p.y - 1, p.z) else {
                    setBlock(p.x, p.y, p.z, .air)
                    pops.append((p, b))
                    continue
                }
            case let d where d.isDoor && !d.doorTop:
                let powered = isPowered(p.x, p.y, p.z) || isPowered(p.x, p.y + 1, p.z)
                let held = redstoneOpenDoors.contains(p)
                if powered && !held {
                    redstoneOpenDoors.insert(p)
                    if !d.doorOpen { setDoor(p, open: true) }
                } else if !powered && held {
                    redstoneOpenDoors.remove(p)
                    if block(p.x, p.y, p.z).doorOpen { setDoor(p, open: false) }
                }
            case .tnt:
                if isPowered(p.x, p.y, p.z) { primed.append(p) }
            default:
                break
            }
        }
        return (pops, primed)
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

/// Deterministic sequence generator for cave carving: every draw is a pure
/// function of (seed, chunk) and the draw count, so any chunk replaying a
/// neighbor's caves consumes the identical sequence.
private struct CaveRand {
    private var s: UInt64

    init(_ seed: UInt64, _ x: Int, _ z: Int) {
        s = seed &+ 0xCA7E_5EED_0000_0001
        s ^= UInt64(bitPattern: Int64(x)) &* 0xBF58476D1CE4E5B9
        s ^= UInt64(bitPattern: Int64(z)) &* 0x94D049BB133111EB
        _ = next()
        _ = next()
    }

    mutating func next() -> UInt64 {
        s &+= 0x9E3779B97F4A7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func float() -> Float { Float(next() & 0xFFFFFF) * (1.0 / 16777216.0) }
    mutating func int(_ n: Int) -> Int { n <= 0 ? 0 : Int(next() % UInt64(n)) }
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

        carveCaves(into: chunk, at: c)

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

    // MARK: - Caves (beta-style worm carvers)

    /// Blocks a tunnel may eat through — terrain and ores, never bedrock,
    /// water or vegetation.
    private static let carvableRaw: Set<UInt8> = {
        let blocks: [Block] = [.grass, .dirt, .stone, .sand, .snow, .gravel,
                               .coalOre, .ironOre, .goldOre, .diamondOre, .redstoneOre]
        return Set(blocks.map(\.rawValue))
    }()

    /// Each chunk in a radius around this one re-rolls its cave systems from
    /// a per-chunk deterministic RNG; tunnels carve freely through space but
    /// only cells inside this chunk are written, so caves cross chunk
    /// borders seamlessly. About 1 in 15 chunks roots a system, like beta.
    private func carveCaves(into chunk: Chunk, at c: ChunkCoord) {
        let range = 5 // chunks; comfortably beyond the longest tunnel's reach
        for ox in (c.x - range)...(c.x + range) {
            for oz in (c.z - range)...(c.z + range) {
                var rng = CaveRand(seed, ox, oz)
                guard rng.int(15) == 0 else { continue }
                for _ in 0..<(1 + rng.int(2)) {
                    let sx = Float(ox << 4) + rng.float() * 16
                    let sy = Float(6 + rng.int(rng.int(40) + 8)) // depth-biased
                    let sz = Float(oz << 4) + rng.float() * 16
                    for _ in 0..<(1 + rng.int(3)) {
                        carveTunnel(into: chunk, at: c, rng: &rng, x: sx, y: sy, z: sz)
                    }
                }
            }
        }
    }

    /// One wandering worm: the heading drifts each step, the radius swells
    /// toward the middle of the run, and roughly 3 of 4 steps carve.
    private func carveTunnel(into chunk: Chunk, at c: ChunkCoord, rng: inout CaveRand,
                             x: Float, y: Float, z: Float) {
        var px = x, py = y, pz = z
        var yaw = rng.float() * 2 * .pi
        var pitch = (rng.float() - 0.5) * 0.25
        var yawDrift: Float = 0
        var pitchDrift: Float = 0
        let girth = 1.0 + rng.float() * 1.6
        let wide: Float = rng.int(10) == 0 ? 2 : 1 // the occasional cavern
        let length = 30 + rng.int(50)
        for step in 0..<length {
            let swell = sin(Float(step) / Float(length) * .pi)
            px += cos(yaw) * cos(pitch)
            pz += sin(yaw) * cos(pitch)
            py += sin(pitch)
            pitch = pitch * 0.72 + pitchDrift * 0.05
            yaw += yawDrift * 0.05
            pitchDrift = pitchDrift * 0.9 + (rng.float() - rng.float()) * 2
            yawDrift = yawDrift * 0.75 + (rng.float() - rng.float()) * 4
            if rng.int(4) != 0 {
                carveSphere(into: chunk, at: c,
                            cx: px, cy: py, cz: pz, r: (1.3 + swell * girth) * wide)
            }
        }
    }

    /// Hollow a vertically squashed sphere, clipped to this chunk. Cells
    /// directly beneath water stay put so ocean and pond floors hold.
    private func carveSphere(into chunk: Chunk, at c: ChunkCoord,
                             cx: Float, cy: Float, cz: Float, r: Float) {
        let bx = c.x << 4, bz = c.z << 4
        let vr = r * 0.75
        let x0 = max(bx, Int((cx - r).rounded(.down)))
        let x1 = min(bx + 15, Int((cx + r).rounded(.down)))
        let y0 = max(1, Int((cy - vr).rounded(.down)))
        let y1 = min(World.height - 2, Int((cy + vr).rounded(.down)))
        let z0 = max(bz, Int((cz - r).rounded(.down)))
        let z1 = min(bz + 15, Int((cz + r).rounded(.down)))
        guard x0 <= x1, y0 <= y1, z0 <= z1 else { return }
        for y in y0...y1 {
            let dy = (Float(y) + 0.5 - cy) / vr
            for z in z0...z1 {
                let dz = (Float(z) + 0.5 - cz) / r
                for x in x0...x1 {
                    let dx = (Float(x) + 0.5 - cx) / r
                    guard dx * dx + dy * dy + dz * dz < 1 else { continue }
                    let i = Chunk.index(x - bx, y, z - bz)
                    guard Self.carvableRaw.contains(chunk.blocks[i]) else { continue }
                    let above = Block(rawValue: chunk.blocks[Chunk.index(x - bx, y + 1, z - bz)])
                    guard above?.isWater != true else { continue }
                    chunk.blocks[i] = 0
                }
            }
        }
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
