import simd

struct ChunkGeometry {
    var opaqueVertices: [Float] = []
    var opaqueIndices: [UInt32] = []
    var waterVertices: [Float] = []
    var waterIndices: [UInt32] = []
}

/// Copy-on-write snapshot of a chunk and its eight neighbors. The arrays
/// share storage with the live chunks; any later main-thread edit copies on
/// write, so background reads stay consistent without locks.
struct ChunkSnapshot {
    let baseX: Int // block coords of the 3x3 area's min corner
    let baseZ: Int
    private let grids: [[UInt8]?] // indexed [gz * 3 + gx], nil if ungenerated
    private let lightGrids: [[UInt8]?] // sky<<4 | block, same indexing

    init(world: World, center: ChunkCoord) {
        baseX = (center.x - 1) * World.chunkSize
        baseZ = (center.z - 1) * World.chunkSize
        var g: [[UInt8]?] = []
        var l: [[UInt8]?] = []
        for dz in -1...1 {
            for dx in -1...1 {
                let c = ChunkCoord(x: center.x + dx, z: center.z + dz)
                g.append(world.chunkBlocks(c))
                l.append(world.chunkLight(c))
            }
        }
        grids = g
        lightGrids = l
    }

    func block(_ x: Int, _ y: Int, _ z: Int) -> Block {
        guard y >= 0 && y < World.height else { return .air }
        let lx = x - baseX, lz = z - baseZ
        guard lx >= 0, lz >= 0, lx < 48, lz < 48,
              let blocks = grids[(lz >> 4) * 3 + (lx >> 4)] else { return .air }
        return Block(rawValue: blocks[Chunk.index(lx & 15, y, lz & 15)]) ?? .air
    }

    func light(_ x: Int, _ y: Int, _ z: Int) -> SIMD2<Float> {
        if y >= World.height { return SIMD2(15, 0) }
        if y < 0 { return .zero }
        let lx = x - baseX, lz = z - baseZ
        guard lx >= 0, lz >= 0, lx < 48, lz < 48,
              let light = lightGrids[(lz >> 4) * 3 + (lx >> 4)] else { return SIMD2(15, 0) }
        let v = light[Chunk.index(lx & 15, y, lz & 15)]
        return SIMD2(Float(v >> 4), Float(v & 15))
    }
}

/// Builds chunk triangle meshes on the CPU: a quad for every visible block
/// face, lit by a Minecraft-style two-channel voxel light field (skylight +
/// blocklight) computed over the 3×3 snapshot. Vertex layout matches `VIn`:
/// position(3) + normal(3) + uv(2) + tint(3) + light(2) floats.
enum Mesher {
    static let floatsPerVertex = 13

    // 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z
    static let normals: [SIMD3<Float>] = [
        SIMD3(1, 0, 0), SIMD3(-1, 0, 0),
        SIMD3(0, 1, 0), SIMD3(0, -1, 0),
        SIMD3(0, 0, 1), SIMD3(0, 0, -1),
    ]
    static let offsets: [SIMD3<Int>] = [
        SIMD3(1, 0, 0), SIMD3(-1, 0, 0),
        SIMD3(0, 1, 0), SIMD3(0, -1, 0),
        SIMD3(0, 0, 1), SIMD3(0, 0, -1),
    ]
    static let corners: [[SIMD3<Float>]] = [
        [SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(1, 1, 1), SIMD3(1, 0, 1)],
        [SIMD3(0, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 1, 1), SIMD3(0, 1, 0)],
        [SIMD3(0, 1, 0), SIMD3(0, 1, 1), SIMD3(1, 1, 1), SIMD3(1, 1, 0)],
        [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1)],
        [SIMD3(0, 0, 1), SIMD3(1, 0, 1), SIMD3(1, 1, 1), SIMD3(0, 1, 1)],
        [SIMD3(0, 0, 0), SIMD3(0, 1, 0), SIMD3(1, 1, 0), SIMD3(1, 0, 0)],
    ]
    // texture v runs top-to-bottom, so v = 1 - worldY on side faces
    static let uvCorners: [[SIMD2<Float>]] = [
        [SIMD2(0, 1), SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1)], // +X
        [SIMD2(0, 1), SIMD2(1, 1), SIMD2(1, 0), SIMD2(0, 0)], // -X
        [SIMD2(0, 0), SIMD2(0, 1), SIMD2(1, 1), SIMD2(1, 0)], // +Y
        [SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1), SIMD2(0, 1)], // -Y
        [SIMD2(0, 1), SIMD2(1, 1), SIMD2(1, 0), SIMD2(0, 0)], // +Z
        [SIMD2(0, 1), SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1)], // -Z
    ]

    /// Atlas tile (column, row) for a block face — classic terrain.png layout.
    static func tile(_ b: Block, _ dir: Int) -> SIMD2<Float> {
        if b.isWater { return SIMD2(13, 12) }
        switch b {
        case .grass:
            if dir == 2 { return SIMD2(0, 0) }
            if dir == 3 { return SIMD2(2, 0) }
            return SIMD2(3, 0)
        case .dirt: return SIMD2(2, 0)
        case .stone: return SIMD2(1, 0)
        case .sand: return SIMD2(2, 1)
        case .wood:
            return (dir == 2 || dir == 3) ? SIMD2(5, 1) : SIMD2(4, 1)
        case .leaves: return SIMD2(4, 3)
        case .bedrock: return SIMD2(1, 1)
        case .snow:
            if dir == 2 { return SIMD2(2, 4) }
            if dir == 3 { return SIMD2(2, 0) }
            return SIMD2(4, 4)
        case .cactus:
            return (dir == 2 || dir == 3) ? SIMD2(5, 4) : SIMD2(6, 4)
        case .planks: return SIMD2(4, 0)
        case .cobblestone: return SIMD2(0, 1)
        case .coalOre: return SIMD2(2, 2)
        case .ironOre: return SIMD2(1, 2)
        case .goldOre: return SIMD2(0, 2)
        case .diamondOre: return SIMD2(2, 3)
        case .redstoneOre: return SIMD2(3, 3)
        case .gravel: return SIMD2(3, 1)
        case .torch: return SIMD2(0, 5)
        case .wool: return SIMD2(0, 4)
        case .tnt:
            if dir == 2 { return SIMD2(9, 0) }
            if dir == 3 { return SIMD2(10, 0) }
            return SIMD2(8, 0)
        case .plateOff, .plateOn: return SIMD2(1, 0)
        case .craftingTable:
            if dir == 2 { return SIMD2(11, 2) }
            if dir == 3 { return SIMD2(4, 0) }
            return (dir == 4 || dir == 5) ? SIMD2(11, 3) : SIMD2(12, 3)
        case .furnace, .furnaceLit:
            if dir == 2 || dir == 3 { return SIMD2(14, 3) }
            if dir == 5 { return b == .furnaceLit ? SIMD2(13, 3) : SIMD2(12, 2) }
            return SIMD2(13, 2)
        default: return SIMD2(0, 0)
        }
    }

    /// Grass top and leaves are grayscale in classic packs and expect a
    /// biome tint; everything else samples untinted.
    static func tint(_ b: Block, _ dir: Int) -> SIMD3<Float> {
        switch b {
        case .grass where dir == 2: return SIMD3(0.57, 0.74, 0.35)
        case .leaves: return SIMD3(0.45, 0.65, 0.25)
        default: return SIMD3(1, 1, 1)
        }
    }

    static let fullBright = [SIMD2<Float>](repeating: SIMD2(15, 0), count: 4)

    /// `heights` gives the block-top height at each (x,z) corner, indexed by
    /// cornerX * 2 + cornerZ — all 1s for solid cubes, per-corner interpolated
    /// values for water surfaces. `lights` carries per-corner (sky, block)
    /// levels 0-15, matching the corner order.
    static func appendFace(_ verts: inout [Float], _ indices: inout [UInt32],
                           block: Block, dir: Int,
                           origin: SIMD3<Float>, heights: [Float] = [1, 1, 1, 1],
                           lights: [SIMD2<Float>] = fullBright) {
        let n = normals[dir]
        let t = tile(block, dir)
        let tn = tint(block, dir)
        let base = UInt32(verts.count / floatsPerVertex)
        for (ci, corner) in corners[dir].enumerated() {
            let h = heights[Int(corner.x) * 2 + Int(corner.z)]
            verts.append(origin.x + corner.x)
            verts.append(origin.y + corner.y * h)
            verts.append(origin.z + corner.z)
            verts.append(n.x); verts.append(n.y); verts.append(n.z)
            let uv = uvCorners[dir][ci]
            var v = uv.y
            if dir != 2 && dir != 3 && corner.y == 1 {
                v = 1 - h // side textures crop with the lowered surface
            }
            // tiny inset guards against sampling the next tile
            verts.append((t.x + 0.002 + uv.x * 0.996) / 16)
            verts.append((t.y + 0.002 + v * 0.996) / 16)
            verts.append(tn.x); verts.append(tn.y); verts.append(tn.z)
            verts.append(lights[ci].x); verts.append(lights[ci].y)
        }
        indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }

    /// Classic torch: four full-size quads pinched in around the stick (the
    /// texture's alpha trims them) plus the glowing tip's top face.
    static func appendTorch(_ verts: inout [Float], _ indices: inout [UInt32],
                            origin: SIMD3<Float>, light: SIMD2<Float>,
                            tile t: SIMD2<Float> = SIMD2(0, 5)) {
        func quad(_ ps: [SIMD3<Float>], _ uvs: [SIMD2<Float>], _ n: SIMD3<Float>) {
            let base = UInt32(verts.count / floatsPerVertex)
            for (p, uv) in zip(ps, uvs) {
                verts.append(contentsOf: [origin.x + p.x, origin.y + p.y, origin.z + p.z,
                                          n.x, n.y, n.z,
                                          (t.x + 0.002 + uv.x * 0.996) / 16,
                                          (t.y + 0.002 + uv.y * 0.996) / 16,
                                          1, 1, 1,
                                          light.x, light.y])
            }
            indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }
        let sideUV: [SIMD2<Float>] = [SIMD2(0, 1), SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1)]
        let lo: Float = 7 / 16, hi: Float = 9 / 16
        quad([SIMD3(lo, 0, 0), SIMD3(lo, 1, 0), SIMD3(lo, 1, 1), SIMD3(lo, 0, 1)],
             sideUV, SIMD3(-1, 0, 0))
        quad([SIMD3(hi, 0, 0), SIMD3(hi, 1, 0), SIMD3(hi, 1, 1), SIMD3(hi, 0, 1)],
             sideUV, SIMD3(1, 0, 0))
        quad([SIMD3(0, 0, lo), SIMD3(0, 1, lo), SIMD3(1, 1, lo), SIMD3(1, 0, lo)],
             sideUV, SIMD3(0, 0, -1))
        quad([SIMD3(0, 0, hi), SIMD3(0, 1, hi), SIMD3(1, 1, hi), SIMD3(1, 0, hi)],
             sideUV, SIMD3(0, 0, 1))
        // tip: the 2×2 px patch of the texture at (7,6)..(9,8)
        let tipUV: [SIMD2<Float>] = [SIMD2(7.0 / 16, 6.0 / 16), SIMD2(7.0 / 16, 8.0 / 16),
                                     SIMD2(9.0 / 16, 8.0 / 16), SIMD2(9.0 / 16, 6.0 / 16)]
        quad([SIMD3(lo, 10.0 / 16, lo), SIMD3(lo, 10.0 / 16, hi),
              SIMD3(hi, 10.0 / 16, hi), SIMD3(hi, 10.0 / 16, lo)],
             tipUV, SIMD3(0, 1, 0))
    }

    /// One quad of a sub-block box, flat-lit by the cell's light. Local UVs
    /// are clamped into the given 16px tile.
    private static func boxQuad(_ verts: inout [Float], _ indices: inout [UInt32],
                                _ ps: [SIMD3<Float>], _ uvs: [SIMD2<Float>],
                                tile: SIMD2<Float>, n: SIMD3<Float>,
                                origin: SIMD3<Float>, light: SIMD2<Float>) {
        let base = UInt32(verts.count / floatsPerVertex)
        for (p, uv) in zip(ps, uvs) {
            verts.append(contentsOf: [
                origin.x + p.x, origin.y + p.y, origin.z + p.z,
                n.x, n.y, n.z,
                (tile.x + 0.002 + min(max(uv.x, 0), 1) * 0.996) / 16,
                (tile.y + 0.002 + min(max(uv.y, 0), 1) * 0.996) / 16,
                1, 1, 1, light.x, light.y,
            ])
        }
        indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }

    /// Box spanning lo..hi inside one block; each face's UVs project its
    /// extents onto the tile so partial boxes sample matching texture slices.
    /// `tileFor` picks a face's tile, or nil to skip the face.
    static func appendBox(_ verts: inout [Float], _ indices: inout [UInt32],
                          lo: SIMD3<Float>, hi: SIMD3<Float>,
                          origin: SIMD3<Float>, light: SIMD2<Float>,
                          tileFor: (Int) -> SIMD2<Float>?) {
        let size = hi - lo
        for d in 0..<6 {
            guard let tile = tileFor(d) else { continue }
            let ps = corners[d].map { lo + $0 * size }
            let uvs = ps.map { p -> SIMD2<Float> in
                switch d {
                case 0, 1: return SIMD2(p.z, 1 - p.y)
                case 4, 5: return SIMD2(p.x, 1 - p.y)
                default: return SIMD2(p.x, p.z)
                }
            }
            boxQuad(&verts, &indices, ps, uvs, tile: tile, n: normals[d],
                    origin: origin, light: light)
        }
    }

    /// Doors: a 3px-thick panel standing against one cell edge — the placed
    /// facing when closed, swung onto the hinge-side edge when open.
    static func appendDoor(_ verts: inout [Float], _ indices: inout [UInt32],
                           block b: Block, origin: SIMD3<Float>, light: SIMD2<Float>) {
        let t: Float = 3.0 / 16
        let edge = b.doorEdge
        let tile = SIMD2<Float>(b.isIronDoor ? 2 : 1, b.doorTop ? 5 : 6)
        var lo = SIMD3<Float>(0, 0, 0)
        var hi = SIMD3<Float>(1, 1, 1)
        switch edge {
        case 0: hi.z = t
        case 1: lo.x = 1 - t
        case 2: lo.z = 1 - t
        default: hi.x = t
        }
        appendBox(&verts, &indices, lo: lo, hi: hi, origin: origin, light: light) { _ in tile }
    }

    /// Beds: a 9px-tall mattress box. Each half shows its own top, side and
    /// end tiles; the face joining the two halves is skipped.
    static func appendBed(_ verts: inout [Float], _ indices: inout [UInt32],
                          block b: Block, origin: SIMD3<Float>, light: SIMD2<Float>) {
        let h: Float = 9.0 / 16
        let f = b.bedFacing
        let topTile = SIMD2<Float>(b.bedHead ? 7 : 6, 8)
        let sideTile = SIMD2<Float>(b.bedHead ? 7 : 6, 9)
        let endTile = SIMD2<Float>(b.bedHead ? 8 : 5, 9)
        // cardinal index → mesher face dir (0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z)
        let faceOf = [5, 0, 4, 1]
        let endFace = faceOf[b.bedHead ? f : (f + 2) & 3] // outward end
        let innerFace = faceOf[b.bedHead ? (f + 2) & 3 : f] // hidden joint

        let lo = SIMD3<Float>(0, 0, 0), hi = SIMD3<Float>(1, h, 1)
        appendBox(&verts, &indices, lo: lo, hi: hi, origin: origin, light: light) { d in
            if d == 3 { return SIMD2(4, 0) } // planks underside
            if d == 2 { return nil } // top is drawn separately, rotated
            if d == innerFace { return nil }
            return d == endFace ? endTile : sideTile
        }

        // top face with UVs rotated so the pillow points along the facing
        let ps = corners[2].map { SIMD3($0.x, h, $0.z) }
        let uvs = ps.map { p -> SIMD2<Float> in
            var uv = SIMD2(p.x, p.z)
            for _ in 0..<f { uv = SIMD2(1 - uv.y, uv.x) } // 90° per step
            return uv
        }
        boxQuad(&verts, &indices, ps, uvs, tile: topTile, n: SIMD3(0, 1, 0),
                origin: origin, light: light)
    }

    /// Redstone dust: a flat quad just above the floor, tinted by its power
    /// level. Straight runs use the line tile; everything else the cross.
    static func appendWire(_ verts: inout [Float], _ indices: inout [UInt32],
                           level: Int, snap: ChunkSnapshot, x: Int, y: Int, z: Int,
                           origin: SIMD3<Float>, light: SIMD2<Float>) {
        func connects(_ dx: Int, _ dz: Int) -> Bool {
            for dy in -1...1 {
                let n = snap.block(x + dx, y + dy, z + dz)
                if n.isWire { return true }
                if dy == 0 && (n.isRedstoneSource || n == .redstoneTorchOff
                    || n == .leverOff || n == .plateOff) { return true }
            }
            return false
        }
        let n = connects(0, -1), e = connects(1, 0), s = connects(0, 1), w = connects(-1, 0)
        let straightEW = (e || w) && !n && !s
        let straightNS = (n || s) && !e && !w
        let straight = straightNS || straightEW
        let tile: SIMD2<Float> = straight ? SIMD2(5, 10) : SIMD2(4, 10)

        // junctions show the cross tile with its unused arms cropped away;
        // the hub occupies 5-11/16, so quad and uv shrink together
        var x0: Float = 0, x1: Float = 1, z0: Float = 0, z1: Float = 1
        if !straight && (n || e || s || w) {
            if !w { x0 = 5.0 / 16 }
            if !e { x1 = 11.0 / 16 }
            if !n { z0 = 5.0 / 16 }
            if !s { z1 = 11.0 / 16 }
        }

        // the dust texture is grayscale; power drives the red tint
        let c = 0.25 + 0.75 * Float(level) / 15
        let tint = SIMD3<Float>(c, c * 0.15, c * 0.15)
        let lift: Float = 1.0 / 64
        let ps: [SIMD3<Float>] = [SIMD3(x0, lift, z0), SIMD3(x0, lift, z1),
                                  SIMD3(x1, lift, z1), SIMD3(x1, lift, z0)]
        // the line tile runs east-west; rotate the uvs for north-south runs
        let uvs: [SIMD2<Float>] = straightNS
            ? [SIMD2(z0, x0), SIMD2(z1, x0), SIMD2(z1, x1), SIMD2(z0, x1)]
            : [SIMD2(x0, z0), SIMD2(x0, z1), SIMD2(x1, z1), SIMD2(x1, z0)]
        let base = UInt32(verts.count / floatsPerVertex)
        for (p, uv) in zip(ps, uvs) {
            verts.append(contentsOf: [origin.x + p.x, origin.y + p.y, origin.z + p.z,
                                      0, 1, 0,
                                      (tile.x + 0.002 + uv.x * 0.996) / 16,
                                      (tile.y + 0.002 + uv.y * 0.996) / 16,
                                      tint.x, tint.y, tint.z,
                                      light.x, light.y])
        }
        indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }

    /// Lever: a cobblestone base, and the lever-tile handle drawn like a
    /// torch (pinched crossed quads) sheared to lean with its state.
    static func appendLever(_ verts: inout [Float], _ indices: inout [UInt32],
                            on: Bool, origin: SIMD3<Float>, light: SIMD2<Float>) {
        appendBox(&verts, &indices,
                  lo: SIMD3(5.0 / 16, 0, 4.0 / 16), hi: SIMD3(11.0 / 16, 3.0 / 16, 12.0 / 16),
                  origin: origin, light: light) { d in d == 3 ? nil : SIMD2(0, 1) }

        let t = SIMD2<Float>(0, 6)
        let lean: Float = on ? 0.3 : -0.3
        let height: Float = 11.0 / 16
        func quad(_ ps: [SIMD3<Float>], _ uvs: [SIMD2<Float>], _ n: SIMD3<Float>) {
            let base = UInt32(verts.count / floatsPerVertex)
            for (p, uv) in zip(ps, uvs) {
                verts.append(contentsOf: [
                    origin.x + p.x + lean * p.y, // shear: the tip leans over
                    origin.y + p.y * height,
                    origin.z + p.z,
                    n.x, n.y, n.z,
                    (t.x + 0.002 + uv.x * 0.996) / 16,
                    (t.y + 0.002 + uv.y * 0.996) / 16,
                    1, 1, 1, light.x, light.y,
                ])
            }
            indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }
        let sideUV: [SIMD2<Float>] = [SIMD2(0, 1), SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1)]
        let lo: Float = 7.0 / 16, hi: Float = 9.0 / 16
        quad([SIMD3(lo, 0, 0), SIMD3(lo, 1, 0), SIMD3(lo, 1, 1), SIMD3(lo, 0, 1)],
             sideUV, SIMD3(-1, 0, 0))
        quad([SIMD3(hi, 0, 0), SIMD3(hi, 1, 0), SIMD3(hi, 1, 1), SIMD3(hi, 0, 1)],
             sideUV, SIMD3(1, 0, 0))
        quad([SIMD3(0, 0, lo), SIMD3(0, 1, lo), SIMD3(1, 1, lo), SIMD3(1, 0, lo)],
             sideUV, SIMD3(0, 0, -1))
        quad([SIMD3(0, 0, hi), SIMD3(0, 1, hi), SIMD3(1, 1, hi), SIMD3(1, 0, hi)],
             sideUV, SIMD3(0, 0, 1))
    }

    /// Stone pressure plate: a thin pad that sinks while something stands on it.
    static func appendPlate(_ verts: inout [Float], _ indices: inout [UInt32],
                            pressed: Bool, origin: SIMD3<Float>, light: SIMD2<Float>) {
        let h: Float = (pressed ? 0.5 : 1.0) / 16
        appendBox(&verts, &indices,
                  lo: SIMD3(1.0 / 16, 0, 1.0 / 16), hi: SIMD3(15.0 / 16, h, 15.0 / 16),
                  origin: origin, light: light) { d in d == 3 ? nil : SIMD2(1, 0) }
    }

    /// Minecraft-style sloped water: a top corner's height is the average of
    /// the water surface heights of the up-to-4 water cells sharing that
    /// corner. Any of them carrying water above pins the corner to full.
    static func waterCornerHeight(_ snap: ChunkSnapshot, _ x: Int, _ y: Int, _ z: Int,
                                  _ cx: Int, _ cz: Int) -> Float {
        var sum: Float = 0
        var count: Float = 0
        for dx in (cx - 1)...cx {
            for dz in (cz - 1)...cz {
                let nb = snap.block(x + dx, y, z + dz)
                guard nb.isWater else { continue }
                if snap.block(x + dx, y + 1, z + dz).isWater { return 1 }
                sum += (8 - Float(nb.waterLevel)) / 9
                count += 1
            }
        }
        return count > 0 ? sum / count : 1
    }

    /// Face culling uses opacity, not solidity: leaves have see-through holes
    /// and torches are tiny, so faces behind them must still render.
    @inline(__always)
    static func opaqueAt(_ snap: ChunkSnapshot, _ x: Int, _ y: Int, _ z: Int) -> Bool {
        if y < 0 { return true }
        if y >= World.height { return false }
        return snap.block(x, y, z).occludes
    }

    // MARK: - Chunk meshing

    static func buildChunk(snapshot snap: ChunkSnapshot, coord: ChunkCoord) -> ChunkGeometry {
        var geo = ChunkGeometry()
        geo.opaqueVertices.reserveCapacity(16384)
        geo.opaqueIndices.reserveCapacity(4096)

        /// Flat lighting like the classic game: every face takes the single
        /// light value of the cell it opens into — no per-corner averaging,
        /// so no gradients or soft occlusion creeping in at block corners.
        func cornerLights(_ dir: Int, _ x: Int, _ y: Int, _ z: Int) -> [SIMD2<Float>] {
            let o = offsets[dir]
            let l = snap.light(x + o.x, y + o.y, z + o.z)
            return [l, l, l, l]
        }

        let x0 = coord.x * World.chunkSize, z0 = coord.z * World.chunkSize
        for y in 0..<World.height {
            for lz in 0..<World.chunkSize {
                for lx in 0..<World.chunkSize {
                    let x = x0 + lx, z = z0 + lz
                    let b = snap.block(x, y, z)
                    if b == .air { continue }
                    let origin = SIMD3<Float>(Float(x), Float(y), Float(z))

                    if b == .torch || b == .redstoneTorch || b == .redstoneTorchOff {
                        let tile: SIMD2<Float> = b == .torch ? SIMD2(0, 5)
                            : (b == .redstoneTorch ? SIMD2(3, 6) : SIMD2(3, 7))
                        appendTorch(&geo.opaqueVertices, &geo.opaqueIndices,
                                    origin: origin, light: snap.light(x, y, z), tile: tile)
                        continue
                    }
                    if b.isWire {
                        appendWire(&geo.opaqueVertices, &geo.opaqueIndices,
                                   level: b.wireLevel, snap: snap, x: x, y: y, z: z,
                                   origin: origin, light: snap.light(x, y, z))
                        continue
                    }
                    if b == .leverOff || b == .leverOn {
                        appendLever(&geo.opaqueVertices, &geo.opaqueIndices,
                                    on: b == .leverOn, origin: origin,
                                    light: snap.light(x, y, z))
                        continue
                    }
                    if b == .plateOff || b == .plateOn {
                        appendPlate(&geo.opaqueVertices, &geo.opaqueIndices,
                                    pressed: b == .plateOn, origin: origin,
                                    light: snap.light(x, y, z))
                        continue
                    }
                    if b.isDoor {
                        appendDoor(&geo.opaqueVertices, &geo.opaqueIndices,
                                   block: b, origin: origin, light: snap.light(x, y, z))
                        continue
                    }
                    if b.isBed {
                        appendBed(&geo.opaqueVertices, &geo.opaqueIndices,
                                  block: b, origin: origin, light: snap.light(x, y, z))
                        continue
                    }

                    let isWater = b.isWater
                    var heights: [Float] = [1, 1, 1, 1]
                    if isWater && !snap.block(x, y + 1, z).isWater {
                        for cx in 0...1 {
                            for cz in 0...1 {
                                heights[cx * 2 + cz] = waterCornerHeight(snap, x, y, z, cx, cz)
                            }
                        }
                    }

                    for d in 0..<6 {
                        let o = offsets[d]
                        if isWater {
                            let nb = snap.block(x + o.x, y + o.y, z + o.z)
                            guard nb == .air || nb == .leaves || nb == .torch else { continue }
                        } else if opaqueAt(snap, x + o.x, y + o.y, z + o.z) {
                            continue
                        }
                        let lights = cornerLights(d, x, y, z)
                        if isWater {
                            appendFace(&geo.waterVertices, &geo.waterIndices,
                                       block: b, dir: d, origin: origin,
                                       heights: heights, lights: lights)
                        } else {
                            appendFace(&geo.opaqueVertices, &geo.opaqueIndices,
                                       block: b, dir: d, origin: origin, lights: lights)
                        }
                    }
                }
            }
        }

        return geo
    }

    /// Unit cube inflated just past the block it wraps, textured with one of
    /// the 10 destroy-stage tiles (row 15 of terrain.png) — the mining crack
    /// overlay. Packed [x y z u v] for the UI shader; it draws with multiply
    /// blending, so the gray cracks darken the block's own texture.
    static func crackCube(stage: Int) -> (vertices: [Float], indices: [UInt32]) {
        var verts: [Float] = []
        var indices: [UInt32] = []
        let t = SIMD2<Float>(Float(min(max(stage, 0), 9)), 15)
        for d in 0..<6 {
            let base = UInt32(verts.count / 5)
            for (ci, corner) in corners[d].enumerated() {
                let p = corner * 1.004 - SIMD3<Float>(repeating: 0.002)
                let uv = uvCorners[d][ci]
                verts.append(contentsOf: [p.x, p.y, p.z,
                                          (t.x + 0.002 + uv.x * 0.996) / 16,
                                          (t.y + 0.002 + uv.y * 0.996) / 16])
            }
            indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }
        return (verts, indices)
    }

    /// A standalone unit cube for one block type, centered on the origin —
    /// used for dropped-item entities and hotbar icons. `faces` selects which
    /// faces to emit (icons only need the three visible in isometric view).
    static func blockCube(_ b: Block, faces: [Int] = [0, 1, 2, 3, 4, 5]) -> (vertices: [Float], indices: [UInt32]) {
        var verts: [Float] = []
        var indices: [UInt32] = []
        for d in faces {
            appendFace(&verts, &indices, block: b, dir: d,
                       origin: SIMD3(-0.5, -0.5, -0.5))
        }
        return (verts, indices)
    }
}
