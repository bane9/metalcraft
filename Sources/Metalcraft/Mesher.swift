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
    /// The two axes spanning each face, for smooth-light corner sampling.
    static let tangents: [(Int, Int)] = [(1, 2), (1, 2), (0, 2), (0, 2), (0, 1), (0, 1)]

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
                            origin: SIMD3<Float>, light: SIMD2<Float>) {
        let t = SIMD2<Float>(0, 5)
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
        let b = snap.block(x, y, z)
        return b != .air && b != .leaves && b != .torch && !b.isWater
    }

    // MARK: - Chunk meshing

    static func buildChunk(snapshot snap: ChunkSnapshot, coord: ChunkCoord) -> ChunkGeometry {
        var geo = ChunkGeometry()
        geo.opaqueVertices.reserveCapacity(16384)
        geo.opaqueIndices.reserveCapacity(4096)

        /// Smooth lighting: each face corner averages the four cells that
        /// touch it in the face plane; solid cells contribute 0, which doubles
        /// as soft ambient occlusion in corners.
        func cornerLights(_ dir: Int, _ x: Int, _ y: Int, _ z: Int) -> [SIMD2<Float>] {
            let o = offsets[dir]
            let n = SIMD3<Int>(x + o.x, y + o.y, z + o.z)
            let (t1, t2) = tangents[dir]
            return corners[dir].map { corner in
                var e1 = SIMD3<Int>.zero, e2 = SIMD3<Int>.zero
                e1[t1] = corner[t1] > 0.5 ? 1 : -1
                e2[t2] = corner[t2] > 0.5 ? 1 : -1
                let a = snap.light(n.x, n.y, n.z)
                let b = snap.light(n.x + e1.x, n.y + e1.y, n.z + e1.z)
                let c = snap.light(n.x + e2.x, n.y + e2.y, n.z + e2.z)
                let d = snap.light(n.x + e1.x + e2.x, n.y + e1.y + e2.y, n.z + e1.z + e2.z)
                return (a + b + c + d) / 4
            }
        }

        let x0 = coord.x * World.chunkSize, z0 = coord.z * World.chunkSize
        for y in 0..<World.height {
            for lz in 0..<World.chunkSize {
                for lx in 0..<World.chunkSize {
                    let x = x0 + lx, z = z0 + lz
                    let b = snap.block(x, y, z)
                    if b == .air { continue }
                    let origin = SIMD3<Float>(Float(x), Float(y), Float(z))

                    if b == .torch {
                        appendTorch(&geo.opaqueVertices, &geo.opaqueIndices,
                                    origin: origin, light: snap.light(x, y, z))
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
