import simd

struct ChunkGeometry {
    var opaqueVertices: [Float] = []
    var opaqueIndices: [UInt32] = []
    var waterVertices: [Float] = []
    var waterIndices: [UInt32] = []
}

/// Builds chunk triangle meshes on the CPU: a quad for every visible block
/// face. Water goes in a separate mesh so it can render in a blended pass.
/// Vertex layout matches `VIn` in the shader:
/// position(3) + normal(3) + uv(2) + tint(3) floats.
enum Mesher {
    static let floatsPerVertex = 11

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

    /// `heights` gives the block-top height at each (x,z) corner, indexed by
    /// cornerX * 2 + cornerZ — all 1s for solid cubes, per-corner interpolated
    /// values for water surfaces.
    static func appendFace(_ verts: inout [Float], _ indices: inout [UInt32],
                           block: Block, dir: Int,
                           origin: SIMD3<Float>, heights: [Float] = [1, 1, 1, 1]) {
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
        }
        indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }

    /// Minecraft-style sloped water: a top corner's height is the average of
    /// the water surface heights of the up-to-4 water cells sharing that
    /// corner. Any of them carrying water above pins the corner to full.
    /// Adjacent blocks share corner values, so surfaces interpolate smoothly
    /// between flow levels instead of stepping.
    static func waterCornerHeight(_ world: World, _ x: Int, _ y: Int, _ z: Int,
                                  _ cx: Int, _ cz: Int) -> Float {
        var sum: Float = 0
        var count: Float = 0
        for dx in (cx - 1)...cx {
            for dz in (cz - 1)...cz {
                let nb = world.block(x + dx, y, z + dz)
                guard nb.isWater else { continue }
                if world.block(x + dx, y + 1, z + dz).isWater { return 1 }
                sum += (8 - Float(nb.waterLevel)) / 9
                count += 1
            }
        }
        return count > 0 ? sum / count : 1
    }

    /// Face culling uses opacity, not solidity: leaves have see-through holes,
    /// so faces behind them must still render.
    @inline(__always)
    static func opaqueAt(_ world: World, _ x: Int, _ y: Int, _ z: Int) -> Bool {
        if y < 0 { return true }
        if y >= World.height { return false }
        let b = world.block(x, y, z)
        return b != .air && b != .leaves && !b.isWater
    }

    static func buildChunk(world: World, coord: ChunkCoord) -> ChunkGeometry {
        var geo = ChunkGeometry()
        geo.opaqueVertices.reserveCapacity(16384)
        geo.opaqueIndices.reserveCapacity(4096)

        let x0 = coord.x * World.chunkSize, z0 = coord.z * World.chunkSize
        for y in 0..<World.height {
            for lz in 0..<World.chunkSize {
                for lx in 0..<World.chunkSize {
                    let x = x0 + lx, z = z0 + lz
                    let b = world.block(x, y, z)
                    if b == .air { continue }
                    let isWater = b.isWater
                    var heights: [Float] = [1, 1, 1, 1]
                    if isWater && !world.block(x, y + 1, z).isWater {
                        for cx in 0...1 {
                            for cz in 0...1 {
                                heights[cx * 2 + cz] = waterCornerHeight(world, x, y, z, cx, cz)
                            }
                        }
                    }
                    let origin = SIMD3<Float>(Float(x), Float(y), Float(z))

                    for d in 0..<6 {
                        let o = offsets[d]
                        if isWater {
                            let nb = world.block(x + o.x, y + o.y, z + o.z)
                            guard nb == .air || nb == .leaves else { continue }
                        } else if opaqueAt(world, x + o.x, y + o.y, z + o.z) {
                            continue
                        }
                        if isWater {
                            appendFace(&geo.waterVertices, &geo.waterIndices,
                                       block: b, dir: d, origin: origin, heights: heights)
                        } else {
                            appendFace(&geo.opaqueVertices, &geo.opaqueIndices,
                                       block: b, dir: d, origin: origin)
                        }
                    }
                }
            }
        }
        return geo
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
