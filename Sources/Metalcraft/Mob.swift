import simd

enum MobKind: CaseIterable, Hashable {
    case pig, sheep, cow, chicken, zombie, creeper

    var textureNames: [String] {
        switch self {
        case .pig: return ["pig"]
        case .sheep: return ["sheep", "sheep_fur"]
        case .cow: return ["cow"]
        case .chicken: return ["chicken"]
        case .zombie: return ["zombie"]
        case .creeper: return ["creeper"]
        }
    }

    var hostile: Bool { self == .zombie || self == .creeper }

    var halfWidth: Float {
        switch self {
        case .chicken: return 0.18
        case .pig, .sheep, .cow: return 0.45
        case .zombie, .creeper: return 0.3
        }
    }

    var height: Float {
        switch self {
        case .chicken: return 0.7
        case .pig: return 0.9
        case .sheep: return 1.15
        case .cow: return 1.4
        case .zombie: return 1.8
        case .creeper: return 1.65
        }
    }

    var walkSpeed: Float {
        switch self {
        case .chicken: return 1.2
        case .zombie: return 1.6
        case .creeper: return 1.8
        case .pig, .sheep, .cow: return 1.5
        }
    }
}

/// Wandering entity with the same axis-by-axis voxel collision as the player.
/// AI is a two-state machine: stand around, or pick a heading and walk it.
final class Mob {
    let kind: MobKind
    var pos: SIMD3<Float> // feet center
    var vel = SIMD3<Float>.zero
    var yaw: Float
    var onGround = false
    var age: Float = 0

    // walk-cycle phase and intensity, drives limb swing and body bob
    var limbSwing: Float = 0
    var swingAmount: Float = 0

    private var targetYaw: Float
    private var walking = false
    private var stateTimer: Float = 0
    private var hitWall = false

    init(kind: MobKind, pos: SIMD3<Float>) {
        self.kind = kind
        self.pos = pos
        yaw = Float.random(in: 0..<(2 * .pi))
        targetYaw = yaw
    }

    func update(dt: Float, world: World) {
        age += dt
        stateTimer -= dt
        if stateTimer <= 0 {
            if Float.random(in: 0...1) < 0.65 {
                walking = true
                targetYaw += Float.random(in: -2.4...2.4)
                stateTimer = Float.random(in: 1.5...5)
            } else {
                walking = false
                stateTimer = Float.random(in: 1...4)
            }
        }

        // turn smoothly toward the chosen heading
        var dyaw = (targetYaw - yaw).truncatingRemainder(dividingBy: 2 * .pi)
        if dyaw > .pi { dyaw -= 2 * .pi }
        if dyaw < -.pi { dyaw += 2 * .pi }
        yaw += dyaw * min(1, 6 * dt)

        let inWater = world.block(Int(pos.x.rounded(.down)),
                                  Int((pos.y + 0.3).rounded(.down)),
                                  Int(pos.z.rounded(.down))).isWater

        var hvel = SIMD3<Float>(vel.x, 0, vel.z)
        hvel *= exp(-(onGround ? 10 : 2) * dt)
        if walking {
            let fwd = SIMD3<Float>(sin(yaw), 0, -cos(yaw))
            let target = kind.walkSpeed * (inWater ? 0.6 : 1)
            let add = target - simd_dot(hvel, fwd)
            if add > 0 { hvel += fwd * min(40 * dt, add) }
        }
        vel.x = hvel.x
        vel.z = hvel.z

        if inWater {
            vel.y = min(vel.y + 24 * dt, 3) // paddle back up to the surface
        } else {
            vel.y -= 28 * dt
            vel.y = max(vel.y, kind == .chicken ? -3 : -50) // chickens flutter down
        }

        onGround = false
        hitWall = false
        move(axis: 1, by: vel.y * dt, world: world)
        move(axis: 0, by: vel.x * dt, world: world)
        move(axis: 2, by: vel.z * dt, world: world)

        if hitWall && (onGround || inWater) {
            vel.y = max(vel.y, inWater ? 4.5 : 8.2) // hop one-block ledges
        }

        let speed = simd_length(SIMD2(vel.x, vel.z))
        let amt = min(speed / max(kind.walkSpeed, 0.1), 1)
        swingAmount += (amt - swingAmount) * min(1, 10 * dt)
        limbSwing += speed * dt * 3.3
    }

    private func move(axis: Int, by amount: Float, world: World) {
        guard amount != 0 else { return }
        pos[axis] += amount
        let half = kind.halfWidth
        let mn = SIMD3<Float>(pos.x - half, pos.y, pos.z - half)
        let mx = SIMD3<Float>(pos.x + half, pos.y + kind.height, pos.z + half)
        let eps: Float = 0.001
        let x0 = Int((mn.x + eps).rounded(.down)), x1 = Int((mx.x - eps).rounded(.down))
        let y0 = Int((mn.y + eps).rounded(.down)), y1 = Int((mx.y - eps).rounded(.down))
        let z0 = Int((mn.z + eps).rounded(.down)), z1 = Int((mx.z - eps).rounded(.down))

        for y in y0...y1 {
            for z in z0...z1 {
                for x in x0...x1 {
                    guard world.isSolid(x, y, z) else { continue }
                    switch axis {
                    case 0:
                        pos.x = amount > 0 ? Float(x) - half - eps : Float(x + 1) + half + eps
                    case 1:
                        if amount > 0 {
                            pos.y = Float(y) - kind.height - eps
                        } else {
                            pos.y = Float(y + 1) + eps
                            onGround = true
                        }
                    default:
                        pos.z = amount > 0 ? Float(z) - half - eps : Float(z + 1) + half + eps
                    }
                    if axis != 1 { hitWall = true }
                    vel[axis] = 0
                    return
                }
            }
        }
    }
}

// MARK: - Model definitions

enum PartRole {
    case body, head, armR, armL, legFR, legFL, legBR, legBL, wingR, wingL
}

/// One textured cuboid of a model part. Coordinates are in texture pixels
/// (1 px = 1/16 m before the model scale), relative to the part pivot.
/// `rotX` is baked into the geometry — quadruped torsos are authored upright
/// in the texture and rotated 90° to lie horizontally.
struct BoxSpec {
    var tex = 0
    var uv: SIMD2<Float>
    var origin: SIMD3<Float>
    var size: SIMD3<Float>
    var inflate: Float = 0
    var rotX: Float = 0
}

/// A rigid group of boxes that animates as one unit around `pivot`
/// (pixels from the entity's feet center; entity faces -Z at yaw 0).
struct PartSpec {
    var role: PartRole
    var pivot: SIMD3<Float>
    var baseRotX: Float = 0 // resting pose, e.g. zombie arms held out
    var boxes: [BoxSpec]
}

struct ModelSpec {
    var scale: Float // meters per texture pixel
    var parts: [PartSpec]
}

/// Classic 64×32 entity layouts: every mob is a handful of boxes hung off
/// pivots, with the standard cross-shaped box UV unwrap.
enum MobModels {
    static let textureSize = SIMD2<Float>(64, 32)

    static func humanoid(zombieArms: Bool = false) -> ModelSpec {
        ModelSpec(scale: 1.8 / 32, parts: [
            PartSpec(role: .body, pivot: SIMD3(0, 24, 0), boxes: [
                BoxSpec(uv: SIMD2(16, 16), origin: SIMD3(-4, -12, -2), size: SIMD3(8, 12, 4))]),
            PartSpec(role: .head, pivot: SIMD3(0, 24, 0), boxes: [
                BoxSpec(uv: SIMD2(0, 0), origin: SIMD3(-4, 0, -4), size: SIMD3(8, 8, 8)),
                BoxSpec(uv: SIMD2(32, 0), origin: SIMD3(-4, 0, -4), size: SIMD3(8, 8, 8), inflate: 0.5)]),
            PartSpec(role: .armR, pivot: SIMD3(-6, 22, 0), baseRotX: zombieArms ? .pi / 2 * 0.9 : 0, boxes: [
                BoxSpec(uv: SIMD2(40, 16), origin: SIMD3(-2, -10, -2), size: SIMD3(4, 12, 4))]),
            PartSpec(role: .armL, pivot: SIMD3(6, 22, 0), baseRotX: zombieArms ? .pi / 2 * 0.9 : 0, boxes: [
                BoxSpec(uv: SIMD2(40, 16), origin: SIMD3(-2, -10, -2), size: SIMD3(4, 12, 4))]),
            PartSpec(role: .legFR, pivot: SIMD3(-2, 12, 0), boxes: [
                BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4))]),
            PartSpec(role: .legFL, pivot: SIMD3(2, 12, 0), boxes: [
                BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4))]),
        ])
    }

    static func spec(for kind: MobKind) -> ModelSpec {
        switch kind {
        case .zombie:
            return humanoid(zombieArms: true)

        case .creeper:
            return ModelSpec(scale: 1.7 / 26, parts: [
                PartSpec(role: .body, pivot: SIMD3(0, 18, 0), boxes: [
                    BoxSpec(uv: SIMD2(16, 16), origin: SIMD3(-4, -12, -2), size: SIMD3(8, 12, 4))]),
                PartSpec(role: .head, pivot: SIMD3(0, 18, 0), boxes: [
                    BoxSpec(uv: SIMD2(0, 0), origin: SIMD3(-4, 0, -4), size: SIMD3(8, 8, 8))]),
                PartSpec(role: .legFR, pivot: SIMD3(-2, 6, -2), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4))]),
                PartSpec(role: .legFL, pivot: SIMD3(2, 6, -2), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4))]),
                PartSpec(role: .legBR, pivot: SIMD3(-2, 6, 2), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4))]),
                PartSpec(role: .legBL, pivot: SIMD3(2, 6, 2), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4))]),
            ])

        case .pig:
            return ModelSpec(scale: 1.0 / 16, parts: [
                PartSpec(role: .body, pivot: SIMD3(0, 10, 0), boxes: [
                    BoxSpec(uv: SIMD2(28, 8), origin: SIMD3(-5, -8, -4), size: SIMD3(10, 16, 8), rotX: .pi / 2)]),
                PartSpec(role: .head, pivot: SIMD3(0, 10, -8), boxes: [
                    BoxSpec(uv: SIMD2(0, 0), origin: SIMD3(-4, -4, -8), size: SIMD3(8, 8, 8)),
                    BoxSpec(uv: SIMD2(16, 16), origin: SIMD3(-2, -3, -9), size: SIMD3(4, 3, 1))]),
                PartSpec(role: .legFR, pivot: SIMD3(-3, 6, -5), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4))]),
                PartSpec(role: .legFL, pivot: SIMD3(3, 6, -5), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4))]),
                PartSpec(role: .legBR, pivot: SIMD3(-3, 6, 5), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4))]),
                PartSpec(role: .legBL, pivot: SIMD3(3, 6, 5), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4))]),
            ])

        case .cow:
            return ModelSpec(scale: 1.0 / 16, parts: [
                PartSpec(role: .body, pivot: SIMD3(0, 17, 0), boxes: [
                    BoxSpec(uv: SIMD2(18, 4), origin: SIMD3(-6, -9, -5), size: SIMD3(12, 18, 10), rotX: .pi / 2)]),
                PartSpec(role: .head, pivot: SIMD3(0, 20, -9), boxes: [
                    BoxSpec(uv: SIMD2(0, 0), origin: SIMD3(-4, -4, -5), size: SIMD3(8, 8, 6))]),
                PartSpec(role: .legFR, pivot: SIMD3(-4, 12, -6), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4))]),
                PartSpec(role: .legFL, pivot: SIMD3(4, 12, -6), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4))]),
                PartSpec(role: .legBR, pivot: SIMD3(-4, 12, 6), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4))]),
                PartSpec(role: .legBL, pivot: SIMD3(4, 12, 6), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4))]),
            ])

        case .sheep:
            return ModelSpec(scale: 1.0 / 16, parts: [
                PartSpec(role: .body, pivot: SIMD3(0, 15, 0), boxes: [
                    BoxSpec(uv: SIMD2(28, 8), origin: SIMD3(-4, -8, -3), size: SIMD3(8, 16, 6), rotX: .pi / 2),
                    BoxSpec(tex: 1, uv: SIMD2(28, 8), origin: SIMD3(-4, -8, -3), size: SIMD3(8, 16, 6), inflate: 1.75, rotX: .pi / 2)]),
                PartSpec(role: .head, pivot: SIMD3(0, 18, -8), boxes: [
                    BoxSpec(uv: SIMD2(0, 0), origin: SIMD3(-3, -4, -6), size: SIMD3(6, 6, 8)),
                    BoxSpec(tex: 1, uv: SIMD2(0, 0), origin: SIMD3(-3, -4, -4), size: SIMD3(6, 6, 6), inflate: 0.6)]),
                PartSpec(role: .legFR, pivot: SIMD3(-2, 12, -5), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4)),
                    BoxSpec(tex: 1, uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4), inflate: 0.5)]),
                PartSpec(role: .legFL, pivot: SIMD3(2, 12, -5), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4)),
                    BoxSpec(tex: 1, uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4), inflate: 0.5)]),
                PartSpec(role: .legBR, pivot: SIMD3(-2, 12, 5), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4)),
                    BoxSpec(tex: 1, uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4), inflate: 0.5)]),
                PartSpec(role: .legBL, pivot: SIMD3(2, 12, 5), boxes: [
                    BoxSpec(uv: SIMD2(0, 16), origin: SIMD3(-2, -12, -2), size: SIMD3(4, 12, 4)),
                    BoxSpec(tex: 1, uv: SIMD2(0, 16), origin: SIMD3(-2, -6, -2), size: SIMD3(4, 6, 4), inflate: 0.5)]),
            ])

        case .chicken:
            return ModelSpec(scale: 0.05, parts: [
                PartSpec(role: .body, pivot: SIMD3(0, 8, 0), boxes: [
                    BoxSpec(uv: SIMD2(0, 9), origin: SIMD3(-3, -4, -3), size: SIMD3(6, 8, 6), rotX: .pi / 2)]),
                PartSpec(role: .head, pivot: SIMD3(0, 9, -3), boxes: [
                    BoxSpec(uv: SIMD2(0, 0), origin: SIMD3(-2, 0, -3), size: SIMD3(4, 6, 3)),
                    BoxSpec(uv: SIMD2(14, 0), origin: SIMD3(-2, 2, -5), size: SIMD3(4, 2, 2)),
                    BoxSpec(uv: SIMD2(14, 4), origin: SIMD3(-1, 0, -4), size: SIMD3(2, 2, 2))]),
                PartSpec(role: .legFR, pivot: SIMD3(-2, 5, 0), boxes: [
                    BoxSpec(uv: SIMD2(26, 0), origin: SIMD3(-1.5, -5, -2), size: SIMD3(3, 5, 3))]),
                PartSpec(role: .legFL, pivot: SIMD3(2, 5, 0), boxes: [
                    BoxSpec(uv: SIMD2(26, 0), origin: SIMD3(-1.5, -5, -2), size: SIMD3(3, 5, 3))]),
                PartSpec(role: .wingR, pivot: SIMD3(-3, 11, 0), boxes: [
                    BoxSpec(uv: SIMD2(24, 13), origin: SIMD3(-1, -4, -3), size: SIMD3(1, 4, 6))]),
                PartSpec(role: .wingL, pivot: SIMD3(3, 11, 0), boxes: [
                    BoxSpec(uv: SIMD2(24, 13), origin: SIMD3(0, -4, -3), size: SIMD3(1, 4, 6))]),
            ])
        }
    }

    /// Per-part animation transform around its pivot. `swing` is the walk
    /// phase, `amount` the 0-1 walk intensity, `flap` lifts the wings.
    static func animation(for role: PartRole, baseRotX: Float,
                          swing: Float, amount: Float,
                          headPitch: Float, flap: Float) -> simd_float4x4 {
        switch role {
        case .body:
            return matrix_identity_float4x4
        case .head:
            return rotationXMatrix(headPitch)
        case .legFR, .legBL:
            return rotationXMatrix(cos(swing) * 0.75 * amount)
        case .legFL, .legBR:
            return rotationXMatrix(cos(swing + .pi) * 0.75 * amount)
        case .armR:
            return rotationXMatrix(baseRotX + cos(swing + .pi) * 0.6 * amount)
        case .armL:
            return rotationXMatrix(baseRotX + cos(swing) * 0.6 * amount)
        case .wingR:
            return rotationZMatrix(-flap)
        case .wingL:
            return rotationZMatrix(flap)
        }
    }

    /// Triangulates one part, grouped by texture index. Vertex layout matches
    /// the block shader: position(3) + normal(3) + uv(2) + tint(3).
    static func geometry(for part: PartSpec, scale: Float) -> [(tex: Int, verts: [Float], indices: [UInt32])] {
        var byTex: [Int: (verts: [Float], indices: [UInt32])] = [:]
        for box in part.boxes {
            var g = byTex[box.tex] ?? ([], [])
            appendBox(&g.verts, &g.indices, box: box, scale: scale)
            byTex[box.tex] = g
        }
        return byTex.keys.sorted().map { ($0, byTex[$0]!.verts, byTex[$0]!.indices) }
    }

    /// Standard Minecraft box unwrap around uv offset (u,v) for a box of
    /// nominal size (dx,dy,dz) px: the top strip holds top/bottom, the bottom
    /// strip holds the four sides in the order +X, front, -X, back.
    private static func appendBox(_ verts: inout [Float], _ indices: inout [UInt32],
                                  box: BoxSpec, scale: Float) {
        let lo = box.origin - SIMD3(repeating: box.inflate)
        let hi = box.origin + box.size + SIMD3(repeating: box.inflate)
        let u = box.uv.x, v = box.uv.y
        let dx = box.size.x, dy = box.size.y, dz = box.size.z
        let rot = rotationXMatrix(box.rotX)
        let x0 = lo.x, y0 = lo.y, z0 = lo.z
        let x1 = hi.x, y1 = hi.y, z1 = hi.z

        func face(_ n: SIMD3<Float>, _ corners: [(SIMD3<Float>, SIMD2<Float>)]) {
            let base = UInt32(verts.count / Mesher.floatsPerVertex)
            let nr = rot * SIMD4(n, 0)
            let center = corners.reduce(SIMD2<Float>.zero) { $0 + $1.1 } / 4
            for (p, uv) in corners {
                let pr = rot * SIMD4(p, 1)
                verts.append(contentsOf: [pr.x * scale, pr.y * scale, pr.z * scale,
                                          nr.x, nr.y, nr.z])
                // pull uvs a hair toward the region center so nearest sampling
                // never bleeds into the neighboring region
                verts.append((uv.x + (uv.x < center.x ? 0.03 : -0.03)) / textureSize.x)
                verts.append((uv.y + (uv.y < center.y ? 0.03 : -0.03)) / textureSize.y)
                verts.append(contentsOf: [1, 1, 1])
            }
            indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }

        // front (-Z)
        face(SIMD3(0, 0, -1), [
            (SIMD3(x1, y1, z0), SIMD2(u + dz, v + dz)),
            (SIMD3(x0, y1, z0), SIMD2(u + dz + dx, v + dz)),
            (SIMD3(x0, y0, z0), SIMD2(u + dz + dx, v + dz + dy)),
            (SIMD3(x1, y0, z0), SIMD2(u + dz, v + dz + dy)),
        ])
        // back (+Z)
        face(SIMD3(0, 0, 1), [
            (SIMD3(x0, y1, z1), SIMD2(u + 2 * dz + dx, v + dz)),
            (SIMD3(x1, y1, z1), SIMD2(u + 2 * dz + 2 * dx, v + dz)),
            (SIMD3(x1, y0, z1), SIMD2(u + 2 * dz + 2 * dx, v + dz + dy)),
            (SIMD3(x0, y0, z1), SIMD2(u + 2 * dz + dx, v + dz + dy)),
        ])
        // +X
        face(SIMD3(1, 0, 0), [
            (SIMD3(x1, y1, z1), SIMD2(u, v + dz)),
            (SIMD3(x1, y1, z0), SIMD2(u + dz, v + dz)),
            (SIMD3(x1, y0, z0), SIMD2(u + dz, v + dz + dy)),
            (SIMD3(x1, y0, z1), SIMD2(u, v + dz + dy)),
        ])
        // -X
        face(SIMD3(-1, 0, 0), [
            (SIMD3(x0, y1, z0), SIMD2(u + dz + dx, v + dz)),
            (SIMD3(x0, y1, z1), SIMD2(u + 2 * dz + dx, v + dz)),
            (SIMD3(x0, y0, z1), SIMD2(u + 2 * dz + dx, v + dz + dy)),
            (SIMD3(x0, y0, z0), SIMD2(u + dz + dx, v + dz + dy)),
        ])
        // top (+Y)
        face(SIMD3(0, 1, 0), [
            (SIMD3(x1, y1, z0), SIMD2(u + dz, v + dz)),
            (SIMD3(x0, y1, z0), SIMD2(u + dz + dx, v + dz)),
            (SIMD3(x0, y1, z1), SIMD2(u + dz + dx, v)),
            (SIMD3(x1, y1, z1), SIMD2(u + dz, v)),
        ])
        // bottom (-Y)
        face(SIMD3(0, -1, 0), [
            (SIMD3(x1, y0, z0), SIMD2(u + dz + dx, v + dz)),
            (SIMD3(x0, y0, z0), SIMD2(u + dz + 2 * dx, v + dz)),
            (SIMD3(x0, y0, z1), SIMD2(u + dz + 2 * dx, v)),
            (SIMD3(x1, y0, z1), SIMD2(u + dz + dx, v)),
        ])
    }
}
