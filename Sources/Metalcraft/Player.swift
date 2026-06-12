import simd

struct RayHit {
    var block: SIMD3<Int32>
    var normal: SIMD3<Int32>
    var t: Float // distance along the ray to the entered face
}

/// Voxel traversal (Amanatides & Woo DDA): steps the ray cell by cell and
/// returns the first solid block plus the face normal it entered through.
func raycast(origin: SIMD3<Float>, dir: SIMD3<Float>, maxDist: Float, world: World) -> RayHit? {
    var ix = Int32(origin.x.rounded(.down))
    var iy = Int32(origin.y.rounded(.down))
    var iz = Int32(origin.z.rounded(.down))
    let stepX: Int32 = dir.x > 0 ? 1 : -1
    let stepY: Int32 = dir.y > 0 ? 1 : -1
    let stepZ: Int32 = dir.z > 0 ? 1 : -1

    func initialT(_ o: Float, _ d: Float, _ i: Int32) -> Float {
        if d == 0 { return .infinity }
        let boundary = d > 0 ? Float(i + 1) : Float(i)
        return (boundary - o) / d
    }
    var tMaxX = initialT(origin.x, dir.x, ix)
    var tMaxY = initialT(origin.y, dir.y, iy)
    var tMaxZ = initialT(origin.z, dir.z, iz)
    let tDeltaX = dir.x != 0 ? abs(1 / dir.x) : Float.infinity
    let tDeltaY = dir.y != 0 ? abs(1 / dir.y) : Float.infinity
    let tDeltaZ = dir.z != 0 ? abs(1 / dir.z) : Float.infinity

    var t: Float = 0
    while t <= maxDist {
        var normal = SIMD3<Int32>(0, 0, 0)
        if tMaxX < tMaxY && tMaxX < tMaxZ {
            ix += stepX; t = tMaxX; tMaxX += tDeltaX; normal.x = -stepX
        } else if tMaxY < tMaxZ {
            iy += stepY; t = tMaxY; tMaxY += tDeltaY; normal.y = -stepY
        } else {
            iz += stepZ; t = tMaxZ; tMaxZ += tDeltaZ; normal.z = -stepZ
        }
        if t > maxDist { return nil }
        let b = world.block(Int(ix), Int(iy), Int(iz))
        if b != .air && !b.isWater { // mining ray passes through water
            return RayHit(block: SIMD3(ix, iy, iz), normal: normal, t: t)
        }
    }
    return nil
}

final class Player {
    static let halfWidth: Float = 0.3
    static let height: Float = 1.8
    static let eyeHeight: Float = 1.62

    var pos = SIMD3<Float>(8, 50, 8) // feet center
    var vel = SIMD3<Float>.zero
    var yaw: Float = 0
    var pitch: Float = -0.15
    var onGround = false
    var flying = false // spectator: noclip + free vertical movement
    private var hitWall = false

    // walk-cycle phase and 0-1 intensity: drives the first-person view bob
    // and the third-person model's limb swing
    var bobPhase: Float = 0
    var bobAmount: Float = 0

    var eye: SIMD3<Float> { pos + SIMD3(0, Self.eyeHeight, 0) }
    var forward: SIMD3<Float> {
        SIMD3(sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch))
    }

    func spawn(in world: World) {
        let x = 8, z = 8
        let h = world.surfaceHeight(x, z)
        pos = SIMD3(Float(x) + 0.5, Float(h + 1) + 0.01, Float(z) + 0.5)
        vel = .zero
        yaw = 0
        pitch = -0.15
        flying = false
    }

    func look(dx: Float, dy: Float) {
        let sensitivity: Float = 0.0032
        yaw += dx * sensitivity
        pitch -= dy * sensitivity
        pitch = max(-1.55, min(1.55, pitch))
    }

    func update(dt: Float, input: Input, world: World) {
        let fwd = SIMD3<Float>(sin(yaw), 0, -cos(yaw))
        let right = SIMD3<Float>(cos(yaw), 0, sin(yaw))
        var wishDir = SIMD3<Float>.zero
        if input.keys.contains(Keys.w) { wishDir += fwd }
        if input.keys.contains(Keys.s) { wishDir -= fwd }
        if input.keys.contains(Keys.a) { wishDir -= right }
        if input.keys.contains(Keys.d) { wishDir += right }

        if flying {
            // spectator: glide toward the wish velocity, no gravity, noclip
            var wish = wishDir
            if input.keys.contains(Keys.space) { wish.y += 1 }
            if input.sprint { wish.y -= 1 } // shift descends while flying
            if simd_length_squared(wish) > 0 { wish = simd_normalize(wish) }
            let blend = 1 - exp(-8 * dt)
            vel += (wish * 18 - vel) * blend
            pos += vel * dt
            onGround = false
            bobAmount *= exp(-6 * dt)
            if pos.y < -64 { spawn(in: world) }
            return
        }

        if simd_length_squared(wishDir) > 0 { wishDir = simd_normalize(wishDir) }
        let waistBlock = world.block(Int(pos.x.rounded(.down)),
                                     Int((pos.y + 0.9).rounded(.down)),
                                     Int(pos.z.rounded(.down)))
        let inWater = waistBlock.isWater

        // Horizontal movement: friction first, then capped acceleration
        // toward the wish direction — momentum instead of teleport-velocity.
        var hvel = SIMD3<Float>(vel.x, 0, vel.z)
        let drag: Float = onGround ? 10 : (inWater ? 5 : 0.4)
        hvel *= exp(-drag * dt)
        if simd_length_squared(wishDir) > 0 {
            let targetSpeed: Float = inWater ? 3.4 : (input.sprint ? 7.8 : 4.8)
            let currentSpeed = simd_dot(hvel, wishDir)
            let addSpeed = targetSpeed - currentSpeed
            if addSpeed > 0 {
                let accel: Float = onGround ? 55 : (inWater ? 20 : 9)
                hvel += wishDir * min(accel * dt, addSpeed)
            }
        }
        vel.x = hvel.x
        vel.z = hvel.z

        if inWater {
            vel.y -= 10 * dt // buoyancy: sink slowly
            vel.y = max(vel.y, -3.5)
            if input.keys.contains(Keys.space) {
                vel.y = min(vel.y + 35 * dt, 4.4) // swim up
            }
        } else {
            vel.y -= 28 * dt
            vel.y = max(vel.y, -50)
            if input.keys.contains(Keys.space) && onGround {
                vel.y = 9.0
            }
        }

        onGround = false
        hitWall = false
        move(axis: 1, by: vel.y * dt, world: world)
        move(axis: 0, by: vel.x * dt, world: world)
        move(axis: 2, by: vel.z * dt, world: world)

        // Minecraft-style hop out of water: swimming against a ledge boosts
        // you up so you can actually climb onto the shore
        if inWater && hitWall && simd_length_squared(wishDir) > 0 {
            vel.y = max(vel.y, 6.5)
        }

        let hspeed = simd_length(SIMD2(vel.x, vel.z))
        let target = onGround ? min(hspeed / 4.8, 1.2) : 0
        bobAmount += (target - bobAmount) * min(1, 8 * dt)
        bobPhase += hspeed * dt * 2.2

        if pos.y < -12 { spawn(in: world) }
    }

    var aabb: (min: SIMD3<Float>, max: SIMD3<Float>) {
        (SIMD3(pos.x - Self.halfWidth, pos.y, pos.z - Self.halfWidth),
         SIMD3(pos.x + Self.halfWidth, pos.y + Self.height, pos.z + Self.halfWidth))
    }

    /// Move along one axis and clamp against the first solid voxel layer hit.
    private func move(axis: Int, by amount: Float, world: World) {
        guard amount != 0 else { return }
        pos[axis] += amount
        let (mn, mx) = aabb
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
                        pos.x = amount > 0 ? Float(x) - Self.halfWidth - eps
                                           : Float(x + 1) + Self.halfWidth + eps
                    case 1:
                        if amount > 0 {
                            pos.y = Float(y) - Self.height - eps
                        } else {
                            pos.y = Float(y + 1) + eps
                            onGround = true
                        }
                    default:
                        pos.z = amount > 0 ? Float(z) - Self.halfWidth - eps
                                           : Float(z + 1) + Self.halfWidth + eps
                    }
                    if axis != 1 { hitWall = true }
                    vel[axis] = 0
                    return
                }
            }
        }
    }
}
