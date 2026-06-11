import simd

struct ItemStack {
    var block: Block
    var count: Int
}

/// Nine-slot Minecraft-style hotbar. Mining feeds it via item pickups,
/// placing consumes from the selected slot.
final class Inventory {
    static let slotCount = 9
    static let stackLimit = 64

    var slots: [ItemStack?] = Array(repeating: nil, count: slotCount)
    var selected = 0

    var selectedStack: ItemStack? { slots[selected] }

    /// Returns false when no slot can take the item (drop stays on the ground).
    func add(_ b: Block) -> Bool {
        for i in 0..<slots.count {
            if var stack = slots[i], stack.block == b, stack.count < Self.stackLimit {
                stack.count += 1
                slots[i] = stack
                return true
            }
        }
        for i in 0..<slots.count where slots[i] == nil {
            slots[i] = ItemStack(block: b, count: 1)
            return true
        }
        return false
    }

    func consumeSelected() -> Block? {
        guard var stack = slots[selected] else { return nil }
        stack.count -= 1
        slots[selected] = stack.count > 0 ? stack : nil
        return stack.block
    }

    func clear() {
        slots = Array(repeating: nil, count: Self.slotCount)
        selected = 0
    }
}

/// A mined block bouncing on the ground as a mini 3D model, waiting to be
/// picked up.
final class ItemEntity {
    var block: Block
    var pos: SIMD3<Float> // center
    var vel: SIMD3<Float>
    var age: Float = 0

    init(block: Block, pos: SIMD3<Float>, vel: SIMD3<Float>) {
        self.block = block
        self.pos = pos
        self.vel = vel
    }

    func update(dt: Float, world: World) {
        age += dt
        let inWater = world.block(Int(pos.x.rounded(.down)),
                                  Int(pos.y.rounded(.down)),
                                  Int(pos.z.rounded(.down))).isWater
        if inWater {
            vel *= exp(-3 * dt)
            vel.y = min(vel.y + 30 * dt, 1.2) // drops float up to the surface
        } else {
            vel.y -= 22 * dt
        }

        // tiny per-axis collision against the voxel grid
        let half: Float = 0.13
        for axis in 0..<3 {
            guard vel[axis] != 0 else { continue }
            var p = pos
            p[axis] += vel[axis] * dt
            var probe = p
            probe[axis] += (vel[axis] > 0 ? half : -half)
            if world.isSolid(Int(probe.x.rounded(.down)),
                             Int(probe.y.rounded(.down)),
                             Int(probe.z.rounded(.down))) {
                if axis == 1 && vel.y < 0 { // landed: scrub sideways speed
                    vel.x *= 0.7
                    vel.z *= 0.7
                }
                vel[axis] = 0
            } else {
                pos = p
            }
        }
    }
}
