import simd

/// Full Minecraft-style inventory: slots 0-8 are the hotbar, 9-35 the main
/// grid shown in the inventory screen. Mining feeds it via item pickups,
/// placing consumes from the selected hotbar slot.
final class Inventory {
    static let hotbarCount = 9
    static let slotCount = 36
    static let stackLimit = 64

    var slots: [ItemStack?] = Array(repeating: nil, count: slotCount)
    var selected = 0

    var selectedStack: ItemStack? { slots[selected] }

    /// Returns false when nothing can take the items (drop stays on the
    /// ground). Tops up matching stacks before opening new slots.
    @discardableResult
    func add(_ item: Item, count: Int = 1) -> Bool {
        var remaining = count
        for i in 0..<slots.count where remaining > 0 {
            if var stack = slots[i], stack.item == item, stack.count < Self.stackLimit {
                let take = min(Self.stackLimit - stack.count, remaining)
                stack.count += take
                remaining -= take
                slots[i] = stack
            }
        }
        for i in 0..<slots.count where remaining > 0 && slots[i] == nil {
            let take = min(Self.stackLimit, remaining)
            slots[i] = ItemStack(item: item, count: take)
            remaining -= take
        }
        return remaining == 0
    }

    func consumeSelected() -> Item? {
        guard var stack = slots[selected] else { return nil }
        stack.count -= 1
        slots[selected] = stack.count > 0 ? stack : nil
        return stack.item
    }

    func clear() {
        slots = Array(repeating: nil, count: Self.slotCount)
        selected = 0
    }
}

/// A mined drop bouncing on the ground, waiting to be picked up. Blocks show
/// as mini cubes, materials as flat sprites.
final class ItemEntity {
    var item: Item
    var count: Int
    var pos: SIMD3<Float> // center
    var vel: SIMD3<Float>
    var age: Float = 0

    init(item: Item, pos: SIMD3<Float>, vel: SIMD3<Float>, count: Int = 1) {
        self.item = item
        self.count = count
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
