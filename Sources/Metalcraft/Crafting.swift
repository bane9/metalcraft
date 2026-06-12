/// Shaped crafting: a recipe is the trimmed bounding box of its pattern, so
/// it matches anywhere in the 2x2 or 3x3 grid.
struct Recipe {
    let pattern: [[Item?]]
    let result: ItemStack
}

enum Crafting {
    private static let p = Item.block(.planks)
    private static let c = Item.block(.cobblestone)

    static let recipes: [Recipe] = [
        Recipe(pattern: [[.block(.wood)]],
               result: ItemStack(item: .block(.planks), count: 4)),
        Recipe(pattern: [[p], [p]],
               result: ItemStack(item: .stick, count: 4)),
        Recipe(pattern: [[p, p], [p, p]],
               result: ItemStack(item: .block(.craftingTable), count: 1)),
        Recipe(pattern: [[c, c, c], [c, nil, c], [c, c, c]],
               result: ItemStack(item: .block(.furnace), count: 1)),
    ]

    /// `grid` is row-major width×width. Returns what the grid would craft.
    static func match(grid: [ItemStack?], width: Int) -> ItemStack? {
        var minR = Int.max, maxR = -1, minC = Int.max, maxC = -1
        for r in 0..<width {
            for col in 0..<width where grid[r * width + col] != nil {
                minR = min(minR, r); maxR = max(maxR, r)
                minC = min(minC, col); maxC = max(maxC, col)
            }
        }
        guard maxR >= 0 else { return nil }
        let shape: [[Item?]] = (minR...maxR).map { r in
            (minC...maxC).map { col in grid[r * width + col]?.item }
        }
        return recipes.first { $0.pattern == shape }?.result
    }
}

/// Per-placed-furnace smelting state, keyed by block position in the world.
final class FurnaceState {
    static let cookTime: Float = 10

    var input: ItemStack?
    var fuel: ItemStack?
    var output: ItemStack?
    var burnLeft: Float = 0
    var burnTotal: Float = 1
    var cook: Float = 0

    var isBurning: Bool { burnLeft > 0 }

    static func smelt(_ item: Item) -> Item? {
        switch item {
        case .block(.ironOre): return .ironIngot
        case .block(.goldOre): return .goldIngot
        case .block(.cobblestone): return .block(.stone)
        default: return nil
        }
    }

    static func fuelTime(_ item: Item) -> Float? {
        switch item {
        case .coal: return 80
        case .block(.planks), .block(.wood), .block(.craftingTable): return 15
        case .stick: return 5
        default: return nil
        }
    }

    private var canSmelt: Bool {
        guard let input, let out = Self.smelt(input.item) else { return false }
        guard let output else { return true }
        return output.item == out && output.count < Inventory.stackLimit
    }

    func tick(dt: Float) {
        if burnLeft <= 0, canSmelt, let f = fuel, let time = Self.fuelTime(f.item) {
            burnLeft = time
            burnTotal = time
            fuel = f.count > 1 ? ItemStack(item: f.item, count: f.count - 1) : nil
        }
        guard burnLeft > 0 else {
            cook = max(0, cook - dt * 2) // unburnt progress decays
            return
        }
        burnLeft -= dt
        guard canSmelt, let inp = input else {
            cook = 0
            return
        }
        cook += dt
        if cook >= Self.cookTime {
            cook = 0
            let out = Self.smelt(inp.item)!
            if var o = output {
                o.count += 1
                output = o
            } else {
                output = ItemStack(item: out, count: 1)
            }
            input = inp.count > 1 ? ItemStack(item: inp.item, count: inp.count - 1) : nil
        }
    }
}
