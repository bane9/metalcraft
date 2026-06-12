/// Shaped crafting: a recipe is the trimmed bounding box of its pattern, so
/// it matches anywhere in the 2x2 or 3x3 grid.
struct Recipe {
    let pattern: [[Item?]]
    let result: ItemStack
}

enum Crafting {
    private static let p = Item.block(.planks)
    private static let c = Item.block(.cobblestone)
    private static let s = Item.stick
    private static let i = Item.ironIngot

    /// Horizontal flip, so handed recipes (axe, hoe, bow) match either way.
    private static func mirrored(_ pattern: [[Item?]]) -> [[Item?]] {
        pattern.map { $0.reversed() }
    }

    private static func add(_ recipes: inout [Recipe], _ pattern: [[Item?]],
                            _ item: Item, _ count: Int = 1) {
        recipes.append(Recipe(pattern: pattern, result: ItemStack(item: item, count: count)))
        let flipped = mirrored(pattern)
        if flipped != pattern {
            recipes.append(Recipe(pattern: flipped, result: ItemStack(item: item, count: count)))
        }
    }

    static let recipes: [Recipe] = {
        var r: [Recipe] = []
        add(&r, [[.block(.wood)]], .block(.planks), 4)
        add(&r, [[p], [p]], .stick, 4)
        add(&r, [[p, p], [p, p]], .block(.craftingTable))
        add(&r, [[c, c, c], [c, nil, c], [c, c, c]], .block(.furnace))
        add(&r, [[Item.coal], [s]], .block(.torch), 4)

        // tools: same shapes as the real game, one set per material
        for m in ToolMaterial.allCases {
            let x = m.ingredient
            add(&r, [[x], [x], [s]], .tool(.sword, m))
            add(&r, [[x], [s], [s]], .tool(.shovel, m))
            add(&r, [[x, x, x], [nil, s, nil], [nil, s, nil]], .tool(.pickaxe, m))
            add(&r, [[x, x], [x, s], [nil, s]], .tool(.axe, m))
            add(&r, [[x, x], [nil, s], [nil, s]], .tool(.hoe, m))
        }

        // armor (chainmail stays uncraftable, like the real game)
        for m in ArmorMaterial.allCases {
            guard let x = m.ingredient else { continue }
            add(&r, [[x, x, x], [x, nil, x]], .armor(.helmet, m))
            add(&r, [[x, nil, x], [x, x, x], [x, x, x]], .armor(.chestplate, m))
            add(&r, [[x, x, x], [x, nil, x], [x, nil, x]], .armor(.leggings, m))
            add(&r, [[x, nil, x], [x, nil, x]], .armor(.boots, m))
        }

        // utility items
        add(&r, [[i, nil], [nil, .flint]], .flintAndSteel)
        add(&r, [[.flint], [s], [.feather]], .arrow, 4)
        add(&r, [[nil, s, .string], [s, nil, .string], [nil, s, .string]], .bow)
        add(&r, [[nil, nil, s], [nil, s, .string], [s, nil, .string]], .fishingRod)
        add(&r, [[nil, i, nil], [i, .redstone, i], [nil, i, nil]], .compass)
        add(&r, [[p, nil, p], [nil, p, nil]], .bowl, 4)
        add(&r, [[i, nil, i], [nil, i, nil]], .bucket)
        add(&r, [[p, p, p], [p, p, p], [nil, s, nil]], .sign)
        add(&r, [[p, p], [p, p], [p, p]], .doorWood)
        add(&r, [[i, i], [i, i], [i, i]], .doorIron)
        let w = Item.block(.wool)
        add(&r, [[w, w, w], [p, p, p]], .bed)
        add(&r, [[i, nil, i], [i, i, i]], .minecart)
        add(&r, [[p, nil, p], [p, p, p]], .boat)
        return r
    }()

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
        case .porkchopRaw: return .porkchopCooked
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
