import simd

/// Everything a slot can hold: placeable blocks plus crafting materials.
enum Item: Hashable {
    case block(Block)
    case stick, coal, ironIngot, goldIngot, diamond

    /// items.png sprite tile (column, row) for non-block items; blocks render
    /// as isometric mini cubes instead.
    var sprite: SIMD2<Float>? {
        switch self {
        case .block: return nil
        case .stick: return SIMD2(5, 3)
        case .coal: return SIMD2(7, 0)
        case .ironIngot: return SIMD2(7, 1)
        case .goldIngot: return SIMD2(7, 2)
        case .diamond: return SIMD2(7, 3)
        }
    }

    var asBlock: Block? {
        if case .block(let b) = self { return b }
        return nil
    }
}

struct ItemStack {
    var item: Item
    var count: Int
}
