import simd

/// Tool rows in items.png start at row 4; the raw value is the row offset.
enum ToolType: Int, Hashable, CaseIterable {
    case sword = 0, shovel, pickaxe, axe, hoe
}

/// Tool columns in items.png; the raw value is the column.
enum ToolMaterial: Int, Hashable, CaseIterable {
    case wood = 0, stone, iron, diamond, gold

    /// Dig-speed multiplier when the tool matches the block, like the real game.
    var miningSpeed: Float {
        switch self {
        case .wood: return 2
        case .stone: return 4
        case .iron: return 6
        case .diamond: return 8
        case .gold: return 12
        }
    }

    var ingredient: Item {
        switch self {
        case .wood: return .block(.planks)
        case .stone: return .block(.cobblestone)
        case .iron: return .ironIngot
        case .diamond: return .diamond
        case .gold: return .goldIngot
        }
    }
}

/// Armor rows 0-3 in items.png; the raw value is the row.
enum ArmorPiece: Int, Hashable, CaseIterable {
    case helmet = 0, chestplate, leggings, boots
}

/// Armor columns in items.png; the raw value is the column.
enum ArmorMaterial: Int, Hashable, CaseIterable {
    case leather = 0, chain, iron, diamond, gold

    /// nil = not craftable (chainmail), like the real game
    var ingredient: Item? {
        switch self {
        case .leather: return .leather
        case .chain: return nil
        case .iron: return .ironIngot
        case .diamond: return .diamond
        case .gold: return .goldIngot
        }
    }
}

/// Everything a slot can hold: placeable blocks plus items from items.png.
enum Item: Hashable {
    case block(Block)
    case stick, coal, ironIngot, goldIngot, diamond
    case flint, flintAndSteel, bow, arrow, fishingRod
    case string, feather, gunpowder, redstone, leather
    case porkchopRaw, porkchopCooked
    case bowl, bucket, compass
    case sign, doorWood, doorIron, minecart, boat
    case tool(ToolType, ToolMaterial)
    case armor(ArmorPiece, ArmorMaterial)

    /// items.png sprite tile (column, row) for non-block items; blocks render
    /// as isometric mini cubes instead.
    var sprite: SIMD2<Float>? {
        switch self {
        case .block: return nil
        case .tool(let t, let m): return SIMD2(Float(m.rawValue), Float(4 + t.rawValue))
        case .armor(let p, let m): return SIMD2(Float(m.rawValue), Float(p.rawValue))
        case .flintAndSteel: return SIMD2(5, 0)
        case .bow: return SIMD2(5, 1)
        case .arrow: return SIMD2(5, 2)
        case .stick: return SIMD2(5, 3)
        case .fishingRod: return SIMD2(5, 4)
        case .flint: return SIMD2(6, 0)
        case .compass: return SIMD2(6, 3)
        case .coal: return SIMD2(7, 0)
        case .ironIngot: return SIMD2(7, 1)
        case .goldIngot: return SIMD2(7, 2)
        case .diamond: return SIMD2(7, 3)
        case .bowl: return SIMD2(7, 4)
        case .porkchopRaw: return SIMD2(7, 5)
        case .leather: return SIMD2(7, 6)
        case .string: return SIMD2(8, 0)
        case .feather: return SIMD2(8, 1)
        case .gunpowder: return SIMD2(8, 2)
        case .redstone: return SIMD2(8, 3)
        case .porkchopCooked: return SIMD2(8, 5)
        case .sign: return SIMD2(10, 2)
        case .doorWood: return SIMD2(11, 2)
        case .doorIron: return SIMD2(12, 2)
        case .bucket: return SIMD2(10, 4)
        case .minecart: return SIMD2(7, 8)
        case .boat: return SIMD2(8, 8)
        }
    }

    var asBlock: Block? {
        if case .block(let b) = self { return b }
        return nil
    }

    /// Punch damage when held: swords hit like the real game's tiers.
    var attackDamage: Float {
        if case .tool(.sword, let m) = self {
            switch m {
            case .wood, .gold: return 6
            case .stone: return 8
            case .iron: return 10
            case .diamond: return 12
            }
        }
        return 4
    }
}

struct ItemStack {
    var item: Item
    var count: Int
}

/// Progressive-mining data: Minecraft hardness values and the tool class
/// that digs each block faster. Bedrock's -1 means unbreakable.
extension Block {
    var hardness: Float {
        switch self {
        case .air: return 0
        case .torch: return 0 // pops instantly
        case .leaves, .snow: return 0.2
        case .cactus: return 0.4
        case .dirt, .sand: return 0.5
        case .grass, .gravel: return 0.6
        case .stone: return 1.5
        case .wood, .planks, .cobblestone: return 2
        case .craftingTable: return 2.5
        case .coalOre, .ironOre, .goldOre, .diamondOre, .redstoneOre: return 3
        case .furnace, .furnaceLit: return 3.5
        case .bedrock: return -1
        default: return 1
        }
    }

    var preferredTool: ToolType? {
        switch self {
        case .grass, .dirt, .sand, .gravel, .snow: return .shovel
        case .stone, .cobblestone, .coalOre, .ironOre, .goldOre,
             .diamondOre, .redstoneOre, .furnace, .furnaceLit: return .pickaxe
        case .wood, .planks, .craftingTable: return .axe
        default: return nil
        }
    }
}
