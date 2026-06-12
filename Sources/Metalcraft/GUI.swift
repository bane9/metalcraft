import Foundation

/// Which container screen is open. All dialogs are classic 176×166 textures.
enum GUIScreen {
    case inventory          // 2x2 crafting
    case craftingTable      // 3x3 crafting
    case furnace(BlockPos)

    var textureName: String {
        switch self {
        case .inventory: return "inventory"
        case .craftingTable: return "crafting"
        case .furnace: return "furnace"
        }
    }
}

/// One interactive 16×16 slot, positioned in dialog pixels from the top-left
/// of the 176×166 texture (matching the art's slot frames).
struct GUISlot {
    enum Source: Equatable {
        case inventory(Int) // index into Inventory.slots
        case craft(Int)     // index into the crafting grid
        case craftResult
        case furnaceInput, furnaceFuel, furnaceOutput
    }
    var source: Source
    var x: Int
    var y: Int
}

/// Screen state + click handling for the inventory/crafting/furnace dialogs.
/// Pure logic: the renderer asks for `slots` to draw and forwards clicks.
final class GUIState {
    private(set) var screen: GUIScreen?
    private(set) var craftWidth = 2
    var craftGrid: [ItemStack?] = []
    var held: ItemStack? // stack picked up on the cursor

    var craftResult: ItemStack? {
        screen == nil ? nil : Crafting.match(grid: craftGrid, width: craftWidth)
    }

    func open(_ screen: GUIScreen) {
        self.screen = screen
        if case .craftingTable = screen {
            craftWidth = 3
        } else {
            craftWidth = 2
        }
        craftGrid = Array(repeating: nil, count: craftWidth * craftWidth)
        held = nil
    }

    /// Returns everything that must go back to the player (grid + cursor).
    func close() -> [ItemStack] {
        let leftovers = (craftGrid + [held]).compactMap { $0 }
        screen = nil
        craftGrid = []
        held = nil
        return leftovers
    }

    var slots: [GUISlot] {
        guard let screen else { return [] }
        var out: [GUISlot] = []
        // shared player area: main inventory rows + hotbar row
        for row in 0..<3 {
            for col in 0..<9 {
                out.append(GUISlot(source: .inventory(9 + row * 9 + col),
                                   x: 8 + col * 18, y: 84 + row * 18))
            }
        }
        for col in 0..<9 {
            out.append(GUISlot(source: .inventory(col), x: 8 + col * 18, y: 142))
        }
        switch screen {
        case .inventory:
            for r in 0..<2 {
                for c in 0..<2 {
                    out.append(GUISlot(source: .craft(r * 2 + c), x: 88 + c * 18, y: 26 + r * 18))
                }
            }
            out.append(GUISlot(source: .craftResult, x: 144, y: 36))
        case .craftingTable:
            for r in 0..<3 {
                for c in 0..<3 {
                    out.append(GUISlot(source: .craft(r * 3 + c), x: 30 + c * 18, y: 17 + r * 18))
                }
            }
            out.append(GUISlot(source: .craftResult, x: 124, y: 35))
        case .furnace:
            out.append(GUISlot(source: .furnaceInput, x: 56, y: 17))
            out.append(GUISlot(source: .furnaceFuel, x: 56, y: 53))
            out.append(GUISlot(source: .furnaceOutput, x: 116, y: 35))
        }
        return out
    }

    // MARK: - Clicks

    func click(_ source: GUISlot.Source, right: Bool,
               inventory: Inventory, furnace: FurnaceState?) {
        switch source {
        case .craftResult:
            takeCraftResult()
        case .furnaceOutput:
            if let furnace { takeOutput(&furnace.output) }
        default:
            var stack = read(source, inventory: inventory, furnace: furnace)
            interact(slot: &stack, right: right, accepts: accepts(source))
            write(stack, to: source, inventory: inventory, furnace: furnace)
        }
    }

    /// What a slot will admit from the held stack (furnace slots filter).
    private func accepts(_ source: GUISlot.Source) -> (Item) -> Bool {
        switch source {
        case .furnaceFuel: return { FurnaceState.fuelTime($0) != nil }
        case .furnaceInput: return { FurnaceState.smelt($0) != nil }
        default: return { _ in true }
        }
    }

    /// Standard Minecraft slot semantics: left click swaps/merges whole
    /// stacks, right click takes half / places one.
    private func interact(slot: inout ItemStack?, right: Bool, accepts: (Item) -> Bool) {
        switch (held, slot) {
        case (nil, .some(let s)):
            if right {
                let take = (s.count + 1) / 2
                held = ItemStack(item: s.item, count: take)
                let rest = s.count - take
                slot = rest > 0 ? ItemStack(item: s.item, count: rest) : nil
            } else {
                held = s
                slot = nil
            }
        case (.some(let h), nil):
            guard accepts(h.item) else { return }
            if right {
                slot = ItemStack(item: h.item, count: 1)
                held = h.count > 1 ? ItemStack(item: h.item, count: h.count - 1) : nil
            } else {
                slot = h
                held = nil
            }
        case (.some(let h), .some(var s)) where h.item == s.item:
            let move = right ? 1 : h.count
            let take = min(Inventory.stackLimit - s.count, move)
            guard take > 0 else {
                if !right { swap(&held, &slot) } // both full: plain swap
                return
            }
            s.count += take
            slot = s
            held = h.count - take > 0 ? ItemStack(item: h.item, count: h.count - take) : nil
        case (.some(let h), .some(let s)):
            guard accepts(h.item) else { return }
            held = s
            slot = h
        case (nil, nil):
            break
        }
    }

    private func takeCraftResult() {
        guard let result = craftResult else { return }
        if let h = held {
            guard h.item == result.item,
                  h.count + result.count <= Inventory.stackLimit else { return }
            held = ItemStack(item: h.item, count: h.count + result.count)
        } else {
            held = result
        }
        for i in craftGrid.indices {
            guard let s = craftGrid[i] else { continue }
            craftGrid[i] = s.count > 1 ? ItemStack(item: s.item, count: s.count - 1) : nil
        }
    }

    private func takeOutput(_ output: inout ItemStack?) {
        guard let out = output else { return }
        if let h = held {
            guard h.item == out.item else { return }
            let take = min(Inventory.stackLimit - h.count, out.count)
            guard take > 0 else { return }
            held = ItemStack(item: h.item, count: h.count + take)
            let rest = out.count - take
            output = rest > 0 ? ItemStack(item: out.item, count: rest) : nil
        } else {
            held = out
            output = nil
        }
    }

    // MARK: - Slot storage plumbing

    func read(_ source: GUISlot.Source, inventory: Inventory, furnace: FurnaceState?) -> ItemStack? {
        switch source {
        case .inventory(let i): return inventory.slots[i]
        case .craft(let i): return craftGrid[i]
        case .craftResult: return craftResult
        case .furnaceInput: return furnace?.input
        case .furnaceFuel: return furnace?.fuel
        case .furnaceOutput: return furnace?.output
        }
    }

    private func write(_ stack: ItemStack?, to source: GUISlot.Source,
                       inventory: Inventory, furnace: FurnaceState?) {
        switch source {
        case .inventory(let i): inventory.slots[i] = stack
        case .craft(let i): craftGrid[i] = stack
        case .craftResult: break
        case .furnaceInput: furnace?.input = stack
        case .furnaceFuel: furnace?.fuel = stack
        case .furnaceOutput: furnace?.output = stack
        }
    }
}
