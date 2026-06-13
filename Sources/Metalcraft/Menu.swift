import AppKit
import MetalKit
import simd

/// Which full-screen menu is up. nil = playing. `.pause` overlays the world;
/// the others replace the frame with the classic dirt backdrop.
enum MenuScreen {
    case title, selectWorld, confirmDelete, pause
}

struct MenuButton {
    var id: String
    var rect: CGRect // view points, AppKit y-up
    var label: String
    var enabled = true
}

final class MenuState {
    var screen: MenuScreen? = .title
    var saves: [SaveSummary] = []
    var selectedSave: Int?

    func refreshSaves() {
        saves = SaveIO.list()
        selectedSave = nil
    }
}

/// A piece of menu text, drawn as a pooled NSTextField over the Metal view.
struct MenuLabel {
    var text: String
    var center: CGPoint
    var size: CGFloat
    var color: NSColor
}

// MARK: - Menu layout, drawing and clicks

extension Renderer {
    private static let saveDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .short
        return f
    }()

    /// Button layout for the current screen; the same rects drive drawing,
    /// hover, labels and hit-testing.
    func menuButtons(_ bounds: CGSize) -> [MenuButton] {
        guard let screen = menu.screen else { return [] }
        let s = uiScale
        let bw = 200 * s, bh = 20 * s, halfW = 98 * s
        let cx = bounds.width / 2
        let leftX = cx - halfW - 2 * s
        let rightX = cx + 2 * s
        var out: [MenuButton] = []
        switch screen {
        case .title:
            let top = bounds.height * 0.46
            out.append(MenuButton(id: "singleplayer",
                                  rect: CGRect(x: cx - bw / 2, y: top, width: bw, height: bh),
                                  label: "Singleplayer"))
            out.append(MenuButton(id: "quit_game",
                                  rect: CGRect(x: cx - bw / 2, y: top - bh - 8, width: bw, height: bh),
                                  label: "Quit Game"))
        case .selectWorld:
            for (i, save) in menu.saves.prefix(6).enumerated() {
                let y = bounds.height - 64 - CGFloat(i + 1) * (bh + 8)
                let when = Self.saveDateFormatter.string(from: save.lastPlayed)
                out.append(MenuButton(id: "save_\(i)",
                                      rect: CGRect(x: cx - bw / 2, y: y, width: bw, height: bh),
                                      label: "\(save.name)  ·  \(when)"))
            }
            let hasSelection = menu.selectedSave != nil
            let row1 = 24 + bh + 8
            let row2: CGFloat = 24
            out.append(MenuButton(id: "play",
                                  rect: CGRect(x: leftX, y: row1, width: halfW, height: bh),
                                  label: "Play Selected World", enabled: hasSelection))
            out.append(MenuButton(id: "create",
                                  rect: CGRect(x: rightX, y: row1, width: halfW, height: bh),
                                  label: "Create New World"))
            out.append(MenuButton(id: "delete",
                                  rect: CGRect(x: leftX, y: row2, width: halfW, height: bh),
                                  label: "Delete", enabled: hasSelection))
            out.append(MenuButton(id: "cancel_select",
                                  rect: CGRect(x: rightX, y: row2, width: halfW, height: bh),
                                  label: "Cancel"))
        case .confirmDelete:
            let y = bounds.height / 2 - 40
            out.append(MenuButton(id: "confirm_delete",
                                  rect: CGRect(x: leftX, y: y, width: halfW, height: bh),
                                  label: "Delete"))
            out.append(MenuButton(id: "cancel_delete",
                                  rect: CGRect(x: rightX, y: y, width: halfW, height: bh),
                                  label: "Cancel"))
        case .pause:
            let top = bounds.height * 0.5
            out.append(MenuButton(id: "back_to_game",
                                  rect: CGRect(x: cx - bw / 2, y: top, width: bw, height: bh),
                                  label: "Back to Game"))
            out.append(MenuButton(id: "save_quit",
                                  rect: CGRect(x: cx - bw / 2, y: top - bh - 8, width: bw, height: bh),
                                  label: "Save and Quit to Title"))
        }
        return out
    }

    /// Consume this frame's clicks against the given layout. Only the first
    /// hit acts — an action can rebuild the whole screen under the cursor.
    func handleMenuClicks(_ buttons: [MenuButton], bounds: CGSize) {
        let clicks = input.guiLeftClicks
        input.guiLeftClicks.removeAll()
        input.guiRightClicks.removeAll()
        for p in clicks {
            guard let b = buttons.first(where: { $0.enabled && $0.rect.contains(p) })
            else { continue }
            menuAction(b.id)
            break
        }
    }

    private func menuAction(_ id: String) {
        switch id {
        case "singleplayer":
            menu.refreshSaves()
            menu.screen = .selectWorld
        case "quit_game":
            NSApplication.shared.terminate(nil)
        case "play":
            if let i = menu.selectedSave, i < menu.saves.count {
                loadWorld(menu.saves[i])
            }
        case "create":
            startNewWorld()
        case "delete":
            if menu.selectedSave != nil { menu.screen = .confirmDelete }
        case "confirm_delete":
            if let i = menu.selectedSave, i < menu.saves.count {
                SaveIO.delete(menu.saves[i].dir)
            }
            menu.refreshSaves()
            menu.screen = .selectWorld
        case "cancel_delete":
            menu.screen = .selectWorld
        case "cancel_select":
            menu.screen = .title
        case "back_to_game":
            resumeGame()
        case "save_quit":
            quitToTitle()
        default:
            if id.hasPrefix("save_"), let i = Int(id.dropFirst(5)) {
                menu.selectedSave = i
            }
        }
    }

    // MARK: - Drawing

    /// Full-frame pass for the title / select-world / confirm screens:
    /// darkened dirt tiles, the logo, and gui.png buttons.
    func drawFrontMenu(view: MTKView) {
        let bounds = view.bounds.size
        guard bounds.width > 1, bounds.height > 1 else { return }

        for key in input.pressed where key == Keys.escape {
            switch menu.screen {
            case .selectWorld: menu.screen = .title
            case .confirmDelete: menu.screen = .selectWorld
            default: break
            }
        }
        input.pressed.removeAll()
        input.scrollSteps = 0
        input.leftClicks = 0
        input.rightClicks = 0

        handleMenuClicks(menuButtons(bounds), bounds: bounds)

        hud?.isHidden = true
        for label in countLabels { label.isHidden = true }

        // a click may just have started or loaded a world; render it next frame
        guard let screen = menu.screen, screen != .pause else {
            updateMenuLabels(view, [])
            return
        }

        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.setDepthStencilState(depthOff)

        // tiled dirt backdrop (16px tiles at 2× = 32pt), darkened like the
        // real game; the repeat sampler turns one quad into the whole grid
        drawUIQuad(enc, texture: texture("background", "gui"),
                   rect: CGRect(origin: .zero, size: CGSize(width: bounds.width, height: bounds.height)),
                   srcX: 0, srcY: 0,
                   srcW: Float(bounds.width / 32) * 256,
                   srcH: Float(bounds.height / 32) * 256,
                   bounds: bounds, tint: SIMD4(0.25, 0.25, 0.25, 1))

        var labels: [MenuLabel] = []
        let buttons = menuButtons(bounds)

        switch screen {
        case .title:
            let logoW = 256 * uiScale, logoH = 44 * uiScale
            drawUIQuad(enc, texture: texture("logo", "gui"),
                       rect: CGRect(x: (bounds.width - logoW) / 2,
                                    y: bounds.height - 80 - logoH,
                                    width: logoW, height: logoH),
                       srcX: 0, srcY: 0, srcW: 256, srcH: 44, bounds: bounds)
            labels.append(MenuLabel(text: "Metalcraft", center: CGPoint(x: 52, y: 16),
                                    size: 12, color: NSColor(white: 0.85, alpha: 1)))
        case .selectWorld:
            labels.append(MenuLabel(text: "Select World",
                                    center: CGPoint(x: bounds.width / 2, y: bounds.height - 36),
                                    size: 18, color: .white))
            if menu.saves.isEmpty {
                labels.append(MenuLabel(text: "No worlds yet — create one below",
                                        center: CGPoint(x: bounds.width / 2, y: bounds.height / 2),
                                        size: 14, color: NSColor(white: 0.7, alpha: 1)))
            }
            // white frame behind the selected world row
            if let i = menu.selectedSave,
               let row = buttons.first(where: { $0.id == "save_\(i)" }) {
                let r = row.rect.insetBy(dx: -3, dy: -3)
                drawSolidQuad(enc, ndcRect(r.minX, r.minY, r.width, r.height, bounds),
                              SIMD4(0.85, 0.85, 0.85, 1))
            }
        case .confirmDelete:
            let name = menu.selectedSave.flatMap {
                $0 < menu.saves.count ? menu.saves[$0].name : nil
            } ?? "?"
            labels.append(MenuLabel(text: "Are you sure you want to delete '\(name)'?",
                                    center: CGPoint(x: bounds.width / 2, y: bounds.height / 2 + 40),
                                    size: 16, color: .white))
            labels.append(MenuLabel(text: "This world will be lost forever!",
                                    center: CGPoint(x: bounds.width / 2, y: bounds.height / 2 + 8),
                                    size: 13, color: NSColor(white: 0.7, alpha: 1)))
        case .pause:
            break
        }

        for b in buttons {
            labels.append(drawButton(enc, b, bounds: bounds))
        }

        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
        updateMenuLabels(view, labels)
    }

    /// In-game pause overlay, drawn at the end of the world frame's encoder.
    /// Returns the labels for the caller to hand to updateMenuLabels.
    func drawPauseMenu(_ enc: MTLRenderCommandEncoder, view: MTKView) -> [MenuLabel] {
        let bounds = view.bounds.size
        guard bounds.width > 1, bounds.height > 1 else { return [] }
        enc.setDepthStencilState(depthOff)
        drawSolidQuad(enc, ndcRect(0, 0, bounds.width, bounds.height, bounds),
                      SIMD4(0, 0, 0, 0.6))
        var labels = [MenuLabel(text: "Game Menu",
                                center: CGPoint(x: bounds.width / 2, y: bounds.height * 0.5 + 32 * uiScale),
                                size: 18, color: .white)]
        for b in menuButtons(bounds) {
            labels.append(drawButton(enc, b, bounds: bounds))
        }
        return labels
    }

    /// Classic 200×20 button art from gui.png: disabled / normal / hover rows.
    /// Narrower buttons splice the texture's left and right halves so both
    /// end caps survive. Returns the button's centered text label.
    private func drawButton(_ enc: MTLRenderCommandEncoder, _ b: MenuButton,
                            bounds: CGSize) -> MenuLabel {
        let hover = b.enabled && b.rect.contains(input.cursor)
        let srcY: Float = b.enabled ? (hover ? 86 : 66) : 46
        let tex = texture("gui", "gui")
        let halfW = Float(b.rect.width / uiScale) / 2
        drawUIQuad(enc, texture: tex,
                   rect: CGRect(x: b.rect.minX, y: b.rect.minY,
                                width: b.rect.width / 2, height: b.rect.height),
                   srcX: 0, srcY: srcY, srcW: halfW, srcH: 20, bounds: bounds)
        drawUIQuad(enc, texture: tex,
                   rect: CGRect(x: b.rect.midX, y: b.rect.minY,
                                width: b.rect.width / 2, height: b.rect.height),
                   srcX: 200 - halfW, srcY: srcY, srcW: halfW, srcH: 20, bounds: bounds)
        let color: NSColor = b.enabled
            ? (hover ? NSColor(red: 1, green: 1, blue: 0.63, alpha: 1) : NSColor(white: 0.88, alpha: 1))
            : NSColor(white: 0.63, alpha: 1)
        return MenuLabel(text: b.label,
                         center: CGPoint(x: b.rect.midX, y: b.rect.midY),
                         size: b.rect.width < 250 ? 13 : 15, color: color)
    }

    func updateMenuLabels(_ view: MTKView, _ requests: [MenuLabel]) {
        while menuLabels.count < requests.count {
            let label = NSTextField(labelWithString: "")
            let shadow = NSShadow()
            shadow.shadowColor = NSColor.black.withAlphaComponent(0.8)
            shadow.shadowOffset = NSSize(width: 1.5, height: -1.5)
            label.shadow = shadow
            view.addSubview(label)
            menuLabels.append(label)
        }
        for (i, label) in menuLabels.enumerated() {
            if i < requests.count {
                let r = requests[i]
                label.font = .monospacedSystemFont(ofSize: r.size, weight: .bold)
                label.textColor = r.color
                if label.stringValue != r.text { label.stringValue = r.text }
                label.sizeToFit()
                label.setFrameOrigin(NSPoint(x: r.center.x - label.frame.width / 2,
                                             y: r.center.y - label.frame.height / 2))
                label.isHidden = false
            } else {
                label.isHidden = true
            }
        }
    }
}
