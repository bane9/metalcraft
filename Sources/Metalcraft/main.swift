import Cocoa
import MetalKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
let appDelegate = AppDelegate()
app.delegate = appDelegate
app.setActivationPolicy(.regular)

let mainMenu = NSMenu()
let appMenuItem = NSMenuItem()
mainMenu.addItem(appMenuItem)
let appMenu = NSMenu()
appMenu.addItem(NSMenuItem(title: "Quit Metalcraft",
                           action: #selector(NSApplication.terminate(_:)),
                           keyEquivalent: "q"))
appMenuItem.submenu = appMenu
app.mainMenu = mainMenu

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("This machine has no Metal device")
}

let contentRect = NSRect(x: 0, y: 0, width: 1280, height: 800)
let window = NSWindow(contentRect: contentRect,
                      styleMask: [.titled, .closable, .miniaturizable, .resizable],
                      backing: .buffered, defer: false)
window.title = "Metalcraft"

let gameView = GameView(frame: contentRect, device: device)
let renderer = Renderer(device: device, view: gameView, input: gameView.input)
gameView.delegate = renderer

let hud = NSTextField(labelWithString: "")
hud.font = .monospacedSystemFont(ofSize: 13, weight: .semibold)
hud.textColor = .white
hud.backgroundColor = NSColor.black.withAlphaComponent(0.4)
hud.drawsBackground = true
gameView.addSubview(hud)
renderer.hud = hud

window.contentView = gameView
window.center()
window.acceptsMouseMovedEvents = true
window.makeKeyAndOrderFront(nil)
window.makeFirstResponder(gameView)
app.activate(ignoringOtherApps: true)
app.run()
