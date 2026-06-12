import MetalKit

enum Keys {
    static let a: UInt16 = 0
    static let s: UInt16 = 1
    static let d: UInt16 = 2
    static let w: UInt16 = 13
    static let r: UInt16 = 15
    static let one: UInt16 = 18
    static let two: UInt16 = 19
    static let three: UInt16 = 20
    static let four: UInt16 = 21
    static let five: UInt16 = 23
    static let six: UInt16 = 22
    static let seven: UInt16 = 26
    static let eight: UInt16 = 28
    static let nine: UInt16 = 25
    static let e: UInt16 = 14
    static let space: UInt16 = 49
    static let escape: UInt16 = 53
    static let f5: UInt16 = 96
}

final class Input {
    var keys = Set<UInt16>()
    var pressed: [UInt16] = [] // edge-triggered presses, consumed each frame
    var mouseDX: Float = 0
    var mouseDY: Float = 0
    var leftClicks = 0
    var leftDown = false // held = progressive mining
    var rightClicks = 0
    var scrollSteps = 0 // hotbar slot cycling
    var captured = false
    var sprint = false

    // GUI mode: the cursor is released and clicks route to the open screen
    var guiOpen = false
    var cursor = CGPoint.zero // view coords, AppKit y-up
    var guiLeftClicks: [CGPoint] = []
    var guiRightClicks: [CGPoint] = []
}

final class GameView: MTKView {
    let input = Input()

    override var acceptsFirstResponder: Bool { true }

    override init(frame: CGRect, device: MTLDevice?) {
        super.init(frame: frame, device: device)
        colorPixelFormat = .bgra8Unorm
        depthStencilPixelFormat = .depth32Float
        clearColor = MTLClearColor(red: 0.55, green: 0.74, blue: 0.95, alpha: 1)
        NotificationCenter.default.addObserver(
            self, selector: #selector(windowResigned),
            name: NSWindow.didResignKeyNotification, object: nil)
    }

    required init(coder: NSCoder) { fatalError("not supported") }

    @objc private func windowResigned(_ note: Notification) {
        if (note.object as? NSWindow) == window { setCaptured(false) }
    }

    func setCaptured(_ captured: Bool) {
        guard captured != input.captured else { return }
        input.captured = captured
        if captured {
            CGAssociateMouseAndMouseCursorPosition(0)
            NSCursor.hide()
            warpCursorToWindowCenter()
        } else {
            CGAssociateMouseAndMouseCursorPosition(1)
            NSCursor.unhide()
            input.keys.removeAll()
            input.leftDown = false
        }
    }

    private func warpCursorToWindowCenter() {
        guard let window else { return }
        let f = window.frame
        let screenHeight = NSScreen.screens.first?.frame.height ?? 0
        CGWarpMouseCursorPosition(CGPoint(x: f.midX, y: screenHeight - f.midY))
    }

    override func mouseDown(with event: NSEvent) {
        if input.guiOpen {
            input.guiLeftClicks.append(convert(event.locationInWindow, from: nil))
        } else if input.captured {
            input.leftClicks += 1
            input.leftDown = true
        } else {
            setCaptured(true)
        }
    }

    override func mouseUp(with event: NSEvent) {
        input.leftDown = false
    }

    override func rightMouseDown(with event: NSEvent) {
        if input.guiOpen {
            input.guiRightClicks.append(convert(event.locationInWindow, from: nil))
        } else if input.captured {
            input.rightClicks += 1
        }
    }

    override func scrollWheel(with event: NSEvent) {
        guard input.captured else { return }
        if event.scrollingDeltaY > 0.5 { input.scrollSteps -= 1 }
        else if event.scrollingDeltaY < -0.5 { input.scrollSteps += 1 }
    }

    override func mouseMoved(with event: NSEvent) { accumulate(event) }
    override func mouseDragged(with event: NSEvent) { accumulate(event) }
    override func rightMouseDragged(with event: NSEvent) { accumulate(event) }

    private func accumulate(_ event: NSEvent) {
        input.cursor = convert(event.locationInWindow, from: nil)
        guard input.captured else { return }
        input.mouseDX += Float(event.deltaX)
        input.mouseDY += Float(event.deltaY)
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == Keys.escape {
            if input.guiOpen {
                // the renderer closes the open screen and recaptures
                if !event.isARepeat { input.pressed.append(event.keyCode) }
            } else {
                setCaptured(false)
            }
            return
        }
        if !event.isARepeat { input.pressed.append(event.keyCode) }
        input.keys.insert(event.keyCode)
    }

    override func keyUp(with event: NSEvent) {
        input.keys.remove(event.keyCode)
    }

    override func flagsChanged(with event: NSEvent) {
        input.sprint = event.modifierFlags.contains(.shift)
    }
}
