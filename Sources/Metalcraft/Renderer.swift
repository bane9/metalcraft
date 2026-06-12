import MetalKit
import AppKit
import simd

struct Uniforms {
    var viewProj: simd_float4x4
    var model: simd_float4x4
    var camPos: SIMD4<Float>
    var sunDir: SIMD4<Float>
    var fogColor: SIMD4<Float>
    var fogParams: SIMD4<Float>   // x = fog start, y = fog end
    var alphaParams: SIMD4<Float> // x = alpha multiplier, y = discard threshold
}

struct LineUniforms {
    var mvp: simd_float4x4
    var color: SIMD4<Float>
}

final class ChunkMesh {
    var vertexBuffer: MTLBuffer?
    var indexBuffer: MTLBuffer?
    var indexCount = 0
    var waterVertexBuffer: MTLBuffer?
    var waterIndexBuffer: MTLBuffer?
    var waterIndexCount = 0
}

struct CubeMesh {
    var vertexBuffer: MTLBuffer
    var indexBuffer: MTLBuffer
    var indexCount: Int
}

/// GPU-side mob model: one mesh per (part, texture) pair, animated by
/// composing per-part pivot rotations at draw time.
struct MobPartMesh {
    var role: PartRole
    var pivot: SIMD3<Float> // meters
    var baseRotX: Float
    var submeshes: [(tex: Int, mesh: CubeMesh)]
}

struct MobModel {
    var parts: [MobPartMesh]
    var textures: [MTLTexture]
}

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let blockPipeline: MTLRenderPipelineState
    let waterPipeline: MTLRenderPipelineState
    let linePipeline: MTLRenderPipelineState
    let uiPipeline: MTLRenderPipelineState
    let depthOn: MTLDepthStencilState
    let depthReadOnly: MTLDepthStencilState
    let depthOff: MTLDepthStencilState
    let atlas: MTLTexture
    let cubeLineBuffer: MTLBuffer
    let crosshairBuffer: MTLBuffer
    let quadBuffer: MTLBuffer

    let world = World()
    let player = Player()
    let inventory = Inventory()
    var items: [ItemEntity] = []
    var mobs: [Mob] = []
    let input: Input
    weak var hud: NSTextField?
    private var countLabels: [NSTextField] = []

    private var meshes: [ChunkCoord: ChunkMesh] = [:]
    private var itemMeshCache: [UInt8: CubeMesh] = [:]
    private var iconMeshCache: [UInt8: CubeMesh] = [:]
    private var mobModelCache: [MobKind: MobModel] = [:]
    private var mobTextureCache: [String: MTLTexture] = [:]
    private lazy var playerModel = buildModel(MobModels.humanoid(), textureNames: ["char"])
    private var mobSpawnTimer: Float = 0
    private var thirdPerson = false
    private var aspect: Float = 16.0 / 9.0
    private var lastTime = CACurrentMediaTime()
    private var lastSpaceTap: CFTimeInterval = -10
    private var waterTickAccum: Float = 0

    // chunk streaming: generate a bit beyond what we mesh, so every meshed
    // chunk has all four neighbors available for face culling
    private let genRadius = 15
    private let meshRadius = 14
    private let genOffsets: [(Int, Int)]
    private let meshOffsets: [(Int, Int)]
    private let genBudgetPerFrame = 24
    private let meshBudgetPerFrame = 10

    private let skyColor = SIMD4<Float>(0.55, 0.74, 0.95, 1)
    private let waterFogColor = SIMD4<Float>(0.07, 0.20, 0.40, 1)

    init(device: MTLDevice, view: MTKView, input: Input) {
        self.device = device
        self.input = input
        queue = device.makeCommandQueue()!

        let library = try! device.makeLibrary(source: shaderSource, options: nil)

        func makePipeline(vertex: String, fragment: String, blended: Bool) -> MTLRenderPipelineState {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = library.makeFunction(name: vertex)
            desc.fragmentFunction = library.makeFunction(name: fragment)
            let ca = desc.colorAttachments[0]!
            ca.pixelFormat = view.colorPixelFormat
            if blended {
                ca.isBlendingEnabled = true
                ca.rgbBlendOperation = .add
                ca.alphaBlendOperation = .add
                ca.sourceRGBBlendFactor = .sourceAlpha
                ca.destinationRGBBlendFactor = .oneMinusSourceAlpha
                ca.sourceAlphaBlendFactor = .one
                ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            desc.depthAttachmentPixelFormat = view.depthStencilPixelFormat
            return try! device.makeRenderPipelineState(descriptor: desc)
        }
        blockPipeline = makePipeline(vertex: "block_vertex", fragment: "block_fragment", blended: false)
        waterPipeline = makePipeline(vertex: "block_vertex", fragment: "block_fragment", blended: true)
        linePipeline = makePipeline(vertex: "line_vertex", fragment: "line_fragment", blended: false)
        uiPipeline = makePipeline(vertex: "line_vertex", fragment: "line_fragment", blended: true)

        func makeDepthState(compare: MTLCompareFunction, write: Bool) -> MTLDepthStencilState {
            let d = MTLDepthStencilDescriptor()
            d.depthCompareFunction = compare
            d.isDepthWriteEnabled = write
            return device.makeDepthStencilState(descriptor: d)!
        }
        depthOn = makeDepthState(compare: .less, write: true)
        depthReadOnly = makeDepthState(compare: .less, write: false)
        depthOff = makeDepthState(compare: .always, write: false)

        let loader = MTKTextureLoader(device: device)
        let atlasURL = Bundle.module.url(forResource: "terrain", withExtension: "png")!
        atlas = try! loader.newTexture(URL: atlasURL, options: [
            .SRGB: false,
            .generateMipmaps: true,
        ])

        // wireframe unit cube (12 edges, 24 line vertices), slightly inflated
        let lo: Float = -0.004, hi: Float = 1.004
        var cube: [Float] = []
        let edges: [(SIMD3<Float>, SIMD3<Float>)] = [
            (SIMD3(lo, lo, lo), SIMD3(hi, lo, lo)), (SIMD3(hi, lo, lo), SIMD3(hi, lo, hi)),
            (SIMD3(hi, lo, hi), SIMD3(lo, lo, hi)), (SIMD3(lo, lo, hi), SIMD3(lo, lo, lo)),
            (SIMD3(lo, hi, lo), SIMD3(hi, hi, lo)), (SIMD3(hi, hi, lo), SIMD3(hi, hi, hi)),
            (SIMD3(hi, hi, hi), SIMD3(lo, hi, hi)), (SIMD3(lo, hi, hi), SIMD3(lo, hi, lo)),
            (SIMD3(lo, lo, lo), SIMD3(lo, hi, lo)), (SIMD3(hi, lo, lo), SIMD3(hi, hi, lo)),
            (SIMD3(hi, lo, hi), SIMD3(hi, hi, hi)), (SIMD3(lo, lo, hi), SIMD3(lo, hi, hi)),
        ]
        for (a, b) in edges {
            cube.append(contentsOf: [a.x, a.y, a.z, b.x, b.y, b.z])
        }
        cubeLineBuffer = device.makeBuffer(bytes: cube, length: cube.count * 4)!

        let cross: [Float] = [-1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0]
        crosshairBuffer = device.makeBuffer(bytes: cross, length: cross.count * 4)!

        // unit quad for UI rectangles
        let quad: [Float] = [
            0, 0, 0, 1, 0, 0, 1, 1, 0,
            0, 0, 0, 1, 1, 0, 0, 1, 0,
        ]
        quadBuffer = device.makeBuffer(bytes: quad, length: quad.count * 4)!

        func sortedOffsets(radius: Int) -> [(Int, Int)] {
            var offs: [(Int, Int)] = []
            for dz in -radius...radius {
                for dx in -radius...radius { offs.append((dx, dz)) }
            }
            offs.sort { $0.0 * $0.0 + $0.1 * $0.1 < $1.0 * $1.0 + $1.1 * $1.1 }
            return offs
        }
        genOffsets = sortedOffsets(radius: genRadius)
        meshOffsets = sortedOffsets(radius: meshRadius)

        super.init()

        for _ in 0..<Inventory.slotCount {
            let label = NSTextField(labelWithString: "")
            label.font = .monospacedDigitSystemFont(ofSize: 12, weight: .bold)
            label.textColor = .white
            label.isHidden = true
            view.addSubview(label)
            countLabels.append(label)
        }

        resetWorld()
    }

    private var playerChunk: ChunkCoord {
        World.chunkCoord(blockX: Int(player.pos.x.rounded(.down)),
                         blockZ: Int(player.pos.z.rounded(.down)))
    }

    private func resetWorld() {
        world.reset(seed: UInt64.random(in: 0...UInt64.max))
        meshes.removeAll()
        items.removeAll()
        mobs.removeAll()
        inventory.clear()
        for dz in -2...2 {
            for dx in -2...2 { world.generateChunk(ChunkCoord(x: dx, z: dz)) }
        }
        player.spawn(in: world)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspect = Float(size.width / max(size.height, 1))
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let dt = min(Float(now - lastTime), 0.05)
        lastTime = now

        if input.captured {
            player.look(dx: input.mouseDX, dy: input.mouseDY)
        }
        input.mouseDX = 0
        input.mouseDY = 0

        for key in input.pressed { handleKeyPress(key, now: now) }
        input.pressed.removeAll()
        if input.scrollSteps != 0 {
            let n = Inventory.slotCount
            inventory.selected = ((inventory.selected + input.scrollSteps) % n + n) % n
            input.scrollSteps = 0
        }

        streamGeneration()

        if input.captured {
            player.update(dt: dt, input: input, world: world)
        }

        waterTickAccum += dt
        if waterTickAccum >= 0.25 {
            waterTickAccum = 0
            world.tickWater()
        }

        let hit = raycast(origin: player.eye, dir: player.forward, maxDist: 6, world: world)

        if input.leftClicks > 0 {
            // punch a mob if one is in reach and closer than the targeted block
            let reach = min(hit?.t ?? .infinity, 3.5)
            if let mob = nearestMobHit(origin: player.eye, dir: player.forward, maxDist: reach) {
                var kb = SIMD3<Float>(player.forward.x, 0, player.forward.z)
                if simd_length_squared(kb) > 1e-6 { kb = simd_normalize(kb) }
                mob.hurt(damage: 4, direction: kb)
            } else {
                mine(hit)
            }
        }
        if input.rightClicks > 0 { place(hit) }
        input.leftClicks = 0
        input.rightClicks = 0

        updateItems(dt: dt)
        updateMobs(dt: dt)
        remeshDirtyChunks()
        streamMeshes()

        // camera: player eye, pulled back in third person, or swaying with
        // the walk cycle in first person
        var eye = player.eye
        if thirdPerson {
            var dist: Float = 4
            var t: Float = 0.25
            while t < dist {
                let p = player.eye - player.forward * t
                if world.isSolid(Int(p.x.rounded(.down)),
                                 Int(p.y.rounded(.down)),
                                 Int(p.z.rounded(.down))) {
                    dist = max(t - 0.35, 0.4)
                    break
                }
                t += 0.1
            }
            eye -= player.forward * dist
        } else {
            let right = SIMD3<Float>(cos(player.yaw), 0, sin(player.yaw))
            let b = min(player.bobAmount, 1.2) * 0.045
            eye += SIMD3(0, abs(cos(player.bobPhase)) * b, 0)
                + right * sin(player.bobPhase) * b * 0.5
        }
        let eyeInWater = world.block(Int(eye.x.rounded(.down)),
                                     Int(eye.y.rounded(.down)),
                                     Int(eye.z.rounded(.down))).isWater
        let fogColor = eyeInWater ? waterFogColor : skyColor
        let fogRange: SIMD2<Float> = eyeInWater
            ? SIMD2(4, 28)
            : SIMD2(Float(meshRadius * 16) - 54, Float(meshRadius * 16) - 4)
        view.clearColor = MTLClearColor(red: Double(fogColor.x), green: Double(fogColor.y),
                                        blue: Double(fogColor.z), alpha: 1)

        updateHUD(view: view)

        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }

        let proj = perspectiveMatrix(fovY: 85 * .pi / 180, aspect: aspect, near: 0.05, far: 600)
        let viewM = lookAtMatrix(eye: eye, center: eye + player.forward, up: SIMD3(0, 1, 0))
        let viewProj = proj * viewM

        var uniforms = Uniforms(
            viewProj: viewProj,
            model: matrix_identity_float4x4,
            camPos: SIMD4(eye, 0),
            sunDir: SIMD4(simd_normalize(SIMD3<Float>(0.45, 0.9, 0.25)), 0),
            fogColor: fogColor,
            fogParams: SIMD4(fogRange.x, fogRange.y, 0, 0),
            alphaParams: SIMD4(1.0, 0.5, 0, 0))

        // opaque terrain (leaf holes via alpha discard)
        enc.setRenderPipelineState(blockPipeline)
        enc.setDepthStencilState(depthOn)
        enc.setCullMode(.none)
        enc.setFragmentTexture(atlas, index: 0)
        enc.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        for mesh in meshes.values where mesh.indexCount > 0 {
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                      indexType: .uint32, indexBuffer: mesh.indexBuffer!,
                                      indexBufferOffset: 0)
        }

        // dropped items: spinning, bobbing mini cubes
        for item in items {
            let bob = sin(item.age * 2.0) * 0.04
            var iu = uniforms
            iu.model = translationMatrix(item.pos + SIMD3(0, bob, 0))
                * rotationYMatrix(item.age * 1.6)
                * scaleMatrix(SIMD3(repeating: 0.25))
            let mesh = cubeMesh(for: item.block, icon: false)
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&iu, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&iu, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                      indexType: .uint32, indexBuffer: mesh.indexBuffer,
                                      indexBufferOffset: 0)
        }

        // mobs: boxed models with swinging limbs and a walk-cycle body bob;
        // dying mobs flash red, roll onto their side, then despawn
        for mob in mobs {
            let bob = abs(cos(mob.limbSwing)) * 0.05 * mob.swingAmount
            var entity = translationMatrix(mob.pos + SIMD3(0, bob, 0))
                * rotationYMatrix(-mob.yaw)
            if mob.dead {
                entity *= rotationZMatrix(min(mob.deathTime / 0.45, 1) * .pi / 2)
            }
            let flap: Float = mob.kind == .chicken && !mob.onGround && !mob.dead
                ? 1.0 + sin(mob.age * 30) * 0.8 : 0
            let hurt: Float = mob.dead ? 0.65 : (mob.hurtTime > 0 ? 0.7 : 0)
            drawModel(enc, model: model(for: mob.kind), entity: entity,
                      uniforms: uniforms, swing: mob.limbSwing,
                      amount: mob.swingAmount, headPitch: 0, flap: flap, hurt: hurt)
        }

        // the player's own model, visible in third person
        if thirdPerson {
            let bob = abs(cos(player.bobPhase)) * 0.06 * min(player.bobAmount, 1)
            let entity = translationMatrix(player.pos + SIMD3(0, bob, 0))
                * rotationYMatrix(-player.yaw)
            drawModel(enc, model: playerModel, entity: entity,
                      uniforms: uniforms, swing: player.bobPhase,
                      amount: min(player.bobAmount, 1.2),
                      headPitch: player.pitch, flap: 0)
        }
        enc.setFragmentTexture(atlas, index: 0) // restore for the passes below

        // targeted-block outline
        if let hit {
            enc.setRenderPipelineState(linePipeline)
            let blockPos = SIMD3<Float>(Float(hit.block.x), Float(hit.block.y), Float(hit.block.z))
            var lu = LineUniforms(mvp: viewProj * translationMatrix(blockPos),
                                  color: SIMD4(0.05, 0.05, 0.05, 1))
            enc.setVertexBuffer(cubeLineBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
            enc.setFragmentBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
            enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 24)
        }

        // water: blended, depth-tested but not depth-written
        var waterUniforms = uniforms
        waterUniforms.alphaParams = SIMD4(1.35, 0.05, 0, 0)
        enc.setRenderPipelineState(waterPipeline)
        enc.setDepthStencilState(depthReadOnly)
        enc.setVertexBytes(&waterUniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&waterUniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        for mesh in meshes.values where mesh.waterIndexCount > 0 {
            enc.setVertexBuffer(mesh.waterVertexBuffer, offset: 0, index: 0)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.waterIndexCount,
                                      indexType: .uint32, indexBuffer: mesh.waterIndexBuffer!,
                                      indexBufferOffset: 0)
        }

        drawHotbar(enc, view: view, baseUniforms: uniforms)

        // crosshair, drawn in NDC with depth test off
        enc.setRenderPipelineState(linePipeline)
        enc.setDepthStencilState(depthOff)
        let s: Float = 0.014
        var cu = LineUniforms(mvp: scaleMatrix(SIMD3(s, s * aspect, 1)),
                              color: SIMD4(1, 1, 1, 1))
        enc.setVertexBuffer(crosshairBuffer, offset: 0, index: 0)
        enc.setVertexBytes(&cu, length: MemoryLayout<LineUniforms>.stride, index: 1)
        enc.setFragmentBytes(&cu, length: MemoryLayout<LineUniforms>.stride, index: 1)
        enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 4)

        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }

    // MARK: - Hotbar

    private let slotSize: CGFloat = 48
    private let slotGap: CGFloat = 6

    private func slotOrigin(_ i: Int, _ bounds: CGSize) -> CGPoint {
        let totalW = slotSize * CGFloat(Inventory.slotCount) + slotGap * CGFloat(Inventory.slotCount - 1)
        let x0 = (bounds.width - totalW) / 2
        return CGPoint(x: x0 + CGFloat(i) * (slotSize + slotGap), y: 10)
    }

    private func ndcRect(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, _ b: CGSize) -> simd_float4x4 {
        translationMatrix(SIMD3(Float(x / b.width) * 2 - 1, Float(y / b.height) * 2 - 1, 0))
            * scaleMatrix(SIMD3(Float(w / b.width) * 2, Float(h / b.height) * 2, 1))
    }

    private func drawHotbar(_ enc: MTLRenderCommandEncoder, view: MTKView, baseUniforms: Uniforms) {
        let bounds = view.bounds.size
        guard bounds.width > 1, bounds.height > 1 else { return }
        enc.setDepthStencilState(depthOff)

        // slot backgrounds + selection frame
        enc.setRenderPipelineState(uiPipeline)
        enc.setVertexBuffer(quadBuffer, offset: 0, index: 0)
        func drawQuad(_ mvp: simd_float4x4, _ color: SIMD4<Float>) {
            var lu = LineUniforms(mvp: mvp, color: color)
            enc.setVertexBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
            enc.setFragmentBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }
        for i in 0..<Inventory.slotCount {
            let o = slotOrigin(i, bounds)
            if i == inventory.selected {
                drawQuad(ndcRect(o.x - 3, o.y - 3, slotSize + 6, slotSize + 6, bounds),
                         SIMD4(1, 1, 1, 0.85))
            }
            drawQuad(ndcRect(o.x, o.y, slotSize, slotSize, bounds), SIMD4(0, 0, 0, 0.55))
        }

        // isometric mini-cube icons
        enc.setRenderPipelineState(blockPipeline)
        enc.setFragmentTexture(atlas, index: 0)
        var iu = baseUniforms
        iu.viewProj = matrix_identity_float4x4
        iu.camPos = SIMD4(0, 0, 0, 0)
        iu.fogParams = SIMD4(1e8, 2e8, 0, 0) // no fog on UI
        iu.alphaParams = SIMD4(1.0, 0.5, 0, 0)
        for i in 0..<Inventory.slotCount {
            guard let stack = inventory.slots[i] else { continue }
            let o = slotOrigin(i, bounds)
            let cx = Float((o.x + slotSize / 2) / bounds.width) * 2 - 1
            let cy = Float((o.y + slotSize / 2) / bounds.height) * 2 - 1
            let sy = Float(slotSize * 0.34 / bounds.height) * 2
            let sx = sy * Float(bounds.height / bounds.width)
            iu.model = translationMatrix(SIMD3(cx, cy, 0.5))
                * scaleMatrix(SIMD3(sx, sy, 0.1))
                * rotationXMatrix(-0.52)
                * rotationYMatrix(.pi / 4)
            let mesh = cubeMesh(for: stack.block, icon: true)
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&iu, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&iu, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                      indexType: .uint32, indexBuffer: mesh.indexBuffer,
                                      indexBufferOffset: 0)
        }
    }

    /// Icon meshes only carry the three faces visible from the fixed
    /// isometric angle (+X, +Y, -Z), so no depth testing is needed.
    private func cubeMesh(for b: Block, icon: Bool) -> CubeMesh {
        if icon, let cached = iconMeshCache[b.rawValue] { return cached }
        if !icon, let cached = itemMeshCache[b.rawValue] { return cached }
        let (verts, indices) = Mesher.blockCube(b, faces: icon ? [0, 2, 5] : [0, 1, 2, 3, 4, 5])
        let mesh = CubeMesh(
            vertexBuffer: device.makeBuffer(bytes: verts, length: verts.count * 4)!,
            indexBuffer: device.makeBuffer(bytes: indices, length: indices.count * 4)!,
            indexCount: indices.count)
        if icon { iconMeshCache[b.rawValue] = mesh } else { itemMeshCache[b.rawValue] = mesh }
        return mesh
    }

    // MARK: - Item entities

    private func updateItems(dt: Float) {
        let playerCenter = player.pos + SIMD3<Float>(0, 0.9, 0)
        var i = 0
        while i < items.count {
            let item = items[i]
            item.update(dt: dt, world: world)
            let d = playerCenter - item.pos
            let dist = simd_length(d)
            if item.age > 0.5 && !player.flying {
                if dist < 1.1 {
                    if inventory.add(item.block) {
                        items.remove(at: i)
                        continue
                    }
                } else if dist < 3.0 {
                    item.vel += simd_normalize(d) * (30 / max(dist, 0.6)) * dt
                }
            }
            if item.pos.y < -40 {
                items.remove(at: i)
                continue
            }
            i += 1
        }
        if items.count > 400 {
            items.removeFirst(items.count - 400)
        }
    }

    // MARK: - Mobs

    private func mobTexture(_ name: String) -> MTLTexture {
        if let t = mobTextureCache[name] { return t }
        let url = Bundle.module.url(forResource: name, withExtension: "png", subdirectory: "mob")!
        let t = try! MTKTextureLoader(device: device).newTexture(URL: url, options: [.SRGB: false])
        mobTextureCache[name] = t
        return t
    }

    private func buildModel(_ spec: ModelSpec, textureNames: [String]) -> MobModel {
        var parts: [MobPartMesh] = []
        for p in spec.parts {
            var subs: [(tex: Int, mesh: CubeMesh)] = []
            for (tex, verts, indices) in MobModels.geometry(for: p, scale: spec.scale) {
                subs.append((tex, CubeMesh(
                    vertexBuffer: device.makeBuffer(bytes: verts, length: verts.count * 4)!,
                    indexBuffer: device.makeBuffer(bytes: indices, length: indices.count * 4)!,
                    indexCount: indices.count)))
            }
            parts.append(MobPartMesh(role: p.role, pivot: p.pivot * spec.scale,
                                     baseRotX: p.baseRotX, submeshes: subs))
        }
        return MobModel(parts: parts, textures: textureNames.map { mobTexture($0) })
    }

    private func model(for kind: MobKind) -> MobModel {
        if let m = mobModelCache[kind] { return m }
        let m = buildModel(MobModels.spec(for: kind), textureNames: kind.textureNames)
        mobModelCache[kind] = m
        return m
    }

    private func drawModel(_ enc: MTLRenderCommandEncoder, model: MobModel,
                           entity: simd_float4x4, uniforms: Uniforms,
                           swing: Float, amount: Float, headPitch: Float, flap: Float,
                           hurt: Float = 0) {
        for part in model.parts {
            var u = uniforms
            u.alphaParams.z = hurt
            u.model = entity * translationMatrix(part.pivot)
                * MobModels.animation(for: part.role, baseRotX: part.baseRotX,
                                      swing: swing, amount: amount,
                                      headPitch: headPitch, flap: flap)
            enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            for (tex, mesh) in part.submeshes {
                enc.setFragmentTexture(model.textures[tex], index: 0)
                enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                          indexType: .uint32, indexBuffer: mesh.indexBuffer,
                                          indexBufferOffset: 0)
            }
        }
    }

    private func nearestMobHit(origin: SIMD3<Float>, dir: SIMD3<Float>, maxDist: Float) -> Mob? {
        var best: Mob?
        var bestT = maxDist
        for mob in mobs where !mob.dead {
            let half = mob.kind.halfWidth
            guard let t = rayAABBIntersect(
                origin: origin, dir: dir,
                mn: mob.pos - SIMD3(half, 0, half),
                mx: mob.pos + SIMD3(half, mob.kind.height, half)),
                t < bestT else { continue }
            bestT = t
            best = mob
        }
        return best
    }

    private func updateMobs(dt: Float) {
        for mob in mobs { mob.update(dt: dt, world: world) }
        mobs.removeAll {
            ($0.dead && $0.deathTime > 1.6)
                || simd_distance($0.pos, player.pos) > 110
                || $0.pos.y < -20
        }

        mobSpawnTimer -= dt
        guard mobSpawnTimer <= 0 else { return }
        mobSpawnTimer = 0.4
        if mobs.count < 26 { attemptMobSpawn() }
    }

    /// One spawn attempt per tick: pick a random surface spot near the player
    /// and roll a mob that fits the ground there. Failures just wait for the
    /// next tick, so the population drifts toward the cap instead of jumping.
    private func attemptMobSpawn() {
        let angle = Float.random(in: 0..<(2 * .pi))
        let dist = Float.random(in: 14...60)
        let x = Int((player.pos.x + cos(angle) * dist).rounded(.down))
        let z = Int((player.pos.z + sin(angle) * dist).rounded(.down))
        guard world.isGenerated(World.chunkCoord(blockX: x, blockZ: z)) else { return }
        let h = world.surfaceHeight(x, z)
        guard h > World.waterLevel, h + 3 < World.height,
              world.isSolid(x, h, z),
              world.block(x, h + 1, z) == .air,
              world.block(x, h + 2, z) == .air else { return }

        let kind: MobKind
        if Float.random(in: 0...1) < 0.18 {
            kind = Bool.random() ? .zombie : .creeper
        } else {
            let ground = world.block(x, h, z)
            guard ground == .grass || ground == .snow else { return } // passive mobs graze
            kind = [.pig, .sheep, .cow, .chicken].randomElement()!
        }
        mobs.append(Mob(kind: kind, pos: SIMD3(Float(x) + 0.5, Float(h + 1) + 0.01, Float(z) + 0.5)))
    }

    // MARK: - Chunk streaming

    private func streamGeneration() {
        let pc = playerChunk
        // the 3x3 around the player is generated synchronously every frame so
        // physics and the mining raycast always have real terrain
        for dz in -1...1 {
            for dx in -1...1 {
                world.generateChunk(ChunkCoord(x: pc.x + dx, z: pc.z + dz))
            }
        }
        var budget = genBudgetPerFrame
        for (dx, dz) in genOffsets {
            let c = ChunkCoord(x: pc.x + dx, z: pc.z + dz)
            if !world.isGenerated(c) {
                world.generateChunk(c)
                budget -= 1
                if budget == 0 { break }
            }
        }
    }

    private func neighborsGenerated(_ c: ChunkCoord) -> Bool {
        world.isGenerated(c)
            && world.isGenerated(ChunkCoord(x: c.x + 1, z: c.z))
            && world.isGenerated(ChunkCoord(x: c.x - 1, z: c.z))
            && world.isGenerated(ChunkCoord(x: c.x, z: c.z + 1))
            && world.isGenerated(ChunkCoord(x: c.x, z: c.z - 1))
    }

    private func rebuildMesh(_ c: ChunkCoord) {
        let geo = Mesher.buildChunk(world: world, coord: c)
        let mesh = meshes[c] ?? ChunkMesh()
        mesh.indexCount = geo.opaqueIndices.count
        mesh.vertexBuffer = geo.opaqueVertices.isEmpty ? nil
            : device.makeBuffer(bytes: geo.opaqueVertices, length: geo.opaqueVertices.count * 4)
        mesh.indexBuffer = geo.opaqueIndices.isEmpty ? nil
            : device.makeBuffer(bytes: geo.opaqueIndices, length: geo.opaqueIndices.count * 4)
        mesh.waterIndexCount = geo.waterIndices.count
        mesh.waterVertexBuffer = geo.waterVertices.isEmpty ? nil
            : device.makeBuffer(bytes: geo.waterVertices, length: geo.waterVertices.count * 4)
        mesh.waterIndexBuffer = geo.waterIndices.isEmpty ? nil
            : device.makeBuffer(bytes: geo.waterIndices, length: geo.waterIndices.count * 4)
        meshes[c] = mesh
    }

    private func remeshDirtyChunks() {
        guard !world.dirtyChunks.isEmpty else { return }
        for c in world.dirtyChunks where meshes[c] != nil && neighborsGenerated(c) {
            rebuildMesh(c)
        }
        world.dirtyChunks.removeAll()
    }

    private func streamMeshes() {
        let pc = playerChunk
        var budget = meshBudgetPerFrame
        for (dx, dz) in meshOffsets {
            let c = ChunkCoord(x: pc.x + dx, z: pc.z + dz)
            if meshes[c] == nil && neighborsGenerated(c) {
                rebuildMesh(c)
                budget -= 1
                if budget == 0 { break }
            }
        }
        // drop GPU meshes far behind us; chunk data is kept so edits persist
        let dropDistance = meshRadius + 2
        for c in Array(meshes.keys) where abs(c.x - pc.x) > dropDistance || abs(c.z - pc.z) > dropDistance {
            meshes.removeValue(forKey: c)
        }
    }

    // MARK: - Game actions

    private func handleKeyPress(_ key: UInt16, now: CFTimeInterval) {
        switch key {
        case Keys.one, Keys.two, Keys.three, Keys.four, Keys.five,
             Keys.six, Keys.seven, Keys.eight, Keys.nine:
            let order: [UInt16] = [Keys.one, Keys.two, Keys.three, Keys.four, Keys.five,
                                   Keys.six, Keys.seven, Keys.eight, Keys.nine]
            inventory.selected = order.firstIndex(of: key)!
        case Keys.space:
            if now - lastSpaceTap < 0.35 {
                player.flying.toggle() // velocity carries over in both directions
                lastSpaceTap = -10
            } else {
                lastSpaceTap = now
            }
        case Keys.f5:
            thirdPerson.toggle()
        case Keys.r:
            resetWorld()
        default:
            break
        }
    }

    private func dropFor(_ b: Block) -> Block? {
        switch b {
        case .grass: return .dirt // like Minecraft, grass blocks drop dirt
        default: return b
        }
    }

    private func mine(_ hit: RayHit?) {
        guard let hit else { return }
        let x = Int(hit.block.x), y = Int(hit.block.y), z = Int(hit.block.z)
        let b = world.block(x, y, z)
        guard b != .air && b != .bedrock && !b.isWater else { return }
        world.setBlock(x, y, z, .air)
        if let drop = dropFor(b) {
            let pos = SIMD3<Float>(Float(x) + 0.5, Float(y) + 0.4, Float(z) + 0.5)
            let vel = SIMD3<Float>(Float.random(in: -1.2...1.2),
                                   Float.random(in: 2.2...3.4),
                                   Float.random(in: -1.2...1.2))
            items.append(ItemEntity(block: drop, pos: pos, vel: vel))
        }
    }

    private func place(_ hit: RayHit?) {
        guard let hit, let stack = inventory.selectedStack else { return }
        let p = hit.block &+ hit.normal
        let x = Int(p.x), y = Int(p.y), z = Int(p.z)
        let target = world.block(x, y, z)
        guard target == .air || target.isWater else { return }

        // don't place a block inside the player
        let bmin = SIMD3<Float>(Float(x), Float(y), Float(z))
        let bmax = bmin + SIMD3<Float>(1, 1, 1)
        let (pmin, pmax) = player.aabb
        let overlaps = pmin.x < bmax.x && pmax.x > bmin.x
            && pmin.y < bmax.y && pmax.y > bmin.y
            && pmin.z < bmax.z && pmax.z > bmin.z
        guard !overlaps || player.flying else { return }

        world.setBlock(x, y, z, stack.block)
        _ = inventory.consumeSelected()
    }

    // MARK: - HUD

    private func updateHUD(view: MTKView) {
        let bounds = view.bounds.size
        for i in 0..<Inventory.slotCount {
            let label = countLabels[i]
            if let stack = inventory.slots[i], stack.count > 1 {
                label.stringValue = "\(stack.count)"
                label.sizeToFit()
                let o = slotOrigin(i, bounds)
                label.setFrameOrigin(NSPoint(x: o.x + slotSize - label.frame.width - 4,
                                             y: o.y + 3))
                label.isHidden = false
            } else {
                label.isHidden = true
            }
        }

        guard let hud else { return }
        var text = "  WASD move · Space jump · 2×Space fly · F5 view · LMB mine · RMB place · 1-9/scroll slots · R new world  "
        if player.flying {
            text = "  ✈ SPECTATOR — Space up · Shift down  " + text
        }
        if !input.captured {
            text = "  ▸ Click to capture mouse (Esc releases)  " + text
        }
        if hud.stringValue != text {
            hud.stringValue = text
            hud.sizeToFit()
        }
        if let superview = hud.superview {
            hud.setFrameOrigin(NSPoint(x: 12, y: superview.bounds.height - hud.frame.height - 12))
        }
    }
}
