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
    var alphaParams: SIMD4<Float> // x = alpha multiplier, y = discard threshold, z = hurt
    var lightParams: SIMD4<Float> // x = day factor, y = mode (0 vertex/1 entity/2 full),
                                  // z/w = entity sky/block light 0-1
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
    let uiTexPipeline: MTLRenderPipelineState
    let cloudPipeline: MTLRenderPipelineState
    let crackPipeline: MTLRenderPipelineState
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
    weak var gameView: GameView?
    private var countLabels: [NSTextField] = [] // pooled stack-count overlays

    private var meshes: [ChunkCoord: ChunkMesh] = [:]
    private var itemMeshCache: [UInt8: CubeMesh] = [:]
    private var iconMeshCache: [UInt8: CubeMesh] = [:]
    private var spriteMeshCache: [Item: CubeMesh] = [:]
    private var mobModelCache: [MobKind: MobModel] = [:]
    private var textureCache: [String: MTLTexture] = [:]
    private lazy var playerModel = buildModel(MobModels.humanoid(), textureNames: ["char"])
    private var mobSpawnTimer: Float = 0
    private var thirdPerson = false

    // progressive mining: the block being held under the crosshair and its
    // 0-1 break progress; resets when the button lifts or the target changes
    private var breakingPos: SIMD3<Int32>?
    private var breakingProgress: Float = 0
    private var crackMeshCache: [Int: CubeMesh] = [:]

    let gui = GUIState()
    var furnaces: [BlockPos: FurnaceState] = [:]
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

    // Terrain gen and mesh building run on a concurrent queue; each frame the
    // main thread integrates finished results and tops the jobs back up,
    // nearest-first, so the render loop never blocks on world streaming.
    private let workQueue = DispatchQueue(label: "metalcraft.chunks",
                                          qos: .userInitiated, attributes: .concurrent)
    private let resultLock = NSLock()
    private var finishedChunks: [(epoch: Int, coord: ChunkCoord, chunk: Chunk)] = []
    private var finishedMeshes: [(epoch: Int, coord: ChunkCoord, mesh: ChunkMesh)] = []
    private var pendingGen = Set<ChunkCoord>()
    private var pendingMesh = Set<ChunkCoord>()
    private var dirtyWhileMeshing = Set<ChunkCoord>()
    private var discardInFlight = Set<ChunkCoord>() // superseded by an urgent sync remesh
    private var epoch = 0 // bumped on world reset; stale results are dropped
    private let maxGenJobs = max(2, ProcessInfo.processInfo.activeProcessorCount - 2)
    private let maxMeshJobs = 6

    private let waterFogColor = SIMD4<Float>(0.07, 0.20, 0.40, 1)

    // day/night cycle: one full day every 10 minutes, starting mid-morning
    private let dayLength: Float = 600
    private var timeOfDay: Float = 0.35 // 0.25 = sunrise, 0.5 = noon, 0.75 = sunset
    private var cloudOffset: Float = 0
    private let starBuffer: MTLBuffer
    private let starVertexCount: Int

    init(device: MTLDevice, view: MTKView, input: Input) {
        self.device = device
        self.input = input
        queue = device.makeCommandQueue()!

        let library = try! device.makeLibrary(source: shaderSource, options: nil)

        func makePipeline(vertex: String, fragment: String, blended: Bool,
                          multiply: Bool = false) -> MTLRenderPipelineState {
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
            if multiply {
                // 2 × src × dst, like the GL_DST_COLOR/GL_SRC_COLOR blend the
                // real game uses for the crack overlay: gray texels darken
                // the block's texture instead of painting over it
                ca.isBlendingEnabled = true
                ca.rgbBlendOperation = .add
                ca.alphaBlendOperation = .add
                ca.sourceRGBBlendFactor = .destinationColor
                ca.destinationRGBBlendFactor = .sourceColor
                ca.sourceAlphaBlendFactor = .zero
                ca.destinationAlphaBlendFactor = .one
            }
            desc.depthAttachmentPixelFormat = view.depthStencilPixelFormat
            return try! device.makeRenderPipelineState(descriptor: desc)
        }
        blockPipeline = makePipeline(vertex: "block_vertex", fragment: "block_fragment", blended: false)
        waterPipeline = makePipeline(vertex: "block_vertex", fragment: "block_fragment", blended: true)
        linePipeline = makePipeline(vertex: "line_vertex", fragment: "line_fragment", blended: false)
        uiPipeline = makePipeline(vertex: "line_vertex", fragment: "line_fragment", blended: true)
        uiTexPipeline = makePipeline(vertex: "ui_vertex", fragment: "ui_fragment", blended: true)
        cloudPipeline = makePipeline(vertex: "cloud_vertex", fragment: "cloud_fragment", blended: true)
        crackPipeline = makePipeline(vertex: "ui_vertex", fragment: "ui_fragment",
                                     blended: false, multiply: true)

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

        // star field: small quads scattered on the celestial sphere, rotated
        // with the sun so they wheel across the night sky like beta's
        var starVerts: [Float] = []
        for _ in 0..<600 {
            var d = SIMD3<Float>.zero
            repeat {
                d = SIMD3(Float.random(in: -1...1), Float.random(in: -1...1),
                          Float.random(in: -1...1))
            } while simd_length_squared(d) > 1 || simd_length_squared(d) < 0.01
            d = simd_normalize(d)
            let size = Float.random(in: 0.5...1.4)
            let ref: SIMD3<Float> = abs(d.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
            let t1 = simd_normalize(simd_cross(d, ref)) * size
            let t2 = simd_normalize(simd_cross(d, t1)) * size
            let c = d * 400
            let corners = [c - t1 - t2, c + t1 - t2, c + t1 + t2, c - t1 + t2]
            for i in [0, 1, 2, 0, 2, 3] {
                starVerts.append(contentsOf: [corners[i].x, corners[i].y, corners[i].z])
            }
        }
        starBuffer = device.makeBuffer(bytes: starVerts, length: starVerts.count * 4)!
        starVertexCount = starVerts.count / 3

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

        resetWorld()
    }

    private var playerChunk: ChunkCoord {
        World.chunkCoord(blockX: Int(player.pos.x.rounded(.down)),
                         blockZ: Int(player.pos.z.rounded(.down)))
    }

    private func resetWorld() {
        epoch += 1
        pendingGen.removeAll()
        pendingMesh.removeAll()
        dirtyWhileMeshing.removeAll()
        resultLock.lock()
        finishedChunks.removeAll()
        finishedMeshes.removeAll()
        resultLock.unlock()

        if gui.screen != nil { closeGUI() }
        world.reset(seed: UInt64.random(in: 0...UInt64.max))
        meshes.removeAll()
        items.removeAll()
        mobs.removeAll()
        furnaces.removeAll()
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
            let n = Inventory.hotbarCount
            inventory.selected = ((inventory.selected + input.scrollSteps) % n + n) % n
            input.scrollSteps = 0
        }
        processGUIClicks(view: view)

        integrateFinishedWork()
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
                let damage = inventory.selectedStack?.item.attackDamage ?? 4
                mob.hurt(damage: damage, direction: kb)
                if mob.dead {
                    for (item, maxCount) in mob.kind.deathDrops {
                        let n = Int.random(in: 0...maxCount)
                        if n > 0 {
                            spawnDrop(item, count: n, at: mob.pos + SIMD3(0, 0.5, 0))
                        }
                    }
                }
            }
        }
        updateMining(hit, dt: dt)
        if input.rightClicks > 0 { rightClick(hit) }
        input.leftClicks = 0
        input.rightClicks = 0

        updateItems(dt: dt)
        updateMobs(dt: dt)
        tickFurnaces(dt: dt)
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
        // beta celestial math: a cosine over the day drives everything.
        // dayBright is beta's "celestial brightness" (0 night, 1 day); the
        // shader subtracts skyDarken whole light levels like beta did
        // (midnight surface sits at level 4).
        timeOfDay = (timeOfDay + dt / dayLength).truncatingRemainder(dividingBy: 1)
        cloudOffset += dt * 0.6
        let sunAngle = (timeOfDay - 0.25) * 2 * .pi
        let sunHeight = sin(sunAngle)
        let sunDir = simd_normalize(SIMD3<Float>(cos(sunAngle), sin(sunAngle), 0.18))
        let dayBright = max(0, min(1, cos((timeOfDay - 0.5) * 2 * .pi) * 2 + 0.5))
        let skyDarken = (1 - dayBright) * 11

        // beta fog/horizon color, with a faint warm band at sunrise/sunset
        var skyc = SIMD3<Float>(0.753, 0.847, 1.0)
            * SIMD3(dayBright * 0.94 + 0.06, dayBright * 0.94 + 0.06, dayBright * 0.91 + 0.09)
        let dusk = exp(-(sunHeight / 0.13) * (sunHeight / 0.13))
        skyc = simd_mix(skyc, SIMD3(0.95, 0.50, 0.25), SIMD3(repeating: dusk * 0.30))

        let eyeInWater = world.block(Int(eye.x.rounded(.down)),
                                     Int(eye.y.rounded(.down)),
                                     Int(eye.z.rounded(.down))).isWater
        let fogColor = eyeInWater
            ? waterFogColor * (0.25 + 0.75 * dayBright)
            : SIMD4(skyc, 1)
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
            sunDir: SIMD4(sunDir, 0),
            fogColor: fogColor,
            fogParams: SIMD4(fogRange.x, fogRange.y, 0, 0),
            alphaParams: SIMD4(1.0, 0.5, 0, 0),
            lightParams: SIMD4(skyDarken, 0, 0, 0))

        // stars, sun and moon: drawn first so terrain occludes them
        if !eyeInWater {
            enc.setRenderPipelineState(uiPipeline)
            enc.setDepthStencilState(depthOff)

            let starAlpha = (1 - dayBright) * (1 - dayBright) * 0.75
            if starAlpha > 0.02 {
                var su = LineUniforms(
                    mvp: viewProj * translationMatrix(eye) * rotationZMatrix(sunAngle),
                    color: SIMD4(1, 1, 1, starAlpha))
                enc.setVertexBuffer(starBuffer, offset: 0, index: 0)
                enc.setVertexBytes(&su, length: MemoryLayout<LineUniforms>.stride, index: 1)
                enc.setFragmentBytes(&su, length: MemoryLayout<LineUniforms>.stride, index: 1)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: starVertexCount)
            }

            enc.setVertexBuffer(quadBuffer, offset: 0, index: 0)
            func skyQuad(_ dir: SIMD3<Float>, _ size: Float, _ color: SIMD4<Float>) {
                guard dir.y > -0.3 else { return }
                let right = simd_normalize(simd_cross(SIMD3<Float>(0, 0, 1), dir))
                let up = simd_cross(dir, right)
                var m = matrix_identity_float4x4
                m.columns.0 = SIMD4(right * size, 0)
                m.columns.1 = SIMD4(up * size, 0)
                m.columns.2 = SIMD4(dir, 0)
                m.columns.3 = SIMD4(eye + dir * 420, 1)
                var lu = LineUniforms(mvp: viewProj * m * translationMatrix(SIMD3(-0.5, -0.5, 0)),
                                      color: color)
                enc.setVertexBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
                enc.setFragmentBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            }
            skyQuad(sunDir, 52, SIMD4(1.0, 0.93, 0.72, 1))
            skyQuad(-sunDir, 32, SIMD4(0.88, 0.92, 1.0, 0.9))
        }

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

        // dropped items: blocks spin as mini cubes, materials as flat sprites
        for item in items {
            let bob = sin(item.age * 2.0) * 0.04
            var iu = uniforms
            iu.model = translationMatrix(item.pos + SIMD3(0, bob, 0))
                * rotationYMatrix(item.age * 1.6)
            let l = world.lightAt(Int(item.pos.x.rounded(.down)),
                                  Int((item.pos.y + 0.2).rounded(.down)),
                                  Int(item.pos.z.rounded(.down)))
            iu.lightParams = SIMD4(skyDarken, 1, l.x, l.y)
            let mesh: CubeMesh
            if let b = item.item.asBlock, b != .torch {
                iu.model *= scaleMatrix(SIMD3(repeating: 0.25))
                mesh = cubeMesh(for: b, icon: false)
                enc.setFragmentTexture(atlas, index: 0)
            } else {
                mesh = spriteMesh(for: item.item)
                enc.setFragmentTexture(item.item == .block(.torch) ? atlas : texture("items", "gui"),
                                       index: 0)
            }
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&iu, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&iu, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                      indexType: .uint32, indexBuffer: mesh.indexBuffer,
                                      indexBufferOffset: 0)
        }
        enc.setFragmentTexture(atlas, index: 0)

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
            let l = world.lightAt(Int(mob.pos.x.rounded(.down)),
                                  Int((mob.pos.y + mob.kind.height * 0.6).rounded(.down)),
                                  Int(mob.pos.z.rounded(.down)))
            var mu = uniforms
            mu.lightParams = SIMD4(skyDarken, 1, l.x, l.y)
            drawModel(enc, model: model(for: mob.kind), entity: entity,
                      uniforms: mu, swing: mob.limbSwing,
                      amount: mob.swingAmount, headPitch: 0, flap: flap, hurt: hurt)
        }

        // the player's own model, visible in third person
        if thirdPerson {
            let bob = abs(cos(player.bobPhase)) * 0.06 * min(player.bobAmount, 1)
            let entity = translationMatrix(player.pos + SIMD3(0, bob, 0))
                * rotationYMatrix(-player.yaw)
            let l = world.lightAt(Int(player.pos.x.rounded(.down)),
                                  Int(player.eye.y.rounded(.down)),
                                  Int(player.pos.z.rounded(.down)))
            var pu = uniforms
            pu.lightParams = SIMD4(skyDarken, 1, l.x, l.y)
            drawModel(enc, model: playerModel, entity: entity,
                      uniforms: pu, swing: player.bobPhase,
                      amount: min(player.bobAmount, 1.2),
                      headPitch: player.pitch, flap: 0)
        }
        enc.setFragmentTexture(atlas, index: 0) // restore for the passes below

        // mining crack overlay: destroy-stage tile multiplied over the block
        if let bp = breakingPos, breakingProgress > 0 {
            let stage = min(Int(breakingProgress * 10), 9)
            let mesh = crackMesh(stage: stage)
            let blockPos = SIMD3<Float>(Float(bp.x), Float(bp.y), Float(bp.z))
            var cu = LineUniforms(mvp: viewProj * translationMatrix(blockPos),
                                  color: SIMD4(1, 1, 1, 1))
            enc.setRenderPipelineState(crackPipeline)
            enc.setDepthStencilState(depthReadOnly)
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&cu, length: MemoryLayout<LineUniforms>.stride, index: 1)
            enc.setFragmentBytes(&cu, length: MemoryLayout<LineUniforms>.stride, index: 1)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                      indexType: .uint32, indexBuffer: mesh.indexBuffer,
                                      indexBufferOffset: 0)
            enc.setDepthStencilState(depthOn)
        }

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
        waterUniforms.alphaParams = SIMD4(0.72, 0.05, 0, 0)
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

        // beta cloud layer: flat texture plane at y=108 drifting east, one
        // cloud pixel = 12 blocks, tinted down with the daylight
        if !eyeInWater {
            let reach: Float = 768
            let cy: Float = 108
            let scale: Float = 12 * 256
            func cv(_ x: Float, _ z: Float) -> [Float] {
                [x, cy, z, (x + cloudOffset) / scale, z / scale]
            }
            let verts = cv(eye.x - reach, eye.z - reach) + cv(eye.x + reach, eye.z - reach)
                + cv(eye.x + reach, eye.z + reach) + cv(eye.x - reach, eye.z - reach)
                + cv(eye.x + reach, eye.z + reach) + cv(eye.x - reach, eye.z + reach)
            let cb = dayBright * 0.9 + 0.1
            var cu = LineUniforms(mvp: viewProj, color: SIMD4(cb, cb, cb, 0.8))
            enc.setRenderPipelineState(cloudPipeline)
            enc.setDepthStencilState(depthReadOnly)
            enc.setFragmentTexture(texture("clouds", "environment"), index: 0)
            enc.setVertexBytes(verts, length: verts.count * 4, index: 0)
            enc.setVertexBytes(&cu, length: MemoryLayout<LineUniforms>.stride, index: 1)
            enc.setFragmentBytes(&cu, length: MemoryLayout<LineUniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }

        drawHotbar(enc, view: view, baseUniforms: uniforms)

        if gui.screen == nil {
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
        }

        updateCountLabels(view: view)

        enc.endEncoding()

        // Container screens render in a second pass with a cleared depth
        // buffer, so the inventory's player preview can depth-test against
        // itself while drawing over the finished world frame.
        if gui.screen != nil, let depthTex = rpd.depthAttachment.texture {
            let rpd2 = MTLRenderPassDescriptor()
            rpd2.colorAttachments[0].texture = drawable.texture
            rpd2.colorAttachments[0].loadAction = .load
            rpd2.colorAttachments[0].storeAction = .store
            rpd2.depthAttachment.texture = depthTex
            rpd2.depthAttachment.loadAction = .clear
            rpd2.depthAttachment.clearDepth = 1
            rpd2.depthAttachment.storeAction = .dontCare
            if let enc2 = cmd.makeRenderCommandEncoder(descriptor: rpd2) {
                drawGUI(enc2, view: view, baseUniforms: uniforms)
                enc2.endEncoding()
            }
        }

        cmd.present(drawable)
        cmd.commit()
    }

    // MARK: - UI (hotbar + container screens)

    /// Classic GUI pixel scale: dialog art is 176×166, hotbar 182×22.
    private let uiScale: CGFloat = 2

    private func texture(_ name: String, _ subdirectory: String) -> MTLTexture {
        let key = subdirectory + "/" + name
        if let t = textureCache[key] { return t }
        let url = Bundle.module.url(forResource: name, withExtension: "png",
                                    subdirectory: subdirectory)!
        let t = try! MTKTextureLoader(device: device).newTexture(URL: url, options: [.SRGB: false])
        textureCache[key] = t
        return t
    }

    private func ndcRect(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, _ b: CGSize) -> simd_float4x4 {
        translationMatrix(SIMD3(Float(x / b.width) * 2 - 1, Float(y / b.height) * 2 - 1, 0))
            * scaleMatrix(SIMD3(Float(w / b.width) * 2, Float(h / b.height) * 2, 1))
    }

    private func drawSolidQuad(_ enc: MTLRenderCommandEncoder, _ mvp: simd_float4x4,
                               _ color: SIMD4<Float>) {
        enc.setRenderPipelineState(uiPipeline)
        enc.setVertexBuffer(quadBuffer, offset: 0, index: 0)
        var lu = LineUniforms(mvp: mvp, color: color)
        enc.setVertexBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
        enc.setFragmentBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
    }

    /// Blit a pixel region of a 256×256 GUI texture into a view rect.
    /// `rect` is in view points (AppKit y-up).
    private func drawUIQuad(_ enc: MTLRenderCommandEncoder, texture tex: MTLTexture,
                            rect: CGRect, srcX: Float, srcY: Float, srcW: Float, srcH: Float,
                            bounds: CGSize) {
        let nx0 = Float(rect.minX / bounds.width) * 2 - 1
        let ny0 = Float(rect.minY / bounds.height) * 2 - 1
        let nx1 = Float(rect.maxX / bounds.width) * 2 - 1
        let ny1 = Float(rect.maxY / bounds.height) * 2 - 1
        let u0 = srcX / 256, v0 = srcY / 256
        let u1 = (srcX + srcW) / 256, v1 = (srcY + srcH) / 256
        // texture v runs top-down, view y runs bottom-up
        let verts: [Float] = [
            nx0, ny0, 0, u0, v1,
            nx1, ny0, 0, u1, v1,
            nx1, ny1, 0, u1, v0,
            nx0, ny0, 0, u0, v1,
            nx1, ny1, 0, u1, v0,
            nx0, ny1, 0, u0, v0,
        ]
        var lu = LineUniforms(mvp: matrix_identity_float4x4, color: SIMD4(1, 1, 1, 1))
        enc.setRenderPipelineState(uiTexPipeline)
        enc.setFragmentTexture(tex, index: 0)
        enc.setVertexBytes(verts, length: verts.count * 4, index: 0)
        enc.setVertexBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
        enc.setFragmentBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
    }

    /// A 16×16 item graphic centered at `center`: blocks as isometric mini
    /// cubes, materials as items.png sprites.
    private func drawItemIcon(_ enc: MTLRenderCommandEncoder, item: Item,
                              center: CGPoint, size: CGFloat,
                              bounds: CGSize, baseUniforms: Uniforms) {
        if item == .block(.torch) {
            // torches draw flat from their terrain tile, like the real game
            drawUIQuad(enc, texture: atlas,
                       rect: CGRect(x: center.x - size / 2, y: center.y - size / 2,
                                    width: size, height: size),
                       srcX: 0, srcY: 5 * 16, srcW: 16, srcH: 16, bounds: bounds)
        } else if let b = item.asBlock {
            enc.setRenderPipelineState(blockPipeline)
            enc.setFragmentTexture(atlas, index: 0)
            var iu = baseUniforms
            iu.viewProj = matrix_identity_float4x4
            iu.camPos = SIMD4(0, 0, 0, 0)
            iu.fogParams = SIMD4(1e8, 2e8, 0, 0) // no fog on UI
            iu.alphaParams = SIMD4(1.0, 0.5, 0, 0)
            iu.lightParams = SIMD4(1, 2, 0, 0) // full bright
            let cx = Float(center.x / bounds.width) * 2 - 1
            let cy = Float(center.y / bounds.height) * 2 - 1
            let sy = Float(size * 0.36 / bounds.height) * 2
            let sx = sy * Float(bounds.height / bounds.width)
            iu.model = translationMatrix(SIMD3(cx, cy, 0.5))
                * scaleMatrix(SIMD3(sx, sy, 0.1))
                * rotationXMatrix(-0.52)
                * rotationYMatrix(.pi / 4)
            let mesh = cubeMesh(for: b, icon: true)
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&iu, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&iu, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                      indexType: .uint32, indexBuffer: mesh.indexBuffer,
                                      indexBufferOffset: 0)
        } else if let t = item.sprite {
            drawUIQuad(enc, texture: texture("items", "gui"),
                       rect: CGRect(x: center.x - size / 2, y: center.y - size / 2,
                                    width: size, height: size),
                       srcX: t.x * 16, srcY: t.y * 16, srcW: 16, srcH: 16, bounds: bounds)
        }
    }

    // MARK: - Hotbar

    private func hotbarOrigin(_ bounds: CGSize) -> CGPoint {
        CGPoint(x: (bounds.width - 182 * uiScale) / 2, y: 8)
    }

    /// Center of hotbar slot i's 16×16 interior (slots start at x=3+20i, y=3).
    private func hotbarSlotCenter(_ i: Int, _ bounds: CGSize) -> CGPoint {
        let o = hotbarOrigin(bounds)
        return CGPoint(x: o.x + (3 + CGFloat(i) * 20 + 8) * uiScale,
                       y: o.y + 11 * uiScale)
    }

    private func drawHotbar(_ enc: MTLRenderCommandEncoder, view: MTKView, baseUniforms: Uniforms) {
        let bounds = view.bounds.size
        guard bounds.width > 1, bounds.height > 1 else { return }
        enc.setDepthStencilState(depthOff)

        let guiTex = texture("gui", "gui")
        let o = hotbarOrigin(bounds)
        drawUIQuad(enc, texture: guiTex,
                   rect: CGRect(x: o.x, y: o.y, width: 182 * uiScale, height: 22 * uiScale),
                   srcX: 0, srcY: 0, srcW: 182, srcH: 22, bounds: bounds)
        drawUIQuad(enc, texture: guiTex,
                   rect: CGRect(x: o.x + (CGFloat(inventory.selected) * 20 - 1) * uiScale,
                                y: o.y - 1 * uiScale,
                                width: 24 * uiScale, height: 24 * uiScale),
                   srcX: 0, srcY: 22, srcW: 24, srcH: 24, bounds: bounds)

        for i in 0..<Inventory.hotbarCount {
            guard let stack = inventory.slots[i] else { continue }
            drawItemIcon(enc, item: stack.item, center: hotbarSlotCenter(i, bounds),
                         size: 16 * uiScale, bounds: bounds, baseUniforms: baseUniforms)
        }
    }

    // MARK: - Container screens

    /// Dialog px (top-left origin, y down) → view rect (y up). All dialogs
    /// are 176×166, centered on screen.
    private func guiRect(_ dx: CGFloat, _ dy: CGFloat, _ w: CGFloat, _ h: CGFloat,
                         _ bounds: CGSize) -> CGRect {
        let ox = (bounds.width - 176 * uiScale) / 2
        let oyTop = (bounds.height + 166 * uiScale) / 2
        return CGRect(x: ox + dx * uiScale, y: oyTop - (dy + h) * uiScale,
                      width: w * uiScale, height: h * uiScale)
    }

    private var currentFurnace: FurnaceState? {
        if case .furnace(let pos)? = gui.screen { return furnaces[pos] }
        return nil
    }

    private func drawGUI(_ enc: MTLRenderCommandEncoder, view: MTKView, baseUniforms: Uniforms) {
        guard let screen = gui.screen else { return }
        let bounds = view.bounds.size
        guard bounds.width > 1, bounds.height > 1 else { return }
        enc.setDepthStencilState(depthOff)

        // dimmed world behind the dialog
        drawSolidQuad(enc, ndcRect(0, 0, bounds.width, bounds.height, bounds),
                      SIMD4(0, 0, 0, 0.55))

        let dialogTex = texture(screen.textureName, "gui")
        drawUIQuad(enc, texture: dialogTex, rect: guiRect(0, 0, 176, 166, bounds),
                   srcX: 0, srcY: 0, srcW: 176, srcH: 166, bounds: bounds)

        if case .inventory = screen {
            drawPlayerPreview(enc, bounds: bounds, baseUniforms: baseUniforms)
        }

        // furnace flame + progress arrow, cropped by their fractions
        if let f = currentFurnace {
            if f.isBurning {
                let frac = max(0, min(1, f.burnLeft / f.burnTotal))
                let h = CGFloat((14 * frac).rounded())
                drawUIQuad(enc, texture: dialogTex,
                           rect: guiRect(56, 36 + (14 - h), 14, h, bounds),
                           srcX: 176, srcY: Float(14 - h), srcW: 14, srcH: Float(h),
                           bounds: bounds)
            }
            let cookW = (24 * CGFloat(f.cook / FurnaceState.cookTime)).rounded()
            if cookW > 0 {
                drawUIQuad(enc, texture: dialogTex,
                           rect: guiRect(79, 34, cookW, 17, bounds),
                           srcX: 176, srcY: 14, srcW: Float(cookW), srcH: 17, bounds: bounds)
            }
        }

        // slot contents
        for slot in gui.slots {
            guard let stack = gui.read(slot.source, inventory: inventory, furnace: currentFurnace)
            else { continue }
            let r = guiRect(CGFloat(slot.x), CGFloat(slot.y), 16, 16, bounds)
            drawItemIcon(enc, item: stack.item, center: CGPoint(x: r.midX, y: r.midY),
                         size: 16 * uiScale, bounds: bounds, baseUniforms: baseUniforms)
        }

        // stack picked up on the cursor
        if let held = gui.held {
            drawItemIcon(enc, item: held.item, center: input.cursor,
                         size: 16 * uiScale, bounds: bounds, baseUniforms: baseUniforms)
        }
    }

    /// The character in the inventory screen's preview box, turning to track
    /// the cursor like the real game.
    private func drawPlayerPreview(_ enc: MTLRenderCommandEncoder, bounds: CGSize,
                                   baseUniforms: Uniforms) {
        let pxPerMeter = 30 * uiScale // 1.8 m model ≈ 54 px in the 72 px box
        let feetRect = guiRect(52, 76, 0, 0, bounds)
        let feet = CGPoint(x: feetRect.minX, y: feetRect.minY)
        let cx = Float(feet.x / bounds.width) * 2 - 1
        let cy = Float(feet.y / bounds.height) * 2 - 1
        let sx = Float(pxPerMeter / bounds.width) * 2
        let sy = Float(pxPerMeter / bounds.height) * 2

        let eyeY = feet.y + 1.62 * pxPerMeter
        let dx = Float(input.cursor.x - feet.x)
        let dy = Float(input.cursor.y - eyeY)
        let bodyYaw = max(-0.7, min(0.7, dx / 400))
        let headPitch = max(-0.6, min(0.6, dy / 400))

        var u = baseUniforms
        // orthographic blow-up of the model into the preview box; negative z
        // scale keeps "toward the viewer" at smaller depth
        u.viewProj = translationMatrix(SIMD3(cx, cy, 0.5))
            * scaleMatrix(SIMD3(sx, sy, -0.05))
        u.camPos = SIMD4(0, 0, 0, 0)
        u.sunDir = SIMD4(simd_normalize(SIMD3<Float>(0.3, 0.6, 0.9)), 0)
        u.fogParams = SIMD4(1e8, 2e8, 0, 0)
        u.alphaParams = SIMD4(1.0, 0.5, 0, 0)
        u.lightParams = SIMD4(1, 2, 0, 0) // full bright

        enc.setRenderPipelineState(blockPipeline)
        enc.setDepthStencilState(depthOn)
        drawModel(enc, model: playerModel, entity: rotationYMatrix(.pi + bodyYaw),
                  uniforms: u, swing: 0, amount: 0, headPitch: headPitch, flap: 0)
        enc.setDepthStencilState(depthOff)
    }

    private func slotAt(_ p: CGPoint, bounds: CGSize) -> GUISlot.Source? {
        for slot in gui.slots
        where guiRect(CGFloat(slot.x), CGFloat(slot.y), 16, 16, bounds).contains(p) {
            return slot.source
        }
        return nil
    }

    private func processGUIClicks(view: MTKView) {
        defer {
            input.guiLeftClicks.removeAll()
            input.guiRightClicks.removeAll()
        }
        guard gui.screen != nil else { return }
        let bounds = view.bounds.size
        for (points, right) in [(input.guiLeftClicks, false), (input.guiRightClicks, true)] {
            for p in points {
                guard let source = slotAt(p, bounds: bounds) else { continue }
                gui.click(source, right: right, inventory: inventory, furnace: currentFurnace)
            }
        }
    }

    func openGUI(_ screen: GUIScreen) {
        gui.open(screen)
        input.guiOpen = true
        gameView?.setCaptured(false)
    }

    func closeGUI() {
        for stack in gui.close() where !inventory.add(stack.item, count: stack.count) {
            items.append(ItemEntity(item: stack.item, pos: player.eye,
                                    vel: player.forward * 3, count: stack.count))
        }
        input.guiOpen = false
        gameView?.setCaptured(true)
    }

    // MARK: - Stack count labels

    private func updateCountLabels(view: MTKView) {
        let bounds = view.bounds.size
        var requests: [(text: String, x: CGFloat, y: CGFloat)] = []

        for i in 0..<Inventory.hotbarCount {
            if let stack = inventory.slots[i], stack.count > 1 {
                let c = hotbarSlotCenter(i, bounds)
                requests.append(("\(stack.count)", c.x + 8 * uiScale, c.y - 8 * uiScale))
            }
        }
        if gui.screen != nil {
            for slot in gui.slots {
                guard let stack = gui.read(slot.source, inventory: inventory,
                                           furnace: currentFurnace), stack.count > 1
                else { continue }
                let r = guiRect(CGFloat(slot.x), CGFloat(slot.y), 16, 16, bounds)
                requests.append(("\(stack.count)", r.maxX, r.minY))
            }
            if let held = gui.held, held.count > 1 {
                requests.append(("\(held.count)", input.cursor.x + 8 * uiScale,
                                 input.cursor.y - 8 * uiScale))
            }
        }

        while countLabels.count < requests.count {
            let label = NSTextField(labelWithString: "")
            label.font = .monospacedDigitSystemFont(ofSize: 12, weight: .bold)
            label.textColor = .white
            view.addSubview(label)
            countLabels.append(label)
        }
        for (i, label) in countLabels.enumerated() {
            if i < requests.count {
                let r = requests[i]
                if label.stringValue != r.text {
                    label.stringValue = r.text
                    label.sizeToFit()
                }
                label.setFrameOrigin(NSPoint(x: r.x - label.frame.width, y: r.y))
                label.isHidden = false
            } else {
                label.isHidden = true
            }
        }
    }

    /// Flat double-sided quad showing a 16px sprite tile, for dropped
    /// materials like coal or ingots (and torches, which use the atlas).
    private func spriteMesh(for item: Item) -> CubeMesh {
        if let cached = spriteMeshCache[item] { return cached }
        let t = item == .block(.torch) ? SIMD2<Float>(0, 5) : (item.sprite ?? SIMD2(0, 0))
        let u0 = (t.x * 16 + 0.1) / 256, u1 = (t.x * 16 + 15.9) / 256
        let v0 = (t.y * 16 + 0.1) / 256, v1 = (t.y * 16 + 15.9) / 256
        let s: Float = 0.26
        let verts: [Float] = [
            -s, -s, 0, 0, 0, 1, u0, v1, 1, 1, 1, 15, 0,
             s, -s, 0, 0, 0, 1, u1, v1, 1, 1, 1, 15, 0,
             s,  s, 0, 0, 0, 1, u1, v0, 1, 1, 1, 15, 0,
            -s,  s, 0, 0, 0, 1, u0, v0, 1, 1, 1, 15, 0,
        ]
        let indices: [UInt32] = [0, 1, 2, 0, 2, 3]
        let mesh = CubeMesh(
            vertexBuffer: device.makeBuffer(bytes: verts, length: verts.count * 4)!,
            indexBuffer: device.makeBuffer(bytes: indices, length: indices.count * 4)!,
            indexCount: indices.count)
        spriteMeshCache[item] = mesh
        return mesh
    }

    /// One of the 10 destroy-stage overlay cubes, built lazily and cached.
    private func crackMesh(stage: Int) -> CubeMesh {
        if let cached = crackMeshCache[stage] { return cached }
        let (verts, indices) = Mesher.crackCube(stage: stage)
        let mesh = CubeMesh(
            vertexBuffer: device.makeBuffer(bytes: verts, length: verts.count * 4)!,
            indexBuffer: device.makeBuffer(bytes: indices, length: indices.count * 4)!,
            indexCount: indices.count)
        crackMeshCache[stage] = mesh
        return mesh
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
                    if inventory.add(item.item, count: item.count) {
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
        return MobModel(parts: parts, textures: textureNames.map { texture($0, "mob") })
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

    /// Adopt chunks and meshes finished by background workers since last frame.
    private func integrateFinishedWork() {
        resultLock.lock()
        let chunks = finishedChunks
        finishedChunks.removeAll()
        let built = finishedMeshes
        finishedMeshes.removeAll()
        resultLock.unlock()

        for r in chunks {
            pendingGen.remove(r.coord)
            guard r.epoch == epoch else { continue }
            world.insertChunk(r.chunk, at: r.coord)
        }
        for m in built {
            pendingMesh.remove(m.coord)
            guard m.epoch == epoch else { continue }
            if discardInFlight.remove(m.coord) == nil {
                meshes[m.coord] = m.mesh
            }
            if dirtyWhileMeshing.remove(m.coord) != nil {
                submitMeshJob(m.coord) // edited while the job ran; refresh again
            }
        }
    }

    private func streamGeneration() {
        let pc = playerChunk
        // the 3x3 around the player is generated synchronously so physics and
        // the mining raycast always have real terrain (no-op once generated)
        for dz in -1...1 {
            for dx in -1...1 {
                world.generateChunk(ChunkCoord(x: pc.x + dx, z: pc.z + dz))
            }
        }

        // top background generation back up, nearest chunks first
        guard pendingGen.count < maxGenJobs else { return }
        let gen = world.generator
        let ep = epoch
        for (dx, dz) in genOffsets {
            let c = ChunkCoord(x: pc.x + dx, z: pc.z + dz)
            guard !world.isGenerated(c), !pendingGen.contains(c) else { continue }
            pendingGen.insert(c)
            workQueue.async { [weak self] in
                let chunk = gen.buildChunk(c)
                guard let self else { return }
                self.resultLock.lock()
                self.finishedChunks.append((ep, c, chunk))
                self.resultLock.unlock()
            }
            if pendingGen.count >= maxGenJobs { break }
        }
    }

    private func neighborsGenerated(_ c: ChunkCoord) -> Bool {
        world.isGenerated(c)
            && world.isGenerated(ChunkCoord(x: c.x + 1, z: c.z))
            && world.isGenerated(ChunkCoord(x: c.x - 1, z: c.z))
            && world.isGenerated(ChunkCoord(x: c.x, z: c.z + 1))
            && world.isGenerated(ChunkCoord(x: c.x, z: c.z - 1))
    }

    /// MTLDevice is thread-safe, so buffer upload happens wherever the
    /// geometry was built — main thread for edits, workers for streaming.
    private func uploadGeometry(_ geo: ChunkGeometry, into mesh: ChunkMesh) {
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
    }

    /// Mesh + light job for one chunk; results land in integrateFinishedWork.
    private func submitMeshJob(_ c: ChunkCoord) {
        pendingMesh.insert(c)
        let snapshot = ChunkSnapshot(world: world, center: c)
        let ep = epoch
        workQueue.async { [weak self] in
            guard let self else { return }
            let geo = Mesher.buildChunk(snapshot: snapshot, coord: c)
            let mesh = ChunkMesh()
            self.uploadGeometry(geo, into: mesh)
            self.resultLock.lock()
            self.finishedMeshes.append((ep, c, mesh))
            self.resultLock.unlock()
        }
    }

    /// Instant remesh of the chunks whose geometry a player edit visibly
    /// changed: the edited cell's chunk plus border neighbors. The wider 3×3
    /// light refresh stays on the background queue, where the few-ms latency
    /// is invisible. Any in-flight result for these chunks is now stale and
    /// gets dropped on arrival.
    private func remeshUrgent(_ x: Int, _ z: Int) {
        let cc = World.chunkCoord(blockX: x, blockZ: z)
        var coords = [cc]
        let lx = x & 15, lz = z & 15
        if lx == 0 { coords.append(ChunkCoord(x: cc.x - 1, z: cc.z)) }
        if lx == 15 { coords.append(ChunkCoord(x: cc.x + 1, z: cc.z)) }
        if lz == 0 { coords.append(ChunkCoord(x: cc.x, z: cc.z - 1)) }
        if lz == 15 { coords.append(ChunkCoord(x: cc.x, z: cc.z + 1)) }
        for c in coords where meshes[c] != nil && neighborsGenerated(c) {
            if pendingMesh.contains(c) { discardInFlight.insert(c) }
            let snapshot = ChunkSnapshot(world: world, center: c)
            let geo = Mesher.buildChunk(snapshot: snapshot, coord: c)
            let mesh = meshes[c] ?? ChunkMesh()
            uploadGeometry(geo, into: mesh)
            meshes[c] = mesh
            world.dirtyChunks.remove(c) // already fresh; skip the background pass
        }
    }

    /// Edits remesh through the background pipeline (lighting makes a rebuild
    /// a few ms, too slow for the render loop). A chunk already being meshed
    /// is remembered and refreshed when the stale result lands.
    private func remeshDirtyChunks() {
        guard !world.dirtyChunks.isEmpty else { return }
        for c in world.dirtyChunks {
            if pendingMesh.contains(c) {
                dirtyWhileMeshing.insert(c)
            } else if meshes[c] != nil && neighborsGenerated(c) {
                submitMeshJob(c)
            }
        }
        world.dirtyChunks.removeAll()
    }

    private func streamMeshes() {
        let pc = playerChunk
        if pendingMesh.count < maxMeshJobs {
            for (dx, dz) in meshOffsets {
                let c = ChunkCoord(x: pc.x + dx, z: pc.z + dz)
                guard meshes[c] == nil, !pendingMesh.contains(c), neighborsGenerated(c) else { continue }
                submitMeshJob(c)
                if pendingMesh.count >= maxMeshJobs { break }
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
        case Keys.e:
            if gui.screen != nil { closeGUI() } else { openGUI(.inventory) }
        case Keys.escape:
            if gui.screen != nil { closeGUI() }
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

    private func dropFor(_ b: Block) -> (item: Item, count: Int)? {
        switch b {
        case .grass: return (.block(.dirt), 1) // like Minecraft, grass drops dirt
        case .stone: return (.block(.cobblestone), 1)
        case .coalOre: return (.coal, 1)
        case .diamondOre: return (.diamond, 1)
        case .redstoneOre: return (.redstone, 4)
        case .gravel: return Float.random(in: 0...1) < 0.3 ? (.flint, 1) : (.block(.gravel), 1)
        case .furnaceLit: return (.block(.furnace), 1)
        default: return (.block(b), 1)
        }
    }

    private func spawnDrop(_ item: Item, count: Int = 1, at p: SIMD3<Float>) {
        let vel = SIMD3<Float>(Float.random(in: -1.2...1.2),
                               Float.random(in: 2.2...3.4),
                               Float.random(in: -1.2...1.2))
        items.append(ItemEntity(item: item, pos: p, vel: vel, count: count))
    }

    /// Seconds to break a block by hand, or faster with the matching tool —
    /// the real game's hardness × 1.5, divided by the tool's dig speed.
    private func breakTime(_ b: Block) -> Float {
        var speed: Float = 1
        if case .tool(let type, let material)? = inventory.selectedStack?.item,
           type == b.preferredTool {
            speed = material.miningSpeed
        }
        return b.hardness * 1.5 / speed
    }

    /// Progressive mining: holding the button charges the targeted block's
    /// break progress; the block pops once it reaches 1. Switching targets or
    /// releasing the button forfeits the progress, like the real game.
    private func updateMining(_ hit: RayHit?, dt: Float) {
        guard input.leftDown, input.captured, gui.screen == nil, let hit else {
            breakingPos = nil
            breakingProgress = 0
            return
        }
        // a mob in front catches the swing instead of the block behind it
        let x = Int(hit.block.x), y = Int(hit.block.y), z = Int(hit.block.z)
        let b = world.block(x, y, z)
        guard b != .air && b != .bedrock && !b.isWater,
              nearestMobHit(origin: player.eye, dir: player.forward,
                            maxDist: min(hit.t, 3.5)) == nil else {
            breakingPos = nil
            breakingProgress = 0
            return
        }
        if breakingPos != hit.block {
            breakingPos = hit.block
            breakingProgress = 0
        }
        let time = breakTime(b)
        breakingProgress = time <= 0 ? 1 : breakingProgress + dt / time
        if breakingProgress >= 1 {
            mine(hit)
            breakingPos = nil
            breakingProgress = 0
        }
    }

    private func mine(_ hit: RayHit?) {
        guard let hit else { return }
        let x = Int(hit.block.x), y = Int(hit.block.y), z = Int(hit.block.z)
        let b = world.block(x, y, z)
        guard b != .air && b != .bedrock && !b.isWater else { return }
        world.setBlock(x, y, z, .air)
        remeshUrgent(x, z)
        let center = SIMD3<Float>(Float(x) + 0.5, Float(y) + 0.4, Float(z) + 0.5)
        // a mined furnace spills whatever it held
        if b == .furnace || b == .furnaceLit,
           let f = furnaces.removeValue(forKey: BlockPos(x: x, y: y, z: z)) {
            for stack in [f.input, f.fuel, f.output].compactMap({ $0 }) {
                spawnDrop(stack.item, count: stack.count, at: center)
            }
        }
        if let drop = dropFor(b) {
            spawnDrop(drop.item, count: drop.count, at: center)
        }
    }

    /// Right click: open interactive blocks, otherwise place the selected one.
    private func rightClick(_ hit: RayHit?) {
        guard let hit else { return }
        let x = Int(hit.block.x), y = Int(hit.block.y), z = Int(hit.block.z)
        switch world.block(x, y, z) {
        case .craftingTable:
            openGUI(.craftingTable)
        case .furnace, .furnaceLit:
            let pos = BlockPos(x: x, y: y, z: z)
            if furnaces[pos] == nil { furnaces[pos] = FurnaceState() }
            openGUI(.furnace(pos))
        default:
            place(hit)
        }
    }

    private func place(_ hit: RayHit?) {
        guard let hit, let stack = inventory.selectedStack,
              let block = stack.item.asBlock else { return }
        let p = hit.block &+ hit.normal
        let x = Int(p.x), y = Int(p.y), z = Int(p.z)
        let target = world.block(x, y, z)
        guard target == .air || target.isWater else { return }

        if block == .torch {
            // torches need solid ground and have no collision box
            guard world.isSolid(x, y - 1, z) else { return }
        } else {
            // don't place a block inside the player
            let bmin = SIMD3<Float>(Float(x), Float(y), Float(z))
            let bmax = bmin + SIMD3<Float>(1, 1, 1)
            let (pmin, pmax) = player.aabb
            let overlaps = pmin.x < bmax.x && pmax.x > bmin.x
                && pmin.y < bmax.y && pmax.y > bmin.y
                && pmin.z < bmax.z && pmax.z > bmin.z
            guard !overlaps || player.flying else { return }
        }

        world.setBlock(x, y, z, block)
        remeshUrgent(x, z)
        _ = inventory.consumeSelected()
    }

    // MARK: - Furnaces

    private func tickFurnaces(dt: Float) {
        for (pos, f) in furnaces {
            f.tick(dt: dt)
            // keep the block's lit face in sync with the burn state
            let current = world.block(pos.x, pos.y, pos.z)
            guard current == .furnace || current == .furnaceLit else { continue }
            let wanted: Block = f.isBurning ? .furnaceLit : .furnace
            if current != wanted {
                world.setBlock(pos.x, pos.y, pos.z, wanted)
            }
        }
    }

    // MARK: - HUD

    private func updateHUD(view: MTKView) {
        guard let hud else { return }
        var text = "  WASD move · Space jump · 2×Space fly · E inventory · F5 view · LMB mine · RMB place/use · R new world  "
        if gui.screen != nil {
            text = "  ▸ Click slots to move items · right-click splits · E/Esc closes  "
        } else if player.flying {
            text = "  ✈ SPECTATOR — Space up · Shift down  " + text
        }
        if !input.captured && gui.screen == nil {
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
