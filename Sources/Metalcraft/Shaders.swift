// Metal Shading Language source, compiled at runtime via device.makeLibrary(source:).
let shaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 viewProj;
    float4x4 model;     // identity for terrain; item entities and UI icons set it
    float4 camPos;
    float4 sunDir;
    float4 fogColor;
    float4 fogParams;   // x = fog start, y = fog end
    float4 alphaParams; // x = alpha multiplier, y = discard threshold, z = hurt flash
    float4 lightParams; // x = skylight subtracted 0-11 (beta night), y = mode
                        // (0 vertex / 1 entity / 2 full), z/w = entity sky/block 0-15
};

struct VIn {
    packed_float3 pos;
    packed_float3 normal;
    packed_float2 uv;
    packed_float3 tint;
    packed_float2 light; // (sky, block) 0-15
};

struct VOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float2 uv;
    float3 tint;
    float2 light;
};

vertex VOut block_vertex(uint vid [[vertex_id]],
                         const device VIn *verts [[buffer(0)]],
                         constant Uniforms &u [[buffer(1)]]) {
    VIn v = verts[vid];
    VOut out;
    float3 wp = (u.model * float4(float3(v.pos), 1.0)).xyz;
    out.position = u.viewProj * float4(wp, 1.0);
    out.worldPos = wp;
    out.normal = (u.model * float4(float3(v.normal), 0.0)).xyz;
    out.uv = float2(v.uv);
    out.tint = float3(v.tint);
    out.light = float2(v.light);
    return out;
}

// 256px atlas of 16px tiles stays tile-pure down to mip 4 (1px per tile),
// so clamp the lod there to avoid cross-tile bleeding.
constexpr sampler atlas_sampler(mag_filter::nearest,
                                min_filter::nearest,
                                mip_filter::linear,
                                address::clamp_to_edge,
                                lod_clamp(0.0f, 4.0f));

fragment float4 block_fragment(VOut in [[stage_in]],
                               constant Uniforms &u [[buffer(1)]],
                               texture2d<float> atlas [[texture(0)]]) {
    float4 tex = atlas.sample(atlas_sampler, in.uv);
    float alpha = min(tex.a * u.alphaParams.x, 1.0);
    if (alpha < u.alphaParams.y) {
        discard_fragment();
    }
    float3 n = normalize(in.normal);

    // beta 1.7.3 face shading: top 1.0, bottom 0.5, x sides 0.6, z sides 0.8
    float face = fabs(n.y) > 0.5 ? (n.y > 0.0 ? 1.0 : 0.5)
                                 : (fabs(n.x) > 0.5 ? 0.6 : 0.8);

    // beta lighting is monochrome: night subtracts whole skylight levels,
    // the channels combine by max, and one brightness ramp does the rest
    float level;
    if (u.lightParams.y > 1.5) {
        level = 15.0;
    } else {
        float sky = u.lightParams.y > 0.5 ? u.lightParams.z : in.light.x;
        float blk = u.lightParams.y > 0.5 ? u.lightParams.w : in.light.y;
        level = max(sky - u.lightParams.x, blk);
    }
    float x = clamp(level / 15.0, 0.0, 1.0);
    float br = x / (4.0 - 3.0 * x); // beta brightness table; level 0 is black

    float3 c = tex.rgb * in.tint * face * br;
    // hurt flash: damaged/dying mobs go Minecraft red
    c = mix(c, c * float3(1.0, 0.3, 0.3), u.alphaParams.z);

    float dist = distance(in.worldPos, u.camPos.xyz);
    float f = clamp((dist - u.fogParams.x) / (u.fogParams.y - u.fogParams.x), 0.0, 1.0);
    c = mix(c, u.fogColor.rgb, f);
    alpha = mix(alpha, 1.0, f); // distant water fades into solid fog
    return float4(c, alpha);
}

struct LineUniforms {
    float4x4 mvp;
    float4 color;
};

// beta cloud layer: a huge quad tiling clouds.png, tinted by daylight
struct CloudVOut {
    float4 position [[position]];
    float2 uv;
};

vertex CloudVOut cloud_vertex(uint vid [[vertex_id]],
                              const device float *verts [[buffer(0)]],
                              constant LineUniforms &u [[buffer(1)]]) {
    CloudVOut out;
    out.position = u.mvp * float4(verts[vid * 5], verts[vid * 5 + 1], verts[vid * 5 + 2], 1.0);
    out.uv = float2(verts[vid * 5 + 3], verts[vid * 5 + 4]);
    return out;
}

constexpr sampler cloud_sampler(mag_filter::nearest, min_filter::nearest,
                                address::repeat);

fragment float4 cloud_fragment(CloudVOut in [[stage_in]],
                               constant LineUniforms &u [[buffer(1)]],
                               texture2d<float> tex [[texture(0)]]) {
    float4 c = tex.sample(cloud_sampler, in.uv);
    if (c.a < 0.1) {
        discard_fragment();
    }
    return c * u.color;
}

// Textured UI quads (GUI dialogs, hotbar, item sprites): unlit, nearest
// sampling, transparent texels discarded. Vertices are packed [x y z u v].
struct UIVOut {
    float4 position [[position]];
    float2 uv;
};

vertex UIVOut ui_vertex(uint vid [[vertex_id]],
                        const device float *verts [[buffer(0)]],
                        constant LineUniforms &u [[buffer(1)]]) {
    UIVOut out;
    out.position = u.mvp * float4(verts[vid * 5], verts[vid * 5 + 1], verts[vid * 5 + 2], 1.0);
    out.uv = float2(verts[vid * 5 + 3], verts[vid * 5 + 4]);
    return out;
}

constexpr sampler ui_sampler(mag_filter::nearest, min_filter::nearest, address::repeat);

fragment float4 ui_fragment(UIVOut in [[stage_in]],
                            constant LineUniforms &u [[buffer(1)]],
                            texture2d<float> tex [[texture(0)]]) {
    float4 c = tex.sample(ui_sampler, in.uv);
    if (c.a < 0.01) {
        discard_fragment();
    }
    return c * u.color;
}

vertex float4 line_vertex(uint vid [[vertex_id]],
                          const device packed_float3 *verts [[buffer(0)]],
                          constant LineUniforms &u [[buffer(1)]]) {
    return u.mvp * float4(float3(verts[vid]), 1.0);
}

fragment float4 line_fragment(constant LineUniforms &u [[buffer(1)]]) {
    return u.color;
}
"""
