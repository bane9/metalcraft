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
    float4 alphaParams; // x = alpha multiplier, y = discard threshold
};

struct VIn {
    packed_float3 pos;
    packed_float3 normal;
    packed_float2 uv;
    packed_float3 tint;
};

struct VOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float2 uv;
    float3 tint;
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
    float diff = max(dot(n, normalize(u.sunDir.xyz)), 0.0);
    float light = 0.45 + 0.55 * diff;
    float3 c = tex.rgb * in.tint * light;

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

vertex float4 line_vertex(uint vid [[vertex_id]],
                          const device packed_float3 *verts [[buffer(0)]],
                          constant LineUniforms &u [[buffer(1)]]) {
    return u.mvp * float4(float3(verts[vid]), 1.0);
}

fragment float4 line_fragment(constant LineUniforms &u [[buffer(1)]]) {
    return u.color;
}
"""
