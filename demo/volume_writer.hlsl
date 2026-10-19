// Demo shader for dxbc-replay-verify.
//
// A minimal but genuine "volume writer": it rasterizes a 16x16x16 RGBA8
// volume into a 16^3 R8G8B8A8_UNORM 3D texture, driven by a gain constant in
// the pixel CB0 and a small dependency texture.
//
// The layout contract below matches what replay/worker_replay.py sets up, and
// keeping it explicit is the point: manifest-driven replay is only meaningful
// when the draw's shape is stated, not assumed.
//
//   * 4 vertices, one full-viewport triangle strip, no vertex buffer attribute
//     reads (positions come from SV_VertexID).
//   * 16 instances; SV_InstanceID selects the output z slice.
//   * Render target is a 16^3 texture viewed across all 16 slices, viewport 16x16.
//   * Within one slice: SV_Position.x -> R axis, SV_Position.y -> G axis.
//   * Packed uint32 output: bits 0-7 R, 8-15 G, 16-23 B, 24-31 A.
//
// The gain constant lives in the pixel CB0 (slot 0) as a float32 at byte
// offset 0. Changing it changes the volume, which is exactly what the
// dirty-rebuild cache and the baseline/modified/rollback check exist to prove.
//
// Arithmetic note: the pixel stage deliberately uses only + - * /, saturate()
// and an exact-texel-center texture fetch. demo/expected_volume.py
// re-implements the same math on the CPU with every intermediate rounded to
// float32, so the expected bytes can be compared against a hardware replay
// bit for bit. Avoid pow/log/exp here: their GPU precision is not specified
// tightly enough for a byte-exact expectation.

#define VOLUME_SIZE 16

// ---------------------------------------------------------------------------
// Shared vertex/geometry signature.
// ---------------------------------------------------------------------------
struct VSOut
{
    float4 position : SV_Position;
    uint   slice    : TEXCOORD0;   // SV_InstanceID = output z slice
};

// ---------------------------------------------------------------------------
// Vertex stage: emit the strip quad and carry the instance id (= z slice).
// ---------------------------------------------------------------------------
VSOut vs_main(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    VSOut o;
    float2 corner = float2((vertex_id & 1) ? 1.0 : -1.0,
                           (vertex_id & 2) ? -1.0 : 1.0);
    o.position = float4(corner, 0.0, 1.0);
    o.slice = instance_id;
    return o;
}

// ---------------------------------------------------------------------------
// Geometry stage: the worker binds all three stages, and the render target is
// a 16^3 texture viewed across all 16 slices at once, so *something* has to
// route each instance to its own z slice -- that is what
// SV_RenderTargetArrayIndex is for. A pass-through geometry shader that only
// copies TEXCOORDs would let every instance rasterize into every slice, with
// the last instance silently winning; the slice index must travel as a system
// value, not just as data.
//
// Saying so explicitly is cheaper than discovering an implicit requirement at
// replay time: the first version of this demo made exactly that mistake and
// the byte comparison caught it (every slice showed the last instance's
// values).
// ---------------------------------------------------------------------------
struct GSOut
{
    float4 position : SV_Position;
    uint   slice    : TEXCOORD0;
    uint   rt_index : SV_RenderTargetArrayIndex;
};

[maxvertexcount(3)]
void gs_main(triangle VSOut input[3], inout TriangleStream<GSOut> stream)
{
    for (int i = 0; i < 3; ++i)
    {
        GSOut o;
        o.position = input[i].position;
        o.slice = input[i].slice;
        o.rt_index = input[i].slice;
        stream.Append(o);
    }
}

// ---------------------------------------------------------------------------
// Pixel stage: rebuild (R, G, B) from the coordinate, apply the gain constant
// and a per-axis factor fetched from the dependency texture, then pack to
// RGBA8. Writing the packed bits back out through a float4 keeps the byte
// layout of the render target obvious in the readback.
//
// The dependency texture is 4x4 and is always sampled at exact texel centres
// ((i + 0.5) / 4), so linear filtering returns the texel value itself and the
// CPU expectation needs no interpolation model.
// ---------------------------------------------------------------------------
struct CBParams
{
    float gain;       // byte offset 0
    float _pad0;
    float _pad1;
    float _pad2;
};

cbuffer CBParams : register(b0)
{
    float g_gain;
};

Texture2D      g_source  : register(t0);
SamplerState   g_sampler : register(s0);

float4 ps_main(VSOut input, float4 screen : SV_Position) : SV_Target
{
    uint r = (uint) screen.x;   // 0..15
    uint g = (uint) screen.y;   // 0..15
    uint b = input.slice;       // 0..15

    float3 base = float3(r, g, b) / (float) (VOLUME_SIZE - 1);

    // Exact texel-centre fetch from the 4x4 dependency texture.
    float2 uv = (float2(b & 3u, r & 3u) + 0.5) / 4.0;
    float3 tex = g_source.SampleLevel(g_sampler, uv, 0).rgb;

    // Demo transform: tint the entry by the per-axis factor and the gain
    // constant. A real writer would do something application specific; what
    // matters here is that the output depends on a CB constant the controller
    // can patch and on a texture the worker must bind.
    float3 rgb = saturate(base * tex * g_gain);

    uint packed = (uint) (rgb.r * 255.0 + 0.5)
                | ((uint) (rgb.g * 255.0 + 0.5) << 8)
                | ((uint) (rgb.b * 255.0 + 0.5) << 16);

    // Emit the exact packed components; the UNORM target stores them 8-bit.
    return float4((packed & 0xFFu) / 255.0,
                  ((packed >> 8) & 0xFFu) / 255.0,
                  ((packed >> 16) & 0xFFu) / 255.0,
                  1.0);
}
