"""Build the demo manifest and its input blobs.

The manifest is the whole interface between a caller and the replay worker: it
declares which compiled shaders, constant buffers, vertex buffers and textures
participate in one draw, plus the draw's shape (viewport, scissor, topology)
and the output texture it writes. Nothing here knows about any particular
game -- point it at different bytes and the same worker replays that draw
instead.

Run:

    python demo/build_demo.py

Outputs (all inside demo/build/):

    vs.dxbc / gs.dxbc / ps.dxbc   compiled bytecode, from demo/volume_writer.hlsl
    vertex_buffer.bin             4 vertices x 16 bytes (layout-compatible padding)
    pixel_cb0.bin                 16 bytes; float32 gain at offset 0
    source_texture.bin            4x4 R8G8B8A8_UNORM, the pixel SRV dependency
    expected_output_r8g8b8a8.bin  16384 bytes of expected output (CPU-side)
    manifest.json                 what the worker reads

The expected-output row is what makes this a verification rather than a
screenshot: demo/expected_volume.py re-implements the shader's math on the CPU
with float32 rounding at every step, and the worker compares the hardware
replay against those bytes voxel by voxel.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
VOLUME_SIZE = 16
BASE_GAIN = 1.0


def write(path: Path, data: bytes) -> dict:
    path.write_bytes(data)
    return {
        "path": path.name,
        "byte_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def source_texture_bytes() -> bytes:
    """4x4 RGBA8 gradient. Non-uniform so the dependency actually matters:
    every output channel reads a different texel axis."""
    tex = bytearray()
    for y in range(4):
        for x in range(4):
            tex += bytes(((x * 60 + 30) & 0xFF,
                          (y * 60 + 30) & 0xFF,
                          ((x + y) * 28 + 24) & 0xFF,
                          255))
    return bytes(tex)


def main() -> None:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent))
    from expected_volume import BASE_GAIN as EXPECT_BASE
    from expected_volume import expected_volume, load_source_texture
    from replay.compile_shader import compile_hlsl

    assert EXPECT_BASE == BASE_GAIN, "gain constants drifted apart"

    BUILD.mkdir(parents=True, exist_ok=True)
    source = (HERE / "volume_writer.hlsl").read_text(encoding="utf-8")

    # The worker binds all three stages, so a slice-routing geometry shader is
    # compiled as part of the draw.
    stages = {
        "vertex": compile_hlsl(source, "vs_main", "vs_5_0"),
        "geometry": compile_hlsl(source, "gs_main", "gs_5_0"),
        "pixel": compile_hlsl(source, "ps_main", "ps_5_0"),
    }
    rows = {stage: write(BUILD / f"{stage}.dxbc", data) for stage, data in stages.items()}
    for stage, row in rows.items():
        row["stage"] = stage

    # Vertex buffer: the worker's input layout declares two RG32_FLOAT
    # elements with stride 16, so provide 4 such vertices (4 x 16 = 64 bytes).
    # The demo shader reads no attributes, but replaying the real
    # input-assembler path is the point.
    vertices = b"".join(struct.pack("<4f", 0.0, 0.0, 0.0, 1.0) for _ in range(4))
    vb_row = write(BUILD / "vertex_buffer.bin", vertices)
    vb_row.update(stride=16, offset=0)

    # Constant buffer: 16 bytes, float32 gain at offset 0. This is the knob
    # the dirty-rebuild cache and the B/M/R check patch.
    cb_row = write(BUILD / "pixel_cb0.bin", struct.pack("<4f", BASE_GAIN, 0.0, 0.0, 0.0))
    cb_row.update(stage="pixel", slot=0, role="demo_gain")

    # Source texture: the worker binds a pixel SRV at slot 0. 4x4 RGBA8_UNORM.
    tex_row = write(BUILD / "source_texture.bin", source_texture_bytes())
    tex_row.update(stage="pixel", slot=0, width=4, height=4,
                   format="R8G8B8A8_UNORM", row_pitch=16, slice_pitch=64,
                   role="source_pixel_input", resource_id="demo_dependency")

    # Expected output: the CPU-side mirror of the shader, generated from the
    # same gain and the same texture bytes. The worker compares the hardware
    # replay against this file voxel by voxel.
    expected = expected_volume(BASE_GAIN, load_source_texture(BUILD / "source_texture.bin"))
    ref_row = write(BUILD / "expected_output_r8g8b8a8.bin", expected)
    ref_row.update(role="expected_writer_output", stage="pixel", slot=1,
                   width=VOLUME_SIZE, height=VOLUME_SIZE, depth=VOLUME_SIZE,
                   format="DXGI_FORMAT_R8G8B8A8_UNORM",
                   resource_id="demo_expected_output")
    # slot=1 keeps this row out of the worker's pixel-SRV picker (slot 0), so
    # the source texture is selected by construction, not by list order.

    manifest = {
        "schema": "dxbc-replay-draw/1",
        "capture": "demo: self-compiled HLSL, no external capture",
        "event_id": "demo_volume_writer",
        "draw": {
            "api": "D3D11",
            "topology": "TriangleStrip",
            "vertex_count": 4,
            "instance_count": VOLUME_SIZE,
            "start_vertex": 0,
            "start_instance": 0,
            "note": "4 strip vertices x 16 instances; SV_InstanceID selects the output z slice",
        },
        "shaders": [rows["vertex"], rows["geometry"], rows["pixel"]],
        "constant_buffers": [cb_row],
        "vertex_buffers": [vb_row],
        "textures": [tex_row, ref_row],
        "input_layout": [
            {"semantic": "ATTRIBUTE", "semantic_index": 0, "format": "RG32_FLOAT",
             "slot": 0, "offset": 0, "perInstance": False},
            {"semantic": "ATTRIBUTE", "semantic_index": 1, "format": "RG32_FLOAT",
             "slot": 0, "offset": 8, "perInstance": False},
        ],
        "viewport": {"x": 0, "y": 0, "width": VOLUME_SIZE, "height": VOLUME_SIZE,
                     "minDepth": 0.0, "maxDepth": 1.0},
        "scissors": [{"x": 0, "y": 0, "width": VOLUME_SIZE, "height": VOLUME_SIZE}],
        "rasterizer": {"fillMode": "Solid", "cullMode": "NoCull", "frontCCW": True,
                       "depthClip": True, "scissorEnable": True,
                       "multisampleEnable": True, "antialiasedLineEnable": False,
                       "depthBias": 0, "depthBiasClamp": 0, "slopeScaledDepthBias": 0},
        "depth_stencil": {"depthEnable": False, "stencilEnable": False},
        "blend_state": {"targets": [{"enabled": False}]},
        "output": {"width": VOLUME_SIZE, "height": VOLUME_SIZE, "depth": VOLUME_SIZE,
                   "format": "R8G8B8A8_UNORM",
                   "packing": "bits 0-7 R, 8-15 G, 16-23 B, 24-31 A"},
    }
    (BUILD / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"manifest -> {BUILD / 'manifest.json'}")
    print(f"  shaders : {[(r['stage'], r['byte_size']) for r in manifest['shaders']]}")
    print(f"  cb0     : {cb_row['byte_size']} bytes, gain at offset 0")
    print(f"  texture : 4x4 RGBA8_UNORM pixel SRV")
    print(f"  expected: {ref_row['byte_size']} bytes, sha256 {ref_row['sha256'][:16]}...")
    print(f"  output  : {VOLUME_SIZE}^3 R8G8B8A8_UNORM")


if __name__ == "__main__":
    main()
