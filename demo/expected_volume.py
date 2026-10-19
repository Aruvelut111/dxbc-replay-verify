"""Independent CPU expectation for the demo volume writer.

demo/volume_writer.hlsl is the shader under test; this module is the
*expectation* side of the verification: it re-implements the same transform on
the CPU, rounding every intermediate to float32, so the expected 16^3
R8G8B8A8 bytes can be compared against a hardware replay byte for byte.

The mirroring is deliberately literal -- same operation order, same rounding
points, no reassociation. If the replay disagrees, either the replay pipeline
is wrong or the shader was optimised differently than written; both are
exactly the kind of finding this tool exists to surface.

Deliberately absent: pow/log/exp and any interpolation model. Those have
GPU precision the D3D11 spec does not pin down tightly enough for a
byte-exact expectation, so the demo shader avoids them too.
"""
from __future__ import annotations

import struct
from pathlib import Path

VOLUME_SIZE = 16
BASE_GAIN = 1.0


def f32(x: float) -> float:
    """Round a Python float to the nearest float32, as GPU math would."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def load_source_texture(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def _texel(texture: bytes, b: int, r: int) -> tuple[float, float, float]:
    """UNORM8 texel at exact centre (b % 4, r % 4) -> float32 triple.

    The shader samples at ((i + 0.5) / 4), an exact texel centre, so linear
    filtering returns the texel value itself and no interpolation happens.
    """
    x, y = b & 3, r & 3
    offset = (y * 4 + x) * 4
    return tuple(f32(texture[offset + c] / 255.0) for c in range(3))  # type: ignore[return-value]


def expected_voxel_code(r: int, g: int, b: int, gain: float,
                        texture: bytes) -> int:
    """One packed uint32: bits 0-7 R, 8-15 G, 16-23 B, 24-31 A (mirrors ps_main).

    The shader emits alpha 1.0, which the 8-bit UNORM alpha of R8G8B8A8
    stores as 255.
    """
    base = tuple(f32(v / (VOLUME_SIZE - 1)) for v in (r, g, b))
    tex = _texel(texture, b, r)

    codes = []
    for c in range(3):
        rgb = f32(f32(base[c] * tex[c]) * gain)
        rgb = min(1.0, max(0.0, rgb))  # saturate
        codes.append(int(f32(f32(rgb * 255.0) + 0.5)))  # (uint)(x * 255 + 0.5)
    return codes[0] | (codes[1] << 8) | (codes[2] << 16) | (255 << 24)


def expected_volume(gain: float, texture: bytes) -> bytes:
    """Full 16^3 packed volume, z-major then y then x (worker readback order)."""
    out = bytearray()
    for b in range(VOLUME_SIZE):
        for g in range(VOLUME_SIZE):
            for r in range(VOLUME_SIZE):
                out += struct.pack("<I", expected_voxel_code(r, g, b, gain, texture))
    return bytes(out)


def expected_volume_from_paths(gain: float, texture_path: str | Path) -> bytes:
    return expected_volume(gain, load_source_texture(texture_path))
