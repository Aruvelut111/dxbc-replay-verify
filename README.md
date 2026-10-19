# dxbc-replay-verify

Replays a compiled D3D11 draw on a real device (or the WARP software
rasterizer) and checks the output against an expectation you write yourself.

It covers the case where you refactored a shader, changed the compiler or its
flags, or moved a draw to another backend, and want a straight answer on
whether the rendered result changed. The tool runs the draw and compares raw
bytes, so the answer comes from the output itself instead of a screenshot.

## Requirements

Windows, Python 3.10+. Nothing to install: it uses d3d11.dll and
d3dcompiler_47.dll, both of which ship with Windows.

## Demo

```bat
python demo\build_demo.py                          :: compile the demo shader, build the manifest
python -c "from replay.controller import rebuild; r = rebuild('demo/build/manifest.json', 'demo/build/checks'); print(r['status'], r['reference_status'])"
python tests\test_baseline_modified_rollback.py    :: the full B/M/R verification
```

The demo compiles a three-stage shader that writes a 16x16x16 RGBA8 volume,
replays it, and compares all 4096 voxels against a CPU implementation of the
same math, rounding every intermediate to float32. Measured on this machine
(RTX-class hardware, then repeated on WARP):

```
baseline                          rebuilt    EXACT_REFERENCE_MATCHED, 4096/4096 voxels, max code error 0
CB0 gain 1.0 -> 1.5                rebuilt    different volume, matches its own expectation
gain back to 1.0                   cache_hit  bytes identical to baseline, writer not re-run
force re-run of baseline          rebuilt    same SHA-256 as baseline
manifest missing reference row    failed     failed_retaining_last_success, previous success kept
```

Hardware and WARP produce identical bytes. The test script covers these cases
and exits non-zero on any failure.

## Using it with your own draw

The worker runs exactly what one JSON manifest declares; see
`demo/build/manifest.json` for a complete example. Two parts of the draw are
declared by you rather than fixed:

- the output texture: width, height, depth and format (`R8G8B8A8_UNORM` or
  `R10G10B10A2_UNORM`), plus a row with `role: "expected_writer_output"`
  holding the bytes you expect the draw to produce;
- the draw call itself: topology (triangle strip/list, lines, points),
  vertex and instance counts, start offsets.

The harness side is fixed and validated by
`replay/controller.py:validate_profile`: three shader stages (vertex /
geometry / pixel), constant-buffer rows on stages `vertex` or `pixel`, one
vertex buffer (stride 16, two `RG32_FLOAT` input-layout elements), one pixel
SRV texture at slot 0, one viewport, solid/no-cull raster, depth & stencil
off, no blend.

Two things to know before writing your own expectation:

1. The z-slice routing has to come out of the geometry shader. A 3D render
   target is viewed across all its slices at once, so a pass-through GS that
   only copies TEXCOORDs lets every instance write every slice and the last
   one wins. The slice index has to travel as
   `SV_RenderTargetArrayIndex`. The first version of the demo had this bug,
   and the byte comparison showed every slice holding the last instance's
   values.

2. Keep the math something a CPU can reproduce exactly. pow, log and exp are
   not pinned to the bit by the GPU precision spec, and non-texel-centre
   sampling has no interpolation model to copy. The demo sticks to `+ - * /`,
   `saturate()` and exact texel-centre fetches.

## Scope

One draw, on the device you run it on. The demo happens to match across
hardware and WARP here, but treat that as one data point. No claim about
full-pipeline or engine equivalence, and nothing here is meant to run
per-frame.

## Layout

```
replay/
  worker_replay.py     ctypes D3D11 replay of a manifest-described draw
  controller.py        identity = SHA-256(effective input bytes), dirty-rebuild
                       cache, three-state result (rebuilt / cache_hit /
                       failed_retaining_last_success)
  compile_shader.py    HLSL -> DXBC via d3dcompiler_47.dll (demo bootstrapping only)
demo/
  volume_writer.hlsl   the shader under test (vs / gs / ps)
  build_demo.py        compiles the shader, writes the manifest and input blobs
  expected_volume.py   CPU expectation, float32 at every step
tests/
  test_baseline_modified_rollback.py   baseline / modified / rollback / force / failure
```

Cache keys are computed from the effective input bytes after overrides, not
from the requested labels, so two requests that produce the same inputs share
one cache entry, and a request that changes nothing is a `cache_hit` even
under a different label.

## License

MIT, see LICENSE.
