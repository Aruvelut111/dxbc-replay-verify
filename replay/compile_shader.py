"""Compile HLSL to DXBC using only Windows' own d3dcompiler_47.dll.

The replay tool deliberately consumes *compiled* bytecode, because compiled
bytecode is the artifact under test: whatever the compiler produced is what
gets replayed. This module closes the loop for the demo: HLSL source -> DXBC
container, with no external SDK, no fxc.exe on PATH and no
DirectXShaderCompiler NuGet package.

ID3DBlob is reached through its vtable (QueryInterface/AddRef/Release/
GetBufferPointer/GetBufferSize) rather than through comtypes, so nothing beyond
ctypes is required.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path

D3DCOMPILER = "d3dcompiler_47.dll"

# D3DCompile: HRESULT D3DCompile(pSrcData, SrcDataSize, pSourceName, pDefines,
#   pInclude, pEntrypoint, pTarget, Flags1, Flags2, ppCode, ppErrorMsgs)
D3DCOMPILE_PROTO = C.WINFUNCTYPE(
    C.c_long,                        # HRESULT
    C.c_void_p, C.c_size_t, C.c_char_p,
    C.c_void_p, C.c_void_p,
    C.c_char_p, C.c_char_p,
    C.c_uint, C.c_uint,
    C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
)


def _blob_bytes(blob: int) -> bytes:
    """Read an ID3DBlob through its vtable."""
    if not blob:
        raise RuntimeError("null ID3DBlob")
    vtable = C.cast(blob, C.POINTER(C.c_void_p))[0]
    get_pointer = C.WINFUNCTYPE(C.c_void_p, C.c_void_p)(C.cast(vtable, C.POINTER(C.c_void_p))[3])
    get_size = C.WINFUNCTYPE(C.c_size_t, C.c_void_p)(C.cast(vtable, C.POINTER(C.c_void_p))[4])
    ptr = get_pointer(blob)
    size = get_size(blob)
    return C.string_at(ptr, size) if ptr and size else b""


def compile_hlsl(source: str, entry_point: str, target: str, flags: int = 0) -> bytes:
    """Compile one HLSL entry point to DXBC bytes. Raises with the compiler log."""
    lib = C.WinDLL(D3DCOMPILER)
    compile_fn = D3DCOMPILE_PROTO(("D3DCompile", lib))

    src = source.encode("utf-8")
    code, errors = C.c_void_p(), C.c_void_p()
    hr = compile_fn(src, len(src), b"<demo>", None, None,
                    entry_point.encode("ascii"), target.encode("ascii"),
                    flags, 0, C.byref(code), C.byref(errors))

    error_log = _blob_bytes(errors.value) if errors else b""
    if hr < 0:
        raise RuntimeError(
            f"D3DCompile failed for {entry_point} ({target}), HRESULT=0x{hr & 0xFFFFFFFF:08X}\n"
            + error_log.decode("utf-8", "replace")
        )
    if error_log.strip():
        # Warnings are surfaced rather than swallowed.
        print(error_log.decode("utf-8", "replace").strip())

    bytecode = _blob_bytes(code.value)
    if not bytecode:
        raise RuntimeError("D3DCompile returned an empty blob")
    if bytecode[:4] != b"DXBC":
        raise RuntimeError(f"unexpected container magic {bytecode[:4]!r}, expected b'DXBC'")
    return bytecode


def compile_file(hlsl_path: Path, entry_point: str, target: str) -> bytes:
    return compile_hlsl(Path(hlsl_path).read_text(encoding="utf-8"), entry_point, target)


if __name__ == "__main__":
    import sys

    src = Path(sys.argv[1])
    print(f"compiling {src.name} ...", flush=True)
    for entry, target in (("vs_main", "vs_5_0"), ("gs_main", "gs_5_0"), ("ps_main", "ps_5_0")):
        blob = compile_file(src, entry, target)
        out = src.with_name(f"{src.stem}_{entry}.dxbc")
        out.write_bytes(blob)
        print(f"  {entry:8s} {target}  ->  {out.name}  {len(blob)} bytes")
