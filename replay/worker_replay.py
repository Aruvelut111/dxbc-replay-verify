"""D3D11 replay of pre-compiled DXBC bytecode with caller-supplied input bytes.

Uses the current Python process as a native D3D11 host: no engine process and
no capture tooling is involved at replay time. Everything is driven by a JSON
manifest (shaders / constant buffers / vertex buffers / textures / draw /
viewport / samplers), so the same worker replays any compatible D3D11 draw
whose compiled bytecode and inputs you can point at.

The point of this tool is verification, not rendering: if you believe a compiled
shader computes X, you can replay it on real hardware and compare the output
bytes against your expectation, byte for byte.
"""
import ctypes as C
import datetime as DT
import json
from pathlib import Path
import struct
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--backend', choices=('hardware', 'warp'), required=True)
args = parser.parse_args()
INPUT_MANIFEST = Path(args.manifest).resolve()
EXPORT = INPUT_MANIFEST.parent
HERE = Path(args.output_dir).resolve()
HERE.mkdir(parents=True, exist_ok=True)
HARDWARE = args.backend == 'hardware'
P, U, F, HR = C.c_void_p, C.c_uint, C.c_float, C.c_long

# Output textures the worker can create, by manifest-declared DXGI format name.
# Each entry: (dxgi enum, per-channel bit widths). 32bpp packed formats only,
# which keeps the readback layout (width*4 bytes per row) uniform.
OUTPUT_FORMATS = {
    'R8G8B8A8_UNORM': (28, (8, 8, 8, 8)),
    'R10G10B10A2_UNORM': (24, (10, 10, 10, 2)),
}
TOPOLOGIES = {'PointList': 1, 'LineList': 2, 'LineStrip': 3, 'TriangleList': 4, 'TriangleStrip': 5}


def method(obj, index, result, *args):
    return C.WINFUNCTYPE(result, P, *args)(C.cast(obj, C.POINTER(C.POINTER(P))).contents[index])


def check(hr, label):
    if hr < 0:
        raise RuntimeError(f'{label}: HRESULT 0x{hr & 0xffffffff:08x}')


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


class Sample(C.Structure):
    _fields_ = [('Count', U), ('Quality', U)]


class Tex2(C.Structure):
    _fields_ = [('Width', U), ('Height', U), ('MipLevels', U), ('ArraySize', U), ('Format', U),
                ('SampleDesc', Sample), ('Usage', U), ('BindFlags', U), ('CPUAccessFlags', U), ('MiscFlags', U)]


class Tex3(C.Structure):
    _fields_ = [(k, U) for k in ['Width', 'Height', 'Depth', 'MipLevels', 'Format', 'Usage', 'BindFlags', 'CPUAccessFlags', 'MiscFlags']]


class BufferDesc(C.Structure):
    _fields_ = [(k, U) for k in ['ByteWidth', 'Usage', 'BindFlags', 'CPUAccessFlags', 'MiscFlags', 'StructureByteStride']]


class Init(C.Structure):
    _fields_ = [('pSysMem', P), ('SysMemPitch', U), ('SysMemSlicePitch', U)]


class InputElement(C.Structure):
    _fields_ = [('SemanticName', C.c_char_p), ('SemanticIndex', U), ('Format', U), ('InputSlot', U),
                ('AlignedByteOffset', U), ('InputSlotClass', U), ('InstanceDataStepRate', U)]


class Raster(C.Structure):
    _fields_ = [('FillMode', U), ('CullMode', U), ('FrontCounterClockwise', C.c_int),
                ('DepthBias', C.c_int), ('DepthBiasClamp', F), ('SlopeScaledDepthBias', F),
                ('DepthClipEnable', C.c_int), ('ScissorEnable', C.c_int),
                ('MultisampleEnable', C.c_int), ('AntialiasedLineEnable', C.c_int)]


class Sampler(C.Structure):
    _fields_ = [('Filter', U), ('AddressU', U), ('AddressV', U), ('AddressW', U), ('MipLODBias', F),
                ('MaxAnisotropy', U), ('ComparisonFunc', U), ('BorderColor', F * 4), ('MinLOD', F), ('MaxLOD', F)]


class Viewport(C.Structure):
    _fields_ = [(k, F) for k in ['TopLeftX', 'TopLeftY', 'Width', 'Height', 'MinDepth', 'MaxDepth']]


class Rect(C.Structure):
    _fields_ = [(k, C.c_long) for k in ['left', 'top', 'right', 'bottom']]


class RTVDesc(C.Structure):
    _fields_ = [(k, U) for k in ['Format', 'ViewDimension', 'MipSlice', 'FirstWSlice', 'WSize']]


class Mapped(C.Structure):
    _fields_ = [('pData', P), ('RowPitch', U), ('DepthPitch', U)]


class SRVDesc(C.Structure):
    _fields_ = [(k, U) for k in ['Format', 'ViewDimension', 'MostDetailedMip', 'MipLevels', 'Unused0', 'Unused1']]


class GUID(C.Structure):
    _fields_ = [('Data1', U), ('Data2', C.c_ushort), ('Data3', C.c_ushort), ('Data4', C.c_ubyte * 8)]


class AdapterDesc(C.Structure):
    _fields_ = [('Description', C.c_wchar * 128), ('VendorId', U), ('DeviceId', U), ('SubSysId', U), ('Revision', U),
                ('DedicatedVideoMemory', C.c_size_t), ('DedicatedSystemMemory', C.c_size_t), ('SharedSystemMemory', C.c_size_t),
                ('AdapterLuidLow', U), ('AdapterLuidHigh', C.c_long)]


def adapter_info(device):
    import uuid
    guid = GUID.from_buffer_copy(uuid.UUID('54ec77fa-1377-44e6-8c32-88fd5f44c84c').bytes_le)
    dxgi, adapter = P(), P()
    check(method(device, 0, HR, C.POINTER(GUID), C.POINTER(P))(device, C.byref(guid), C.byref(dxgi)), 'Query IDXGIDevice')
    try:
        check(method(dxgi, 7, HR, C.POINTER(P))(dxgi, C.byref(adapter)), 'IDXGIDevice.GetAdapter')
        desc = AdapterDesc()
        check(method(adapter, 8, HR, C.POINTER(AdapterDesc))(adapter, C.byref(desc)), 'IDXGIAdapter.GetDesc')
        return dict(description=desc.Description, vendor_id=desc.VendorId, device_id=desc.DeviceId,
                    dedicated_video_memory=desc.DedicatedVideoMemory, shared_system_memory=desc.SharedSystemMemory)
    finally:
        for obj in [adapter, dxgi]:
            if obj:
                method(obj, 2, U)(obj)


def normalize_manifest(manifest):
    """Read the manifest interface; explicit schema errors remain actionable."""
    shaders = manifest['shaders']
    if isinstance(shaders, list):
        shaders = {r['stage']: r for r in shaders}
    buffers = manifest['constant_buffers']
    if isinstance(buffers, dict):
        buffers = list(buffers.values())
    return shaders, buffers


def path_of(row):
    for key in ['raw_path', 'path', 'file', 'output_file', 'raw_file', 'dxbc_path']:
        if row.get(key):
            path = Path(row[key])
            return path if path.is_absolute() else EXPORT / path
    raise KeyError(f'No known path key in {list(row)}')


def main():
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding='utf-8-sig'))
    report = dict(schema='d3d11_dxbc_replay/1', status='starting', capture=manifest.get('capture'), event=manifest.get('event_id'),
                  backend=f'D3D11 {"HARDWARE" if HARDWARE else "WARP"} FL11_0 inside python.exe; compiled DXBC as-is; no assembler or recompile', events=[],
                  scope='Single draw declared by the manifest; no full-pipeline or cross-GPU exactness implied.', input_manifest=str(INPUT_MANIFEST))
    objects, backing = [], []
    device, context = P(), P()

    def event(kind, **fields):
        report['events'].append(dict(kind=kind, **fields))
        save('REPLAY.json', report)

    def keep(obj):
        objects.append(obj)
        return obj

    def buffer(data, bind):
        storage = C.create_string_buffer(data)
        backing.append(storage)
        desc = BufferDesc(len(data), 1, bind, 0, 0, 0)
        initial, obj = Init(C.cast(storage, P), 0, 0), P()
        check(method(device, 3, HR, C.POINTER(BufferDesc), C.POINTER(Init), C.POINTER(P))(
            device, C.byref(desc), C.byref(initial), C.byref(obj)), 'CreateBuffer')
        return keep(obj)

    try:
        shaders, buffers = normalize_manifest(manifest)
        create = C.WinDLL('d3d11.dll').D3D11CreateDevice
        create.restype, create.argtypes = HR, [P, U, P, U, C.POINTER(U), U, U, C.POINTER(P), C.POINTER(U), C.POINTER(P)]
        level, feature = (U * 1)(0xb000), U()
        check(create(None, 1 if HARDWARE else 5, None, 0, level, 1, 7, C.byref(device), C.byref(feature), C.byref(context)), 'D3D11CreateDevice')
        event('device_created', driver_type=1 if HARDWARE else 5, feature_level=feature.value, adapter=adapter_info(device))
        shader_objects, bytecodes = {}, {}
        for stage, index, setter in [('vertex', 12, 11), ('geometry', 13, 23), ('pixel', 15, 9)]:
            data = path_of(shaders[stage]).read_bytes()
            assert data[:4] == b'DXBC', (stage, 'Not a DXBC container', len(data))
            bytecodes[stage] = data
            shader = P()
            check(method(device, index, HR, P, C.c_size_t, P, C.POINTER(P))(
                device, data, len(data), None, C.byref(shader)), 'Create ' + stage + ' shader')
            keep(shader)
            shader_objects[stage] = shader
            method(context, setter, None, P, P, U)(context, shader, None, 0)
            event('shader_bound', stage=stage, bytes=len(data), source=str(path_of(shaders[stage])))
        elements = (InputElement * 2)(InputElement(b'ATTRIBUTE', 0, 16, 0, 0, 0, 0), InputElement(b'ATTRIBUTE', 1, 16, 0, 8, 0, 0))
        layout = P()
        check(method(device, 11, HR, P, U, P, C.c_size_t, C.POINTER(P))(
            device, elements, 2, bytecodes['vertex'], len(bytecodes['vertex']), C.byref(layout)), 'Create input layout')
        keep(layout)
        method(context, 17, None, P)(context, layout)
        vbs = manifest['vertex_buffers']
        if isinstance(vbs, dict):
            vbs = list(vbs.values())
        vbrow = vbs[0]
        vb = buffer(path_of(vbrow).read_bytes(), 1)
        stride, offset = U(vbrow.get('stride', 16)), U(vbrow.get('offset', 0))
        method(context, 18, None, U, U, C.POINTER(P), C.POINTER(U), C.POINTER(U))(
            context, 0, 1, (P * 1)(vb), C.byref(stride), C.byref(offset))
        draw = manifest['draw']
        method(context, 24, None, U)(context, TOPOLOGIES[draw['topology']])
        for row in buffers:
            stage, slot = row['stage'], int(row.get('slot', 0))
            if stage not in ('vertex', 'pixel'):
                continue
            data = path_of(row).read_bytes()
            assert len(data) == int(row['byte_size']) and len(data) % 16 == 0
            # Constant-buffer patching is the controller's job: it hands this
            # worker already-effective bytes, so the worker never rewrites a CB
            # itself and can stay a faithful executor of whatever it is given.
            cb = buffer(data, 4)
            setter = 7 if stage == 'vertex' else 16
            method(context, setter, None, U, U, C.POINTER(P))(context, slot, 1, (P * 1)(cb))
            event('constant_buffer_bound', stage=stage, slot=slot, bytes=len(data), path=str(path_of(row)))

        # Exact texture bytes are required input: absent bytes are an
        # actionable input error rather than an invented placeholder texture.
        textures = manifest['textures']
        if isinstance(textures, dict):
            textures = list(textures.values())
        texture = next(r for r in textures if r.get('stage', 'pixel') == 'pixel' and int(r.get('slot', 0)) == 0)
        width, height = int(texture['width']), int(texture['height'])
        format_value = texture.get('srv', {}).get('viewFormat', texture.get('dxgi_format', texture.get('view_format', texture.get('format'))))
        formats = {'R8G8B8A8_UNORM': 28, 'R8G8B8A8_UNORM_SRGB': 29, 'B8G8R8A8_UNORM': 87,
                   'B8G8R8A8_UNORM_SRGB': 91, 'R16G16B16A16_FLOAT': 10, 'R32G32B32A32_FLOAT': 2,
                   'R10G10B10A2_UNORM': 24, 'BC1_UNORM': 71, 'BC1_UNORM_SRGB': 72,
                   'BC7_UNORM': 98, 'BC7_UNORM_SRGB': 99}
        if format_value == 'DXGI_FORMAT_R8G8B8A8_TYPELESS':
            # The saved scan omitted the typed SRV field. At this exact captured
            # 1x1 endpoint byte texture, SRGB and UNORM have identical values.
            endpoint_bytes = path_of(texture).read_bytes()
            assert set(endpoint_bytes) <= {0, 255}, 'Typed view format required for non-endpoint values'
            format_value = 28
        if isinstance(format_value, str):
            format_value = {'RGBA8_UNORM': 'R8G8B8A8_UNORM', 'RGBA8_UNORM_SRGB': 'R8G8B8A8_UNORM_SRGB'}.get(format_value, format_value)
            format_value = formats[format_value.removeprefix('DXGI_FORMAT_')]
        format_value = int(format_value)
        raw = path_of(texture).read_bytes()
        storage = C.create_string_buffer(raw)
        backing.append(storage)
        if format_value in (71, 72, 98, 99):
            pitch = max(1, (width + 3) // 4) * (8 if format_value in (71, 72) else 16)
            minimum = pitch * max(1, (height + 3) // 4)
        else:
            bpp = 16 if format_value == 2 else 8 if format_value == 10 else 4
            pitch, minimum = width * bpp, width * height * bpp
        assert len(raw) >= minimum, (len(raw), minimum)
        initial = Init(C.cast(storage, P), int(texture.get('row_pitch', pitch)), int(texture.get('slice_pitch', minimum)))
        td, tex = Tex2(width, height, 1, 1, format_value, Sample(1, 0), 1, 8, 0, 0), P()
        check(method(device, 5, HR, C.POINTER(Tex2), P, C.POINTER(P))(device, C.byref(td), C.byref(initial), C.byref(tex)), 'Create source texture')
        keep(tex)
        srv = P()
        check(method(device, 7, HR, P, P, C.POINTER(P))(device, tex, None, C.byref(srv)), 'Create texture SRV')
        keep(srv)
        method(context, 8, None, U, U, C.POINTER(P))(context, 0, 1, (P * 1)(srv))
        sd = Sampler(0x14, 3, 3, 3, 0, 1, 1, (F * 4)(0, 0, 0, 0), 0, 3.4028234663852886e38)
        sampler = P()
        check(method(device, 23, HR, C.POINTER(Sampler), C.POINTER(P))(device, C.byref(sd), C.byref(sampler)), 'Create sampler')
        keep(sampler)
        method(context, 10, None, U, U, C.POINTER(P))(context, 0, 1, (P * 1)(sampler))
        event('texture_dependency_bound', resource_id=str(texture.get('resource_id')), format=format_value, dimensions=[width, height], mip_scope='exported mip0', bytes=len(raw))

        target, rtv, staging = P(), P(), P()
        # The output texture is declared by the manifest (width/height/depth/
        # format); the worker creates exactly that and nothing else.
        out_spec = manifest['output']
        width, height, depth = int(out_spec['width']), int(out_spec['height']), int(out_spec['depth'])
        format_name = str(out_spec['format']).removeprefix('DXGI_FORMAT_')
        assert format_name in OUTPUT_FORMATS, ('Unsupported output format', format_name, sorted(OUTPUT_FORMATS))
        format_value, channel_bits = OUTPUT_FORMATS[format_name]
        assert width >= 1 and height >= 1 and depth >= 1
        td = Tex3(width, height, depth, 1, format_value, 0, 32, 0, 0)
        check(method(device, 6, HR, C.POINTER(Tex3), P, C.POINTER(P))(device, C.byref(td), None, C.byref(target)), 'Create output texture')
        keep(target)
        rd = RTVDesc(format_value, 8, 0, 0, depth)
        check(method(device, 9, HR, P, P, C.POINTER(P))(device, target, C.byref(rd), C.byref(rtv)), 'Create 3D render target view')
        keep(rtv)
        td = Tex3(width, height, depth, 1, format_value, 3, 0, 0x20000, 0)
        check(method(device, 6, HR, C.POINTER(Tex3), P, C.POINTER(P))(device, C.byref(td), None, C.byref(staging)), 'Create staging texture')
        keep(staging)
        raster, raster_desc = P(), Raster(3, 1, 1, 0, 0, 0, 1, 1, 1, 0)
        check(method(device, 22, HR, C.POINTER(Raster), C.POINTER(P))(device, C.byref(raster_desc), C.byref(raster)), 'Create raster state')
        keep(raster)
        method(context, 43, None, P)(context, raster)
        # Viewport/scissor come from the manifest when present; otherwise the
        # full-RT default below is used.
        vp = manifest.get('viewport', manifest.get('viewports', [{}])[0] if manifest.get('viewports') else dict(x=0, y=0, width=width, height=height, minDepth=0, maxDepth=1))
        viewport = Viewport(vp.get('x', 0), vp.get('y', 0), vp['width'], vp['height'], vp.get('minDepth', 0), vp.get('maxDepth', 1))
        scissor = manifest.get('scissors', [dict(x=0, y=0, width=width, height=height)])[0]
        rect = Rect(scissor['x'], scissor['y'], scissor['x']+scissor['width'], scissor['y']+scissor['height'])
        method(context, 44, None, U, C.POINTER(Viewport))(context, 1, C.byref(viewport))
        method(context, 45, None, U, C.POINTER(Rect))(context, 1, C.byref(rect))
        method(context, 33, None, U, C.POINTER(P), P)(context, 1, (P * 1)(rtv), None)
        method(context, 50, None, P, C.POINTER(F))(context, rtv, (F * 4)(0, 0, 0, 0))
        vertex_count, instance_count = int(draw['vertex_count']), int(draw['instance_count'])
        event('command_execution', command=f'ID3D11DeviceContext::DrawInstanced({vertex_count},{instance_count},{draw.get("start_vertex", 0)},{draw.get("start_instance", 0)})', topology=draw['topology'], viewport=vp, scissor=scissor,
              viewport_source='manifest' if 'viewport' in manifest or 'viewports' in manifest else 'full-RT default, not separately evidenced')
        method(context, 21, None, U, U, U, U)(context, vertex_count, instance_count, int(draw.get('start_vertex', 0)), int(draw.get('start_instance', 0)))
        method(context, 33, None, U, P, P)(context, 0, None, None)
        method(context, 47, None, P, P)(context, staging, target)
        mapped = Mapped()
        check(method(context, 14, HR, P, U, U, U, C.POINTER(Mapped))(context, staging, 0, 1, 0, C.byref(mapped)), 'Map staging texture')
        row_bytes = width * 4
        generated = b''.join(C.string_at(mapped.pData + z * mapped.DepthPitch + y * mapped.RowPitch, row_bytes) for z in range(depth) for y in range(height))
        row_pitch, depth_pitch = mapped.RowPitch, mapped.DepthPitch
        method(context, 15, None, P, U)(context, staging, 0)
        output = HERE / 'generated_output.bin'
        output.write_bytes(generated)
        reference_row = next(r for r in textures if r.get('role') == 'expected_writer_output')
        reference = path_of(reference_row).read_bytes()
        voxels = width * height * depth
        assert len(generated) == len(reference) == voxels * 4
        ga, ra = struct.unpack(f'<{voxels}I', generated), struct.unpack(f'<{voxels}I', reference)
        mismatch = [i for i, (g, r) in enumerate(zip(ga, ra)) if g != r]
        channels = []
        shift = 0
        for name, bits in zip('rgba', channel_bits):
            mask = (1 << bits) - 1
            gv, rv = [(v >> shift) & mask for v in ga], [(v >> shift) & mask for v in ra]
            delta = [abs(g - r) for g, r in zip(gv, rv)]
            greatest = max(delta)
            channels.append(dict(channel=name, bits=bits, compared=len(delta), max_abs_code_error=greatest,
                                 mean_abs_code_error=sum(delta) / len(delta), unequal_count=sum(d != 0 for d in delta),
                                 code_error_histogram={str(v): delta.count(v) for v in sorted(set(delta))},
                                 count_error_greater_than_one=sum(d > 1 for d in delta),
                                 largest_error_locations=[dict(index=i, xyz=[i%width,(i//width)%height,i//(width*height)], generated=gv[i], reference=rv[i]) for i,d in enumerate(delta) if d == greatest][:20],
                                 generated_range=[min(gv), max(gv)], reference_range=[min(rv), max(rv)],
                                 max_abs_normalized_error=greatest / mask))
            shift += bits
        report.update(status='draw_executed_and_full_output_compared', exact_reference_status='EXACT_REFERENCE_NOT_MATCHED' if mismatch else 'EXACT_REFERENCE_MATCHED', output=str(output), output_bytes=len(generated),
                      output_dimensions=[width, height, depth], output_format=format_name,
                      staging_pitch=dict(row=row_pitch, depth=depth_pitch), output_layout='packed z-major, then y, then x; each voxel uint32 little-endian',
                      comparison=dict(voxels=voxels, identical_packed_voxels=voxels-len(mismatch), unequal_packed_voxels=len(mismatch),
                                      raw_byte_unequal=sum(g != r for g, r in zip(generated, reference)), channels=channels,
                                      first_mismatches=[dict(index=i, xyz=[i%width,(i//width)%height,i//(width*height)], generated_uint=ga[i], reference_uint=ra[i]) for i in mismatch[:20]]),
                      statement='Observed native D3D11 draw and complete output comparison; no general GPU bitwise or full-pipeline equivalence implied.')
        save('REPLAY.json', report)
        print(json.dumps(dict(status=report['status'], unequal_voxels=len(mismatch), max_channel_error=[c['max_abs_code_error'] for c in channels[:3]])))
        return 0
    except Exception:
        report.update(status='execution_error', error=traceback.format_exc())
        save('REPLAY.json', report)
        print(report['error'], file=sys.stderr)
        return 1
    finally:
        for obj in reversed(objects):
            if obj:
                method(obj, 2, U)(obj)
        for obj in [context, device]:
            if obj:
                method(obj, 2, U)(obj)


if __name__ == '__main__':
    raise SystemExit(main())
