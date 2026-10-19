"""Dirty rebuild controller for pre-compiled D3D11 draws.

Replays a manifest-described D3D11 draw on real hardware (or WARP), with optional
typed constant-buffer patches, and caches artifacts by the SHA-256 of the
*effective* inputs rather than the requested labels. Rebuilds are
offline/on-demand: this is a verification tool, never a per-frame render
producer. Import is inert, needs no engine modules, and all output paths belong
to the caller.
"""
import argparse
import copy
import ctypes
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
PATH_KEYS = ('raw_path','path','file','output_file','raw_file','dxbc_path')
# Output texture formats the replay worker can create; must stay in sync with
# OUTPUT_FORMATS in replay/worker_replay.py.
SUPPORTED_OUTPUT_FORMATS = {'R8G8B8A8_UNORM', 'R10G10B10A2_UNORM'}

def sha(data): return hashlib.sha256(data).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)
def stamp(): return datetime.datetime.now().astimezone().isoformat()
def rows(value): return list(value.values()) if isinstance(value,dict) else list(value)
def source_path(row, root):
    for key in PATH_KEYS:
        if row.get(key):
            p=Path(row[key]);return p if p.is_absolute() else root/p
    raise ValueError('Input row has no byte-file path')
def check_power():
    if sys.platform!='win32':return
    class Power(ctypes.Structure):
        _fields_=[('ac',ctypes.c_ubyte),('flag',ctypes.c_ubyte),('percent',ctypes.c_ubyte),
                  ('reserved',ctypes.c_ubyte),('life',ctypes.c_ulong),('full',ctypes.c_ulong)]
    value=Power()
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)) and value.percent!=255 and value.percent<=5:
        raise RuntimeError('Battery <=5%; pause requested by user')

def validate_profile(manifest):
    target=next(r for r in rows(manifest['textures']) if r.get('role')=='expected_writer_output')
    out=manifest.get('output')
    assert out, 'Manifest must declare the output texture (width/height/depth/format)'
    assert [int(out[k]) for k in ('width','height','depth')]==[int(target[k]) for k in ('width','height','depth')], 'Output block and expected row disagree on dimensions'
    fmt=str(out['format']).removeprefix('DXGI_FORMAT_')
    assert fmt in SUPPORTED_OUTPUT_FORMATS, ('Unsupported output format', fmt, sorted(SUPPORTED_OUTPUT_FORMATS))
    assert str(target['format']).removeprefix('DXGI_FORMAT_')==fmt, 'Expected row format mismatch'
    draw=manifest['draw']
    assert draw['api']=='D3D11' and draw['topology'] in ('TriangleStrip','TriangleList','LineStrip','LineList','PointList')
    assert int(draw['vertex_count'])>=1 and int(draw['instance_count'])>=1
    assert int(draw.get('start_vertex',0))>=0 and int(draw.get('start_instance',0))>=0
    assert {r['stage'] for r in rows(manifest['shaders'])}=={'vertex','geometry','pixel'}
    assert all(r['stage'] in ('vertex','pixel') for r in rows(manifest['constant_buffers'])), 'Unsupported CB stage'
    layout=manifest['input_layout'];assert len(layout)==2
    assert [int(r['offset']) for r in layout]==[0,8]
    assert all(r['format']=='RG32_FLOAT' and r['slot']==0 and not r.get('perInstance') for r in layout)
    vb=rows(manifest['vertex_buffers']);assert len(vb)==1 and vb[0]['stride']==16 and vb[0].get('offset',0)==0
    raster=manifest.get('rasterizer',{})
    assert raster.get('fillMode')=='Solid' and raster.get('cullMode')=='NoCull'
    assert raster.get('frontCCW') and raster.get('depthClip') and raster.get('scissorEnable') and raster.get('multisampleEnable')
    assert not raster.get('antialiasedLineEnable') and all(raster.get(k,0)==0 for k in ('depthBias','depthBiasClamp','slopeScaledDepthBias'))
    assert not manifest.get('depth_stencil',{}).get('depthEnable') and not manifest.get('depth_stencil',{}).get('stencilEnable')
    assert not any(r.get('enabled') for r in manifest.get('blend_state',{}).get('targets',[]))
    if manifest.get('viewports'):
        assert sum(bool(r.get('enabled')) for r in manifest['viewports'])==1
    samplers=manifest.get('samplers',[])
    if samplers:
        s=samplers[0]
        assert all(s.get(k) in ('Clamp','ClampEdge') for k in ('addressU','addressV','addressW')), 'Unsupported sampler addressing'
        assert [s.get(k) for k in ('minFilter','magFilter','mipFilter')]==['Linear','Linear','Point'], 'Unsupported sampler filtering'
    return target

def prepare(manifest_path, overrides, backend):
    path=Path(manifest_path).resolve();manifest=load(path);validate_profile(manifest)
    patched=copy.deepcopy(manifest);files=[];ident=[];cb_data={};changes=[]
    for section in ('shaders','constant_buffers','vertex_buffers','textures'):
        original=rows(manifest[section]);new=rows(patched[section]);patched[section]=new
        for index,(src,dst) in enumerate(zip(original,new)):
            p=source_path(src,path.parent);data=p.read_bytes()
            if 'byte_size' in src:assert len(data)==int(src['byte_size']),(p,len(data),src['byte_size'])
            if section=='constant_buffers':cb_data[(src['stage'],int(src['slot']))]=(bytearray(data),src,dst)
            files.append((section,index,p,data,dst))
    occupied=set()
    for change in overrides:
        stage=change['stage'];slot=int(change['slot']);offset=int(change['byte_offset'])
        fmt={'float32':'f','uint32':'I','int32':'i'}[change.get('type','float32')]
        data,src,dst=cb_data[(stage,slot)]
        assert offset>=0 and offset%4==0 and offset+4<=len(data),'CB override outside 4-byte aligned range'
        tag=(stage,slot,offset);assert tag not in occupied,'Overlapping CB overrides';occupied.add(tag)
        before=struct.unpack_from('<'+fmt,data,offset)[0]
        value=before+change['delta'] if 'delta' in change else change['value']
        if fmt=='f':assert math.isfinite(float(value))
        struct.pack_into('<'+fmt,data,offset,value)
        after=struct.unpack_from('<'+fmt,data,offset)[0]
        changes.append(dict(stage=stage,slot=slot,byte_offset=offset,type=change.get('type','float32'),
            symbol=change.get('symbol'),before=before,after=after,source=str(source_path(src,path.parent))))
    for section,index,p,data,dst in files:
        if section=='constant_buffers':data=bytes(cb_data[(dst['stage'],int(dst['slot']))][0])
        for k in PATH_KEYS:dst.pop(k,None)
        dst['path']=str(p);dst['raw_path']=str(p);dst['byte_size']=len(data);dst['sha256']=sha(data)
        # Cache identity uses effective input bytes, not requested delta/labels.
        semantics={k:dst[k] for k in ('stage','slot','role','resource_id','width','height','depth','format','srv','stride','offset','row_pitch','slice_pitch') if k in dst}
        ident.append(dict(section=section,index=index,bytes=len(data),sha256=sha(data),semantics=semantics))
    identity=dict(schema='effective_input_identity/1',capture=manifest.get('capture'),event_id=manifest.get('event_id'),
        backend=backend,worker_sha256=sha((HERE/'worker_replay.py').read_bytes()),inputs=ident,
        pipeline={k:manifest.get(k) for k in ('draw','input_layout','viewport','viewports','scissors','samplers','rasterizer')})
    key=sha(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    return key,patched,files,cb_data,changes,identity

def rebuild(manifest_path, output_root, overrides=(), backend='hardware', force=False, timeout_seconds=45):
    root=Path(output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    state_path=root/'STATE.json';state=load(state_path) if state_path.is_file() else {'schema':'dirty_rebuild_controller/1','last_success':None}
    previous=copy.deepcopy(state.get('last_success'));started=stamp();began=time.monotonic()
    attempt=root/'attempts'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);attempt.mkdir(parents=True)
    try:
        check_power()
        key,manifest,files,cb_data,changes,identity=prepare(manifest_path,overrides,backend)
        entry=root/'entries'/key;receipt=entry/'SUCCESS.json'
        if receipt.is_file() and not force:
            report=load(receipt)
            for field in ('output',):
                file=Path(report[field]['path']);assert file.is_file() and sha(file.read_bytes())==report[field]['sha256'],'Cached artifact changed'
            report=copy.deepcopy(report);report.update(status='cache_hit',draw_executed_this_call=False,started=started,ended=stamp())
        else:
            entry.mkdir(parents=True,exist_ok=True);inputs=attempt/'inputs';inputs.mkdir()
            for section,index,p,data,dst in files:
                if section=='constant_buffers':
                    data=bytes(cb_data[(dst['stage'],int(dst['slot']))][0]);copy_path=inputs/(dst['stage']+'_cb'+str(dst['slot'])+'.bin');copy_path.write_bytes(data)
                    dst['path']=dst['raw_path']=str(copy_path)
            input_path=inputs/'INPUT_MANIFEST.json';dump(input_path,manifest);dump(attempt/'IDENTITY.json',identity)
            worker_dir=attempt/'writer';command=[sys.executable,'-B',str(HERE/'worker_replay.py'),'--manifest',str(input_path),'--output-dir',str(worker_dir),'--backend',backend]
            process=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=timeout_seconds)
            execution=dict(command=command,started=started,ended=stamp(),stdout=process.stdout,stderr=process.stderr,exit_code=process.returncode)
            dump(attempt/'COMMAND_EXECUTION.json',execution)
            assert process.returncode==0,'Writer process failed: '+process.stderr[-3000:]
            replay=load(worker_dir/'REPLAY.json');assert replay['status']=='draw_executed_and_full_output_compared'
            out=manifest['output']
            voxels=int(out['width'])*int(out['height'])*int(out['depth'])
            raw=Path(replay['output']).read_bytes();assert len(raw)==voxels*4 and replay['output_bytes']==len(raw)
            # Validate in the attempt before publishing/reusing immutable cache files.
            out_path=entry/'output.bin'
            if out_path.exists():assert out_path.read_bytes()==raw,'Same input identity produced different output; preserve prior success and choose a new backend/cache context'
            else:out_path.write_bytes(raw)
            report=dict(schema='dxbc_replay_result/1',status='rebuilt',identity=key,capture=manifest.get('capture'),event_id=manifest.get('event_id'),
                backend=backend,draw_executed_this_call=True,started=started,ended=stamp(),elapsed_seconds=time.monotonic()-began,
                changes=changes,output=dict(path=str(out_path),bytes=len(raw),sha256=sha(raw),format=replay['output_format'],dimensions=replay['output_dimensions']),
                reference_status=replay['exact_reference_status'],reference_comparison=replay['comparison'],
                execution=str(attempt/'COMMAND_EXECUTION.json'),replay=str(worker_dir/'REPLAY.json'),
                per_frame_engine_production=False, engine_bound=False)
            dump(receipt,report)
        # Publish only after draw/readback checks succeeded. Failure keeps previous.
        state.update(last_success=report,last_attempt=dict(status=report['status'],identity=key,started=started,ended=stamp()))
        dump(state_path,state);dump(attempt/'RESULT.json',report)
        return report
    except Exception:
        error=traceback.format_exc();report=dict(schema='dxbc_replay_result/1',status='failed_retaining_last_success',
            started=started,ended=stamp(),error=error,last_success=previous,draw_executed_this_call='see COMMAND_EXECUTION when present')
        state.update(last_success=previous,last_attempt=report);dump(state_path,state);dump(attempt/'RESULT.json',report)
        return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--output-root',required=True)
    parser.add_argument('--overrides',help='JSON list: stage,slot,byte_offset,type,value or delta,symbol')
    parser.add_argument('--backend',choices=('hardware','warp'),default='hardware');parser.add_argument('--force',action='store_true')
    args=parser.parse_args();overrides=load(args.overrides) if args.overrides else []
    result=rebuild(args.manifest,args.output_root,overrides,args.backend,args.force)
    print(json.dumps(result,ensure_ascii=False));return 1 if result['status'].startswith('failed') else 0

if __name__=='__main__':raise SystemExit(main())
