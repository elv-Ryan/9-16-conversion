#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, fcntl, json, os, re, subprocess, sys, time, traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

VIDEO_SUFFIXES={'.mp4','.mov','.m4v','.mkv','.avi','.webm'}

def utcnow(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def atomic_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2)+'\n')
    tmp.replace(path)

def ordinal(path: Path) -> int:
    m=re.match(r'^(\d+)(?:[-_]|$)',path.stem)
    if not m: raise ValueError(f'cannot parse shot ordinal: {path.name}')
    return int(m.group(1))

def duration_ms(path: Path) -> int:
    m=re.match(r'^\d+-(\d+)_(\d+)$',path.stem)
    if not m: return 1
    return max(1,int(m.group(2))-int(m.group(1)))

def discover(root: Path, expected: int) -> List[Path]:
    vids=[p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and not p.name.startswith('.')]
    by={ordinal(p):p for p in vids}
    missing=[i for i in range(expected) if i not in by]
    if len(vids)!=expected or missing:
        raise ValueError(f'shot corpus invalid files={len(vids)} missing={missing[:20]}')
    return [by[i] for i in range(expected)]

def gpu_rows() -> List[Dict]:
    inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    try:
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    except subprocess.CalledProcessError:
        procs=''
    pmap={}
    for line in procs.splitlines():
        ps=[x.strip() for x in line.split(',',3)]
        if len(ps)==4:
            pmap.setdefault(ps[0],[]).append({'pid':int(ps[1]),'name':ps[2],'memory_mib':int(ps[3].split()[0])})
    out=[]
    for line in inv.splitlines():
        ps=[x.strip() for x in line.split(',',4)]
        if len(ps)!=5: continue
        idx,uuid,used,free,util=ps
        out.append({'index':int(idx),'uuid':uuid,'memory_used_mib':int(used),'memory_free_mib':int(free),'utilization_percent':int(util),'compute_processes':pmap.get(uuid,[])})
    return out

def is_free(row: Dict) -> bool:
    return (not row['compute_processes'] and row['memory_used_mib']<=1024 and row['memory_free_mib']>=20000 and row['utilization_percent']<=10)

def params(batch_size:int,inference_fps:float)->str:
    return json.dumps({
        'device':'0','input_mode':'shot_file','imgsz':1280,'inference_fps':inference_fps,
        'batch_size':batch_size,'use_fp16':True,'continue_on_error':True,
        'emit_progress_ratio':False,'emit_focus_track':False,'include_focus_samples':True,
    },separators=(',',':'),sort_keys=True)

def count_terminals(path:Path)->Tuple[int,int,int]:
    if not path.exists(): return 0,0,0
    p=e=t=0
    try:
        for raw in path.open():
            if not raw.strip(): continue
            row=json.loads(raw); typ=row.get('type'); data=row.get('data') or {}
            if typ=='progress': p+=1
            elif typ=='error': e+=1
            elif typ=='tag' and data.get('track')=='vertical_video': t+=1
    except Exception: pass
    return p,e,t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--shot-dir',required=True,type=Path)
    ap.add_argument('--image',default='nba-yolo-shot-tagger:joe518-v3.1')
    ap.add_argument('--output-root',required=True,type=Path)
    ap.add_argument('--expected-count',type=int,default=518)
    ap.add_argument('--gpu-candidates',default='2,3,4,5')
    ap.add_argument('--min-gpus',type=int,default=2)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--inference-fps',type=float,default=10.0)
    args=ap.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True)
    status_path=args.output_root/'STATUS.json'
    def state(stage,**kw):
        payload={'stage':stage,'updated_utc':utcnow(),'output_root':str(args.output_root),'image':args.image,**kw}
        atomic_json(status_path,payload)
    locks=[]
    try:
        shots=discover(args.shot_dir,args.expected_count)
        candidates=[int(x) for x in args.gpu_candidates.split(',') if x.strip()]
        rows=gpu_rows(); by={r['index']:r for r in rows}
        free=[g for g in candidates if g in by and is_free(by[g])]
        state('GPU_PREFLIGHT',gpu_inventory=rows,candidates=candidates,selected_free=free)
        selected=[]
        for gpu in free:
            fh=Path(f'/tmp/nba-yolo-shot-tagger-gpu-{gpu}.lock').open('w')
            try: fcntl.flock(fh.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close(); continue
            # Recheck after lock.
            row={r['index']:r for r in gpu_rows()}.get(gpu)
            if not row or not is_free(row):
                fcntl.flock(fh.fileno(),fcntl.LOCK_UN); fh.close(); continue
            locks.append((gpu,fh)); selected.append(gpu)
        if len(selected)<args.min_gpus:
            raise RuntimeError(f'need at least {args.min_gpus} free GPUs; selected={selected}')

        # Greedy duration-balanced assignment.
        bins={g:[] for g in selected}; loads={g:0 for g in selected}
        for shot in sorted(shots,key=duration_ms,reverse=True):
            gpu=min(selected,key=lambda g:loads[g])
            bins[gpu].append(shot); loads[gpu]+=duration_ms(shot)
        for g in selected: bins[g].sort(key=ordinal)
        state('LAUNCHING',selected_gpus=selected,estimated_source_ms_by_gpu=loads)

        processes={}; started=time.monotonic()
        for gpu in selected:
            shard=args.output_root/f'gpu_{gpu}'
            shard.mkdir(parents=True,exist_ok=True); os.chmod(shard,0o777)
            inputs=shard/'expected_inputs.txt'
            inputs.write_text(''.join(f'/elv/input/{p.name}\n' for p in bins[gpu]))
            log=(shard/'container.log').open('wb',buffering=0)
            stdin=inputs.open('rb')
            cmd=['podman','run','--rm','-i','--device',f'nvidia.com/gpu={gpu}',
                 '-v',f'{args.shot_dir}:/elv/input:ro','-v',f'{shard}:/elv/output',args.image,
                 '--output-path','/elv/output/out.jsonl','--params',params(args.batch_size,args.inference_fps)]
            proc=subprocess.Popen(cmd,stdin=stdin,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[gpu]={'proc':proc,'stdin':stdin,'log':log,'cmd':cmd,'count':len(bins[gpu]),'shard':shard}
        while any(v['proc'].poll() is None for v in processes.values()):
            progress={}
            for gpu,v in processes.items():
                p,e,t=count_terminals(v['shard']/'out.jsonl')
                progress[str(gpu)]={'inputs':v['count'],'complete':p+e,'progress':p,'errors':e,'vertical_tags':t,'running':v['proc'].poll() is None}
            state('RUNNING_PARALLEL',selected_gpus=selected,elapsed_seconds=time.monotonic()-started,progress=progress)
            time.sleep(10)
        wall=time.monotonic()-started
        for v in processes.values(): v['stdin'].close(); v['log'].close()
        bad={g:v['proc'].returncode for g,v in processes.items() if v['proc'].returncode!=0}
        if bad: raise RuntimeError(f'container shard failures: {bad}')

        state('VALIDATING_SHARDS',selected_gpus=selected,wall_seconds=wall)
        summaries=[]
        for gpu,v in processes.items():
            shard=v['shard']
            cp=subprocess.run([sys.executable,str(args.repo/'scripts/validate_shot_directory_jsonl.py'),str(shard/'out.jsonl'),
                               '--expected-inputs',str(shard/'expected_inputs.txt'),'--output-dir',str(shard),
                               '--expected-count',str(v['count']),'--wall-seconds',str(wall)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (shard/'validator.log').write_text(cp.stdout or '')
            if cp.returncode!=0: raise RuntimeError(f'GPU {gpu} validation failed: {(cp.stdout or "")[-4000:]}')
            summaries.append((gpu,json.loads((shard/'summary.json').read_text())))

        family=Counter(); category=Counter(); source_seconds=0.0; success=errors=missing=0
        shots_out=[]; per_rows=[]; error_payload={}
        for gpu,s in summaries:
            family.update(s.get('family_counts') or {}); category.update(s.get('category_counts') or {})
            source_seconds+=float(s.get('successful_source_seconds') or 0); success+=int(s.get('successful_shots') or 0)
            errors+=int(s.get('error_shots') or 0); missing+=int(s.get('missing_or_invalid_shots') or 0)
            shard=args.output_root/f'gpu_{gpu}'
            x=json.loads((shard/'x-output.json').read_text()); shots_out.extend(x.get('shots') or [])
            with (shard/'per_shot_summary.csv').open(newline='') as h: per_rows.extend(csv.DictReader(h))
            ep=json.loads((shard/'errors.json').read_text()); error_payload.update(ep)
        def source_ord(src:str)->int:
            m=re.match(r'^(\d+)',Path(src).name); return int(m.group(1)) if m else 10**9
        shots_out.sort(key=lambda x:source_ord(str(x.get('source_media','')))); per_rows.sort(key=lambda x:source_ord(x['source_media']))
        (args.output_root/'x-output.json').write_text(json.dumps({'schema_version':'eluvio.nba-yolo-x-json.v1','shots':shots_out},indent=2)+'\n')
        if per_rows:
            with (args.output_root/'per_shot_summary.csv').open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=list(per_rows[0])); w.writeheader(); w.writerows(per_rows)
        (args.output_root/'errors.json').write_text(json.dumps(error_payload,indent=2)+'\n')
        strict=(success==args.expected_count and errors==0 and missing==0)
        result={'status':'PASS' if strict else 'FAIL','mode':'parallel_corpus_qualification_not_single_container_latency',
                'selected_gpus':selected,'expected_shots':args.expected_count,'successful_shots':success,'error_shots':errors,
                'missing_or_invalid_shots':missing,'wall_seconds':wall,'successful_source_seconds':source_seconds,
                'aggregate_video_seconds_per_wall_second':source_seconds/wall if wall>0 else None,
                'family_counts':dict(sorted(family.items())),'category_counts':dict(sorted(category.items())),
                'shards':{str(g):s for g,s in summaries}}
        atomic_json(args.output_root/'PARALLEL_RESULT.json',result)
        state('COMPLETE',result=result)
        print(json.dumps(result,indent=2))
        if not strict: raise SystemExit(1)
    except Exception as exc:
        state('FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        for gpu,fh in locks:
            try: fcntl.flock(fh.fileno(),fcntl.LOCK_UN)
            except Exception: pass
            fh.close()

if __name__=='__main__': main()
