#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('jsonl');p.add_argument('--expected-source');a=p.parse_args();path=Path(a.jsonl)
 rows=[]
 for n,line in enumerate(path.read_text().splitlines(),1):
  if not line.strip():continue
  try:r=json.loads(line)
  except json.JSONDecodeError as e:raise SystemExit(f'line {n}: invalid JSON: {e}')
  if r.get('type') not in {'tag','progress','error'} or not isinstance(r.get('data'),dict):raise SystemExit(f'line {n}: bad message')
  rows.append(r)
 tags=[r['data'] for r in rows if r['type']=='tag' and r['data'].get('track')=='vertical_video']
 if not tags:raise SystemExit('no vertical_video tags')
 expected_frame=None;source=None
 for d in tags:
  if a.expected_source and d.get('source_media')!=a.expected_source:raise SystemExit('source_media mismatch')
  if source is None:source=d.get('source_media')
  elif d.get('source_media')!=source:raise SystemExit('mixed source_media')
  fi=d.get('frame_info') or {};idx=fi.get('frame_idx')
  if not isinstance(idx,int):raise SystemExit('frame_info.frame_idx must be int')
  xs=(d.get('additional_info') or {}).get('x-coordinates')
  if not isinstance(xs,list) or not xs:raise SystemExit('missing x-coordinates')
  if expected_frame is not None and idx!=expected_frame:raise SystemExit(f'non-contiguous frame_idx: {idx} != {expected_frame}')
  expected_frame=idx+len(xs)
  for i,x in enumerate(xs):
   if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise SystemExit(f'bad x[{i}]')
   if not 0<=x<=1:raise SystemExit(f'x[{i}] outside normalized source width')
 progress=[r['data'] for r in rows if r['type']=='progress' and r['data'].get('source_media')==source]
 if len(progress)!=1:raise SystemExit(f'expected exactly one terminal progress for {source}, got {len(progress)}')
 last_index=max(i for i,r in enumerate(rows) if r['type']=='tag' and r['data'].get('source_media')==source)
 prog_index=max(i for i,r in enumerate(rows) if r['type']=='progress' and r['data'].get('source_media')==source)
 if prog_index<last_index:raise SystemExit('progress must be terminal after tags')
 errors=[r for r in rows if r['type']=='error']
 if errors:raise SystemExit(f'error messages present: {errors}')
 print(json.dumps({'status':'PASS','vertical_tags':len(tags),'frames':sum(len(d['additional_info']['x-coordinates']) for d in tags),'source_media':source},indent=2))
if __name__=='__main__':main()
