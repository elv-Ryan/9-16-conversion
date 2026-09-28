#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('path');a=p.parse_args();rows=[json.loads(x) for x in Path(a.path).read_text().splitlines() if x.strip()]
if not rows or rows[0].get('type')!='progress':raise SystemExit('first row must be progress')
tags=rows[1:]
if not tags or any(x.get('type')!='tag' for x in tags):raise SystemExit('remaining rows must be tags')
expected=0
for i,r in enumerate(tags):
 d=r['data'];info=d['additional_info'];xs=info['x-coordinates']
 if d.get('track')!='vertical_video':raise SystemExit('wrong track')
 if set(info)!={'x-coordinates','source_iq','fps','provenance'}:raise SystemExit(f'wrong additional_info keys: {set(info)}')
 if set(d)!={'tag','start_time','end_time','track','frame_info','additional_info'}:raise SystemExit(f'wrong tag keys: {set(d)}')
 if d['frame_info']!={'frame_idx':expected}:raise SystemExit('non-contiguous frame_idx')
 if not xs or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in xs):raise SystemExit('bad x')
 expected+=len(xs)
if expected!=rows[0]['data']['frame_count']:raise SystemExit('frame_count mismatch')
print(json.dumps({'status':'PASS','frames':expected,'tags':len(tags)}))
