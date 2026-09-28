#!/usr/bin/env python3
"""Convert strict Eluvio common-ml output to the exact supplied recovered JSONL shape.
This is an offline renderer-interchange export, not the Tagger wire protocol.
"""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('-o','--output',required=True);p.add_argument('--source-iq',required=True);p.add_argument('--title',default='');p.add_argument('--recovered-from',default='nba_reviewed_round01_oneeuro_v1');a=p.parse_args()
rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
tags=[r for r in rows if r.get('type')=='tag' and r.get('data',{}).get('track')=='vertical_video']
if not tags:raise SystemExit('no vertical_video tags')
frame_count=sum(len(t['data']['additional_info']['x-coordinates']) for t in tags);fps=tags[0]['data']['additional_info']['fps']
out=[{'type':'progress','data':{'source_iq':a.source_iq,'title':a.title,'frame_count':frame_count,'fps':fps,'recovered_from':a.recovered_from}}]
for r in tags:
    d=dict(r['data']);d.pop('source_media',None)
    d['additional_info']=dict(d['additional_info']);d['additional_info']['source_iq']=a.source_iq
    out.append({'type':'tag','data':d})
Path(a.output).write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in out)+'\n')
