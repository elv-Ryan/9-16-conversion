from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .reviewed_core import FeatureState, OneEuro, NAMES, source_frame_x

class ReviewedRound01Runtime:
    """Frozen reviewed Round01 controller + One Euro 0.4/1.0."""
    def __init__(self,manifest_path:str,width:int,height:int):
        self.cfg=json.loads(Path(manifest_path).read_text())
        if self.cfg.get('selected',{}).get('kind')!='filter_only':raise ValueError('runtime manifest must select filter_only')
        self.features=FeatureState(self.cfg['controller_params'],width,height)
        f=self.cfg['filter_config']
        if f.get('kind')!='one_euro':raise ValueError('reviewed handoff requires one_euro')
        self.euro=OneEuro(f['min_cutoff'],f['beta'])
    def reset(self):self.features.reset();self.euro.reset()
    def update(self,candidates,timestamp_seconds,shot_start=False):
        if shot_start:self.reset()
        feature,family,event=self.features.step(candidates,float(timestamp_seconds))
        if event==2:self.euro.reset()
        # feature[-4] is controller_x in the reviewed feature contract.
        x=self.euro.step(float(timestamp_seconds),float(feature[-4]))
        x=max(self.features.half,min(1-self.features.half,float(x)))
        return x,int(family),int(event)

def reconstruct_source_frames(sample_times,sample_x,sample_events,start_frame,end_frame,fps_num,fps_den):
    t=np.asarray(sample_times,float);x=np.asarray(sample_x,float)
    if len(t)==1:return np.full(end_frame-start_frame,x[0],np.float32)
    reset=np.asarray([True]+[e==2 for e in sample_events[1:]],bool)
    frames=np.arange(start_frame,end_frame,dtype=np.int64)
    return source_frame_x(t,x,frames,int(fps_num),int(fps_den),reset)
