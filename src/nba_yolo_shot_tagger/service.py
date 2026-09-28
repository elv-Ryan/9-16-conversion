from __future__ import annotations
from collections import Counter,defaultdict
import json,re
from pathlib import Path
from typing import Dict,List
from .config import RuntimeConfig
from .model import EXPECTED_FAMILIES,YoloStudentModel
from .reviewed_runtime import ReviewedRound01Runtime,reconstruct_source_frames
from .types import FocusSample,FrameEvidence,ShotAnalysis,ShotInterval
from .video import VideoSource

class ShotManifest:
    def __init__(self,path:str):self.payload=json.loads(Path(path).read_text())
    @staticmethod
    def _parse(items):
        out=[]
        for i,x in enumerate(items):out.append(ShotInterval(str(x.get('shot_id',f'shot_{i:06d}')),int(x['start_ms']),int(x['end_ms'])))
        return out
    def intervals_for(self,source):
        p=self.payload
        if isinstance(p,list):return self._parse(p)
        if 'shots' in p:return self._parse(p['shots'])
        for k in (source,str(Path(source).resolve()),Path(source).name):
            if k in p:return self._parse(p[k])
        raise ValueError(f'no shot manifest entry for {source}')

def _source_iq(path:str,override:str)->str:
    if override:return override
    m=re.search(r'iq__[A-Za-z0-9]+',path)
    return m.group(0) if m else ''

class ShotFocusService:
    def __init__(self,config:RuntimeConfig):
        self.config=config
        self.model=YoloStudentModel(model_path=config.model_path,manifest_path=config.model_manifest_path,verify_sha256=config.verify_model_sha256,device=config.device,imgsz=config.imgsz,min_confidence=config.min_detection_confidence,use_fp16=config.use_fp16,iou=config.iou,max_det=config.max_det,top_k=config.top_k)
        self.manifest=ShotManifest(config.shot_manifest_path) if config.input_mode=='shot_manifest' else None
    def intervals_for(self,video):
        return self.manifest.intervals_for(video.path) if self.manifest else [ShotInterval('shot_000000',0,video.info.duration_ms)]
    def analyze_file(self,source_media:str)->List[ShotAnalysis]:
        video=VideoSource(source_media);return [self._analyze_interval(video,x) for x in self.intervals_for(video)]
    def _analyze_interval(self,video,interval):
        start,end=video.validate_interval(interval,self.config.max_shot_seconds);evidence=[]
        for inds,frames in video.iter_sample_batches(start_frame=start,end_frame=end,inference_fps=self.config.inference_fps,batch_size=self.config.batch_size):
            evidence.extend(self.model.infer_batch(frame_indices=inds,frames=frames,source_fps=video.info.fps))
        if not evidence:raise ValueError(f'{interval.shot_id} produced no evidence')
        runtime=ReviewedRound01Runtime(self.config.runtime_manifest_path,video.info.width,video.info.height)
        xs=[];times=[];events=[];families=[];focus=[];votes=defaultdict(float)
        for i,frame in enumerate(evidence):
            candidates=[]
            for c in frame.candidates:
                cls=EXPECTED_FAMILIES.index(c.family);candidates.append({'class_id':cls,'family':c.family,'confidence':c.confidence,'bbox_xyxy_norm':list(c.bbox)})
                votes[c.family]+=max(.001,c.confidence)**1.5
            t=frame.frame_index*video.info.fps_den/video.info.fps_num
            x,fam,event=runtime.update(candidates,t,shot_start=(i==0));xs.append(x);times.append(t);events.append(event);families.append(fam)
            top=frame.candidates[0] if frame.candidates else None
            focus.append(FocusSample(frame.frame_index,frame.timestamp_ms,top.family if top else 'safe_center',top.confidence if top else 0.0,top.bbox if top else None,top.x_center if top else None))
        full=reconstruct_source_frames(times,xs,events,start,end,video.info.fps_num,video.info.fps_den)
        family=max(votes,key=votes.get) if votes else 'safe_center';total=sum(votes.values());fconf=votes.get(family,0)/total if total else 1.0
        crop_width=min(1.0,video.info.height*self.config.target_aspect_width_over_height/video.info.width)
        return ShotAnalysis(video.path,_source_iq(video.path,self.config.source_iq),interval.shot_id,interval.start_ms,interval.end_ms,start,end-start,video.info.fps,video.info.fps_num,video.info.fps_den,video.info.fps_text,video.info.width,video.info.height,family,fconf,crop_width,tuple(float(x) for x in full),tuple(focus),self.model.identity)
