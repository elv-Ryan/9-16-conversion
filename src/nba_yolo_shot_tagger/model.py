from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
import numpy as np
from .types import Candidate, FrameEvidence, ModelIdentity

EXPECTED_FAMILIES=("active_speaker","gameplay_follow","graphic_text_lock","person_subject","safe_center","split_screen","static_composition")

class YoloStudentModel:
    def __init__(self,*,model_path:str,manifest_path:str,verify_sha256:bool,device:str,imgsz:int,min_confidence:float,use_fp16:bool,iou:float,max_det:int,top_k:int):
        self.model_path=Path(model_path);self.manifest_path=Path(manifest_path);self.verify_sha256=verify_sha256
        self.device=str(device);self.imgsz=int(imgsz);self.min_confidence=float(min_confidence);self.use_fp16=bool(use_fp16)
        self.iou=float(iou);self.max_det=int(max_det);self.top_k=int(top_k);self._model=None
        self._manifest=json.loads(self.manifest_path.read_text());self._sha256=self._hash_file(self.model_path)
        if self._manifest.get("class_names")!=list(EXPECTED_FAMILIES):raise ValueError("seven-family manifest mismatch")
        if verify_sha256 and self._manifest.get("sha256")!=self._sha256:raise ValueError("model sha256 mismatch")
    @staticmethod
    def _hash_file(path:Path)->str:
        if not path.is_file():raise FileNotFoundError(path)
        h=sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        return h.hexdigest()
    @property
    def identity(self):return ModelIdentity(str(self.model_path),self._sha256,str(self._manifest.get("artifact_version","unknown")),tuple(EXPECTED_FAMILIES))
    def _ensure_loaded(self):
        if self._model is not None:return
        from ultralytics import YOLO
        m=YOLO(str(self.model_path));names=m.names
        names=[str(names[i]) for i in sorted(names)] if isinstance(names,Mapping) else [str(x) for x in names]
        if names!=list(EXPECTED_FAMILIES):raise ValueError(f"checkpoint class names mismatch: {names}")
        self._model=m
    @staticmethod
    def _np(v):
        if v is None:return np.empty((0,),np.float32)
        if hasattr(v,'detach'):v=v.detach()
        if hasattr(v,'cpu'):v=v.cpu()
        if hasattr(v,'numpy'):v=v.numpy()
        return np.asarray(v)
    def _predict(self,frames:Sequence[np.ndarray]):
        self._ensure_loaded();common=dict(source=list(frames),imgsz=self.imgsz,conf=self.min_confidence,iou=self.iou,max_det=self.max_det,device=self.device,verbose=False,save=False,stream=False)
        try:return self._model.predict(half=self.use_fp16,**common)
        except (TypeError,ValueError) as e:
            if 'half' not in str(e).lower() and 'quantize' not in str(e).lower():raise
            return self._model.predict(quantize=16 if self.use_fp16 and self.device!='cpu' else 32,**common)
    def infer_batch(self,*,frame_indices:Sequence[int],frames:Sequence[np.ndarray],source_fps:float)->List[FrameEvidence]:
        if len(frame_indices)!=len(frames):raise ValueError("frame_indices/frames mismatch")
        results=self._predict(frames);out=[]
        for fi,r in zip(frame_indices,results):
            cs=[];boxes=getattr(r,'boxes',None)
            if boxes is not None and len(boxes):
                xy=self._np(getattr(boxes,'xyxyn',None));cl=self._np(getattr(boxes,'cls',None)).reshape(-1);cf=self._np(getattr(boxes,'conf',None)).reshape(-1)
                for b,k,c in zip(xy,cl,cf):
                    k=int(k)
                    if not 0<=k<7:continue
                    x1,y1,x2,y2=[float(x) for x in b];bb=(max(0,min(1,x1)),max(0,min(1,y1)),max(0,min(1,x2)),max(0,min(1,y2)))
                    if bb[2]<=bb[0] or bb[3]<=bb[1]:continue
                    cs.append(Candidate(EXPECTED_FAMILIES[k],float(c),bb))
            cs.sort(key=lambda x:x.confidence,reverse=True);cs=cs[:self.top_k]
            out.append(FrameEvidence(int(fi),int(round(1000.0*fi/source_fps)),tuple(cs)))
        return out
