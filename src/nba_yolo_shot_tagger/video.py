from __future__ import annotations
from fractions import Fraction
from pathlib import Path
from typing import Iterator, List, Tuple
import cv2
import numpy as np
from .types import ShotInterval, VideoInfo

class VideoSource:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"input media does not exist: {self.path}")
        cap=cv2.VideoCapture(self.path)
        if not cap.isOpened(): raise ValueError(f"OpenCV could not open input media: {self.path}")
        fps=float(cap.get(cv2.CAP_PROP_FPS)); frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        if not np.isfinite(fps) or fps<=0 or frame_count<=0 or width<=0 or height<=0:
            raise ValueError(f"invalid video metadata: fps={fps} frames={frame_count} size={width}x{height}")
        frac=Fraction(fps).limit_denominator(100000)
        fps_num,fps_den=frac.numerator,frac.denominator
        duration_ms=int(round(frame_count*1000.0*fps_den/fps_num))
        self.info=VideoInfo(fps=fps,fps_num=fps_num,fps_den=fps_den,fps_text=f"{fps_num}/{fps_den}",frame_count=frame_count,width=width,height=height,duration_ms=duration_ms)

    def validate_interval(self, interval: ShotInterval, max_seconds: float) -> tuple[int,int]:
        if interval.start_ms<0 or interval.end_ms<=interval.start_ms: raise ValueError("invalid shot interval")
        if interval.end_ms>self.info.duration_ms+100: raise ValueError("shot exceeds media duration")
        if (interval.end_ms-interval.start_ms)/1000.0>max_seconds: raise ValueError("shot exceeds max_shot_seconds")
        s=max(0,int(round(interval.start_ms*self.info.fps_num/(1000*self.info.fps_den))))
        e=min(self.info.frame_count,max(s+1,int(round(interval.end_ms*self.info.fps_num/(1000*self.info.fps_den)))))
        return s,e

    @staticmethod
    def sample_indices(start_frame:int,end_frame:int,source_fps:float,inference_fps:float)->List[int]:
        if end_frame<=start_frame:return []
        step=source_fps/min(source_fps,inference_fps)
        vals=np.arange(start_frame,end_frame,step,dtype=np.float64)
        idx=sorted({min(end_frame-1,max(start_frame,int(round(v)))) for v in vals})
        if not idx: idx=[start_frame]
        if idx[-1]!=end_frame-1: idx.append(end_frame-1)
        return idx

    def iter_sample_batches(self,*,start_frame:int,end_frame:int,inference_fps:float,batch_size:int)->Iterator[Tuple[List[int],List[np.ndarray]]]:
        targets=self.sample_indices(start_frame,end_frame,self.info.fps,inference_fps)
        cap=cv2.VideoCapture(self.path)
        if not cap.isOpened():raise ValueError(f"OpenCV could not reopen {self.path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame);current=start_frame;pos=0;inds=[];frames=[]
        try:
            while pos<len(targets):
                target=targets[pos]
                while current<target:
                    if not cap.grab():raise EOFError(f"unexpected EOF at {current}")
                    current+=1
                ok,frame=cap.read()
                if not ok or frame is None:raise EOFError(f"failed to decode frame {target}")
                current+=1;inds.append(target);frames.append(frame);pos+=1
                if len(frames)>=batch_size:
                    yield inds,frames;inds,frames=[],[]
            if frames:yield inds,frames
        finally:cap.release()
