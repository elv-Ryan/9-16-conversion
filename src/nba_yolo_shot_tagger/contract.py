from __future__ import annotations
import math
from typing import List
from common_ml.tagging.messages import Tag
from .config import RuntimeConfig
from .types import ShotAnalysis

def _r(x,d):return round(float(x),d)
def validate_analysis(a:ShotAnalysis):
    if a.frame_count<=0 or len(a.x_coordinates)!=a.frame_count:raise ValueError('one X is required per source frame')
    lo=.5*a.crop_width_norm;hi=1-lo
    for x in a.x_coordinates:
        if not isinstance(x,(int,float)) or not math.isfinite(float(x)) or x<lo-1e-9 or x>hi+1e-9:raise ValueError(f'illegal x-coordinate {x}')

def tags_for_analysis(a:ShotAnalysis,c:RuntimeConfig)->List[Tag]:
    """Attached-JSON compatible X chunks, plus source_media required by Eluvio."""
    validate_analysis(a);out=[];n=c.chunk_frames;d=c.coordinate_decimals
    for off in range(0,a.frame_count,n):
        vals=a.x_coordinates[off:off+n];frame=a.start_frame+off
        start=int(round(frame*1000*a.fps_den/a.fps_num));end=int(round((frame+len(vals))*1000*a.fps_den/a.fps_num))
        out.append(Tag(tag=c.tag_name,start_time=start,end_time=end,source_media=a.source_media,track=c.output_track,
            frame_info={'frame_idx':int(frame)},
            additional_info={'x-coordinates':[_r(x,d) for x in vals],'source_iq':a.source_iq,'fps':a.fps_text,'provenance':c.provenance}))
    return out
