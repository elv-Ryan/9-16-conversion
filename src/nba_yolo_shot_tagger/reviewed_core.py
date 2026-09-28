"""Causal feature extraction, filters, alignment and metrics. No CUDA imports."""
from __future__ import annotations
import csv, hashlib, json, math, os, statistics, time
from pathlib import Path
import numpy as np

NAMES = ['active_speaker','gameplay_follow','graphic_text_lock','person_subject','safe_center','split_screen','static_composition']
# No game identity, absolute time, teacher features, or future observations.
FEATURE_NAMES = [f'{name}_{k}_{field}' for name in NAMES for k in range(2)
                 for field in ('confidence','cx','cy','width','height')]
FEATURE_NAMES += ['raw_x','top_confidence','controller_x','anchor_one_euro_x','dt_over_0p1','no_detection']
FEATURES = len(FEATURE_NAMES)
BASE_INDEX = FEATURE_NAMES.index('anchor_one_euro_x')


def utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def objhash(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def number(value, default=None):
    try:
        out=float(value)
        return out if math.isfinite(out) else default
    except (ValueError,TypeError): return default


def true(value): return str(value).strip().lower() in {'true','1','yes'}

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def half_width(width, height):
    h=(float(height)*9/16)/(2*float(width))
    if not 0 < h < .5: raise ValueError('Expected a source wider than the requested full-height 9:16 crop.')
    return h


class OneEuro:
    def __init__(self, min_cutoff=.15, beta=.25, d_cutoff=1.):
        self.min_cutoff=min_cutoff; self.beta=beta; self.d_cutoff=d_cutoff
        self.reset()
    def reset(self): self.t=None; self.raw=None; self.dx=0.; self.y=None
    @staticmethod
    def alpha(cutoff, dt): return 1/(1+1/(2*math.pi*max(cutoff,1e-8)*dt))
    def step(self, t, x):
        t=float(t); x=float(x)
        if not math.isfinite(x): raise ValueError('Nonfinite X')
        if self.t is None:
            self.t=t; self.raw=x; self.y=x; return x
        dt=t-self.t
        if dt <= 0: raise ValueError('Timestamps must strictly increase.')
        a=self.alpha(self.d_cutoff,dt)
        self.dx=a*(x-self.raw)/dt+(1-a)*self.dx
        a=self.alpha(self.min_cutoff+self.beta*abs(self.dx),dt)
        self.y=a*x+(1-a)*self.y; self.t=t; self.raw=x
        return self.y


class TrailingHampel:
    """Trailing RAW history. A robust scale floor handles a constant history."""
    def __init__(self, window=5, nsigma=3., floor=.003):
        self.window=int(window); self.nsigma=float(nsigma); self.floor=float(floor); self.hist=[]
    def reset(self): self.hist=[]
    def step(self,x):
        x=float(x); vals=(self.hist+[x])[-self.window:]
        med=statistics.median(vals); sigma=1.4826*statistics.median(abs(v-med) for v in vals)
        reject=len(vals)>=3 and abs(x-med)>max(self.floor,self.nsigma*sigma)
        self.hist=vals
        return med if reject else x


class Controller:
    """Existing rules with actual dt and explicit external-shot resets."""
    def __init__(self, params, half): self.p=params; self.half=half; self.reset()
    def reset(self):
        self.x=.5; self.t=None; self.last_good=-1e12; self.pending=None; self.count=0; self.last_jump=-1e12
    def step(self,t,raw,confidence,family):
        dt=.1 if self.t is None else float(t)-self.t
        if dt<=0: raise ValueError('Non-monotone controller input')
        self.t=float(t); p=self.p; gl=p['global']; event=0
        if family=='gameplay_follow':
            v=p[family]; acq,maint,tau,hold=v['acquire_confidence'],v['maintain_confidence'],v['pan_time_constant_seconds'],v['occlusion_hold_seconds']
        elif family=='person_subject':
            v=p[family]; acq,maint,tau,hold=v['acquire_confidence'],v['maintain_confidence'],v['pan_time_constant_seconds'],v['identity_hold_seconds']
        elif family=='active_speaker':
            v=p[family]; acq,maint,tau,hold=v['acquire_confidence'],v['maintain_confidence'],.25,gl['max_missing_target_hold_seconds']
        else: acq,maint,tau,hold=.10,.04,.35,gl['max_missing_target_hold_seconds']
        reliable=confidence >= (maint if t-self.last_good<=hold else acq)
        if reliable:
            self.last_good=t
            if family=='active_speaker':
                v=p[family]
                if self.pending is None or abs(raw-self.pending)>.06: self.pending=raw; self.count=1
                else: self.count+=1; self.pending=.7*self.pending+.3*raw
                if self.count>=v['speaker_change_confirmation_observations'] and abs(self.pending-self.x)>=v['speaker_jump_distance_x'] and t-self.last_jump>=v['speaker_jump_cooldown_seconds']:
                    self.x=self.pending; self.last_jump=t; event=2
            else: self.x+=dt/(tau+dt)*(raw-self.x); event=1
        elif t-self.last_good>gl['center_fallback_timeout_seconds']:
            self.x+=dt/(1.5+dt)*(.5-self.x); event=3
        self.x=float(np.clip(self.x,self.half,1-self.half))
        return self.x,event


class FeatureState:
    def __init__(self, params, width, height):
        self.half=half_width(width,height); self.ctrl=Controller(params,self.half); self.euro=OneEuro(); self.t=None
    def reset(self): self.ctrl.reset(); self.euro.reset(); self.t=None
    def step(self,candidates,t,reset=False):
        if reset: self.reset()
        out=np.zeros(FEATURES, np.float32)
        cs=sorted(candidates,key=lambda c:float(c['confidence']),reverse=True)
        counts=[0]*7
        for c in cs:
            cls=int(c['class_id']); conf=float(c['confidence']); box=np.asarray(c['bbox_xyxy_norm'],float)
            if not 0<=cls<7 or c['family']!=NAMES[cls]: raise ValueError('Seven-class contract mismatch')
            if box.shape!=(4,) or not np.isfinite(box).all() or not 0<=conf<=1: raise ValueError('Bad detection')
            if counts[cls]>=2: continue
            k=(cls*2+counts[cls])*5; counts[cls]+=1
            x1,y1,x2,y2=np.clip(box,0,1)
            out[k:k+5]=conf,(x1+x2)/2,(y1+y2)/2,max(0,x2-x1),max(0,y2-y1)
        if cs:
            c=cs[0]; box=c['bbox_xyxy_norm']; raw=float((box[0]+box[2])/2); conf=float(c['confidence']); family=c['family']; cls=int(c['class_id'])
        else: raw=.5; conf=0.; family='none'; cls=-1
        dt=.1 if self.t is None else float(t)-self.t
        base,event=self.ctrl.step(t,raw,conf,family)
        if event==2: self.euro.reset() # preserve an explicit current-controller jump
        anchor=float(np.clip(self.euro.step(t,base),self.half,1-self.half))
        out[-6:]=raw,conf,base,anchor,min(dt/.1,5.),float(not cs)
        self.t=float(t)
        return out,cls,event


def filter_path(t, x, resets, events, config):
    kind=config['kind']; out=np.empty(len(x),np.float32)
    euro=OneEuro(config.get('min_cutoff',.15),config.get('beta',.25),config.get('d_cutoff',1.))
    hp=TrailingHampel(config.get('window',5),config.get('nsigma',3.),config.get('floor',.003))
    for i,(ti,xi) in enumerate(zip(t,x)):
        if resets[i] or events[i]==2: euro.reset(); hp.reset()
        y=float(xi)
        if kind in ('hampel','hampel_one_euro'): y=hp.step(y)
        if kind in ('one_euro','hampel_one_euro'): y=euro.step(ti,y)
        out[i]=y
    return out


def teacher_ok(row):
    """Reject explicitly invalid/unresolved or future-aware teacher targets."""
    if row.get('split') not in {'train','validation'}: return False,'split'
    x=number(row.get('crop_center_x_norm_causal'))
    if x is None or not 0<=x<=1: return False,'x'
    if row.get('runtime_family') not in NAMES: return False,'family'
    for k in ('future_frames_used_for_student_target','future_context_used_for_causal_semantics'):
        if true(row.get(k)): return False,'future_context'
    if 'student_supervision_is_causal' in row and not true(row['student_supervision_is_causal']): return False,'not_causal'
    if 'student_supervision_reference' in row and row['student_supervision_reference'].lower()!='causal': return False,'not_causal'
    conf=number(row.get('causal_focus_confidence',row.get('focus_confidence')),0.)
    if conf<.5: return False,'low_confidence'
    if row.get('causal_target_reliable') and not true(row['causal_target_reliable']): return False,'unreliable'
    action=(row.get('human_recommended_action','')+' '+row.get('human_team_status','')).lower()
    if any(w in action for w in ('quarantine','unresolved','exclude','reject','disagree','needs_review','repair_teacher','wrong_teacher')): return False,'human_flag'
    return True,'accepted'


def align_teacher(t, shot_ids, records):
    """Offline target reconstruction inside a single trusted window, never across gaps/cuts.
    Teacher targets are labels, not model inputs. No extrapolation into uncovered time.
    """
    n=len(t); y=np.full(n,.5,np.float32); conf=np.zeros(n,np.float32); family=np.full(n,-1,np.int16)
    block=np.full(n,-1,np.int32); vod=np.full(n,np.nan,np.float32)
    groups={}
    for r in records:
        ok,_=teacher_ok(r)
        if not ok: continue
        key=(r.get('shot_id',''),r.get('window_id',''))
        if not key[0] or not key[1]: continue
        groups.setdefault(key,[]).append(r)
    bn=0
    for (sid,wid), rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda r:(float(r['timestamp_seconds']),-float(r.get('causal_focus_confidence',r.get('focus_confidence',0)))))
        uniq={}
        for r in rr: uniq.setdefault(float(r['timestamp_seconds']),r)
        rr=list(uniq.values()); tt=np.array([float(r['timestamp_seconds']) for r in rr])
        xx=np.array([float(r['crop_center_x_norm_causal']) for r in rr]); cc=np.array([float(r.get('causal_focus_confidence',r.get('focus_confidence',0))) for r in rr])
        ff=np.array([NAMES.index(r['runtime_family']) for r in rr])
        vv=np.array([number(r.get('crop_center_x_norm_vod_reference'),np.nan) for r in rr])
        rate=max(.25,min(float(r.get('focus_rate_hz') or 4) for r in rr)); max_gap=min(1.1,max(.30,1.5/rate))
        idx=np.flatnonzero((t>=tt[0]-1e-7)&(t<=tt[-1]+1e-7)&(shot_ids==sid))
        if len(tt)<2 or not len(idx): continue
        hi=np.clip(np.searchsorted(tt,t[idx],side='right'),1,len(tt)-1); lo=hi-1
        dt=tt[hi]-tt[lo]; a=(t[idx]-tt[lo])/np.maximum(dt,1e-9)
        discontinuity=np.array([any(v in str(r.get('event_causal','')).lower() for v in ('jump','cut','reset')) for r in rr])
        good=(dt>0)&(dt<=max_gap)&(ff[hi]==ff[lo])&(~discontinuity[hi])
        confidence=np.minimum(cc[hi],cc[lo]); good &= confidence>conf[idx]
        use=idx[good]; lo=lo[good]; hi=hi[good]; a=a[good]; confidence=confidence[good]
        # A distinct block at each gap, event, family change, or teacher-window boundary.
        breaks=np.r_[True,(np.diff(tt)>max_gap)|(np.diff(ff)!=0)|discontinuity[1:]]
        sub=np.cumsum(breaks)
        y[use]=xx[lo]+a*(xx[hi]-xx[lo]); conf[use]=confidence; family[use]=ff[lo]; block[use]=bn+sub[lo]
        vod[use]=vv[lo]+a*(vv[hi]-vv[lo]); bn+=int(sub[-1])+1
    return y,conf,family,block,vod


def contiguous_edges(t,mask,block,reset):
    dt=np.diff(t); e=mask[1:]&mask[:-1]&(block[1:]==block[:-1])&(~reset[1:])&(dt>0)&(dt<.16)
    return e,dt


def metrics(d,x,mask_override=None):
    t=d['t']; y=d['target']; mask=(d['weight']>0) if mask_override is None else mask_override
    if np.count_nonzero(mask)<3: return None
    x=np.asarray(x,float); y=np.asarray(y,float); er=np.abs(x-y); edge,dt=contiguous_edges(t,mask,d['block'],d['reset'])
    dx=np.diff(x)/np.maximum(dt,1e-9); dy=np.diff(y)/np.maximum(dt,1e-9)
    stable=edge&(np.abs(dy)<.02); moving=edge&(np.abs(dy)>.05)
    ae=edge[1:]&edge[:-1]
    def mean(v): return float(np.mean(v)) if len(v) else 0.
    acceleration=np.abs(np.diff(dx)-np.diff(dy))/np.maximum((dt[1:]+dt[:-1])/2,1e-9)
    m={'n':int(mask.sum()),'mae':mean(er[mask]),'p95_abs_error':float(np.quantile(er[mask],.95)),
       'stable_mean_abs_step':mean(np.abs(np.diff(x))[stable]),'velocity_mae':mean(np.abs(dx-dy)[edge]),
       'acceleration_mae':mean(acceleration[ae]),'moving_mae':mean(er[1:][moving]),
       'stable_edges':int(stable.sum()),'moving_edges':int(moving.sum())}
    # Delay estimate using within-block shifted teacher positions during teacher motion.
    delays=[]
    for k in range(11):
        ids=np.flatnonzero(moving)+1; ids=ids[ids>=k]
        if len(ids):
            valid=mask[ids-k]&(d['block'][ids-k]==d['block'][ids])
            ids=ids[valid]
            if len(ids)>=20: delays.append((mean(np.abs(x[ids]-y[ids-k])),k))
    m['estimated_lag_ms']=int(min(delays)[1]*100) if delays else None
    m['objective']=m['mae']+.35*m['p95_abs_error']+1.8*m['stable_mean_abs_step']+.06*m['velocity_mae']+.008*m['acceleration_mae']+.25*m['moving_mae']
    return m


def aggregate(game_metrics):
    ms=[m for m in game_metrics if m is not None]
    if not ms: raise ValueError('No supervised games for metrics.')
    # Macro-average games. p95 is mean per-game p95, not a pooled quantile.
    fields=['mae','p95_abs_error','stable_mean_abs_step','velocity_mae','acceleration_mae','moving_mae','objective']
    out={k:float(np.mean([m[k] for m in ms])) for k in fields}
    out.update(n=sum(m['n'] for m in ms),games=len(ms),aggregation='macro_game; p95 is mean per-game p95')
    ls=[m['estimated_lag_ms'] for m in ms if m.get('estimated_lag_ms') is not None]
    out['estimated_lag_ms']=float(np.mean(ls)) if ls else None
    return out


def acceptable(candidate,base):
    checks={
        'mae':candidate['mae']<=base['mae']+max(.0015,.02*base['mae']),
        'moving_mae':candidate['moving_mae']<=base['moving_mae']+max(.002,.03*base['moving_mae']),
        'p95':candidate['p95_abs_error']<=base['p95_abs_error']+.005,
        'stable_jitter':candidate['stable_mean_abs_step']<=base['stable_mean_abs_step']*1.05+.0002,
        'objective':candidate['objective']<=base['objective']+1e-8}
    if candidate.get('estimated_lag_ms') is not None and base.get('estimated_lag_ms') is not None:
        checks['lag_diagnostic']=candidate['estimated_lag_ms']<=base['estimated_lag_ms']+100
    return all(checks.values()),checks


def source_frame_x(t,x,frames,fps_num,fps_den,reset):
    """VOD reconstruction only: linear between known updates, no interpolation across cuts."""
    query=np.asarray(frames,dtype=float)*fps_den/fps_num
    hi=np.clip(np.searchsorted(t,query,side='right'),1,len(t)-1); lo=hi-1
    a=np.clip((query-t[lo])/np.maximum(t[hi]-t[lo],1e-9),0,1)
    out=x[lo]+a*(x[hi]-x[lo]); out=np.where(reset[hi],x[lo],out)
    return np.where(query<=t[0],x[0],np.where(query>=t[-1],x[-1],out))
