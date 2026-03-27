#!/usr/bin/env python3
import argparse, json, math
from collections import defaultdict
import cv2

def _box_from_region(r):
    keys = [
        ("x_min","y_min","x_max","y_max"),
        ("xmin","ymin","xmax","ymax"),
        ("x0","y0","x1","y1"),
        ("left","top","right","bottom"),
    ]
    for a,b,c,d in keys:
        if a in r and b in r and c in r and d in r:
            return float(r[a]), float(r[b]), float(r[c]), float(r[d])
    return None

def _to_px(box, W, H):
    x0,y0,x1,y1 = box
    if 0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= y1 <= 1.0:
        return x0*W, y0*H, x1*W, y1*H
    return x0, y0, x1, y1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--shots_json", required=True)
    ap.add_argument("--signals_json", required=True)
    ap.add_argument("--policy_json", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--target_w", type=int, default=406)
    ap.add_argument("--target_h", type=int, default=720)
    ap.add_argument("--fps", type=float, default=16.0)
    args=ap.parse_args()

    shots=json.load(open(args.shots_json,"r",encoding="utf-8")).get("shots")
    pols=json.load(open(args.policy_json,"r",encoding="utf-8"))["policies"]
    sig_root=json.load(open(args.signals_json,"r",encoding="utf-8")).get("signals",{})

    pol_by_id={int(p["shot_id"]):p for p in pols}

    reg_by_shot=defaultdict(lambda: defaultdict(list))
    for sig_name, blob in sig_root.items():
        regs=blob.get("regions",[]) if isinstance(blob,dict) else []
        for r in regs:
            sid=r.get("shot_id")
            if sid is None:
                continue
            reg_by_shot[int(sid)][sig_name].append(r)

    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {args.video}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_w=int(round(H * 9.0/16.0))
    crop_h=H

    centers=[]
    last_cx=W/2.0
    for s in shots:
        sid=int(s["shot_id"])
        p=pol_by_id.get(sid,{})
        prios=p.get("prioritized_signals",[])
        cx=None

        for pr in prios:
            sig=pr.get("signal")
            regs=reg_by_shot.get(sid,{}).get(sig,[])
            best=None
            best_score=-1.0
            for r in regs:
                box=_box_from_region(r)
                if not box:
                    continue
                score=float(r.get("score", 0.0))
                if score > best_score:
                    best_score=score
                    best=r
            if best is not None:
                box=_box_from_region(best)
                x0,y0,x1,y1=_to_px(box, W, H)
                cx=(x0+x1)/2.0
                break

        if cx is None:
            cx=last_cx

        sigma=float(p.get("crop_params",{}).get("smoothing_sigma_sec", 0.25))
        dur=max(1e-3, float(s["t_end_sec"])-float(s["t_start_sec"]))
        alpha=1.0-math.exp(-dur/max(1e-3,sigma))
        cx=alpha*cx + (1.0-alpha)*last_cx

        centers.append((float(s["t_start_sec"]), float(s["t_end_sec"]), cx))
        last_cx=cx

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    out=cv2.VideoWriter(args.out_mp4, fourcc, args.fps, (args.target_w, args.target_h))
    if not out.isOpened():
        raise SystemExit("failed to open VideoWriter (mp4v).")

    shot_i=0
    nshots=len(centers)
    frame_i=0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_i / args.fps
        while shot_i+1 < nshots and t >= centers[shot_i][1]:
            shot_i += 1
        cx = centers[shot_i][2]

        left=int(round(cx - crop_w/2.0))
        left=max(0, min(left, W - crop_w))
        crop=frame[0:crop_h, left:left+crop_w]
        resized=cv2.resize(crop, (args.target_w, args.target_h), interpolation=cv2.INTER_LINEAR)
        out.write(resized)
        frame_i += 1

    out.release()
    cap.release()
    print(args.out_mp4)

if __name__=="__main__":
    main()
