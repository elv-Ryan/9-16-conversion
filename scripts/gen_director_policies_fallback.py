#!/usr/bin/env python3
import argparse, json, os, hashlib, datetime
from collections import Counter

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_shots(raw):
    if isinstance(raw, dict):
        shots = raw.get("shots") or raw.get("segments") or raw.get("items") or raw.get("shot_list")
        if shots is None:
            raise ValueError(f"Unrecognized shots json dict keys: {list(raw.keys())[:30]}")
    elif isinstance(raw, list):
        shots = raw
    else:
        raise ValueError("Unrecognized shots json format")
    out = []
    for i, s in enumerate(shots):
        if not isinstance(s, dict):
            continue
        sid = int(s.get("shot_id", s.get("id", i)))
        t0 = s.get("t_start_sec", s.get("start_sec", s.get("start", s.get("t0", 0.0))))
        t1 = s.get("t_end_sec", s.get("end_sec", s.get("end", s.get("t1", t0))))
        out.append({"shot_id": sid, "t_start_sec": float(t0), "t_end_sec": float(t1)})
    return out

def summarize_signals(signals):
    if not isinstance(signals, dict):
        return {}
    sig_root = signals.get("signals", signals)
    if not isinstance(sig_root, dict):
        return {}
    by_shot = {}
    for sig_name, blob in sig_root.items():
        if not isinstance(blob, dict):
            continue
        regions = blob.get("regions")
        if not isinstance(regions, list):
            continue
        for r in regions:
            if not isinstance(r, dict):
                continue
            sid = r.get("shot_id")
            if sid is None:
                continue
            sid = int(sid)
            score = float(r.get("score", 0.0))
            by_shot.setdefault(sid, {})
            by_shot[sid][sig_name] = max(by_shot[sid].get(sig_name, 0.0), score)
    return by_shot

def infer_content_type(clip_id: str, sigmax: dict) -> str:
    cid = clip_id.lower()
    if "spider" in cid or "spiderman" in cid:
        return "music_video_stylized"
    if sigmax.get("ball", 0.0) >= 0.45:
        return "sports_gameplay"
    if max(sigmax.get("person", 0.0), sigmax.get("face", 0.0)) >= 0.45:
        return "interview"
    return "generic"

def defaults_for_type(content_type: str):
    safe_zone = {"top": 0.10, "bottom": 0.14, "left": 0.08, "right": 0.08}
    hard = {
        "target_aspect": "9:16",
        "preserve_full_height": True,
        "disallow_zoom": True,
        "never_clip_primary_subject": True,
        "keep_text_safe_zone": content_type in ("text_heavy", "music_video_stylized"),
        "safe_zone_norm": safe_zone
    }
    if content_type == "sports_gameplay":
        crop = {"smoothing_sigma_sec": 0.18, "max_pan_norm_per_sec": 0.75, "max_jerk_norm_per_sec2": 4.0, "lookahead_sec": 0.25, "deadband_norm": 0.01}
        prio = [
            {"signal":"ball","weight":1.30,"required":False,"min_score":0.45,"max_age_sec":0.40},
            {"signal":"person","weight":1.00,"required":True,"min_score":0.35,"max_age_sec":1.20}
        ]
    elif content_type == "interview":
        crop = {"smoothing_sigma_sec": 0.35, "max_pan_norm_per_sec": 0.35, "max_jerk_norm_per_sec2": 2.0, "lookahead_sec": 0.00, "deadband_norm": 0.02}
        prio = [
            {"signal":"face","weight":1.20,"required":False,"min_score":0.45,"max_age_sec":1.50},
            {"signal":"person","weight":1.00,"required":True,"min_score":0.35,"max_age_sec":1.50}
        ]
    elif content_type == "music_video_stylized":
        crop = {"smoothing_sigma_sec": 0.22, "max_pan_norm_per_sec": 0.60, "max_jerk_norm_per_sec2": 3.5, "lookahead_sec": 0.10, "deadband_norm": 0.02}
        prio = [
            {"signal":"person","weight":1.10,"required":True,"min_score":0.35,"max_age_sec":1.20},
            {"signal":"text","weight":0.90,"required":False,"min_score":0.35,"max_age_sec":2.00}
        ]
    else:
        crop = {"smoothing_sigma_sec": 0.25, "max_pan_norm_per_sec": 0.50, "max_jerk_norm_per_sec2": 3.0, "lookahead_sec": 0.05, "deadband_norm": 0.02}
        prio = [{"signal":"person","weight":1.00,"required":False,"min_score":0.35,"max_age_sec":1.20}]
    return crop, hard, prio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--shots_json", required=True)
    ap.add_argument("--signals_json", default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--user_intent_prompt", default=None)
    args = ap.parse_args()

    shots = normalize_shots(load_json(args.shots_json))
    sig_data = load_json(args.signals_json) if args.signals_json else {}
    sig_by_shot = summarize_signals(sig_data)

    policies = []
    for s in shots:
        sid = s["shot_id"]
        sigmax = sig_by_shot.get(sid, {})
        ctype = infer_content_type(args.clip_id, sigmax)
        crop, hard, prio = defaults_for_type(ctype)

        if ctype == "sports_gameplay" and sigmax.get("ball", 0.0) >= 0.55:
            for it in prio:
                if it["signal"] == "ball":
                    it["required"] = True

        policies.append({
            "shot_id": sid,
            "t_start_sec": s["t_start_sec"],
            "t_end_sec": s["t_end_sec"],
            "content_type": ctype,
            "prioritized_signals": prio,
            "crop_params": crop,
            "hard_constraints": hard,
            "rationale": None
        })

    payload = {
        "schema": "director.policy.v1",
        "clip_id": args.clip_id,
        "generated_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "generator": "fallback_v1",
        "source_inputs": {
            "shots_json": args.shots_json,
            "shots_sha256": sha256_file(args.shots_json),
            "signals_json": args.signals_json,
            "signals_sha256": sha256_file(args.signals_json) if args.signals_json else None
        },
        "user_intent": {"prompt": args.user_intent_prompt, "hard_overrides": None},
        "policies": policies
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    ct = Counter([p["content_type"] for p in policies])
    print(args.out_json)
    print("policy_count:", len(policies))
    print("content_type_counts:", dict(ct))

if __name__ == "__main__":
    main()
