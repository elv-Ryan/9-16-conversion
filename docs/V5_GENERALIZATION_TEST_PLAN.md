# YOLO26 v5 generalization test plan

## Purpose

The original v4 clips are regression examples, not a training set and not the
sole acceptance set. V5 behavior must be justified by reusable evidence:
detection confidence, geometry, track continuity, scene state, subject
identity, and temporal stability. No behavior rule may depend on a source
filename, title, clip timestamp, venue, actor, team, or remembered screen
coordinate.

## Dataset discipline

Use three source-level partitions:

1. **Regression:** the existing v4 examples. They must not regress.
2. **Development:** new clips used to diagnose and make generic changes.
3. **Holdout:** clips from different original games/programs/films that are not
   reviewed while tuning a change. Freeze code and configuration before
   opening their v5 outputs.

Split by original source, not by extracted clip. Two excerpts from the same
movie or broadcast belong to the same partition. This prevents adjacent scenes,
broadcast graphics, cast, camera grammar, or detector artifacts from leaking
from development into holdout.

An initial expansion target is at least 12 sports clips and 12 movie clips from
at least three independent original sources per mode. This is an engineering
checkpoint, not a statistical guarantee. Continue adding clips when a new
camera grammar, edit style, venue, graphic package, subject scale, or failure
mode appears.

## Sports coverage strata

Include examples of:

- half-court possession, dribble, pass, shot, rebound, and loose ball;
- transition and fast breaks with long horizontal ball movement;
- ball occlusion by players and referees;
- wide, medium, and close replay shots;
- timeout, bench, free-throw setup, inbound setup, and dead-ball periods;
- announcer/studio, crowd, scoreboard, full-screen graphic, and commercial-like
  transitions;
- low-contrast, motion-blurred, very small, edge-clamped, and temporarily
  undetected balls;
- false sports-ball candidates from logos, heads, lights, score bugs, court
  marks, and crowd objects when present.

Report live-play metrics separately from replay and non-gameplay metrics. A
high aggregate score must not hide failure in transition play or graphics.

## Movie coverage strata

Include examples of:

- single-character dialogue and shot/reverse-shot editing;
- two-shots, over-the-shoulder compositions, and group dialogue;
- entrances, exits, pans, dollies, and slow blocking within a continuous shot;
- face loss, profile views, back-of-head foreground occlusion, and partial
  person boxes;
- posters, framed photographs, screens, reflections, crowds, and background
  faces;
- action without a stable face, dark scenes, animation, and extreme close-ups;
- hard cuts with similar color palettes, flashes, dissolves, and rapid edits.

Report stable-dialogue, moving-subject, group-composition, action, and hard-cut
metrics separately.

## Annotation and review

For every audited sports frame, record whether a real ball is visible and
credible, its center/box, whether the selected ball is real, and whether edge
clamping makes perfect centering impossible. For every audited movie interval,
record the narratively primary subject or interaction group, whether the crop
includes it, whether subject movement justifies camera movement, and whether a
background or nonfrontal region took control.

Use the QA debug sidecar and side-by-side preview to distinguish:

1. detector miss or false positive;
2. tracker association/continuity failure;
3. scene-state or candidate-generation error;
4. candidate-scoring or hysteresis error;
5. missed/false shot boundary;
6. smoothing of an otherwise correct target;
7. rendering or frame-index mismatch.

Do not tune smoothing to conceal a detector or policy failure.

## Acceptance reporting

Always report:

- aggregate result;
- median, p95, and worst-clip result;
- result by coverage stratum;
- count and duration of critical failures;
- detector/tracker/policy/shot/smoothing attribution;
- regressions relative to v4 and the previous v5 iteration.

Sports gates remain those defined from the observed v4 baseline: visible-ball
inclusion, crop-to-ball error, false-ball focus, reacquisition latency,
no-evidence drift, reversal count, and crop-limit occupancy. Movie gates remain
primary-subject inclusion, background/poster/nonfrontal takeover count,
unexplained dialogue movement, unwanted switches, cut-to-lock time, and
no-evidence drift.

A change is eligible for implementation when it is either a clear invariant
(for example, no MediaPipe path, no no-evidence drift, or a ball must outrank a
person during trusted live-play evidence) or is supported by failures from at
least two independent source videos. A one-clip exception belongs in the
failure log, not in behavior code.

## Iteration procedure

1. Tag the exact code/config state locally; do not push.
2. Run every regression, development, and unopened holdout clip through the
   same command and model weights.
3. Validate common-ml JSONL, one X per frame, audio, dimensions, duration, and
   all QA sidecars.
4. Review side-by-side and pure-vertical outputs.
5. Classify failures before changing code.
6. Make the smallest generic change and add a synthetic invariant test.
7. Rerun the entire suite, not only the failing clips.
8. Compare aggregate, stratum, and worst-clip metrics.
9. Review the complete diff and results before commit or push.
