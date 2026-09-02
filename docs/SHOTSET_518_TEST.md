# Full 518-shot AI-03 test

## Inputs

```text
Host: mltrain@63.247.64.18:5055
Directory: /home/mltrain/elv-joe/shots/iq__2q6ZyYAmFfKDBJeWMGsWLP549twd
Expected files: 518, numbered 0 through 517
```

The directory is mounted read-only at `/elv/input`. One newline-separated list
of all 518 container paths is sent to one long-lived tagger container. This tests
the actual Eluvio stdin queue behavior rather than launching one container per
shot.

## Strict pass gate

- exactly 518 stable video files discovered;
- only a genuinely free selected GPU is exposed to the container; no process is killed or reset;
- three-shot smoke passes first;
- exactly one terminal `progress` or `error` per input;
- zero `error` messages for a clean pass;
- exactly one `vertical_video` tag per successful shot;
- family belongs to the seven-class checkpoint contract;
- `x-coordinates` is non-empty and its length equals `frame_count`;
- every X is finite and inside the legal 9:16 crop-center interval;
- all 518 sources are accounted for.

## Outputs

```text
STATUS.json
full_518/out.jsonl
full_518/x-output.json
full_518/summary.json
full_518/per_shot_summary.csv
full_518/review_candidates.csv
full_518/errors.json
full_518/validator.log
logs/controller.log
logs/full_518_shot_run.log
```

`summary.json` reports total processed source time, wall time, and aggregate
multiples of real time. `review_candidates.csv` prioritizes low-confidence and
high-motion outputs for later visual inspection.
