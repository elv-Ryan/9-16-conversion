# Eluvio NBA YOLO shot-to-X tagger — v3.3 handoff

YOLO-only Eluvio tagger MVP for already-segmented NBA shots.

## Production contract

```text
one shot file on stdin
  -> one custom seven-family NBA YOLO Student
  -> shot family + semantic focus bbox
  -> full-shot horizontal trajectory controller
  -> normalized source-frame X trajectory
  -> Eluvio tag JSONL + terminal progress
```

Production runtime does **not** use Qwen, shot detection, a second YOLO pass,
vertical rendering, or video encoding.

The primary output track is `vertical_video`. The tag's
`additional_info["x-coordinates"]` contains exactly one legal normalized crop
center per source frame.

## Runtime families

- `active_speaker`
- `gameplay_follow`
- `graphic_text_lock`
- `person_subject`
- `safe_center`
- `split_screen`
- `static_composition`

The current checkpoint does not expose the finer Category-Gold taxonomy as a
separate head. `category` is therefore the coarse policy mapping of the runtime
family.

## Eluvio protocol implementation

The entrypoint delegates the daemon/protocol machinery to current pinned
`common-ml`:

- `catch_errors()`
- `get_params()`
- `TagMessageProducer`
- `Tag`, `Progress`, `Error`, `FrameInfo`
- `run_default(...)`

`ProgressRatio` is supported by the pinned `common-ml`, but it is disabled by
default because the current `elv-ml` protocol document explicitly documents
`tag`, `progress`, and `error` as the protocol message types. Runtime progress
is available in logs/test-controller status without requiring the extension.

See:

- `docs/ELUVIO_COMPLIANCE_CHECKLIST.md`
- `docs/ELUVIO_TAGGER_AUDIT.md`
- `docs/OUTPUT_CONTRACT.md`

## Runtime parameters

```json
{
  "device": "0",
  "imgsz": 1280,
  "inference_fps": 10.0,
  "batch_size": 8,
  "use_fp16": true,
  "input_mode": "shot_file",
  "output_track": "vertical_video",
  "include_focus_samples": true,
  "emit_focus_track": false,
  "emit_progress_ratio": false,
  "continue_on_error": true
}
```

Backup externally supplied shot intervals are supported with
`input_mode=shot_manifest`; the container still does not detect shots itself.

## Clone / model / build

```bash
git clone --recurse-submodules git@github.com:elv-Ryan/9-16-conversion.git
cd 9-16-conversion
git checkout nba-yolo-shot-tagger-v3.3
git submodule update --init --recursive

./scripts/fetch_current_model.sh
make unit-test
make build
```

`best.pt` is intentionally not committed. `model_manifest.json` is committed and
pins its SHA-256. On AI-03, `fetch_current_model.sh` copies the trusted checkpoint
locally. Elsewhere it can fetch it over SSH using `REMOTE`, `SSH_PORT`, and
`MODEL_SOURCE`.

## Black-box Podman test

```bash
DEVICE=2 IMAGE=nba-yolo-shot-tagger:latest \
  ./scripts/test_podman_shot.sh /absolute/path/to/one-shot.mp4
```

The test requires a successful `vertical_video` tag, one X per source frame,
legal normalized crop centers, and exactly one terminal `Progress` message.

## Standard Eluvio buildscripts

The repository includes the normal `qluvio/buildscripts` submodule and
`Makefile.tagger-model` integration. `build.sh` calls
`buildscripts/build_container.bash`, so normal build metadata/annotations are
added by the Eluvio build tooling. The release helper also prepares `test-files/`
and runs the unmodified official `make test` before the stricter X-semantic
black-box test and GitHub push.

Typical targets:

```bash
make build
make test
make deploy
```

`make deploy` should only be used after the repository is clean, the model
artifact is present, the Podman test passes, and registry authentication is
configured.

## Joe 518-shot qualification

Source corpus on AI-03:

```text
/home/mltrain/elv-joe/shots/iq__2q6ZyYAmFfKDBJeWMGsWLP549twd
```

The real v3.1 image has already passed a three-shot, zero-error Podman smoke on
this corpus. A full single-container 518-shot run is used to qualify actual
Eluvio-like long-lived behavior and measure single-container throughput.

For faster corpus-only qualification, `scripts/parallel_joe518_qualify.py` may
run independent shot shards on multiple genuinely free GPUs. That result is
**not** a substitute for the single-container latency/throughput measurement;
it is only a faster corpus correctness gate.

## Model currently pinned

Trusted checkpoint source on AI-03:

```text
/home/mltrain/elv-ryan/projects/9-16-conversion-ryan-v2.1-student-mvp/output/
generic_basketball_v11_student_round00_v1/yolo_fp32_6gpu_round00_v7/long_run/
focus_target_round00_fp32/weights/best.pt
```

Current verified SHA-256 from the v3.1 image build:

```text
e1f498b0447f77c30d2b82210367b556d5f649caeee370f5fd3d20f7a0fec770
```

Replace the checkpoint later only after a newer Student champion passes the same
seven-class artifact, protocol, Podman, and corpus gates.


## Output consumed by the renderer

The Fabric-facing file is JSONL, as required by the Tagger protocol. For each
shot, the renderer consumes `data.additional_info["x-coordinates"]`: one
normalized source-width horizontal crop center per source frame. Other fields
are diagnostic/routing metadata and do not change the X contract. The local
`x-output.json` helper is a convenience export, not the Fabric wire format.
