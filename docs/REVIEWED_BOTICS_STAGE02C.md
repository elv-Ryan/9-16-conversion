# Reviewed Round01 integration into botics (Stage 2C)

## Scope and status

This integration activates the frozen v4 `ReviewedRound01Runtime` using
`runtime_manifest.json`, including One Euro min_cutoff=0.4 and beta=1.0.
The weights remain Round01 schemafix v3. This does not add a ball model,
retrain any model, tune camera parameters, add a 150 ms buffer, or deploy a container.

Botics remains the host application. Its sampling, shot boundaries, commit horizon,
metadata vote, finalization, tags, and live output encoding are retained.
`run.py`, `producer.py`, `contract.py`, `types.py`, `live.py`, `video.py`,
`shot_boundaries.py`, dependencies, and the container files are not changed.

## Active behavior

Each shot owns a persistent reviewed runtime, created on its first committed observation.
It receives ALL retained candidates, not just candidates matching the locked shot family.
The locked family remains botics metadata and does not gate reviewed steering.
Each observation is processed exactly once. Segment and batch joins do not reset state.
A real shot cut or the existing maximum-shot close starts new runtime state.
The frozen runtime owns its confirmed-speaker jumps and One Euro resets.

Internal update times use shot-relative frame indices and a rationalized source FPS,
as in v4. Serialized botics millisecond timestamps and file-relative offsets are unchanged.
Source-frame expansion holds the previous X before event 2, then switches at the jump frame.
This avoids interpolating a speaker cut into a pan. Ordinary updates still interpolate.

Botics' legal crop geometry remains authoritative. On standard widescreen-to-9:16 input
it is identical to v4. Full-width/narrow inputs remain centered at 0.5; custom target
aspects preserve their legal crop interval. A geometry or FPS change within an already
open reviewed shot is rejected rather than silently reusing incompatible controller state.
This guard does not add a new shot-detection or discontinuity protocol.

The legacy selection and smooth_samples helpers remain available for reference but are
not called by active steering. `hold_and_cut_families` and `min_hold_seconds` continue to
parse for interface compatibility; they no longer override the reviewed runtime.
The imported reviewed_core.py, reviewed_runtime.py and runtime_manifest.json are unchanged.

## Output compatibility

Final JSON tags still contain one normalized X per source frame.
The live sink still receives committed inference-sample X values, not expanded frame X.
Its four-byte little-endian fixed-point encoding is unchanged. No extra tags or fields
are introduced by this stage. These are unit/fixture contract guarantees until real-video
and container checks complete.

## Tests and acceptance

65 reviewed tests: 2 original runtime tests plus 63 integration tests.
The integration suite covers all seven families against independent runtime instances,
explicit jumps, source-frame expansion, segment persistence, no replay of committed
observations, pending future evidence, deferred cuts, final flush, metadata offsets,
live sample count/encoding, geometry safety and old model-constructor compatibility.
The source hash guard verifies that all unrelated files remain unchanged.

The original 56-test suite retains the same nine known stale-default/timing failures.
Only two old behavior tests are adapted: the former whole-shot legacy-smoothing
comparison and the static constant-lock comparison now compare against the frozen
reviewed runtime. Their IDs remain unchanged for auditability. The nine unrelated
failures are not relabeled or suppressed; exact failure IDs are checked before/after.

The CPU checkpoint also loads the actual checksum-verified Round01 model and passes one
synthetic frame through the active service. This is not a real-video quality benchmark.
Next: rerun the same six shots, then consecutive segment inputs and the container.
No quality improvement is guaranteed. In particular, the gameplay diagnostic already
showed that the reviewed controller can reduce an excursion while adding lag.
That is recorded, not tuned away in this migration. Keep the previous artifacts for A/B review.

## Provenance

Target committed checkpoint state: a16b97b51a8be74c2b0ba5d049b21b25bd9bc7d2.
V4 source: 19b535c6b98245b707061a5fd0f8802f9eeb980e.
Model SHA-256: 0921e4843864557bc1e5141f655a461af24c758d7390f2ea0553091df743865a.
The installer preserves Stage 2B as a rollback point and never commits or pushes.
The repository's older MANIFEST.sha256 remains stale during migration and must be
regenerated for the accepted release; the installer instead creates its own checked
source identity and test report. Do not use legacy setup/push scripts as a release workflow.
