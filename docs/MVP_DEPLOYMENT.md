# Fast MVP deployment

## 1. Apply this overlay to a clean worktree

```bash
git -C /path/to/9-16-conversion worktree add \
  -b nba-yolo-shot-tagger-mvp \
  /path/to/nba-yolo-shot-tagger-mvp \
  ryan-v2.1

cd /path/to/nba-yolo-shot-tagger-mvp
/path/to/this-bundle/apply_overlay.sh .
```

## 2. Fetch the current trusted checkpoint

```bash
./scripts/fetch_current_model.sh
```

The source is the current valid Round00 checkpoint, not any invalid six-class or
failed continuation checkpoint.

## 3. Build

```bash
IMAGE=nba-yolo-shot-tagger:mvp ./scripts/build_podman.sh
```

## 4. Test one actual shot

```bash
DEVICE=0 IMAGE=nba-yolo-shot-tagger:mvp \
  ./scripts/test_podman_shot.sh /absolute/path/to/shot.mp4
```

The test validates:

- valid JSONL messages;
- a `vertical_video` tag;
- exact source path;
- non-empty X output;
- one X per source frame;
- legal normalized crop centers;
- terminal progress.

## 5. Commit and push

```bash
git status --short
git add .
git commit -m "Add NBA YOLO shot-to-X Eluvio tagger MVP"
git push -u origin nba-yolo-shot-tagger-mvp
```

The ChatGPT GitHub integration could read the repository but received HTTP 403
on branch creation, so it did not push this change.

## One-command AI-03 deployment

From this bundle on the Mac:

```bash
BUILD_IMAGE=1 RUN_TEST=0 PUSH_GITHUB=0 \
  ./scripts/deploy_ai03_mvp.sh
```

This creates an isolated Git worktree, applies the overlay, copies the current
trusted Round00 checkpoint, writes its SHA-256 manifest, runs the unit suite, and
builds the Podman image. It does not modify the active training worktree.

To include a CPU protocol smoke test:

```bash
BUILD_IMAGE=1 RUN_TEST=1 TEST_DEVICE=cpu \
  ./scripts/deploy_ai03_mvp.sh
```

To commit and push from AI-03 using the user's existing Git credentials:

```bash
PUSH_GITHUB=1 ./scripts/deploy_ai03_mvp.sh
```
