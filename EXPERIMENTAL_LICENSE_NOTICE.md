# YOLO26 v5 licensing gate

This worktree is a technical evaluation. Do not build, publish, merge into a
production image, or deploy it to Content Fabric until Eluvio confirms the
applicable Ultralytics commercial license or elects to comply with AGPL-3.0.

The runtime is YOLO26-only:

- `yolo26s.pt` supplies COCO person and sports-ball detections.
- `yolo26n-pose.pt` supplies person pose and pose-derived frontal-face/head evidence.
- Eluvio's tracker, separate sports/movie policies, smoother, and common-ml JSONL contract remain in place.
- There is no MediaPipe detector, dependency, import, or fallback path.
