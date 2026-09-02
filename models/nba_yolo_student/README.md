# Model artifact

The OCI build requires two local files in this directory:

- `best.pt` — trusted current NBA YOLO Student checkpoint
- `model_manifest.json` — SHA-256 and locked seven-class contract

Do not commit `best.pt` directly to ordinary Git history. For the first MVP, run:

```bash
./scripts/fetch_current_model.sh
```

The script copies the current trusted Round00 checkpoint from AI-03 and writes the manifest. A Git LFS, release-artifact, or internal model-registry workflow should replace this local step before CI/CD publication.
