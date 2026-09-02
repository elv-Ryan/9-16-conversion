#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

EXPECTED = [
    "active_speaker",
    "gameplay_follow",
    "graphic_text_lock",
    "person_subject",
    "safe_center",
    "split_screen",
    "static_composition",
]


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    model = Path(args.model)
    manifest = Path(args.manifest)
    if not model.is_file() or model.stat().st_size < 1024 * 1024:
        raise SystemExit(f"model artifact is missing or too small: {model}")
    if not manifest.is_file():
        raise SystemExit(f"model manifest is missing: {manifest}")
    payload = json.loads(manifest.read_text())
    actual = digest(model)
    if payload.get("sha256") != actual:
        raise SystemExit(
            f"model sha mismatch: expected={payload.get('sha256')} actual={actual}"
        )
    if payload.get("bytes") != model.stat().st_size:
        raise SystemExit(
            f"model size mismatch: expected={payload.get('bytes')} actual={model.stat().st_size}"
        )
    if payload.get("class_names") != EXPECTED:
        raise SystemExit(f"invalid class contract: {payload.get('class_names')!r}")
    print(json.dumps({"status": "PASS", "model": str(model), "sha256": actual}))


if __name__ == "__main__":
    main()
