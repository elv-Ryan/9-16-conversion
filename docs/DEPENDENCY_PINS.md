# Dependency and protocol pins

This branch was prepared against the supplied September 2026 Eluvio source snapshots.

- `common-ml`: `05045caef7753ce6b1cc1a3333de92a394e46dff` (the supplied `common-ml-main (4).zip` key protocol files match this commit byte-for-byte by Git blob SHA)
- `buildscripts`: update the existing submodule to `a2f6cb2b7cd021e1d21fad49dfcfa5ce6c1e79f7`
- `elv-ml` protocol/docs reference: `7e4e40942e14540fa16d42952e2b7118174ceef0`

The strict Tagger wire protocol requires `source_media` on tags and terminal `progress`. The supplied Spider-Verse recovered JSONL omits `source_media` and places its metadata progress row first, so literal offline compatibility is provided by `scripts/export_spiderverse_format.py`; production wire output remains protocol-compliant.
