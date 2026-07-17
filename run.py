from __future__ import annotations

import json
import sys
from typing import Any, Dict

from loguru import logger

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from vertical_focus.config import runtime_params_from_dict
from vertical_focus.protocol import VerticalFocusProducer


def _load_params() -> Dict[str, Any]:
    params = get_params()
    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object")
    return dict(params)


def _force_continue_on_error(params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the long-lived tagger alive after one malformed input."""
    effective = dict(params)
    effective["continue_on_error"] = True
    encoded = json.dumps(effective, separators=(",", ":"), sort_keys=True)
    rewritten = []
    found = False
    index = 0
    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg == "--params":
            rewritten.extend(["--params", encoded])
            found = True
            index += 2
            continue
        if arg.startswith("--params="):
            rewritten.append(f"--params={encoded}")
            found = True
            index += 1
            continue
        rewritten.append(arg)
        index += 1
    if not found:
        rewritten.extend(["--params", encoded])
    sys.argv[:] = rewritten
    return effective


if __name__ == "__main__":
    catch_errors()
    params = _force_continue_on_error(_load_params())
    runtime = runtime_params_from_dict(params)
    logger.info(
        "vertical focus runtime: {}",
        json.dumps(
            {
                "mode": runtime.mode,
                "delegate": runtime.delegate,
                "detection_fps": runtime.detection_fps,
                "face_model": runtime.face_model,
                "shot_track": runtime.shot_track,
                "continue_on_error": True,
            },
            sort_keys=True,
        ),
    )
    run_default(VerticalFocusProducer(runtime), batch_limit=1)
