from __future__ import annotations

import json
import sys
from typing import Any, Dict
import os

from loguru import logger

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from nba_yolo_shot_tagger.config import config_from_params
from nba_yolo_shot_tagger.producer import NbaShotFocusProducer
from nba_yolo_shot_tagger.live import FabricSink, NoopSink
from nba_yolo_shot_tagger.service import ShotFocusService


def _replace_params_argument(params: Dict[str, Any]) -> None:
    encoded = json.dumps(params, separators=(",", ":"), sort_keys=True)
    rewritten = []
    index = 0
    found = False
    while index < len(sys.argv):
        argument = sys.argv[index]
        if argument == "--params":
            rewritten.extend(["--params", encoded])
            index += 2
            found = True
            continue
        if argument.startswith("--params="):
            rewritten.append(f"--params={encoded}")
            index += 1
            found = True
            continue
        rewritten.append(argument)
        index += 1
    if not found:
        rewritten.extend(["--params", encoded])
    sys.argv[:] = rewritten


def main() -> None:
    catch_errors()
    params = get_params()
    if not isinstance(params, dict):
        raise ValueError("--params must decode to a JSON object")

    # The production container is long-lived: one malformed shot must not end
    # the daemon. Respect an explicit caller setting, otherwise default true.
    effective = dict(params)
    effective.setdefault("continue_on_error", True)
    _replace_params_argument(effective)

    config = config_from_params(effective)
    logger.info(
        "NBA YOLO shot tagger starting: {}",
        json.dumps(
            {
                "input_mode": config.input_mode,
                "model_path": config.model_path,
                "device": config.device,
                "imgsz": config.imgsz,
                "inference_fps": config.inference_fps,
                "batch_size": config.batch_size,
                "output_track": config.output_track,
                "continue_on_error": effective["continue_on_error"],
            },
            sort_keys=True,
        ),
    )

    if config.live_data_stream:
        live_q = os.getenv("ELV_CONTENT") or ""
        tok = os.getenv("ELV_TOKEN") or ""

        assert live_q, "container received live_data_stream argument but is missing ELV_CONTENT environment variable"
        assert tok, "container received live_data_stream argument but is missing ELV_AUTH environment variable"

        sink = FabricSink(
            base_url="https://host-76-74-29-13.contentfabric.io",
            live_q=live_q,
            tok=tok,
            data_stream=config.live_data_stream
        )
    else:
        sink = NoopSink()

    service = ShotFocusService(config, sink)

    producer = NbaShotFocusProducer(config, service)

    run_default(producer, batch_limit=1)


if __name__ == "__main__":
    main()
