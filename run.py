from __future__ import annotations

import json
import sys
from typing import Any, Dict
import os
from urllib.error import HTTPError

from loguru import logger

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from nba_yolo_shot_tagger.config import config_from_params
from nba_yolo_shot_tagger.producer import NbaShotFocusProducer
from nba_yolo_shot_tagger.live import FabricSink, FileSink, NoopSink
from nba_yolo_shot_tagger.service import ShotFocusService
from elv_client_py import ElvClient

def is_live_content(q: str, tok: str) -> bool:
    try:
        client = ElvClient.from_configuration_url(
                        os.environ.get("ELV_CONFIG_URL", "https://main.net955305.contentfabric.io/config"), static_token=tok)
        edge_write_token = client.content_object_metadata(
            object_id=q,
            metadata_subtree="live_recording/status/edge_write_token",
            resolve_links=False,
        )
    except:
        return False

    return isinstance(edge_write_token, str) and edge_write_token.startswith("tqw__")

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

    cont = os.getenv("ELV_CONTENT", None)
    if cont is not None and is_live_content(cont, os.getenv("ELV_TOKEN") or ""):
        logger.info("Detected live content: {}", cont)
        live_q = cont
        tok = os.getenv("ELV_TOKEN") or ""

        assert tok, "container received ELV_CONTENT but missing ELV_TOKEN environment variable"

        sink = FabricSink(
            base_url=config.fabric_live_post,
            live_q=live_q,
            tok=tok,
            data_stream=config.live_data_stream or "vertical"
        )
    else:
        logger.info("Not live, not writing to fabric bin API")
        sink = NoopSink()

    if os.getenv("HARDCODE_FILE_SINK", None) is not None:
        sink = FileSink(os.getenv("HARDCODE_FILE_SINK", "default.bin"))
    
    service = ShotFocusService(config, sink)

    producer = NbaShotFocusProducer(config, service)

    run_default(producer, batch_limit=1)


if __name__ == "__main__":
    main()
