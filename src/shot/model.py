
import cv2
from typing import List, Optional, Iterator
from loguru import logger

from common_ml.tagging.models.av import AVModel
from common_ml.tagging.models.tag_types import Tag

from .transnet import TransNetV2

import torch

class ShotDetector(AVModel):
    shot_types = ["black", "test card"]

    def __init__(
        self, 
        transnet_path: str,
        contiguous: bool
    ):
        if torch.cuda.is_available():
            logger.info("cuda is available, using it")
            device = "cuda"
        else:
            logger.warning("cuda not available, using cpu (still faster than realtime)")
            device = "cpu"

        self.transnet = TransNetV2(transnet_path, device=device)

        self.last_fps = None
        self.abs_curr_start = 0
        self.abs_next_start = 0
        self.contiguous = contiguous
        # the final shot has no closing transition boundary. In contiguous mode it may
        # span the remaining input files, so hold it here and emit it in on_completion().
        self._pending_final: Optional[Tag] = None

    def tag(self, fpath: str) -> List[Tag]:
        logger.debug(f"Running shot detection on {fpath}")

        # get fps
        cap = cv2.VideoCapture(fpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = (frame_count / fps) * 1000
        print(duration_ms)
        cap.release()

        return self.tag_file_given_info(cap, fpath, fps, frame_count, duration_ms)

    def tag_file_given_info(self, fpath: str, fps: float, frame_count:int, duration_ms:int ) -> List[Tag]:
        
        frame_time = 1000 / fps

        if fps != self.last_fps and self.last_fps is not None:
            logger.warning(f"Video fps changed from {self.last_fps} to {fps}")
            self.last_fps = fps

        transition_idx = self.transnet.predict_shots(fpath)

        res = []

        for idx in transition_idx:
            relative_end_ts = (idx / fps) * 1000
            # might be from an earlier segment
            relative_start_ts = self.abs_next_start - self.abs_curr_start

            start_time = int(relative_start_ts)+frame_time
            end_time = int(relative_end_ts)

            if start_time >= end_time:
                # happens with black frames at the beginning
                continue

            res.append(Tag(
                tag="",
                source_media=fpath,
                start_time=int(relative_start_ts+frame_time),
                end_time=int(relative_end_ts),
            ))

            self.abs_next_start = relative_end_ts + self.abs_curr_start

        # The segment after the last transition boundary has no closing boundary,
        # so it is never emitted by the loop above. Close it here.
        relative_start_ts = self.abs_next_start - self.abs_curr_start
        final_start = int(relative_start_ts + frame_time)
        final_end = int(duration_ms)
        final_shot = None
        if final_start < final_end:
            final_shot = Tag(
                tag="",
                source_media=fpath,
                start_time=final_start,
                end_time=final_end,
            )

        if self.contiguous:
            # The final shot may continue into a later file, so defer it to on_completion();
            # if a subsequent file closes it, this pending shot is overwritten.
            self._pending_final = final_shot
            self.abs_curr_start += duration_ms
        else:
            # Each file is independent, so its trailing shot is complete and should be emitted now.
            if final_shot is not None:
                res.append(final_shot)
            self.abs_next_start = 0
            self.abs_curr_start = 0

        return res

    def on_completion(self) -> Iterator[Tag]:
        # Emit the final shot of a contiguous stream, which spans to the end of the
        # last file and therefore cannot be closed inside any single tag() call.
        if self._pending_final is not None:
            yield self._pending_final
            self._pending_final = None
