from __future__ import annotations

from typing import Iterator, List

from loguru import logger

from common_ml.tagging.messages import Error, Message, Progress, Tag
from common_ml.tagging.producer import TagMessageProducer

from .config import RuntimeParams, load_configuration
from .contract import build_shot_tag_specs, validate_tag_spec
from .engine import VerticalFocusEngine
from .shots import TagStoreShots
from .video import VideoReader


class VerticalFocusProducer(TagMessageProducer):
    """Raw-video stdin adapter using common-ml for protocol serialization."""

    def __init__(self, runtime: RuntimeParams) -> None:
        self.runtime = runtime
        policy, output, metadata = load_configuration(runtime)
        self.output_config = output
        self.policy_metadata = metadata
        # The vertical_video track is a single, whole-file focus trajectory, so
        # we never consult the tagstore for shot boundaries (nor run local shot
        # detection, nor infer anything from the filename). An empty shot set
        # keeps the engine on its single-shot-per-file path.
        self.shots = TagStoreShots([])
        self.engine = VerticalFocusEngine(
            mode=runtime.mode,
            policy_config=policy,
            object_model=runtime.object_model,
            face_model=runtime.face_model,
            delegate=runtime.delegate,
            detector_backend=runtime.detector_backend,
            yolo_detect_model=runtime.yolo_detect_model,
            yolo_pose_model=runtime.yolo_pose_model,
            yolo_device=runtime.yolo_device,
            yolo_imgsz=runtime.yolo_imgsz,
            yolo_half=runtime.yolo_half,
            yolo_end2end=runtime.yolo_end2end,
            shots=self.shots,
            progress_log_interval_seconds=runtime.progress_log_interval_seconds,
            debug_jsonl_path=runtime.debug_jsonl_path,
        )
        self.content_offset_ms = int(runtime.initial_content_offset_ms)

    def produce(self, files: List[str]) -> Iterator[Message]:
        for source_media in files:
            duration_for_offset = 0
            try:
                result = self.engine.process_file(source_media, self.content_offset_ms)
                duration_for_offset = result.duration_ms
                for shot in result.shots:
                    for spec in build_shot_tag_specs(
                        shot,
                        output_config=self.output_config,
                        policy_metadata=self.policy_metadata,
                    ):
                        validate_tag_spec(spec)
                        yield Tag(
                            tag=spec.tag,
                            start_time=spec.start_time,
                            end_time=spec.end_time,
                            source_media=source_media,
                            track=spec.track,
                            frame_info=spec.frame_info,
                            additional_info=spec.additional_info,
                        )
                # Terminal progress is emitted immediately for each stdin item,
                # after all tags attributable to that item have been flushed.
                yield Progress(source_media=source_media)
            except Exception as exc:
                logger.opt(exception=exc).error("vertical focus failed for {}", source_media)
                if duration_for_offset <= 0:
                    duration_for_offset = self._best_effort_duration(source_media)
                yield Error(message=f"{type(exc).__name__}: {exc}", source_media=source_media)
            finally:
                self.content_offset_ms += max(0, int(duration_for_offset))

    def on_completion(self) -> Iterator[Message]:
        self.engine.close()
        yield from ()

    @staticmethod
    def _best_effort_duration(source_media: str) -> int:
        try:
            reader = VideoReader(source_media)
            duration = reader.info.duration_ms
            reader.close()
            return duration
        except Exception:
            return 0
