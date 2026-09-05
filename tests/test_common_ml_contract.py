from dataclasses import dataclass
import importlib
import sys
import types
import unittest


@dataclass(frozen=True)
class FakeFrameInfo:
    frame_idx: int
    box: dict


@dataclass(frozen=True)
class FakeTag:
    start_time: int
    end_time: int
    tag: str
    source_media: str
    track: str = ""
    additional_info: dict | None = None
    frame_info: object | None = None


@dataclass(frozen=True)
class FakeProgress:
    source_media: str


@dataclass(frozen=True)
class FakeProgressRatio:
    progress: float


@dataclass(frozen=True)
class FakeError:
    message: str
    source_media: str | None = None


class FakeMessage:
    pass


class FakeProducerBase:
    pass


messages = types.ModuleType("common_ml.tagging.messages")
messages.FrameInfo = FakeFrameInfo
messages.Tag = FakeTag
messages.Progress = FakeProgress
messages.ProgressRatio = FakeProgressRatio
messages.Error = FakeError
messages.Message = FakeMessage
producer_module = types.ModuleType("common_ml.tagging.producer")
producer_module.TagMessageProducer = FakeProducerBase
common_ml = types.ModuleType("common_ml")
tagging = types.ModuleType("common_ml.tagging")

sys.modules.setdefault("common_ml", common_ml)
sys.modules.setdefault("common_ml.tagging", tagging)
sys.modules.setdefault("common_ml.tagging.messages", messages)
sys.modules.setdefault("common_ml.tagging.producer", producer_module)

from nba_yolo_shot_tagger.config import RuntimeConfig
from nba_yolo_shot_tagger.contract import tags_for_analysis
from nba_yolo_shot_tagger.producer import NbaShotFocusProducer
from nba_yolo_shot_tagger.types import FocusSample, ModelIdentity, ShotAnalysis


class ContractIntegrationTests(unittest.TestCase):
    def analysis(self):
        return ShotAnalysis(
            source_media="/elv/input/shot.mp4",
            shot_id="shot_000000",
            start_ms=0,
            end_ms=50,
            start_frame=0,
            frame_count=3,
            source_fps=59.94,
            source_width=1920,
            source_height=1080,
            family="gameplay_follow",
            category="gameplay",
            family_confidence=0.8,
            crop_width_norm=0.31640625,
            x_coordinates=(0.5, 0.55, 0.6),
            focus_samples=(
                FocusSample(
                    frame_index=0,
                    timestamp_ms=0,
                    family="gameplay_follow",
                    confidence=0.8,
                    bbox=(0.4, 0.2, 0.6, 0.8),
                    raw_x_center_norm=0.5,
                ),
            ),
            model=ModelIdentity(
                path="/elv/model/best.pt",
                sha256="a" * 64,
                artifact_version="test",
                class_names=(
                    "active_speaker",
                    "gameplay_follow",
                    "graphic_text_lock",
                    "person_subject",
                    "safe_center",
                    "split_screen",
                    "static_composition",
                ),
            ),
        )

    def test_main_tag_matches_eluvio_track_contract(self):
        tags = tags_for_analysis(self.analysis(), RuntimeConfig(verify_model_sha256=False))
        self.assertEqual(len(tags), 1)
        tag = tags[0]
        self.assertEqual(tag.track, "vertical_video")
        self.assertEqual(tag.tag, "gameplay_follow")
        self.assertEqual(tag.source_media, "/elv/input/shot.mp4")
        self.assertEqual(tag.start_time, 0)
        self.assertEqual(tag.end_time, 50)
        self.assertEqual(tag.additional_info["x-coordinates"], [0.5, 0.55, 0.6])
        self.assertEqual(tag.additional_info["x_coordinate_alignment"], "source_frame")
        self.assertEqual(tag.frame_info, {"frame_idx": 0})
        self.assertNotIn("focus_samples", tag.additional_info)
        self.assertEqual(tag.additional_info["focus_sample_fps"], 10.0)

    def test_include_focus_samples_is_opt_in(self):
        tags = tags_for_analysis(
            self.analysis(),
            RuntimeConfig(verify_model_sha256=False, include_focus_samples=True),
        )
        tag = tags[0]
        self.assertIn("focus_samples", tag.additional_info)
        self.assertEqual(len(tag.additional_info["focus_samples"]), 1)
        self.assertEqual(tag.additional_info["focus_sample_fps"], 10.0)

    def test_producer_emits_terminal_progress(self):
        class FakeService:
            def analyze_file(self, source_media):
                return [self_outer.analysis()]

        self_outer = self
        producer = NbaShotFocusProducer.__new__(NbaShotFocusProducer)
        producer.config = RuntimeConfig(verify_model_sha256=False)
        producer.service = FakeService()
        output = list(producer.produce(["/elv/input/shot.mp4"]))
        self.assertFalse(any(isinstance(item, FakeProgressRatio) for item in output))
        self.assertIsInstance(output[-1], FakeProgress)
        self.assertTrue(any(isinstance(item, FakeTag) for item in output))

    def test_progress_ratio_is_opt_in_only(self):
        class FakeService:
            def analyze_file(self, source_media):
                return [self_outer.analysis()]

        self_outer = self
        producer = NbaShotFocusProducer.__new__(NbaShotFocusProducer)
        producer.config = RuntimeConfig(verify_model_sha256=False, emit_progress_ratio=True)
        producer.service = FakeService()
        output = list(producer.produce(["/elv/input/shot.mp4"]))
        self.assertTrue(any(isinstance(item, FakeProgressRatio) for item in output))
        self.assertIsInstance(output[-1], FakeProgress)

    def test_on_completion_emits_the_final_shot(self):
        class FakeService:
            def finalize(self):
                return [self_outer.analysis()]

        self_outer = self
        producer = NbaShotFocusProducer.__new__(NbaShotFocusProducer)
        producer.config = RuntimeConfig(verify_model_sha256=False, input_mode="segment_file")
        producer.service = FakeService()
        output = list(producer.on_completion())
        self.assertTrue(any(isinstance(item, FakeTag) for item in output))

    def test_on_completion_is_quiet_when_nothing_is_open(self):
        class FakeService:
            def finalize(self):
                return []

        producer = NbaShotFocusProducer.__new__(NbaShotFocusProducer)
        producer.config = RuntimeConfig(verify_model_sha256=False)
        producer.service = FakeService()
        self.assertEqual(list(producer.on_completion()), [])

    def test_producer_turns_file_failure_into_source_scoped_error(self):
        class BrokenService:
            def analyze_file(self, source_media):
                raise ValueError("bad shot")

        producer = NbaShotFocusProducer.__new__(NbaShotFocusProducer)
        producer.config = RuntimeConfig(verify_model_sha256=False)
        producer.service = BrokenService()
        output = list(producer.produce(["/elv/input/bad.mp4"]))
        self.assertIsInstance(output[-1], FakeError)
        self.assertEqual(output[-1].source_media, "/elv/input/bad.mp4")
        self.assertIn("bad shot", output[-1].message)


if __name__ == "__main__":
    unittest.main()
