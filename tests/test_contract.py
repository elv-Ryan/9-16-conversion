import unittest

from vertical_focus.contract import build_shot_tag_specs, validate_tag_spec
from vertical_focus.types import FocusBox, FrameDecision, ShotOutput


class ContractTests(unittest.TestCase):
    def test_three_tracks_and_frame_level_x(self):
        box = FocusBox("person:7", "person", 0.91, (0.2, 0.1, 0.4, 0.9))
        shot = ShotOutput(
            shot_id="shot-1",
            start_ms=0,
            end_ms=100,
            start_frame_idx=0,
            crop_width=0.316,
            decisions=[
                FrameDecision(0, 0, 0.45, "primary_person", 0.8, (box,)),
                FrameDecision(1, 42, 0.46, "primary_person", 0.82, (box,)),
                FrameDecision(2, 84, 0.47, "primary_person", 0.84, (box,)),
            ],
        )
        specs = build_shot_tag_specs(
            shot,
            output_config={
                "vertical_track": "vertical_video",
                "focus_track": "focus",
                "bbox_track": "focus_bbox",
                "coordinate_decimals": 6,
                "max_boxes_per_shot": 3,
            },
            policy_metadata={"policy": "movie", "policy_version": 2},
        )
        self.assertEqual({"vertical_video", "focus", "focus_bbox"}, {spec.track for spec in specs})
        vertical = next(spec for spec in specs if spec.track == "vertical_video")
        self.assertEqual(3, len(vertical.additional_info["x-coordinates"]))
        for spec in specs:
            validate_tag_spec(spec)

    def test_safe_center_still_emits_bbox_track(self):
        shot = ShotOutput(
            shot_id="shot-center",
            start_ms=10,
            end_ms=60,
            start_frame_idx=4,
            crop_width=0.316,
            decisions=[FrameDecision(4, 10, 0.5, "safe_center", 0.05, ())],
        )
        specs = build_shot_tag_specs(
            shot,
            output_config={},
            policy_metadata={"policy": "sports"},
        )
        bbox = next(spec for spec in specs if spec.track == "focus_bbox")
        self.assertEqual("safe_center", bbox.tag)
        self.assertEqual(0.0, bbox.frame_info["box"]["y1"])
        self.assertEqual(1.0, bbox.frame_info["box"]["y2"])


if __name__ == "__main__":
    unittest.main()
