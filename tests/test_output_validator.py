import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class OutputValidatorTests(unittest.TestCase):
    def test_valid_protocol_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            messages = [
                {
                    "type": "tag",
                    "data": {
                        "tag": "gameplay_follow",
                        "start_time": 0,
                        "end_time": 50,
                        "source_media": "/elv/input/shot.mp4",
                        "track": "vertical_video",
                        "additional_info": {
                            "frame_count": 3,
                            "legal_x_center_min": 0.15,
                            "legal_x_center_max": 0.85,
                            "x-coordinates": [0.5, 0.55, 0.6],
                        },
                    },
                },
                {
                    "type": "progress",
                    "data": {"source_media": "/elv/input/shot.mp4"},
                },
            ]
            path.write_text("\n".join(json.dumps(item) for item in messages) + "\n")
            script = Path(__file__).resolve().parents[1] / "scripts" / "validate_tagger_jsonl.py"
            completed = subprocess.run(
                [sys.executable, str(script), str(path), "--expected-source", "/elv/input/shot.mp4"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn('"status": "PASS"', completed.stdout)

    def test_illegal_x_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "tag",
                        "data": {
                            "tag": "gameplay_follow",
                            "start_time": 0,
                            "end_time": 50,
                            "source_media": "/elv/input/shot.mp4",
                            "track": "vertical_video",
                            "additional_info": {
                                "frame_count": 1,
                                "legal_x_center_min": 0.15,
                                "legal_x_center_max": 0.85,
                                "x-coordinates": [0.99],
                            },
                        },
                    }
                )
                + "\n"
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "validate_tagger_jsonl.py"
            completed = subprocess.run(
                [sys.executable, str(script), str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("outside", completed.stdout)


if __name__ == "__main__":
    unittest.main()
