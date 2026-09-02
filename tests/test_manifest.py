import json
import tempfile
import unittest
from pathlib import Path

from nba_yolo_shot_tagger.service import ShotManifest


class ShotManifestTests(unittest.TestCase):
    def test_mapping_by_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shots.json"
            path.write_text(
                json.dumps(
                    {
                        "full.mp4": [
                            {"shot_id": "s1", "start_ms": 100, "end_ms": 900}
                        ]
                    }
                )
            )
            intervals = ShotManifest(str(path)).intervals_for("/input/full.mp4")
            self.assertEqual(len(intervals), 1)
            self.assertEqual(intervals[0].shot_id, "s1")
            self.assertEqual(intervals[0].start_ms, 100)

    def test_invalid_interval_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shots.json"
            path.write_text(json.dumps([{"start_ms": "0", "end_ms": 10}]))
            with self.assertRaisesRegex(ValueError, "start_ms"):
                ShotManifest(str(path)).intervals_for("/input/full.mp4")


if __name__ == "__main__":
    unittest.main()
