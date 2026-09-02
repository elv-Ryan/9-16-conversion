import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_shot_directory_test_controller.py'
spec = importlib.util.spec_from_file_location('joe_controller', SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class JoeShotFilenameParserTests(unittest.TestCase):
    def test_leading_ordinal_is_used(self):
        self.assertEqual(module.numeric_index(Path('000-16_1818.mp4')), 0)
        self.assertEqual(module.numeric_index(Path('099-123_456.mp4')), 99)
        self.assertEqual(module.numeric_index(Path('517-6803180_6821564.mp4')), 517)

if __name__ == '__main__':
    unittest.main()
