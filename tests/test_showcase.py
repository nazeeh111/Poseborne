"""Check actual data flow, missing-data rejection and safe export behavior."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("showcase", ROOT / "scripts/showcase.py")
showcase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(showcase)


class ShowcaseTests(unittest.TestCase):
    def test_seeded_signal_uses_actual_window_and_missing_data_gate(self):
        import numpy as np
        a, b = showcase.signal_trace(), showcase.signal_trace()
        self.assertEqual(a[-1], b[-1])
        self.assertEqual(a[5].shape, (80, 280, 3, 3))
        self.assertEqual(a[5].dtype, np.float16)
        self.assertTrue(np.isfinite(a[5]).all())
        self.assertEqual(a[-1]["rejected_windows"], 1)
        self.assertEqual(a[-1]["valid_windows"], 79)
        self.assertFalse(a[-1]["pose_inference"])
        self.assertFalse(a[-1]["hardware_measurement"])

    def test_existing_output_refused_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "keep.txt"
            marker.write_text("preserve me", encoding="utf-8")
            with self.assertRaises(SystemExit) as error:
                showcase.main(["--out", tmp])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")
            self.assertEqual(list(Path(tmp).iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
