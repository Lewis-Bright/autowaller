import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np


DETECTOR_PATH = (
    Path(__file__).parents[1] / "aws" / "functions" / "worker" / "detector.py"
)
SPEC = importlib.util.spec_from_file_location("autowaller_detector", DETECTOR_PATH)
detector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detector)


def encode(image):
    ok, data = cv2.imencode(".png", image)
    assert ok
    return data.tobytes()


class DetectorTests(unittest.TestCase):
    def test_detects_a_long_straight_boundary(self):
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        cv2.line(image, (30, 100), (370, 100), (0, 0, 0), thickness=5)

        result = detector.detect_walls(encode(image), 800, 600)

        self.assertEqual(result["schemaVersion"], 1)
        self.assertEqual(result["scene"], {"width": 800, "height": 600})
        self.assertTrue(result["walls"])
        self.assertTrue(
            any(
                abs(wall["c"][1] - 200) < 15 and abs(wall["c"][3] - 200) < 15
                for wall in result["walls"]
            )
        )

    def test_rejects_non_image_bytes(self):
        with self.assertRaisesRegex(ValueError, "readable image"):
            detector.detect_walls(b"not an image", 100, 100)


if __name__ == "__main__":
    unittest.main()
