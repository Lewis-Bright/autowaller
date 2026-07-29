import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np


DETECTOR_PATH = (
    Path(__file__).parents[1]
    / "aws"
    / "functions"
    / "worker"
    / "hybrid_detector.py"
)
SPEC = importlib.util.spec_from_file_location("hybrid_detector", DETECTOR_PATH)
detector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detector)


class FakeBedrock:
    def __init__(self):
        self.request = None

    def converse(self, **request):
        self.request = request
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "select_wall_candidates",
                                "input": {
                                    "selected": [
                                        {
                                            "id": "C01",
                                            "confidence": 0.91,
                                            "reason": "Interior structural boundary",
                                        }
                                    ],
                                    "openings": [],
                                    "summary": "Selected test boundary",
                                },
                            }
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 200, "outputTokens": 30},
        }


class HybridDetectorTests(unittest.TestCase):
    def test_extracts_candidates_and_uses_their_exact_geometry(self):
        image = np.full((500, 700, 3), 230, dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (600, 400), (25, 25, 25), 18)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        client = FakeBedrock()

        result = detector.detect_walls_hybrid(
            encoded.tobytes(), 1400, 1000, bedrock_client=client
        )

        self.assertTrue(result["walls"])
        self.assertEqual(result["diagnostics"]["detector"], "opencv-bedrock-hybrid-v1")
        self.assertEqual(result["diagnostics"]["selectedCandidateIds"], ["C01"])
        content = client.request["messages"][0]["content"]
        self.assertEqual(sum("image" in block for block in content), 2)
        self.assertEqual(
            client.request["toolConfig"]["toolChoice"]["tool"]["name"],
            "select_wall_candidates",
        )

    def test_removes_segments_crossing_a_selected_opening(self):
        candidates = [
            {
                "id": "C01",
                "closed": False,
                "points": np.array([[100.0, 100.0], [250.0, 100.0], [400.0, 100.0]]),
            }
        ]
        plan = {
            "selected": [{"id": "C01", "confidence": 0.9}],
            "openings": [{"c": [200, 150, 500, 250], "confidence": 0.95}],
        }

        walls = detector._selected_walls(plan, candidates, 500, 500, (500, 500, 3))

        self.assertEqual(len(walls), 1)
        self.assertEqual(walls[0]["c"], [250.0, 100.0, 400.0, 100.0])


if __name__ == "__main__":
    unittest.main()
