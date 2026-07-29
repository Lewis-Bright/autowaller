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
    / "bedrock_detector.py"
)
SPEC = importlib.util.spec_from_file_location("bedrock_detector", DETECTOR_PATH)
detector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detector)


def encode(image):
    ok, data = cv2.imencode(".png", image)
    assert ok
    return data.tobytes()


class FakeBedrock:
    def __init__(self, walls):
        self.walls = walls
        self.request = None

    def converse(self, **request):
        self.request = request
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "submit_wall_plan",
                                "input": {
                                    "walls": self.walls,
                                    "summary": "Test perimeter",
                                },
                            }
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 20},
        }


class BedrockDetectorTests(unittest.TestCase):
    def setUp(self):
        self.image = encode(np.full((300, 400, 3), 255, dtype=np.uint8))

    def test_converts_normalized_coordinates_to_scene_coordinates(self):
        client = FakeBedrock(
            [{"c": [100, 200, 900, 200], "confidence": 0.92}]
        )

        result = detector.detect_walls_with_bedrock(
            self.image, 800, 600, bedrock_client=client
        )

        self.assertEqual(result["walls"][0]["c"], [80.0, 120.0, 720.0, 120.0])
        self.assertEqual(result["diagnostics"]["inputTokens"], 100)
        self.assertEqual(
            client.request["toolConfig"]["toolChoice"]["tool"]["name"],
            "submit_wall_plan",
        )
        prompt = client.request["messages"][0]["content"][1]["text"]
        self.assertIn("where that floor ends", prompt)
        self.assertIn("not on the structure's exterior silhouette", prompt)

    def test_rejects_out_of_range_coordinates(self):
        client = FakeBedrock(
            [{"c": [-1, 100, 900, 100], "confidence": 0.9}]
        )

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            detector.detect_walls_with_bedrock(
                self.image, 800, 600, bedrock_client=client
            )

    def test_rejects_excessive_wall_count(self):
        client = FakeBedrock(
            [{"c": [0, 0, 100, 100], "confidence": 0.9}] * 65
        )

        with self.assertRaisesRegex(ValueError, "maximum is 64"):
            detector.detect_walls_with_bedrock(
                self.image, 800, 600, bedrock_client=client
            )

    def test_simplifies_connected_nearly_collinear_segments(self):
        walls = [
            {"c": [0, 0, 100, 2], "type": "wall", "confidence": 0.9},
            {"c": [100, 2, 200, 0], "type": "wall", "confidence": 0.9},
            {"c": [200, 0, 300, 1], "type": "wall", "confidence": 0.9},
        ]

        result = detector._simplify_walls(walls, 1000, 1000)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["c"], [0.0, 0.0, 300.0, 1.0])

    def test_preserves_enough_segments_for_a_curved_boundary(self):
        center = np.array([500.0, 500.0])
        radius = 300.0
        points = [
            center
            + radius
            * np.array(
                [
                    np.cos(2 * np.pi * index / 48),
                    np.sin(2 * np.pi * index / 48),
                ]
            )
            for index in range(48)
        ]
        walls = []
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            walls.append(
                {
                    "c": [*start.tolist(), *end.tolist()],
                    "type": "wall",
                    "confidence": 0.9,
                }
            )

        result = detector._simplify_walls(walls, 1000, 1000)

        self.assertGreaterEqual(len(result), 16)
        self.assertLessEqual(len(result), 24)


if __name__ == "__main__":
    unittest.main()
