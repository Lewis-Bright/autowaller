import cv2
import numpy as np


def _canonical_line(line):
    x1, y1, x2, y2 = (int(value) for value in line)
    if (x2, y2) < (x1, y1):
        return x2, y2, x1, y1
    return x1, y1, x2, y2


def _length(line):
    x1, y1, x2, y2 = line
    return float(np.hypot(x2 - x1, y2 - y1))


def _angle(line):
    x1, y1, x2, y2 = line
    return float(np.arctan2(y2 - y1, x2 - x1))


def _endpoint_distance(a, b):
    endpoints_a = ((a[0], a[1]), (a[2], a[3]))
    endpoints_b = ((b[0], b[1]), (b[2], b[3]))
    return min(
        float(np.hypot(ax - bx, ay - by))
        for ax, ay in endpoints_a
        for bx, by in endpoints_b
    )


def _deduplicate(lines, angle_tolerance=0.05, endpoint_tolerance=12):
    accepted = []
    for line in sorted(lines, key=_length, reverse=True):
        angle = _angle(line)
        duplicate = any(
            abs(np.sin(angle - _angle(other))) < angle_tolerance
            and _endpoint_distance(line, other) < endpoint_tolerance
            for other in accepted
        )
        if not duplicate:
            accepted.append(line)
    return accepted


def detect_walls(image_bytes, scene_width, scene_height, options=None):
    options = options or {}
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    image_height, image_width = image.shape[:2]
    max_analysis_dimension = int(options.get("maxAnalysisDimension", 2400))
    scale = min(1.0, max_analysis_dimension / max(image_width, image_height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(image_width * scale), round(image_height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )

    minimum_length = max(24, round(min(image.shape[:2]) * 0.025))
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=45,
        minLineLength=minimum_length,
        maxLineGap=12,
    )
    lines = []
    if raw_lines is not None:
        lines = [_canonical_line(line[0]) for line in raw_lines]
    lines = _deduplicate(lines)

    x_scale = scene_width / image.shape[1]
    y_scale = scene_height / image.shape[0]
    walls = []
    for line in lines[:2000]:
        x1, y1, x2, y2 = line
        length_score = min(1.0, _length(line) / (minimum_length * 4))
        walls.append(
            {
                "c": [
                    round(x1 * x_scale, 2),
                    round(y1 * y_scale, 2),
                    round(x2 * x_scale, 2),
                    round(y2 * y_scale, 2),
                ],
                "type": "wall",
                "confidence": round(0.55 + 0.4 * length_score, 3),
            }
        )

    return {
        "schemaVersion": 1,
        "scene": {"width": int(scene_width), "height": int(scene_height)},
        "walls": walls,
        "diagnostics": {
            "detector": "opencv-hough-v1",
            "sourceImage": {"width": image_width, "height": image_height},
            "analysisImage": {"width": image.shape[1], "height": image.shape[0]},
            "candidateCount": len(lines),
        },
    }

