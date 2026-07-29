import os

import boto3
import cv2
import numpy as np


MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
)
MAX_IMAGE_DIMENSION = 1568
MAX_CANDIDATES = 32
MAX_WALLS = 128

SYSTEM_PROMPT = """You select real structural walls from OpenCV edge candidates on
top-down tabletop battle maps. You receive the original map and a second copy with
candidate contours drawn in distinct colours and labelled C01, C02, and so on.

Keep only candidates that correspond to the floor-facing inner boundary of a building,
tent, cave, cliff, fence, or other movement-blocking structure. Reject furniture,
carpets, shadows, decorations, texture, seams, exterior silhouettes, and the image
border. When both inner and outer edges of a thick structure are candidates, select only
the inner edge adjoining the walkable area. Preserve doors, gates, cave mouths, tent
entrances, and other visibly passable gaps. OpenCV owns the geometry: select candidate
IDs rather than inventing replacement wall coordinates."""

SELECTION_TOOL = {
    "toolSpec": {
        "name": "select_wall_candidates",
        "description": "Select structural OpenCV candidates and mark passable openings.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "selected": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "confidence": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["id", "confidence", "reason"],
                            "additionalProperties": False,
                        },
                    },
                    "openings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "c": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": (
                                        "Exactly [x1,y1,x2,y2] across the passable gap, "
                                        "normalized from 0 through 1000."
                                    ),
                                },
                                "confidence": {"type": "number"},
                                "description": {"type": "string"},
                            },
                            "required": ["c", "confidence", "description"],
                            "additionalProperties": False,
                        },
                    },
                    "summary": {"type": "string"},
                },
                "required": ["selected", "openings", "summary"],
                "additionalProperties": False,
            }
        },
    }
}


def _decode_and_resize(image_bytes):
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    source = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("The uploaded file is not a readable image.")
    source_height, source_width = source.shape[:2]
    scale = min(1.0, MAX_IMAGE_DIMENSION / max(source_width, source_height))
    image = source
    if scale < 1.0:
        image = cv2.resize(
            source,
            (round(source_width * scale), round(source_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image, source_width, source_height


def _candidate_contours(image):
    height, width = image.shape[:2]
    shorter = min(width, height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    median = float(np.median(gray))
    lower = max(25, int(median * 0.55))
    upper = min(220, max(lower + 40, int(median * 1.35)))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    candidates = []
    minimum_length = shorter * 0.10
    epsilon = shorter * 0.004
    for contour in contours:
        length = float(cv2.arcLength(contour, False))
        if length < minimum_length:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) < shorter * 0.06:
            continue
        closed = np.linalg.norm(contour[0, 0] - contour[-1, 0]) <= shorter * 0.02
        approximated = cv2.approxPolyDP(contour, epsilon, closed).reshape((-1, 2))
        if len(approximated) < (3 if closed else 2):
            continue
        candidates.append(
            {
                "length": length,
                "closed": bool(closed),
                "points": approximated.astype(float),
                "bbox": (x, y, w, h),
            }
        )

    candidates.sort(key=lambda candidate: candidate["length"], reverse=True)
    accepted = []
    for candidate in candidates:
        x, y, w, h = candidate["bbox"]
        duplicate = False
        for other in accepted:
            ox, oy, ow, oh = other["bbox"]
            intersection = max(0, min(x + w, ox + ow) - max(x, ox)) * max(
                0, min(y + h, oy + oh) - max(y, oy)
            )
            union = w * h + ow * oh - intersection
            if union and intersection / union > 0.94:
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)
        if len(accepted) == MAX_CANDIDATES:
            break

    for index, candidate in enumerate(accepted, 1):
        candidate["id"] = f"C{index:02d}"
    return accepted


def _annotated_image(image, candidates):
    overlay = image.copy()
    palette = [
        (0, 255, 255),
        (255, 80, 80),
        (80, 255, 80),
        (255, 80, 255),
        (80, 180, 255),
        (255, 255, 80),
    ]
    font_scale = max(0.45, min(image.shape[:2]) / 1400)
    for index, candidate in enumerate(candidates):
        points = candidate["points"].round().astype(np.int32).reshape((-1, 1, 2))
        color = palette[index % len(palette)]
        cv2.polylines(overlay, [points], candidate["closed"], color, 3, cv2.LINE_AA)
        anchor = tuple(points[0, 0])
        cv2.putText(
            overlay,
            candidate["id"],
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            candidate["id"],
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def _jpeg(image):
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("The analysis image could not be encoded.")
    return data.tobytes()


def _tool_input(response):
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    for block in blocks:
        tool_use = block.get("toolUse")
        if tool_use and tool_use.get("name") == "select_wall_candidates":
            return tool_use.get("input")
    raise ValueError("The vision model did not select wall candidates.")


def _point_segment_distance(point, start, end):
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator == 0:
        return float(np.linalg.norm(point - start))
    position = np.clip(float(np.dot(point - start, direction) / denominator), 0, 1)
    return float(np.linalg.norm(point - (start + position * direction)))


def _selected_walls(plan, candidates, scene_width, scene_height, image_shape):
    selected = plan.get("selected", []) if isinstance(plan, dict) else []
    lookup = {candidate["id"]: candidate for candidate in candidates}
    chosen = []
    for item in selected:
        candidate_id = item.get("id") if isinstance(item, dict) else None
        if candidate_id not in lookup:
            continue
        confidence = float(item.get("confidence", 0))
        if not np.isfinite(confidence) or confidence < 0.5 or confidence > 1:
            continue
        candidate = lookup[candidate_id]
        points = candidate["points"]
        pairs = list(zip(points, points[1:]))
        if candidate["closed"]:
            pairs.append((points[-1], points[0]))
        for start, end in pairs:
            chosen.append((start, end, confidence, candidate_id))

    image_height, image_width = image_shape[:2]
    opening_midpoints = []
    for opening in plan.get("openings", []):
        if float(opening.get("confidence", 0)) < 0.8:
            continue
        coordinates = opening.get("c")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            continue
        values = np.array([float(value) for value in coordinates])
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > 1000):
            continue
        opening_midpoints.append(
            np.array(
                [
                    (values[0] + values[2]) * image_width / 2000,
                    (values[1] + values[3]) * image_height / 2000,
                ]
            )
        )

    tolerance = min(image_width, image_height) * 0.035
    walls = []
    for start, end, confidence, candidate_id in chosen:
        if any(
            _point_segment_distance(midpoint, start, end) <= tolerance
            for midpoint in opening_midpoints
        ):
            continue
        walls.append(
            {
                "c": [
                    round(float(start[0]) * scene_width / image_width, 2),
                    round(float(start[1]) * scene_height / image_height, 2),
                    round(float(end[0]) * scene_width / image_width, 2),
                    round(float(end[1]) * scene_height / image_height, 2),
                ],
                "type": "wall",
                "confidence": round(confidence, 3),
                "candidateId": candidate_id,
            }
        )
    if len(walls) > MAX_WALLS:
        raise ValueError(f"Hybrid selection produced {len(walls)} walls; maximum is {MAX_WALLS}.")
    return walls


def detect_walls_hybrid(image_bytes, scene_width, scene_height, bedrock_client=None):
    image, source_width, source_height = _decode_and_resize(image_bytes)
    candidates = _candidate_contours(image)
    if not candidates:
        raise ValueError("OpenCV found no plausible wall candidates.")
    annotated = _annotated_image(image, candidates)
    client = bedrock_client or boto3.client("bedrock-runtime")
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": _jpeg(image)}}},
                    {"image": {"format": "jpeg", "source": {"bytes": _jpeg(annotated)}}},
                    {
                        "text": (
                            "The first image is the original map. The second contains "
                            "colour-coded OpenCV candidates. Select only candidates on real "
                            "floor-facing structural boundaries. Prefer the inner edge of "
                            "thick walls and list every visible passable opening."
                        )
                    },
                ],
            }
        ],
        toolConfig={
            "tools": [SELECTION_TOOL],
            "toolChoice": {"tool": {"name": "select_wall_candidates"}},
        },
        inferenceConfig={"maxTokens": 2500, "temperature": 0},
    )
    plan = _tool_input(response)
    walls = _selected_walls(plan, candidates, scene_width, scene_height, image.shape)
    usage = response.get("usage", {})
    return {
        "schemaVersion": 1,
        "scene": {"width": int(scene_width), "height": int(scene_height)},
        "walls": walls,
        "diagnostics": {
            "detector": "opencv-bedrock-hybrid-v1",
            "modelId": MODEL_ID,
            "candidateCount": len(candidates),
            "selectedCandidateIds": sorted(
                {wall["candidateId"] for wall in walls}
            ),
            "summary": str(plan.get("summary", ""))[:500],
            "inputTokens": int(usage.get("inputTokens", 0)),
            "outputTokens": int(usage.get("outputTokens", 0)),
            "sourceImage": {"width": source_width, "height": source_height},
            "analysisImage": {"width": image.shape[1], "height": image.shape[0]},
        },
    }
