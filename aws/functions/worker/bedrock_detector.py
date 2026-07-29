import os

import boto3
import cv2
import numpy as np


MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
)
MAX_IMAGE_DIMENSION = 1568
MAX_WALLS = 64
GRID_DIVISIONS = 10

SYSTEM_PROMPT = """You convert top-down tabletop battle maps into Foundry VTT walls.
Draw boundaries where the map artwork visibly changes from walkable floor into a solid
physical structure such as a building wall, cave wall, cliff, fence, or tent wall.
The rectangular image/canvas edge is NOT a wall. Never return the image corners or image
border unless a visible physical structure genuinely follows that border. Look inward
from the canvas edge to find the illustrated play area's real structural boundary.
Ignore furniture, tables, chairs, rugs, fires, shadows, artwork, ropes, plants, props,
texture lines, and decorative details. Prefer the perimeter of the traversable play area
and genuine internal room dividers. Preserve visible doors, gates, cave mouths, tent
openings, and entrances as open gaps. Approximate curves with enough connected straight
segments to follow the visible boundary closely. A round or irregular enclosed structure
will normally need 12 to 24 segments; a mostly rectangular structure may need far fewer.
Return the smallest accurate outline a human GM would reasonably draw, but prioritize
matching the artwork over minimizing the segment count. Do not reduce the count so far
that segments cut across walkable floor or visibly drift away from a curved wall.
Boundary segments should connect
endpoint-to-endpoint except where an intentional entrance gap exists. Never trace both
edges of a thick wall. A continuous floor, path, carpet, stairs, or corridor crossing
through the structural boundary indicates an opening even when there is no conventional
door. Tent flaps and open entry corridors must remain passable. Do not infer an opening
merely because a structure is cropped by or approaches the image edge. Declare only
clearly visible, high-confidence openings."""

WALL_PLAN_TOOL = {
    "toolSpec": {
        "name": "submit_wall_plan",
        "description": (
            "Submit the minimal structural wall plan. Coordinates are normalized to "
            "the image: x=0 is left, x=1000 is right, y=0 is top, y=1000 is bottom."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "object",
                        "description": (
                            "Concise visible observations made before choosing coordinates."
                        ),
                        "properties": {
                            "primaryStructure": {
                                "type": "string",
                                "description": (
                                    "The main structure whose boundary needs walls."
                                ),
                            },
                            "boundary": {
                                "type": "string",
                                "description": (
                                    "Where its blocking boundary lies relative to the grid."
                                ),
                            },
                            "openings": {
                                "type": "array",
                                "description": (
                                    "All visible entrances, doors, gates, or deliberate "
                                    "gaps that must remain open."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "c": {
                                            "type": "array",
                                            "description": (
                                                "Exactly [x1,y1,x2,y2] marking the "
                                                "boundary gap from one side to the other, "
                                                "normalized from 0 through 1000."
                                            ),
                                            "items": {"type": "number"},
                                        },
                                        "description": {"type": "string"},
                                        "confidence": {
                                            "type": "number",
                                            "description": (
                                                "Confidence from 0 through 1 that this "
                                                "is a real passable opening."
                                            ),
                                        },
                                    },
                                    "required": ["c", "description", "confidence"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["primaryStructure", "boundary", "openings"],
                        "additionalProperties": False,
                    },
                    "walls": {
                        "type": "array",
                        "description": (
                            "Minimal blocking segments. Keep entrances open and omit "
                            "decorative or furnishing edges."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "c": {
                                    "type": "array",
                                    "description": (
                                        "Exactly [x1,y1,x2,y2], each normalized from "
                                        "0 through 1000."
                                    ),
                                    "items": {"type": "number"},
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": (
                                        "Confidence from 0 through 1 that this is a "
                                        "structural blocking boundary."
                                    ),
                                },
                            },
                            "required": ["c", "confidence"],
                            "additionalProperties": False,
                        },
                    },
                    "summary": {
                        "type": "string",
                        "description": "A short description of the chosen structure.",
                    },
                },
                "required": ["analysis", "walls", "summary"],
                "additionalProperties": False,
            }
        },
    }
}


def _prepare_image(image_bytes):
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    source_height, source_width = image.shape[:2]
    scale = min(1.0, MAX_IMAGE_DIMENSION / max(source_width, source_height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(source_width * scale), round(source_height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    height, width = image.shape[:2]
    overlay = image.copy()
    grid_color = (255, 255, 0)
    for index in range(1, GRID_DIVISIONS):
        x = round(width * index / GRID_DIVISIONS)
        y = round(height * index / GRID_DIVISIONS)
        cv2.line(overlay, (x, 0), (x, height - 1), grid_color, 1, cv2.LINE_AA)
        cv2.line(overlay, (0, y), (width - 1, y), grid_color, 1, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)

    font_scale = max(0.35, min(width, height) / 1800)
    for index in range(GRID_DIVISIONS + 1):
        value = str(index * 100)
        x = min(width - 45, round(width * index / GRID_DIVISIONS) + 3)
        y = min(height - 7, round(height * index / GRID_DIVISIONS) + 15)
        cv2.putText(
            image,
            value,
            (x, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            value,
            (3, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("The map could not be prepared for visual analysis.")
    return jpeg.tobytes(), {
        "sourceImage": {"width": source_width, "height": source_height},
        "analysisImage": {"width": image.shape[1], "height": image.shape[0]},
    }


def _tool_input(response):
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    for block in blocks:
        tool_use = block.get("toolUse")
        if tool_use and tool_use.get("name") == "submit_wall_plan":
            return tool_use.get("input")
    raise ValueError(
        f"The vision model did not return a wall plan: {repr(blocks)[:500]}"
    )


def _validated_walls(plan, scene_width, scene_height):
    raw_walls = plan.get("walls") if isinstance(plan, dict) else None
    if not isinstance(raw_walls, list):
        raise ValueError("The vision model returned invalid walls.")
    if len(raw_walls) > MAX_WALLS:
        raise ValueError(
            f"The vision model returned {len(raw_walls)} walls; maximum is {MAX_WALLS}."
        )

    walls = []
    for raw_wall in raw_walls:
        coordinates = raw_wall.get("c") if isinstance(raw_wall, dict) else None
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ValueError(
                f"The vision model returned invalid wall coordinates: "
                f"{repr(raw_walls)[:1500]}"
            )
        values = [float(value) for value in coordinates]
        if any(not np.isfinite(value) or value < 0 or value > 1000 for value in values):
            raise ValueError("The vision model returned out-of-range wall coordinates.")
        confidence = float(raw_wall.get("confidence", 0))
        if not np.isfinite(confidence) or confidence < 0 or confidence > 1:
            raise ValueError("The vision model returned invalid wall confidence.")

        x1, y1, x2, y2 = values
        if np.hypot(x2 - x1, y2 - y1) < 5:
            continue
        walls.append(
            {
                "c": [
                    round(x1 * scene_width / 1000, 2),
                    round(y1 * scene_height / 1000, 2),
                    round(x2 * scene_width / 1000, 2),
                    round(y2 * scene_height / 1000, 2),
                ],
                "type": "wall",
                "confidence": round(confidence, 3),
            }
        )
    return walls


def _validated_openings(plan, scene_width, scene_height):
    analysis = plan.get("analysis") if isinstance(plan, dict) else None
    raw_openings = analysis.get("openings", []) if isinstance(analysis, dict) else []
    openings = []
    for opening in raw_openings:
        confidence = float(opening.get("confidence", 0))
        if not np.isfinite(confidence) or confidence < 0 or confidence > 1:
            raise ValueError("The vision model returned invalid opening confidence.")
        if confidence < 0.8:
            continue
        coordinates = opening.get("c") if isinstance(opening, dict) else None
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ValueError("The vision model returned invalid opening coordinates.")
        values = [float(value) for value in coordinates]
        if any(not np.isfinite(value) or value < 0 or value > 1000 for value in values):
            raise ValueError("The vision model returned out-of-range opening coordinates.")
        x1, y1, x2, y2 = values
        openings.append(
            [
                x1 * scene_width / 1000,
                y1 * scene_height / 1000,
                x2 * scene_width / 1000,
                y2 * scene_height / 1000,
            ]
        )
    return openings


def _point_segment_distance(point, start, end):
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator == 0:
        return float(np.linalg.norm(point - start))
    position = np.clip(float(np.dot(point - start, direction) / denominator), 0, 1)
    return float(np.linalg.norm(point - (start + position * direction)))


def _remove_opening_crossings(walls, openings, scene_width, scene_height):
    tolerance = min(scene_width, scene_height) * 0.06
    filtered = []
    for wall in walls:
        start = np.array(wall["c"][0:2], dtype=float)
        end = np.array(wall["c"][2:4], dtype=float)
        crosses_opening = False
        for opening in openings:
            gap_start = np.array(opening[0:2], dtype=float)
            gap_end = np.array(opening[2:4], dtype=float)
            gap_midpoint = (gap_start + gap_end) / 2
            if _point_segment_distance(gap_midpoint, start, end) <= tolerance:
                crosses_opening = True
                break
        if not crosses_opening:
            filtered.append(wall)
    return filtered


def _simplify_walls(walls, scene_width, scene_height):
    if len(walls) < 3:
        return walls

    join_tolerance = max(2.0, min(scene_width, scene_height) * 0.005)
    components = []
    current = []
    for wall in walls:
        if not current:
            current = [wall]
            continue
        previous_end = np.array(current[-1]["c"][2:4], dtype=float)
        next_start = np.array(wall["c"][0:2], dtype=float)
        if np.linalg.norm(previous_end - next_start) <= join_tolerance:
            current.append(wall)
        else:
            components.append(current)
            current = [wall]
    if current:
        components.append(current)

    simplified = []
    # Preserve useful curvature while still collapsing genuinely straight runs.
    # At 0.5% of the shorter scene edge, a circular outline normally retains
    # roughly 16-24 segments instead of becoming a visibly coarse polygon.
    epsilon = min(scene_width, scene_height) * 0.006
    for component in components:
        if len(component) < 3:
            simplified.extend(component)
            continue
        vertices = [component[0]["c"][0:2]]
        vertices.extend(wall["c"][2:4] for wall in component)
        closed = (
            np.linalg.norm(
                np.array(vertices[0], dtype=float)
                - np.array(vertices[-1], dtype=float)
            )
            <= join_tolerance
        )
        points = np.array(vertices[:-1] if closed else vertices, dtype=np.float32)
        approximated = cv2.approxPolyDP(
            points.reshape((-1, 1, 2)), epsilon, closed
        ).reshape((-1, 2))
        if len(approximated) < (3 if closed else 2):
            simplified.extend(component)
            continue

        confidence = round(
            min(float(wall["confidence"]) for wall in component), 3
        )
        pairs = list(zip(approximated, approximated[1:]))
        if closed:
            pairs.append((approximated[-1], approximated[0]))
        for start, end in pairs:
            simplified.append(
                {
                    "c": [
                        round(float(start[0]), 2),
                        round(float(start[1]), 2),
                        round(float(end[0]), 2),
                        round(float(end[1]), 2),
                    ],
                    "type": "wall",
                    "confidence": confidence,
                }
            )
    return simplified


def detect_walls_with_bedrock(
    image_bytes, scene_width, scene_height, bedrock_client=None
):
    image, image_diagnostics = _prepare_image(image_bytes)
    client = bedrock_client or boto3.client("bedrock-runtime")
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": image}}},
                    {
                        "text": (
                            "Create an accurate, economical Foundry wall plan for this "
                            "battle map. "
                            "Call submit_wall_plan once. Be conservative: an omitted "
                            "decorative edge is better than a false wall. First locate "
                            "the visibly walkable floor, then trace only its real physical "
                            "boundary; do not use the rectangular image boundary. A cyan "
                            "coordinate grid is overlaid every 100 normalized units, with "
                            "x labels along the top and y labels along the left. Use it to "
                            "place accurate connected endpoints. Curved tent, cave, and "
                            "tower walls need enough segments to follow the artwork; do "
                            "not turn them into coarse polygons merely to reduce the "
                            "segment count. No segment may cut across walkable floor. Complete "
                            "the analysis fields first: inspect the entire perimeter and "
                            "list every visible entrance or deliberate gap. The wall "
                            "segments must stop on each side of those openings."
                        )
                    },
                ],
            }
        ],
        toolConfig={
            "tools": [WALL_PLAN_TOOL],
            "toolChoice": {"tool": {"name": "submit_wall_plan"}},
        },
        inferenceConfig={"maxTokens": 3000, "temperature": 0},
    )
    plan = _tool_input(response)
    raw_walls = _validated_walls(plan, scene_width, scene_height)
    openings = _validated_openings(plan, scene_width, scene_height)
    walls_without_openings = _remove_opening_crossings(
        raw_walls, openings, scene_width, scene_height
    )
    walls = _simplify_walls(walls_without_openings, scene_width, scene_height)
    usage = response.get("usage", {})
    return {
        "schemaVersion": 1,
        "scene": {"width": int(scene_width), "height": int(scene_height)},
        "walls": walls,
        "diagnostics": {
            "detector": "bedrock-vision-v1",
            "modelId": MODEL_ID,
            "summary": str(plan.get("summary", ""))[:500],
            "analysis": plan.get("analysis", {}),
            "rawWallCount": len(raw_walls),
            "openingCount": len(openings),
            "simplifiedWallCount": len(walls),
            "inputTokens": int(usage.get("inputTokens", 0)),
            "outputTokens": int(usage.get("outputTokens", 0)),
            **image_diagnostics,
        },
    }
