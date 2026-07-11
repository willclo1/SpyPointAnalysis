from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

from google.cloud import vision

MIN_CONF_OBJ_ANIMAL = 0.45
MIN_CONF_OBJ_VEHICLE = 0.55
MIN_CONF_OBJ_PERSON = 0.55
MIN_CONF_FALLBACK_ANIMAL_LABEL = 0.60
MIN_CONF_FALLBACK_PERSON_ANYWHERE = 0.70
DEFAULT_GATE_ROI = (0.35, 0.25, 0.98, 0.95)
MIN_GATE_INTERSECTION_RATIO = 0.08

ANIMAL_KEYWORDS = {
    "deer", "hog", "pig", "boar", "coyote", "fox", "bobcat", "raccoon",
    "armadillo", "skunk", "opossum", "possum", "ringtail", "squirrel", "cow",
    "cattle", "bull", "calf", "bovine", "livestock", "horse", "goat", "sheep",
    "donkey", "turkey", "hawk", "owl", "vulture", "heron", "egret", "crane",
    "roadrunner", "animal", "mammal", "bird",
}
VEHICLE_KEYWORDS = {"car", "truck", "vehicle", "van", "suv", "pickup", "motorcycle", "atv", "tractor", "golf cart", "utv"}
PERSON_KEYWORDS = {"person", "human", "man", "woman"}


@dataclass(frozen=True)
class Detected:
    name: str
    score: float
    bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Decision:
    keep: bool
    reason: str
    animals: List[Detected]
    vehicles_at_gate: List[Detected]
    people_at_gate: List[Detected]
    all_objects: List[Detected]


@lru_cache(maxsize=1)
def _client() -> vision.ImageAnnotatorClient:
    return vision.ImageAnnotatorClient()


def _bbox_from_vertices(vertices) -> Tuple[float, float, float, float]:
    xs = [float(getattr(v, "x", 0.0) or 0.0) for v in vertices]
    ys = [float(getattr(v, "y", 0.0) or 0.0) for v in vertices]
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))


def _intersection_ratio(bbox, roi) -> float:
    x0, y0, x1, y1 = bbox
    rx0, ry0, rx1, ry1 = roi
    intersection = max(0.0, min(x1, rx1) - max(x0, rx0)) * max(0.0, min(y1, ry1) - max(y0, ry0))
    object_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return intersection / object_area if object_area else 0.0


def _matches(name: str, keywords: Iterable[str]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return any(re.search(rf"\b{re.escape(keyword)}\b", normalized) for keyword in keywords)


def detect_objects(image_path: str) -> List[Detected]:
    content = Path(image_path).read_bytes()
    response = _client().object_localization(image=vision.Image(content=content))
    if response.error.message:
        raise RuntimeError(f"Google Vision object localization failed: {response.error.message}")

    return [
        Detected(
            name=(obj.name or "").strip().lower(),
            score=float(obj.score),
            bbox=_bbox_from_vertices(obj.bounding_poly.normalized_vertices),
        )
        for obj in response.localized_object_annotations
    ]


def decide_keep(image_path: str, gate_roi=DEFAULT_GATE_ROI) -> Decision:
    objects = detect_objects(image_path)
    animals: List[Detected] = []
    vehicles_at_gate: List[Detected] = []
    people_at_gate: List[Detected] = []

    for detection in objects:
        if detection.score >= MIN_CONF_OBJ_ANIMAL and _matches(detection.name, ANIMAL_KEYWORDS):
            if detection.name != "animal" or detection.score >= MIN_CONF_FALLBACK_ANIMAL_LABEL:
                animals.append(detection)

        if detection.score >= MIN_CONF_OBJ_VEHICLE and _matches(detection.name, VEHICLE_KEYWORDS):
            if _intersection_ratio(detection.bbox, gate_roi) >= MIN_GATE_INTERSECTION_RATIO:
                vehicles_at_gate.append(detection)

        if detection.score >= MIN_CONF_OBJ_PERSON and _matches(detection.name, PERSON_KEYWORDS):
            if _intersection_ratio(detection.bbox, gate_roi) >= MIN_GATE_INTERSECTION_RATIO:
                people_at_gate.append(detection)

    if animals:
        reason = "animal_detected"
    elif vehicles_at_gate or people_at_gate:
        reason = "gate_activity_detected"
    elif any(d.name == "person" and d.score >= MIN_CONF_FALLBACK_PERSON_ANYWHERE for d in objects):
        reason = "person_detected"
    else:
        reason = "no_animal_and_no_gate_activity"

    return Decision(reason != "no_animal_and_no_gate_activity", reason, animals, vehicles_at_gate, people_at_gate, objects)
