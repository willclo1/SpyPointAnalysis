from dataclasses import dataclass
from typing import List, Tuple

from google.cloud import vision

# -----------------------
# TUNABLE SETTINGS
# -----------------------
MIN_CONF_OBJ_ANIMAL = 0.45
MIN_CONF_OBJ_VEHICLE = 0.55
MIN_CONF_OBJ_PERSON = 0.55

# Broad fallback thresholds (when Vision only says "animal" or misses gate ROI)
MIN_CONF_FALLBACK_ANIMAL_LABEL = 0.60   # for object name == "animal"
MIN_CONF_FALLBACK_PERSON_ANYWHERE = 0.70

# Gate region of interest (ROI) in normalized image coords (0..1): (x0, y0, x1, y1)
# Slightly widened to better cover the gate/right-side activity in your sample images.
DEFAULT_GATE_ROI = (0.35, 0.25, 0.98, 0.95)
MIN_GATE_IOU = 0.005

# -----------------------
# La Grange / Fayette County, TX tuned keywords
# -----------------------
ANIMAL_KEYWORDS = {
    "deer", "hog", "pig", "boar",
    "coyote", "fox", "bobcat",
    "raccoon", "armadillo",
    "skunk", "opossum", "possum",
    "ringtail", "squirrel",
    "cow", "cattle", "bull", "calf", "bovine", "livestock",
    "horse", "goat", "sheep", "donkey",
    "turkey", "hawk", "owl", "vulture",
    "heron", "egret", "crane", "roadrunner",
}

VEHICLE_KEYWORDS = {
    "car", "truck", "vehicle", "van", "suv", "pickup",
    "motorcycle", "atv", "tractor", "golf cart", "utv"
}
PERSON_KEYWORDS = {"person", "human", "man", "woman"}


@dataclass
class Detected:
    name: str
    score: float
    bbox: Tuple[float, float, float, float]  # normalized (x0,y0,x1,y1)


@dataclass
class Decision:
    keep: bool
    reason: str
    animals: List[Detected]
    vehicles_at_gate: List[Detected]
    people_at_gate: List[Detected]
    all_objects: List[Detected]


def _bbox_from_vertices(vertices) -> Tuple[float, float, float, float]:
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)

    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _overlaps_gate(bbox, gate_roi) -> bool:
    return _iou(bbox, gate_roi) >= MIN_GATE_IOU


def detect_objects(image_path: str) -> List[Detected]:
    """
    Uses Google Cloud Vision Object Localization.
    Returns list of detected objects with normalized bboxes.
    """
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    resp = client.object_localization(image=image)

    out: List[Detected] = []
    for obj in resp.localized_object_annotations:
        name = (obj.name or "").strip().lower()
        score = float(obj.score)
        bbox = _bbox_from_vertices(obj.bounding_poly.normalized_vertices)
        out.append(Detected(name=name, score=score, bbox=bbox))
    return out


def decide_keep(image_path: str, gate_roi=DEFAULT_GATE_ROI) -> Decision:
    """
    KEEP if:
      - any TX-relevant animal keyword detected (anywhere) above MIN_CONF_OBJ_ANIMAL, OR
      - Vision labels object as 'animal' above MIN_CONF_FALLBACK_ANIMAL_LABEL, OR
      - vehicle/person overlaps gate ROI above threshold, OR
      - fallback: person detected anywhere at high confidence (helps if ROI misses)
    Otherwise DISCARD.
    """
    objs = detect_objects(image_path)

    animals: List[Detected] = []
    vehicles_at_gate: List[Detected] = []
    people_at_gate: List[Detected] = []

    # Object localization based detections
    for d in objs:
        n = d.name

        # Animals anywhere
        if d.score >= MIN_CONF_OBJ_ANIMAL and any(k in n for k in ANIMAL_KEYWORDS):
            animals.append(d)

        # Broad "animal" fallback (Vision often uses generic label)
        if n == "animal" and d.score >= MIN_CONF_FALLBACK_ANIMAL_LABEL:
            animals.append(d)

        # Vehicles only count if near gate
        if d.score >= MIN_CONF_OBJ_VEHICLE and any(k in n for k in VEHICLE_KEYWORDS):
            if _overlaps_gate(d.bbox, gate_roi):
                vehicles_at_gate.append(d)

        # People only count if near gate
        if d.score >= MIN_CONF_OBJ_PERSON and any(k in n for k in PERSON_KEYWORDS):
            if _overlaps_gate(d.bbox, gate_roi):
                people_at_gate.append(d)

    # Decide
    if animals:
        return Decision(True, "animal_detected", animals, vehicles_at_gate, people_at_gate, objs)

    if vehicles_at_gate or people_at_gate:
        return Decision(True, "gate_activity_detected", animals, vehicles_at_gate, people_at_gate, objs)

    # Safety fallback: person anywhere (ROI can be tuned later)
    if any(d.name == "person" and d.score >= MIN_CONF_FALLBACK_PERSON_ANYWHERE for d in objs):
        return Decision(True, "person_detected", animals, vehicles_at_gate, people_at_gate, objs)

    return Decision(False, "no_animal_and_no_gate_activity", animals, vehicles_at_gate, people_at_gate, objs)
