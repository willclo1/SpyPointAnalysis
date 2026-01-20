"""
species_normalization.py

Central Texas Ranch (La Grange / Fayette County) normalization.

Design goals:
- Prefer a confident label (>= threshold) from prediction or top1/top2/top3.
- Normalize to BROAD animal classes (no subspecies obsession).
- Prevent junk labels from polluting the UI.
- Treat domestic dogs as "Other" (per your request).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
import re


# -----------------------------
# Config
# -----------------------------
DEFAULT_STRONG_THRESH = 0.60  # your requested threshold

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s\-]")

def _clean_basic(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    # SpeciesNet taxonomy strings: "a;b;c;white_tailed_deer"
    if ";" in s:
        s = s.split(";")[-1].strip()
    s = s.replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


# -----------------------------
# Junk / artifacts / too-broad
# -----------------------------
JUNK_VALUES = {
    "", " ", "nan", "none", "null", "nil", "n/a", "na", "-", "--", "?", "unknown", "unidentified",
    "blank", "no cv result", "no_cv_result", "nocvresult", "no result", "no_detection", "no detection",
    "empty", "none detected", "nothing", "nothing detected", "not sure", "unsure",
    "background", "motion blur", "false positive", "false alarm", "trigger", "wind", "grass",
}

BROAD_CATEGORIES = {
    "animal", "other animal", "mammal", "bird", "reptile", "amphibian", "fish",
    "rodent", "canid", "felid", "insect", "arthropod",
    "wildlife", "vertebrate",
}

# “too vague to be useful”
BANNED_LABELS = {
    "corvus species",
    "canis species",
    "vulpes species",
    "buteo species",
    "hawk species",
    "owl species",
    "snake species",
    "lizard species",
    "frog species",
    "duck species",
    "goose species",
    "sparrow species",
    "blackbird species",
    "gull species",
    "dove species",
    "pigeon species",
}

def _is_bad_label(s: str) -> bool:
    return (not s) or (s in JUNK_VALUES) or (s in BROAD_CATEGORIES) or (s in BANNED_LABELS)


# -----------------------------
# Broad class mapping (what UI shows)
# -----------------------------
# Keys must be cleaned (lowercase, punctuation removed)
# Values are the broad class labels you want in the UI.
BROAD_MAP = {
    # Humans / vehicles (defensive)
    "human": "Human",
    "person": "Human",
    "people": "Human",
    "man": "Human",
    "woman": "Human",
    "child": "Human",

    "vehicle": "Vehicle",
    "car": "Vehicle",
    "truck": "Vehicle",
    "pickup": "Vehicle",
    "pickup truck": "Vehicle",
    "suv": "Vehicle",
    "van": "Vehicle",
    "atv": "Vehicle",
    "utv": "Vehicle",
    "side by side": "Vehicle",
    "side-by-side": "Vehicle",
    "tractor": "Vehicle",
    "ranger": "Vehicle",

    # Deer (collapse all deer-ish variants)
    "white tailed deer": "Deer",
    "white-tailed deer": "Deer",
    "white tail deer": "Deer",
    "whitetail deer": "Deer",
    "whitetailed deer": "Deer",
    "whitetail": "Deer",
    "deer": "Deer",
    "doe": "Deer",
    "buck": "Deer",
    "fawn": "Deer",
    "odocoileus virginianus": "Deer",
    "mule deer": "Deer",
    "odocoileus hemionus": "Deer",
    "axis deer": "Deer",
    "axis": "Deer",
    "chital": "Deer",
    "fallow deer": "Deer",
    "sika deer": "Deer",

    # Hogs
    "feral hog": "Feral Hog",
    "wild hog": "Feral Hog",
    "wild pig": "Feral Hog",
    "feral pig": "Feral Hog",
    "hog": "Feral Hog",
    "boar": "Feral Hog",
    "sow": "Feral Hog",
    "pig": "Feral Hog",
    "sus scrofa": "Feral Hog",

    # Predators
    "coyote": "Coyote",
    "canis latrans": "Coyote",
    "bobcat": "Bobcat",
    "lynx rufus": "Bobcat",
    "mountain lion": "Mountain Lion",
    "cougar": "Mountain Lion",
    "puma": "Mountain Lion",

    # Small mammals
    "raccoon": "Raccoon",
    "opossum": "Opossum",
    "possum": "Opossum",
    "skunk": "Skunk",
    "armadillo": "Armadillo",
    "rabbit": "Rabbit",
    "cottontail": "Rabbit",
    "eastern cottontail": "Rabbit",
    "jackrabbit": "Rabbit",
    "squirrel": "Squirrel",

    # Livestock
    "cow": "Cattle",
    "cattle": "Cattle",
    "bull": "Cattle",
    "calf": "Cattle",
    "goat": "Goat",
    "sheep": "Sheep",
    "horse": "Horse",
    "donkey": "Donkey",
    "mule": "Mule",
    "chicken": "Chicken",

    # Birds (collapse to “Bird” unless you want separate buckets)
    "bird": "Bird",
    "crow": "Bird",
    "raven": "Bird",
    "hawk": "Bird",
    "owl": "Bird",
    "vulture": "Bird",
    "turkey": "Bird",
    "wild turkey": "Bird",
    "dove": "Bird",
    "quail": "Bird",
    "roadrunner": "Bird",
    "woodpecker": "Bird",

    # Reptiles
    "snake": "Snake",
    "rattlesnake": "Snake",
    "cottonmouth": "Snake",
    "water moccasin": "Snake",
    "turtle": "Turtle",
    "lizard": "Lizard",

    # Amphibians (rare)
    "frog": "Frog",
    "toad": "Toad",
}

# Domestic dog handling (you said: disregard dogs)
DOG_ALIASES = {
    "dog", "domestic dog", "house dog", "pet dog", "stray dog", "canis lupus familiaris"
}
CAT_ALIASES = {
    "cat", "domestic cat", "house cat", "pet cat", "felis catus"
}


def normalize_event_type(raw: Optional[str]) -> str:
    s = _clean_basic(str(raw or ""))
    if not s or s in JUNK_VALUES:
        return "blank"
    if s in ("person", "human", "people"):
        return "human"
    if s in ("vehicle", "car", "truck", "atv", "utv", "tractor", "suv", "van"):
        return "vehicle"
    if s in ("animal", "wildlife", "mammal", "bird", "reptile", "amphibian"):
        return "animal"
    return s


def normalize_species_label_to_broad(raw: Optional[str]) -> str:
    """
    Normalize ONE raw label into a BROAD category.
    Returns 'Other' if unknown / junk / dog/cat.
    """
    s = _clean_basic(str(raw or ""))

    if _is_bad_label(s):
        return "Other"

    # drop dogs/cats (you requested “disregard domestic dog columns”)
    if s in DOG_ALIASES:
        return "Other"
    if s in CAT_ALIASES:
        return "Other"

    # direct map
    if s in BROAD_MAP:
        return BROAD_MAP[s]

    # heuristic containment (catches “white tailed deer buck”, etc.)
    if "vehicle" in s or any(w in s for w in ["truck", "pickup", "atv", "utv", "side by side", "tractor", "suv", "van"]):
        return "Vehicle"
    if any(w in s for w in ["human", "person", "man", "woman"]):
        return "Human"
    if ("white" in s and "tail" in s and "deer" in s) or "whitetail" in s:
        return "Deer"
    if "deer" in s:
        return "Deer"
    if any(w in s for w in ["hog", "boar", "pig"]):
        return "Feral Hog"
    if "coyote" in s:
        return "Coyote"
    if "bobcat" in s:
        return "Bobcat"
    if "lion" in s or "cougar" in s or "puma" in s:
        return "Mountain Lion"
    if "raccoon" in s:
        return "Raccoon"
    if "opossum" in s or "possum" in s:
        return "Opossum"
    if "skunk" in s:
        return "Skunk"
    if "armadillo" in s:
        return "Armadillo"
    if "rabbit" in s or "cottontail" in s or "jackrabbit" in s:
        return "Rabbit"
    if "squirrel" in s:
        return "Squirrel"
    if any(w in s for w in ["hawk", "owl", "raven", "crow", "vulture", "turkey", "dove", "quail", "woodpecker", "bird"]):
        return "Bird"
    if "snake" in s or "rattle" in s or "diamondback" in s or "cottonmouth" in s:
        return "Snake"
    if "turtle" in s:
        return "Turtle"
    if "lizard" in s or "gecko" in s or "anole" in s:
        return "Lizard"
    if "frog" in s:
        return "Frog"
    if "toad" in s:
        return "Toad"

    return "Other"


@dataclass(frozen=True)
class Candidate:
    label: str
    conf: Optional[float]


def choose_best_species_label(
    *,
    prediction_label: str,
    prediction_conf: Optional[float],
    topk: Iterable[Tuple[str, Optional[float]]],
    strong_thresh: float = DEFAULT_STRONG_THRESH,
) -> Tuple[str, Optional[float], str]:
    """
    Decide which label to trust BEFORE normalization.
    Returns: (chosen_label, chosen_conf, source) where source is 'prediction' or 'top1'/'top2'/'top3' or 'none'
    """
    # 1) prediction if strong
    if prediction_label:
        pc = prediction_conf
        if pc is not None and pc >= strong_thresh:
            return prediction_label, pc, "prediction"

    # 2) top-k if strong (first that clears threshold)
    for i, (lbl, conf) in enumerate(topk, start=1):
        if not lbl:
            continue
        c = conf
        if c is not None and c >= strong_thresh:
            return lbl, c, f"top{i}"

    # 3) fallback: return best available by confidence even if below threshold
    # (helps avoid empty if everything is low confidence)
    best: Optional[Candidate] = None
    best_src = "none"

    if prediction_label and prediction_conf is not None:
        best = Candidate(prediction_label, prediction_conf)
        best_src = "prediction"

    for i, (lbl, conf) in enumerate(topk, start=1):
        if not lbl or conf is None:
            continue
        cand = Candidate(lbl, conf)
        if best is None or (cand.conf or 0.0) > (best.conf or 0.0):
            best = cand
            best_src = f"top{i}"

    if best is None:
        return "", None, "none"
    return best.label, best.conf, best_src
