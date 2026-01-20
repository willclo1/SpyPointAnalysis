"""
species_normalization.py

Central Texas ranch-friendly species normalization.

Goals:
- Convert messy model labels into stable, chart-safe categories.
- Consolidate synonyms / near-duplicates into broad categories.
- Keep useful broad classes (Bird, Snake, etc.) but suppress junk (animal, blank, no cv result).
- Always return something predictable for dashboard grouping.

Usage:
    from species_normalization import normalize_species, normalize_event_type
"""

from __future__ import annotations
from typing import Optional
import re


# -----------------------------
# Helpers
# -----------------------------
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

def _title_case(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())


# -----------------------------
# Junk / artifacts
# -----------------------------
JUNK_VALUES = {
    "", " ", "nan", "none", "null", "nil", "n/a", "na", "-", "--", "?", "unknown", "unidentified",
    "blank", "no cv result", "no_cv_result", "nocvresult", "no result", "no_detection", "no detection",
    "empty", "none detected", "nothing", "nothing detected", "not sure", "unsure",
    "background", "motion blur", "false positive", "false alarm", "trigger", "wind", "grass"
}

# Labels that are too vague to trust as a species (we will fall back to broad class)
VAGUE_LABELS = {
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

# Specifically: user wants "animal" to go to Other
FORCE_OTHER = {"animal", "other animal", "wildlife", "vertebrate", "mammal", "rodent", "canid", "felid"}


# -----------------------------
# Broad classes we DO want
# -----------------------------
BROAD_CLASS_MAP = {
    "bird": "Bird",
    "small bird": "Bird",
    "songbird": "Bird",
    "reptile": "Reptile",
    "snake": "Snake",
    "lizard": "Lizard",
    "frog": "Frog",
    "toad": "Toad",
    "turtle": "Turtle",
    "amphibian": "Amphibian",
    "fish": "Fish",
    "insect": "Insect",
    "bug": "Insect",
    "spider": "Insect",
    "scorpion": "Insect",
}


# -----------------------------
# Canonical mapping (cleaned keys)
# -----------------------------
CANONICAL = {
    # Human / vehicle defensive
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

    # Deer (collapse buck/doe/fawn -> deer)
    "white tailed deer": "White-tailed Deer",
    "white tail deer": "White-tailed Deer",
    "white-tailed deer": "White-tailed Deer",
    "whitetail deer": "White-tailed Deer",
    "whitetail": "White-tailed Deer",
    "deer": "White-tailed Deer",
    "buck": "White-tailed Deer",
    "doe": "White-tailed Deer",
    "fawn": "White-tailed Deer",
    "odocoileus virginianus": "White-tailed Deer",

    "axis deer": "Axis Deer",
    "chital": "Axis Deer",
    "axis": "Axis Deer",
    "fallow deer": "Fallow Deer",
    "sika deer": "Sika Deer",
    "mule deer": "Mule Deer",

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

    # Common nocturnals
    "raccoon": "Raccoon",
    "opossum": "Opossum",
    "possum": "Opossum",
    "skunk": "Skunk",
    "armadillo": "Armadillo",

    # Birds (collapse specific -> broad where useful)
    "common raven": "Raven",
    "raven": "Raven",
    "american crow": "Crow",
    "crow": "Crow",
    "turkey vulture": "Vulture",
    "black vulture": "Vulture",
    "vulture": "Vulture",
    "wild turkey": "Wild Turkey",
    "turkey": "Wild Turkey",
    "mourning dove": "Dove",
    "dove": "Dove",
}


def normalize_event_type(raw: Optional[str]) -> str:
    s = _clean_basic(str(raw or ""))
    if not s or s in JUNK_VALUES:
        return "blank"
    if s in ("person", "human", "people"):
        return "human"
    if s in ("vehicle", "car", "truck", "atv", "utv", "tractor", "suv", "van"):
        return "vehicle"
    if s in ("animal", "wildlife", "bird", "reptile", "snake", "lizard", "frog", "toad", "turtle"):
        return "animal"
    return s


def normalize_species(raw: Optional[str]) -> str:
    """
    Normalize a raw label (species/top1/etc.) into a dashboard-safe category.

    Returns one of:
    - Canonical animal names (White-tailed Deer, Feral Hog, Coyote, ...)
    - Broad classes (Bird, Snake, Lizard, Reptile, Insect, ...)
    - Human / Vehicle
    - Other
    """
    s = _clean_basic(str(raw or ""))

    if not s or s in JUNK_VALUES:
        return "Other"

    if s in FORCE_OTHER:
        return "Other"

    if s in VAGUE_LABELS:
        # try to infer a broad class from keywords below
        pass

    if s in CANONICAL:
        return CANONICAL[s]

    if s in BROAD_CLASS_MAP:
        return BROAD_CLASS_MAP[s]

    # Heuristic consolidations (broad but useful)
    if "white" in s and "tail" in s and "deer" in s:
        return "White-tailed Deer"
    if "deer" in s:
        return "White-tailed Deer"

    if "hog" in s or "boar" in s or "pig" in s:
        return "Feral Hog"

    if "coyote" in s:
        return "Coyote"

    if "bobcat" in s:
        return "Bobcat"

    if "raccoon" in s:
        return "Raccoon"

    if "opossum" in s or "possum" in s:
        return "Opossum"

    if "armadillo" in s:
        return "Armadillo"

    if "vulture" in s:
        return "Vulture"

    if "raven" in s:
        return "Raven"

    if "crow" in s:
        return "Crow"

    if "dove" in s:
        return "Dove"

    if "turkey" in s:
        return "Wild Turkey"

    if "snake" in s:
        return "Snake"

    if "lizard" in s or "gecko" in s or "anole" in s:
        return "Lizard"

    if "frog" in s:
        return "Frog"
    if "toad" in s:
        return "Toad"
    if "turtle" in s:
        return "Turtle"

    if "bird" in s or "sparrow" in s or "blackbird" in s or "grackle" in s:
        return "Bird"

    # Default: keep it human readable, but you can tighten to "Other" if you prefer.
    return _title_case(s)
