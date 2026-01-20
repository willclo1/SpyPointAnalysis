"""
species_normalization.py

Central Texas Ranch (La Grange / Fayette County) species normalization.

Design goals:
- "Presentation ready" categories for a ranch dashboard (clear, consistent, not overly detailed).
- Never leak junk labels into the UI (blank/no cv result/animal/bird/canis species/corvus species/etc).
- Consolidate synonyms/misspellings into canonical labels.
- Provide a simple grouping for charts (Deer, Hogs, Predators, Birds, Livestock, Small Mammals, Reptiles, Domestic, Other).

Usage:
    from species_normalization import normalize_species, normalize_event_type
    species_clean, species_group = normalize_species(raw_label)
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s\-]")

def _clean_basic(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    # SpeciesNet taxonomy: "a;b;c;white_tailed_deer"
    if ";" in s:
        s = s.split(";")[-1].strip()
    s = s.replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _title(s: str) -> str:
    # title-case but keep hyphenated words readable
    return " ".join(part.capitalize() for part in s.split(" "))

# -----------------------------
# Junk / artifact values
# -----------------------------
JUNK_VALUES = {
    "", " ", "nan", "none", "null", "nil", "n/a", "na", "-", "--", "?", "unknown", "unidentified",
    "blank", "no cv result", "no_cv_result", "nocvresult", "no result", "no_detection", "no detection",
    "empty", "none detected", "nothing", "nothing detected", "not sure", "unsure",
    "background", "motion blur", "false positive", "false alarm", "trigger", "wind", "grass",
}

# Too broad to display as a "species" label
BROAD_LABELS = {
    "animal", "other animal", "mammal",
    "bird", "small bird", "songbird",
    "reptile", "amphibian", "fish",
    "rodent", "insect", "arthropod",
    "wildlife", "vertebrate",
    # vague taxonomy placeholders we see in real outputs
    "canis species", "corvus species", "buteo species",
    "hawk species", "owl species", "snake species", "lizard species",
    "duck species", "goose species", "sparrow species", "blackbird species",
}

# -----------------------------
# Canonical label map (broad ranch-friendly)
# keys are pre-cleaned lowercase
# -----------------------------
CANONICAL = {
    # Human / vehicle defensive (if they slip in)
    "human": "Human",
    "person": "Human",
    "people": "Human",

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

    # Deer (TX)
    "white tailed deer": "White-tailed Deer",
    "white tail deer": "White-tailed Deer",
    "whitetail deer": "White-tailed Deer",
    "whitetail": "White-tailed Deer",
    "buck": "White-tailed Deer",
    "doe": "White-tailed Deer",
    "fawn": "White-tailed Deer",
    "odocoileus virginianus": "White-tailed Deer",
    "deer": "White-tailed Deer",

    "axis deer": "Axis Deer",
    "axis": "Axis Deer",
    "chital": "Axis Deer",

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

    "gray fox": "Gray Fox",
    "grey fox": "Gray Fox",
    "red fox": "Red Fox",
    "fox": "Fox",

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
    "nine banded armadillo": "Armadillo",
    "rabbit": "Rabbit",
    "cottontail": "Rabbit",
    "jackrabbit": "Jackrabbit",
    "squirrel": "Squirrel",

    # Livestock / domestic
    "cow": "Cattle",
    "cattle": "Cattle",
    "bull": "Cattle",
    "calf": "Cattle",
    "goat": "Goat",
    "sheep": "Sheep",
    "horse": "Horse",
    "donkey": "Donkey",
    "mule": "Mule",

    "domestic dog": "Domestic Dog",
    "dog": "Domestic Dog",
    "domestic cat": "Domestic Cat",
    "cat": "Domestic Cat",

    # Birds (keep broad but useful)
    "common raven": "Raven",
    "raven": "Raven",
    "american crow": "Crow",
    "crow": "Crow",
    "turkey vulture": "Vulture",
    "black vulture": "Vulture",
    "vulture": "Vulture",
    "red tailed hawk": "Hawk",
    "hawk": "Hawk",
    "owl": "Owl",
    "wild turkey": "Wild Turkey",
    "turkey": "Wild Turkey",
    "dove": "Dove",
    "quail": "Quail",
    "roadrunner": "Roadrunner",

    # Reptiles
    "rattlesnake": "Rattlesnake",
    "western diamondback rattlesnake": "Rattlesnake",
    "cottonmouth": "Cottonmouth",
    "water moccasin": "Cottonmouth",
    "snake": "Snake",
    "lizard": "Lizard",
    "turtle": "Turtle",
    "frog": "Frog",
    "toad": "Toad",
}

# Canonical -> chart grouping
GROUPS = {
    # Deer
    "White-tailed Deer": "Deer",
    "Axis Deer": "Deer",
    "Fallow Deer": "Deer",
    "Sika Deer": "Deer",
    "Mule Deer": "Deer",

    # Hogs
    "Feral Hog": "Hogs",

    # Predators
    "Coyote": "Predators",
    "Fox": "Predators",
    "Gray Fox": "Predators",
    "Red Fox": "Predators",
    "Bobcat": "Predators",
    "Mountain Lion": "Predators",

    # Small mammals / small game
    "Raccoon": "Small Mammals",
    "Opossum": "Small Mammals",
    "Skunk": "Small Mammals",
    "Armadillo": "Small Mammals",
    "Rabbit": "Small Game",
    "Jackrabbit": "Small Game",
    "Squirrel": "Small Game",

    # Birds
    "Raven": "Birds",
    "Crow": "Birds",
    "Vulture": "Birds",
    "Hawk": "Birds",
    "Owl": "Birds",
    "Wild Turkey": "Birds",
    "Dove": "Birds",
    "Quail": "Birds",
    "Roadrunner": "Birds",

    # Reptiles
    "Rattlesnake": "Reptiles",
    "Cottonmouth": "Reptiles",
    "Snake": "Reptiles",
    "Lizard": "Reptiles",
    "Turtle": "Reptiles",
    "Frog": "Reptiles",
    "Toad": "Reptiles",

    # Livestock
    "Cattle": "Livestock",
    "Goat": "Livestock",
    "Sheep": "Livestock",
    "Horse": "Livestock",
    "Donkey": "Livestock",
    "Mule": "Livestock",

    # Domestic
    "Domestic Dog": "Domestic",
    "Domestic Cat": "Domestic",

    # Defensive
    "Human": "Human",
    "Vehicle": "Vehicle",
    "Unknown": "Other",
    "Other": "Other",
}

def normalize_event_type(raw: Optional[str]) -> str:
    s = _clean_basic(str(raw or ""))
    if not s or s in JUNK_VALUES:
        return "blank"
    if s in ("person", "people", "human"):
        return "human"
    if s in ("vehicle", "car", "truck", "atv", "utv", "tractor", "suv", "van"):
        return "vehicle"
    if s in ("animal", "wildlife", "mammal", "bird", "reptile", "amphibian"):
        return "animal"
    return s

def normalize_species(raw: Optional[str]) -> Tuple[str, str]:
    """
    Returns (species_clean, species_group)

    - species_clean: canonical ranch-friendly label (e.g., "White-tailed Deer")
    - species_group: broad grouping for charts (e.g., "Deer")
    """
    s = _clean_basic(str(raw or ""))

    if not s or s in JUNK_VALUES or s in BROAD_LABELS:
        return ("Unknown", "Other")

    # direct map
    if s in CANONICAL:
        c = CANONICAL[s]
        return (c, GROUPS.get(c, "Other"))

    # heuristics: keep broad, not subspecies
    if "white" in s and "tail" in s and "deer" in s:
        c = "White-tailed Deer"
        return (c, GROUPS[c])

    if "deer" in s:
        # unknown deer/exotic - keep as "Deer" without over-detail
        return ("Deer (Other)", "Deer")

    if any(tok in s for tok in ("hog", "boar", "pig")):
        c = "Feral Hog"
        return (c, GROUPS[c])

    if "coyote" in s:
        c = "Coyote"
        return (c, GROUPS[c])

    if "bobcat" in s:
        c = "Bobcat"
        return (c, GROUPS[c])

    if "raccoon" in s:
        c = "Raccoon"
        return (c, GROUPS[c])

    if "opossum" in s or "possum" in s:
        c = "Opossum"
        return (c, GROUPS[c])

    if "armadillo" in s:
        c = "Armadillo"
        return (c, GROUPS[c])

    if "vulture" in s:
        c = "Vulture"
        return (c, GROUPS[c])

    if "raven" in s:
        c = "Raven"
        return (c, GROUPS[c])

    if "crow" in s:
        c = "Crow"
        return (c, GROUPS[c])

    if "hawk" in s:
        c = "Hawk"
        return (c, GROUPS[c])

    if "owl" in s:
        c = "Owl"
        return (c, GROUPS[c])

    if "turkey" in s:
        c = "Wild Turkey"
        return (c, GROUPS[c])

    if "snake" in s:
        # consolidate to safe broad label
        if "rattle" in s or "diamond" in s:
            c = "Rattlesnake"
        else:
            c = "Snake"
        return (c, GROUPS[c])

    # default: don't explode the dashboard with weird one-offs
    return ("Other", "Other")
