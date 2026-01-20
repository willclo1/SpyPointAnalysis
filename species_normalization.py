"""
species_normalization.py

Central Texas ranch normalization (La Grange / Fayette County).

We produce TWO outputs:
- species_clean: a readable animal label (ex: "Raven", "White-tailed Deer")
- species_group: a broader bucket (ex: "Birds", "Deer", "Predators", "Other")

Design goals:
- Stable labels for charts (not taxonomy strings, not model artifacts)
- Avoid "bird/animal/corvus species" showing up as species
- Prefer useful broad-ish IDs a rancher cares about (Deer/Hogs/Coyote/Raven/etc.)
- Domestic dog can be treated as "Other" (configurable)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import re


# -----------------------------
# Tuning knobs
# -----------------------------
# If you want dogs to never appear as a species in the dashboard:
HIDE_DOMESTIC_DOG = True


# -----------------------------
# Basic cleaning
# -----------------------------
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s\-']")

def clean_label(raw: Optional[str]) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    # SpeciesNet taxonomy-style strings: "a;b;c;white_tailed_deer"
    if ";" in s:
        s = s.split(";")[-1].strip()
    s = s.replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def title(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())


# -----------------------------
# Junk + broad outputs we never want as species
# -----------------------------
JUNK = {
    "", "nan", "none", "null", "n/a", "na", "-", "--", "?", "unknown",
    "blank", "no cv result", "no result", "no detection", "none detected",
    "background", "false positive", "false alarm", "trigger",
}

# Too broad for species_clean (these should become Other unless we find a better candidate)
BROAD = {
    "animal", "wildlife", "mammal", "bird", "reptile", "amphibian", "fish",
    "rodent", "canid", "felid", "insect", "arthropod",
    "corvus species", "canis species", "vulpes species", "buteo species",
    "hawk species", "owl species", "snake species", "lizard species",
    "duck species", "goose species", "sparrow species", "blackbird species",
    "dove species", "pigeon species",
}


# -----------------------------
# Canonical species + groups
# (keys must be CLEANED via clean_label)
# -----------------------------
@dataclass(frozen=True)
class Canon:
    name: str
    group: str

CANON = {
    # Deer
    "white tailed deer": Canon("White-tailed Deer", "Deer"),
    "white tail deer": Canon("White-tailed Deer", "Deer"),
    "whitetail": Canon("White-tailed Deer", "Deer"),
    "whitetail deer": Canon("White-tailed Deer", "Deer"),
    "buck": Canon("White-tailed Deer", "Deer"),
    "doe": Canon("White-tailed Deer", "Deer"),
    "fawn": Canon("White-tailed Deer", "Deer"),
    "odocoileus virginianus": Canon("White-tailed Deer", "Deer"),

    # Hogs
    "feral hog": Canon("Feral Hog", "Hogs"),
    "wild hog": Canon("Feral Hog", "Hogs"),
    "hog": Canon("Feral Hog", "Hogs"),
    "boar": Canon("Feral Hog", "Hogs"),
    "sow": Canon("Feral Hog", "Hogs"),
    "wild pig": Canon("Feral Hog", "Hogs"),
    "sus scrofa": Canon("Feral Hog", "Hogs"),

    # Predators / meso
    "coyote": Canon("Coyote", "Predators"),
    "canis latrans": Canon("Coyote", "Predators"),

    "bobcat": Canon("Bobcat", "Predators"),
    "lynx rufus": Canon("Bobcat", "Predators"),

    "mountain lion": Canon("Mountain Lion", "Predators"),
    "cougar": Canon("Mountain Lion", "Predators"),
    "puma": Canon("Mountain Lion", "Predators"),

    "fox": Canon("Fox", "Predators"),
    "gray fox": Canon("Gray Fox", "Predators"),
    "grey fox": Canon("Gray Fox", "Predators"),
    "red fox": Canon("Red Fox", "Predators"),

    # Common ranch mammals
    "raccoon": Canon("Raccoon", "Small Mammals"),
    "opossum": Canon("Opossum", "Small Mammals"),
    "possum": Canon("Opossum", "Small Mammals"),
    "skunk": Canon("Skunk", "Small Mammals"),
    "armadillo": Canon("Armadillo", "Small Mammals"),
    "rabbit": Canon("Rabbit", "Small Mammals"),
    "cottontail": Canon("Rabbit", "Small Mammals"),
    "squirrel": Canon("Squirrel", "Small Mammals"),

    # Birds (collapse to “Raven/Crow/Turkey/etc.” + group Birds)
    "common raven": Canon("Raven", "Birds"),
    "raven": Canon("Raven", "Birds"),
    "american crow": Canon("Crow", "Birds"),
    "crow": Canon("Crow", "Birds"),
    "turkey vulture": Canon("Vulture", "Birds"),
    "black vulture": Canon("Vulture", "Birds"),
    "wild turkey": Canon("Wild Turkey", "Birds"),
    "turkey": Canon("Wild Turkey", "Birds"),
    "dove": Canon("Dove", "Birds"),
    "mourning dove": Canon("Dove", "Birds"),
    "roadrunner": Canon("Roadrunner", "Birds"),
    "red tailed hawk": Canon("Hawk", "Birds"),
    "hawk": Canon("Hawk", "Birds"),
    "owl": Canon("Owl", "Birds"),
    "egret": Canon("Wading Bird", "Birds"),
    "heron": Canon("Wading Bird", "Birds"),

    # Reptiles
    "rattlesnake": Canon("Rattlesnake", "Reptiles"),
    "western diamondback rattlesnake": Canon("Rattlesnake", "Reptiles"),
    "cottonmouth": Canon("Cottonmouth", "Reptiles"),
    "water moccasin": Canon("Cottonmouth", "Reptiles"),
    "snake": Canon("Snake", "Reptiles"),
    "turtle": Canon("Turtle", "Reptiles"),
    "lizard": Canon("Lizard", "Reptiles"),

    # Domestic / livestock (optional to show)
    "domestic cattle": Canon("Cattle", "Livestock"),
    "cow": Canon("Cattle", "Livestock"),
    "bull": Canon("Cattle", "Livestock"),

    "horse": Canon("Horse", "Livestock"),
    "goat": Canon("Goat", "Livestock"),
    "sheep": Canon("Sheep", "Livestock"),

    "domestic dog": Canon("Domestic Dog", "Domestic"),
    "dog": Canon("Domestic Dog", "Domestic"),
    "domestic cat": Canon("Domestic Cat", "Domestic"),
    "cat": Canon("Domestic Cat", "Domestic"),
}


def is_junk_or_broad(cleaned: str) -> bool:
    return (not cleaned) or (cleaned in JUNK) or (cleaned in BROAD)


def normalize_species(raw: Optional[str]) -> Tuple[str, str]:
    """
    Returns (species_clean, species_group).
    Always returns something safe for charts.
    """
    s = clean_label(raw)

    if not s or s in JUNK:
        return ("Other", "Other")

    if s in BROAD:
        return ("Other", "Other")

    # Canonical mapping
    if s in CANON:
        canon = CANON[s]
        if HIDE_DOMESTIC_DOG and canon.name == "Domestic Dog":
            return ("Other", "Other")
        return (canon.name, canon.group)

    # Heuristic consolidation (keeps things readable)
    if ("white" in s and "tail" in s and "deer" in s) or s.endswith(" deer"):
        return ("White-tailed Deer", "Deer")
    if any(k in s for k in ("hog", "boar", "pig")):
        return ("Feral Hog", "Hogs")
    if "coyote" in s:
        return ("Coyote", "Predators")
    if "raven" in s:
        return ("Raven", "Birds")
    if "crow" in s:
        return ("Crow", "Birds")
    if "vulture" in s:
        return ("Vulture", "Birds")
    if "hawk" in s:
        return ("Hawk", "Birds")
    if "owl" in s:
        return ("Owl", "Birds")
    if "snake" in s:
        if "rattle" in s or "diamond" in s:
            return ("Rattlesnake", "Reptiles")
        return ("Snake", "Reptiles")

    # If we don’t recognize it, keep it readable but bucket it as Other
    return (title(s), "Other")
