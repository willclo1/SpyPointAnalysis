"""
species_normalization.py

Central Texas Ranch (La Grange / Fayette County) species normalization.

Goal:
- Convert messy model labels into stable, human-friendly categories for charts.
- Ensure that blanks, overly broad terms, and CV artifacts do NOT appear in the UI.
- Consolidate synonyms, scientific names, and common misspellings.

Usage:
    from species_normalization import normalize_species, normalize_vehicle_type, normalize_event_type

    df["species_clean"] = df["species"].apply(normalize_species)
    df["event_type_clean"] = df["event_type"].apply(normalize_event_type)
"""

from __future__ import annotations
from typing import Optional, Tuple
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
    # SpeciesNet often uses taxonomy strings: "a;b;c;white_tailed_deer"
    if ";" in s:
        s = s.split(";")[-1].strip()
    s = s.replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _title_case(s: str) -> str:
    # Keep acronyms and hyphenated words decently readable
    # e.g., "white-tailed deer" -> "White-Tailed Deer" (then we map it anyway)
    return " ".join(w.capitalize() for w in s.split(" "))


# -----------------------------
# Garbage / pipeline artifacts
# -----------------------------
JUNK_VALUES = {
    "", " ", "nan", "none", "null", "nil", "n/a", "na", "-", "--", "?", "unknown", "unidentified",
    "blank", "no cv result", "no_cv_result", "nocvresult", "no result", "no_detection", "no detection",
    "empty", "none detected", "nothing", "nothing detected", "not sure", "unsure",
    "background", "motion blur", "false positive", "false alarm", "trigger", "wind", "grass",
 controlled
}

# Some model outputs that are "too broad" to show as a species
BROAD_CATEGORIES = {
    "animal", "other animal", "mammal", "bird", "reptile", "amphibian", "fish",
    "rodent", "canid", "felid", "insect", "arthropod",
    "wildlife", "vertebrate",
}

# Sometimes taxonomy-y / placeholder-y labels show up
BANNED_LABELS = {
    "corvus species",  # too vague (crow/raven)
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


# -----------------------------
# Canonical mapping
# keys must be pre-cleaned lowercase
# -----------------------------
CANONICAL_SPECIES = {
    # =========================
    # HUMAN / VEHICLE DEFENSIVE
    # =========================
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

    # =========================
    # DEER
    # =========================
    "white tailed deer": "White-tailed Deer",
    "white-tailed deer": "White-tailed Deer",
    "white tail deer": "White-tailed Deer",
    "whitetail deer": "White-tailed Deer",
    "whitetailed deer": "White-tailed Deer",
    "whitetail": "White-tailed Deer",
    "deer": "White-tailed Deer",
    "doe": "White-tailed Deer",
    "buck": "White-tailed Deer",
    "fawn": "White-tailed Deer",
    "odocoileus virginianus": "White-tailed Deer",

    "mule deer": "Mule Deer",
    "odocoileus hemionus": "Mule Deer",

    # Exotics that sometimes appear in TX ranch country
    "axis deer": "Axis Deer",
    "axis": "Axis Deer",
    "chital": "Axis Deer",
    "axis axis": "Axis Deer",

    "fallow deer": "Fallow Deer",
    "dama dama": "Fallow Deer",

    "sika deer": "Sika Deer",
    "cervus nippon": "Sika Deer",

    # =========================
    # HOGS
    # =========================
    "feral hog": "Feral Hog",
    "wild hog": "Feral Hog",
    "wild pig": "Feral Hog",
    "feral pig": "Feral Hog",
    "hog": "Feral Hog",
    "boar": "Feral Hog",
    "sow": "Feral Hog",
    "pig": "Feral Hog",
    "sus scrofa": "Feral Hog",

    # =========================
    # COYOTE / FOX / CANIDS
    # =========================
    "coyote": "Coyote",
    "canis latrans": "Coyote",

    "fox": "Fox",
    "gray fox": "Gray Fox",
    "grey fox": "Gray Fox",
    "urocyon cinereoargenteus": "Gray Fox",
    "red fox": "Red Fox",
    "vulpes vulpes": "Red Fox",

    "dog": "Domestic Dog",
    "domestic dog": "Domestic Dog",
    "canis lupus familiaris": "Domestic Dog",
    "house dog": "Domestic Dog",
    "pet dog": "Domestic Dog",
    "stray dog": "Domestic Dog",

    # =========================
    # CATS
    # =========================
    "cat": "Domestic Cat",
    "domestic cat": "Domestic Cat",
    "house cat": "Domestic Cat",
    "pet cat": "Domestic Cat",
    "felis catus": "Domestic Cat",

    "bobcat": "Bobcat",
    "lynx rufus": "Bobcat",

    "mountain lion": "Mountain Lion",
    "cougar": "Mountain Lion",
    "puma": "Mountain Lion",
    "puma concolor": "Mountain Lion",

    # =========================
    # RACCOON / OPOSSUM / SKUNK
    # =========================
    "raccoon": "Raccoon",
    "procyon lotor": "Raccoon",

    "opossum": "Opossum",
    "possum": "Opossum",
    "virginia opossum": "Opossum",
    "didelphis virginiana": "Opossum",

    "skunk": "Skunk",
    "striped skunk": "Skunk",
    "mephitis mephitis": "Skunk",

    # =========================
    # ARMADILLO
    # =========================
    "armadillo": "Armadillo",
    "nine banded armadillo": "Armadillo",
    "nine-banded armadillo": "Armadillo",
    "dasypus novemcinctus": "Armadillo",

    # =========================
    # RABBITS / HARES
    # =========================
    "rabbit": "Rabbit",
    "cottontail": "Rabbit",
    "eastern cottontail": "Rabbit",
    "sylvilagus floridanus": "Rabbit",
    "jackrabbit": "Jackrabbit",
    "black tailed jackrabbit": "Jackrabbit",
    "black-tailed jackrabbit": "Jackrabbit",
    "lep us californicus": "Jackrabbit",

    # =========================
    # SQUIRRELS / RODENTS
    # =========================
    "squirrel": "Squirrel",
    "fox squirrel": "Fox Squirrel",
    "eastern fox squirrel": "Fox Squirrel",
    "sciurus niger": "Fox Squirrel",
    "gray squirrel": "Gray Squirrel",
    "grey squirrel": "Gray Squirrel",

    "rat": "Rodent",
    "mouse": "Rodent",
    "rodent": "Rodent",

    # =========================
    # LIVESTOCK (common on ranches)
    # =========================
    "cow": "Cattle",
    "cattle": "Cattle",
    "bull": "Cattle",
    "calf": "Cattle",
    "heifer": "Cattle",
    "steer": "Cattle",
    "bos taurus": "Cattle",

    "horse": "Horse",
    "equus ferus caballus": "Horse",

    "goat": "Goat",
    "domestic goat": "Goat",
    "capra hircus": "Goat",

    "sheep": "Sheep",
    "domestic sheep": "Sheep",
    "ovis aries": "Sheep",

    "donkey": "Donkey",
    "burro": "Donkey",
    "equus africanus asinus": "Donkey",

    "mule": "Mule",

    "chicken": "Chicken",
    "rooster": "Chicken",
    "hen": "Chicken",

    "turkey domestic": "Domestic Turkey",
    "domestic turkey": "Domestic Turkey",

    # =========================
    # OTHER MAMMALS COMMON IN TX
    # =========================
    "badger": "Badger",
    "american badger": "Badger",

    "beaver": "Beaver",
    "north american beaver": "Beaver",

    "otter": "Otter",
    "river otter": "Otter",

    "mink": "Mink",

    "weasel": "Weasel",

    "porcupine": "Porcupine",

    "ringtail": "Ringtail",
    "ring-tailed cat": "Ringtail",

    "coati": "Coati",

    # =========================
    # BIRDS — ranch-relevant
    # =========================
    "raven": "Raven",
    "common raven": "Raven",
    "corvus corax": "Raven",

    "crow": "Crow",
    "american crow": "Crow",
    "corvus brachyrhynchos": "Crow",

    "vulture": "Vulture",
    "turkey vulture": "Vulture",
    "black vulture": "Vulture",

    "hawk": "Hawk",
    "red tailed hawk": "Hawk",
    "red-tailed hawk": "Hawk",
    "buteo jamaicensis": "Hawk",
    "cooper hawk": "Hawk",
    "cooper's hawk": "Hawk",

    "eagle": "Eagle",
    "bald eagle": "Eagle",

    "owl": "Owl",
    "great horned owl": "Owl",
    "barred owl": "Owl",
    "screech owl": "Owl",
    "eastern screech owl": "Owl",

    "turkey": "Wild Turkey",
    "wild turkey": "Wild Turkey",
    "meleagris gallopavo": "Wild Turkey",

    "quail": "Quail",
    "northern bobwhite": "Quail",
    "bobwhite": "Quail",

    "dove": "Dove",
    "mourning dove": "Dove",
    "white winged dove": "Dove",
    "white-winged dove": "Dove",

    "pigeon": "Pigeon",

    "roadrunner": "Roadrunner",
    "greater roadrunner": "Roadrunner",

    "woodpecker": "Woodpecker",
    "red bellied woodpecker": "Woodpecker",
    "red-bellied woodpecker": "Woodpecker",

    "blue jay": "Blue Jay",
    "jay": "Blue Jay",

    "cardinal": "Northern Cardinal",
    "northern cardinal": "Northern Cardinal",

    "mockingbird": "Northern Mockingbird",
    "northern mockingbird": "Northern Mockingbird",

    "sparrow": "Small Bird",
    "songbird": "Small Bird",
    "small bird": "Small Bird",
    "bird": "Other",

    "grackle": "Grackle",
    "great tailed grackle": "Grackle",
    "great-tailed grackle": "Grackle",

    "blackbird": "Blackbird",
    "red winged blackbird": "Blackbird",
    "red-winged blackbird": "Blackbird",

    "heron": "Heron",
    "great blue heron": "Heron",

    "egret": "Egret",

    # =========================
    # REPTILES / AMPHIBIANS
    # =========================
    "snake": "Snake",
    "rattlesnake": "Rattlesnake",
    "western diamondback": "Rattlesnake",
    "western diamondback rattlesnake": "Rattlesnake",
    "cottonmouth": "Cottonmouth",
    "water moccasin": "Cottonmouth",

    "lizard": "Lizard",
    "gecko": "Lizard",
    "anole": "Lizard",

    "turtle": "Turtle",
    "box turtle": "Turtle",

    "frog": "Frog",
    "toad": "Toad",

    # =========================
    # INSECTS / VERY SMALL
    # =========================
    "insect": "Other",
    "bug": "Other",
    "spider": "Other",
    "scorpion": "Other",

    # =========================
    # OTHER / FALLBACK-LIKE LABELS
    # =========================
    "other": "Other",
    "unknown animal": "Other",
    "unknown bird": "Other",
    "unknown mammal": "Other",
    "no animal": "Other",
}


# -----------------------------
# Optional vehicle subtyping
# (useful if you later add a vehicle classifier)
# -----------------------------
VEHICLE_ALIASES = {
    "atv": "ATV",
    "utv": "UTV",
    "side by side": "UTV",
    "side-by-side": "UTV",
    "car": "Car",
    "sedan": "Car",
    "suv": "SUV",
    "truck": "Truck",
    "pickup": "Truck",
    "pickup truck": "Truck",
    "tractor": "Tractor",
    "van": "Van",
}


def normalize_vehicle_type(raw: Optional[str]) -> str:
    s = _clean_basic(str(raw or ""))
    if not s or s in JUNK_VALUES:
        return "Other"
    return VEHICLE_ALIASES.get(s, _title_case(s))


def normalize_event_type(raw: Optional[str]) -> str:
    s = _clean_basic(str(raw or ""))
    if not s:
        return "blank"
    if s in ("person", "human", "people"):
        return "human"
    if s in ("vehicle", "car", "truck", "atv", "utv", "tractor"):
        return "vehicle"
    if s in ("animal", "wildlife", "mammal", "bird", "reptile", "amphibian"):
        return "animal"
    if s in JUNK_VALUES:
        return "blank"
    return s


def normalize_species(raw: Optional[str]) -> str:
    """
    Returns canonical name safe for dashboard charts.
    Unknown/vague/junk -> "Other"
    """
    s = _clean_basic(str(raw or ""))

    if not s or s in JUNK_VALUES:
        return "Other"

    # If the model is outputting broad categories, don't chart them
    if s in BROAD_CATEGORIES:
        return "Other"

    # If the label is explicitly banned for being too vague
    if s in BANNED_LABELS:
        return "Other"

    # Canonical mapping
    if s in CANONICAL_SPECIES:
        return CANONICAL_SPECIES[s]

    # Heuristic consolidations (covers tons of “almost” matches)
    # e.g., "white tailed deer buck" -> "White-tailed Deer"
    if "white" in s and "tail" in s and "deer" in s:
        return "White-tailed Deer"
    if "deer" == s or s.endswith(" deer"):
        # If we haven't mapped it, it’s probably an exotic — keep title-cased
        return _title_case(s)

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

    if "hawk" in s:
        return "Hawk"

    if "owl" in s:
        return "Owl"

    if "turkey" in s and "wild" in s:
        return "Wild Turkey"
    if s == "turkey":
        return "Wild Turkey"

    if "dove" in s:
        return "Dove"

    if "snake" in s:
        # if it looks like a rattlesnake mention
        if "diamond" in s or "rattle" in s:
            return "Rattlesnake"
        return "Snake"

    # Default: human-readable label, but keep it from exploding your charts
    # If you want to be stricter, return "Other" here instead.
    return _title_case(s)
