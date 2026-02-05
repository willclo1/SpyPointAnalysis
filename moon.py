# moon.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from zoneinfo import ZoneInfo

from astral import moon

# Ranch timezone
CT = ZoneInfo("America/Chicago")

# Average synodic month
SYNODIC_MONTH = 29.53058867


@dataclass
class MoonInfo:
    phase_name: str
    illumination: float  # 0..1
    age_days: float      # 0..29.53


# ---------------------------
# Helpers
# ---------------------------

def _phase_name(age: float) -> str:
    if age < 1.0 or age > 28.5:
        return "New"
    if age < 6.0:
        return "Waxing Crescent"
    if age < 8.5:
        return "First Quarter"
    if age < 14.0:
        return "Waxing Gibbous"
    if age < 16.0:
        return "Full"
    if age < 21.0:
        return "Waning Gibbous"
    if age < 23.5:
        return "Last Quarter"
    return "Waning Crescent"


def _illumination_from_age(age: float) -> float:
    """
    Illumination formula:
      (1 − cos(2π * age / period)) / 2
    """
    age = age % SYNODIC_MONTH
    angle = 2 * math.pi * (age / SYNODIC_MONTH)
    return (1 - math.cos(angle)) / 2


# ---------------------------
# Main API
# ---------------------------

def moon_info(dt: datetime) -> MoonInfo:
    """
    Uses CENTRAL TIME for phase calculations.

    If dt is naive, we assume it is already Central.
    """

    # attach CT if naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CT)
    else:
        dt = dt.astimezone(CT)

    # Astral gives moon age in days
    age = float(moon.phase(dt))

    illum = _illumination_from_age(age)

    return MoonInfo(
        phase_name=_phase_name(age),
        illumination=illum,
        age_days=age,
    )
