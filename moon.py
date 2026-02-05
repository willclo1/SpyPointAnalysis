from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone

# ✅ NEW imports for Astral 3+
from astral.moon import phase as moon_phase
from astral.moon import illumination as moon_illumination


@dataclass
class MoonInfo:
    phase_name: str
    illumination: float   # 0..1
    age_days: float       # 0..29.53


def _phase_name(age_days: float) -> str:
    """
    Bucket moon age (0..~29.53) into 8 ranch-friendly names.
    """
    if age_days < 1.0 or age_days > 28.5:
        return "New"
    if age_days < 6.0:
        return "Waxing Crescent"
    if age_days < 8.5:
        return "First Quarter"
    if age_days < 14.0:
        return "Waxing Gibbous"
    if age_days < 16.0:
        return "Full"
    if age_days < 21.0:
        return "Waning Gibbous"
    if age_days < 23.5:
        return "Last Quarter"
    return "Waning Crescent"


def moon_info(dt: datetime) -> MoonInfo:
    """
    dt: naive assumed UTC.
    (You can localize to CT later if you want more precision.)
    """

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # ✅ NEW calls
    age = float(moon_phase(dt))            # days since new moon
    illum = float(moon_illumination(dt))   # 0..1

    return MoonInfo(
        phase_name=_phase_name(age),
        illumination=illum,
        age_days=age,
    )
