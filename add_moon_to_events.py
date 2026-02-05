# add_moon_to_events.py
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from moon import moon_info

IN_CSV = Path("events.csv")
OUT_CSV = Path("events.csv")
OUT_TSV = Path("events.tsv")

# Columns we will add/update
MOON_FIELDS = ["moon_phase", "moon_illumination", "moon_age_days"]


def parse_dt(date_s: str, time_s: str) -> datetime | None:
    date_s = (date_s or "").strip()
    time_s = (time_s or "").strip()
    if not date_s or not time_s:
        return None

    # Your pipeline format: 01/18/2026 + 3:58 PM
    try:
        return datetime.strptime(f"{date_s} {time_s}", "%m/%d/%Y %I:%M %p")
    except Exception:
        return None


def write_table(path: Path, rows: List[Dict[str, str]], fieldnames: List[str], delimiter: str):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") or "") for k in fieldnames})


def main():
    if not IN_CSV.exists():
        raise SystemExit("events.csv not found")

    with IN_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        base_fields = reader.fieldnames or []
        rows = list(reader)

    # Ensure moon columns exist (append at end if missing)
    fieldnames = list(base_fields)
    for mf in MOON_FIELDS:
        if mf not in fieldnames:
            fieldnames.append(mf)

    updated = 0
    skipped = 0

    for r in rows:
        dt = parse_dt(r.get("date", ""), r.get("time", ""))
        if dt is None:
            # Leave moon fields blank if timestamp missing
            for mf in MOON_FIELDS:
                r[mf] = r.get(mf, "") or ""
            skipped += 1
            continue

        mi = moon_info(dt)
        r["moon_phase"] = mi.phase_name
        r["moon_illumination"] = f"{mi.illumination:.3f}"
        r["moon_age_days"] = f"{mi.age_days:.2f}"
        updated += 1

    write_table(OUT_CSV, rows, fieldnames, delimiter=",")
    write_table(OUT_TSV, rows, fieldnames, delimiter="\t")

    print(f"Moon updated: {updated}, skipped(no datetime): {skipped}")
    print(f"Wrote {OUT_CSV} and {OUT_TSV}")


if __name__ == "__main__":
    main()
