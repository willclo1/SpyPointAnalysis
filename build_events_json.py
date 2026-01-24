import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

# -----------------------
# Config (tune as you want)
# -----------------------
EVENTS_CSV = Path("events.csv")     # your pipeline output
OUT_JSON = Path("docs/events.json") # consumed by the external site
TIME_GAP_MINUTES = 12              # new session if gap between photos > this
MAX_ITEMS_PER_SESSION = 60         # keep sessions from being huge (optional)

# Maps internal event_type -> site category
TYPE_TO_CATEGORY = {
    "animal": "Wildlife",
    "human": "People",
    "vehicle": "Vehicles",
}

# Label shown in the UI:
# - Wildlife: species_clean (stable + chart-friendly)
# - People/Vehicles: nicer labels
TYPE_TO_LABEL = {
    "human": "Person",
    "vehicle": "Vehicle",
}


def run_rclone_lsjson(cam_folder: str, root_folder_id: str) -> List[dict]:
    """
    Lists files in gdrive:<cam_folder> and returns rclone lsjson output.
    We rely on rclone returning an "ID" field for Drive.
    """
    cmd = [
        "rclone",
        "lsjson",
        f"gdrive:{cam_folder}",
        "--drive-root-folder-id",
        root_folder_id,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # Folder may not exist yet or not accessible
        return []
    out = (p.stdout or "").strip()
    if not out:
        return []
    try:
        return json.loads(out)
    except Exception:
        return []


def build_drive_index(cameras: List[str], root_folder_id: str) -> Dict[str, Dict[str, str]]:
    """
    Returns:
      { camera_name: { filename: file_id } }
    """
    index: Dict[str, Dict[str, str]] = {}
    for cam in cameras:
        items = run_rclone_lsjson(cam, root_folder_id)
        m: Dict[str, str] = {}
        for it in items:
            if it.get("IsDir"):
                continue
            name = it.get("Name")
            fid = it.get("ID") or it.get("Id")  # some versions differ
            if name and fid:
                m[str(name)] = str(fid)
        index[cam] = m
        print(f"[DriveIndex] {cam}: {len(m)} files")
    return index


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # Your CSV has: date = "01/23/2026", time = "03:00 AM"
    dt = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        format="%m/%d/%Y %I:%M %p",
        errors="coerce",
    )
    df = df.copy()
    df["datetime"] = dt
    return df


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_datetime(df)

    # Only keep real detections
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    df = df[df["event_type"].isin(["animal", "human", "vehicle"])].copy()

    # Require datetime + filename + camera
    df["camera"] = df["camera"].astype(str).str.strip()
    df["filename"] = df["filename"].astype(str).str.strip()
    df = df.dropna(subset=["datetime"])
    df = df[(df["camera"] != "") & (df["filename"] != "")]

    # Category + label for site
    df["category"] = df["event_type"].map(TYPE_TO_CATEGORY)

    # Wildlife label
    if "species_clean" in df.columns:
        df["label"] = df.apply(
            lambda r: (r["species_clean"] if r["event_type"] == "animal" else TYPE_TO_LABEL.get(r["event_type"], r["event_type"].title())),
            axis=1,
        )
    else:
        df["label"] = df["event_type"].map(TYPE_TO_LABEL).fillna(df["event_type"].str.title())

    # Clean up label
    df["label"] = df["label"].fillna("Other").astype(str).str.strip()
    df.loc[(df["event_type"] == "animal") & (df["label"] == ""), "label"] = "Other"

    return df


def group_into_sessions(rows: pd.DataFrame, drive_index: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Groups by (camera, category, label) and then splits into sessions by time gaps.
    """
    sessions: List[Dict[str, Any]] = []

    rows = rows.sort_values(["camera", "category", "label", "datetime"]).reset_index(drop=True)

    def file_id_for(cam: str, filename: str) -> Optional[str]:
        return drive_index.get(cam, {}).get(filename)

    # Iterate per group
    for (cam, category, label), g in rows.groupby(["camera", "category", "label"], sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)

        current_items: List[Dict[str, Any]] = []
        start_dt = None
        last_dt = None

        def flush_session():
            nonlocal current_items, start_dt, last_dt
            if not current_items or start_dt is None or last_dt is None:
                return
            session_id = f"{cam}|{label}|{start_dt.isoformat(timespec='minutes')}"
            sessions.append(
                {
                    "session_id": session_id,
                    "camera": cam,
                    "category": category,
                    "label": label,
                    "start": start_dt.isoformat(),
                    "end": last_dt.isoformat(),
                    "count": len(current_items),
                    "items": current_items[:MAX_ITEMS_PER_SESSION],
                }
            )
            current_items = []
            start_dt = None
            last_dt = None

        for _, r in g.iterrows():
            dt = r["datetime"].to_pydatetime()
            fn = str(r["filename"])

            fid = file_id_for(cam, fn)
            # If we can't find a Drive ID, still include it (site will show 📷 placeholder)
            item = {
                "file_id": fid or "",
                "datetime": dt.isoformat(),
                "filename": fn,
            }

            if start_dt is None:
                start_dt = dt
                last_dt = dt
                current_items = [item]
                continue

            gap_min = (dt - last_dt).total_seconds() / 60.0
            if gap_min > TIME_GAP_MINUTES:
                flush_session()
                start_dt = dt
                last_dt = dt
                current_items = [item]
            else:
                current_items.append(item)
                last_dt = dt

        flush_session()

    # Newest sessions first (nice UX)
    sessions.sort(key=lambda s: s["start"], reverse=True)
    return sessions


def main():
    if not EVENTS_CSV.exists():
        raise SystemExit(f"Missing {EVENTS_CSV}. Run your pipeline first.")

    # In GitHub Actions, set this secret/env so we can index Drive properly
    root_folder_id = (Path(".").joinpath(".gdrive_root_id").read_text().strip()
                      if Path(".gdrive_root_id").exists()
                      else None)

    root_folder_id = root_folder_id or (pd.options.mode.chained_assignment is None)  # no-op; keeps lint quiet

    # Prefer env var (recommended)
    import os
    root_folder_id = os.environ.get("GDRIVE_FOLDER_ID") or os.environ.get("GDRIVE_FOLDER") or None
    if not root_folder_id:
        raise SystemExit("Set GDRIVE_FOLDER_ID env var (Drive folder holding camera folders).")

    df = pd.read_csv(EVENTS_CSV)
    df = normalize_rows(df)

    cameras = sorted(df["camera"].dropna().unique().tolist())
    drive_index = build_drive_index(cameras, root_folder_id)

    sessions = group_into_sessions(df, drive_index)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"sessions": sessions}, indent=2), encoding="utf-8")

    # Print quick summary
    total_photos = sum(s["count"] for s in sessions)
    missing_ids = sum(1 for s in sessions for it in s["items"] if not it.get("file_id"))
    print(f"[OK] Wrote {OUT_JSON} with {len(sessions)} sessions, {total_photos} photos")
    print(f"[OK] Missing Drive IDs for {missing_ids} photos (will show placeholders)")


if __name__ == "__main__":
    main()
