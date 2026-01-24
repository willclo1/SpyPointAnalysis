# build_events_json.py
import json
import pandas as pd
from datetime import timedelta
from pathlib import Path

EVENT_GAP_MINUTES = 15  # gap to split events


def group_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").copy()
    df["prev_time"] = df["datetime"].shift()
    df["gap"] = (df["datetime"] - df["prev_time"]).dt.total_seconds() / 60

    df["new_event"] = (df["gap"] > EVENT_GAP_MINUTES) | df["gap"].isna()
    df["event_id"] = df["new_event"].cumsum()

    return df


def make_thumbnail(file_id: str) -> str:
    return f"https://drive.google.com/thumbnail?id={file_id}&sz=w600"


def main():
    df = pd.read_csv("events.csv", parse_dates=["datetime"])

    # -------------------------------------------------
    # FILTER KNOWN JUNK FIRST
    # -------------------------------------------------

    df = df[df["event_type"] == "animal"]              # only wildlife
    df = df[df["wildlife_label"] != "Other"]           # remove Other
    df = df[df["wildlife_label"].notna()]              # sanity

    # -------------------------------------------------
    # GROUP INTO EVENTS
    # -------------------------------------------------

    df = group_events(df)

    events = []

    for eid, g in df.groupby("event_id"):
        if len(g) <= 1:   # remove singles
            continue

        first = g.iloc[0]

        events.append({
            "event_id": int(eid),
            "species": first["wildlife_label"],
            "camera": first["camera"],
            "start": str(g["datetime"].min()),
            "end": str(g["datetime"].max()),
            "count": int(len(g)),
            "thumbnail": make_thumbnail(first["file_id"]),
            "files": g["file_id"].tolist()
        })

    # sort newest first
    events.sort(key=lambda x: x["start"], reverse=True)

    Path("docs").mkdir(exist_ok=True)
    with open("docs/events.json", "w") as f:
        json.dump(events, f, indent=2)

    print(f"✅ Built events.json with {len(events)} clean wildlife events")


if __name__ == "__main__":
    main()
