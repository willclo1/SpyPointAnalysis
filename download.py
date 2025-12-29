import os, json, time, re
from pathlib import Path
import requests
import spypoint

OUT_DIR = Path("images")
STATE_PATH = Path("state.json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLL_LIMIT   = 50      # how many recent photos to look at each run
MAX_PER_RUN  = 25      # safety cap per run
SLEEP_SEC    = 1.0     # pause between downloads

def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()) or "photo.jpg"

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"seen_urls": []}

def save_state(state):
    # keep only last 5000 seen URLs to bound file size
    state["seen_urls"] = state["seen_urls"][-5000:]
    STATE_PATH.write_text(json.dumps(state, indent=2))

def download(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

def main():
    email = os.environ.get("SPYPOINT_EMAIL")
    password = os.environ.get("SPYPOINT_PASSWORD")
    if not email or not password:
        raise SystemExit("Set SPYPOINT_EMAIL and SPYPOINT_PASSWORD env vars.")

    state = load_state()
    seen = set(state["seen_urls"])

    c = spypoint.Client(email, password)
    cams = c.cameras()
    photos = c.photos(cams, limit=POLL_LIMIT)

    downloaded = 0
    for p in photos:
        url = p.url()
        if url in seen:
            continue

        base = url.split("?")[0].split("/")[-1]
        filename = safe_filename(base)
        out_path = OUT_DIR / filename

        # If file already exists, treat as seen
        if out_path.exists():
            seen.add(url)
            continue

        download(url, out_path)
        downloaded += 1
        print(f"Saved: {out_path}")

        seen.add(url)
        if downloaded >= MAX_PER_RUN:
            print(f"Hit MAX_PER_RUN={MAX_PER_RUN}, stopping early.")
            break

        time.sleep(SLEEP_SEC)

    state["seen_urls"] = list(seen)
    save_state(state)
    print(f"Downloaded {downloaded} new images.")

if __name__ == "__main__":
    main()
