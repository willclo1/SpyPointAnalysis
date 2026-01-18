import os, json, time, re
from pathlib import Path
import requests
import spypoint

OUT_DIR = Path("images")
STATE_PATH = Path("state.json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLL_LIMIT   = 50      # how many recent photos per camera to look at each run
MAX_PER_RUN  = 25      # safety cap total per run (across cameras)
SLEEP_SEC    = 1.0     # pause between downloads

def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip()) or "camera"

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"seen_keys": []}

def save_state(state):
    state["seen_keys"] = state["seen_keys"][-5000:]
    STATE_PATH.write_text(json.dumps(state, indent=2))

def download(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

def camera_display_name(cam) -> str:
    # Try common attributes from different pyspypoint versions
    for attr in ("name", "label", "display_name", "cameraName", "nickname"):
        val = getattr(cam, attr, None)
        if val:
            return str(val)
    cam_id = getattr(cam, "id", None) or getattr(cam, "camera_id", None) or "camera"
    return str(cam_id)

def main():
    email = os.environ.get("SPYPOINT_EMAIL")
    password = os.environ.get("SPYPOINT_PASSWORD")
    if not email or not password:
        raise SystemExit("Set SPYPOINT_EMAIL and SPYPOINT_PASSWORD env vars.")

    state = load_state()
    seen = set(state.get("seen_keys", []))

    c = spypoint.Client(email, password)
    cams = c.cameras()

    downloaded = 0

    for cam in cams:
        cam_name = safe_filename(camera_display_name(cam))
        cam_dir = OUT_DIR / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)

        # Pull photos for just this camera
        photos = c.photos([cam], limit=POLL_LIMIT)

        for p in photos:
            url = p.url()
            base = url.split("?")[0].split("/")[-1]
            filename = safe_filename(base)

            key = f"{cam_name}/{filename}"
            if key in seen:
                continue

            out_path = cam_dir / filename

            if out_path.exists():
                seen.add(key)
                continue

            download(url, out_path)
            downloaded += 1
            print(f"Saved: {out_path}")

            seen.add(key)

            if downloaded >= MAX_PER_RUN:
                print(f"Hit MAX_PER_RUN={MAX_PER_RUN}, stopping early.")
                state["seen_keys"] = list(seen)
                save_state(state)
                print(f"Downloaded {downloaded} new images.")
                return

            time.sleep(SLEEP_SEC)

    state["seen_keys"] = list(seen)
    save_state(state)
    print(f"Downloaded {downloaded} new images.")

if __name__ == "__main__":
    main()
