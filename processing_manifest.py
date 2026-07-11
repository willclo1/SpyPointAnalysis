from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class ProcessingManifest:
    def __init__(self, path: Path, pipeline_version: str):
        self.path = path
        self.pipeline_version = pipeline_version
        self.data: Dict[str, Any] = {
            "schema_version": 1,
            "pipeline_version": pipeline_version,
            "updated_at": None,
            "images": {},
        }
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(loaded, dict) and isinstance(loaded.get("images"), dict):
            self.data = loaded

    @staticmethod
    def file_signature(path: Path, chunk_size: int = 1024 * 1024) -> Dict[str, Any]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        stat = path.stat()
        return {
            "sha256": digest.hexdigest(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        value = self.data.get("images", {}).get(key)
        return value if isinstance(value, dict) else None

    def is_current(self, key: str, signature: Dict[str, Any]) -> bool:
        record = self.get(key)
        return bool(
            record
            and record.get("sha256") == signature.get("sha256")
            and record.get("pipeline_version") == self.pipeline_version
            and record.get("status") == "complete"
        )

    def cached_row(self, key: str, signature: Dict[str, Any]) -> Optional[Dict[str, str]]:
        if not self.is_current(key, signature):
            return None
        record = self.get(key) or {}
        row = record.get("row")
        return row if isinstance(row, dict) else None

    def mark_complete(self, key: str, signature: Dict[str, Any], row: Dict[str, str]) -> None:
        self.data.setdefault("images", {})[key] = {
            **signature,
            "pipeline_version": self.pipeline_version,
            "status": "complete",
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "last_error": None,
            "row": row,
        }

    def mark_failed(self, key: str, signature: Dict[str, Any], error: str) -> None:
        previous = self.get(key) or {}
        self.data.setdefault("images", {})[key] = {
            **previous,
            **signature,
            "pipeline_version": self.pipeline_version,
            "status": "failed",
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "last_error": error[:2000],
        }

    def prune(self, valid_keys: set[str]) -> int:
        images = self.data.setdefault("images", {})
        stale = [key for key in images if key not in valid_keys]
        for key in stale:
            del images[key]
        return len(stale)

    def save(self) -> None:
        self.data["pipeline_version"] = self.pipeline_version
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
            json.dump(self.data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(self.path)
