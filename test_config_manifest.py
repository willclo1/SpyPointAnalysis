import json
import tempfile
import unittest
from pathlib import Path

from config import load_config
from processing_manifest import ProcessingManifest


class ConfigAndManifestTests(unittest.TestCase):
    def test_camera_override_inherits_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "default_camera": {"event_gap_minutes": 12, "animal_threshold": 0.25},
                "cameras": {"Gate": {"event_gap_minutes": 7}},
            }))
            config = load_config(path)
            self.assertEqual(config.camera("Gate").event_gap_minutes, 7)
            self.assertEqual(config.camera("Gate").animal_threshold, 0.25)
            self.assertEqual(config.camera("Other").event_gap_minutes, 12)

    def test_manifest_round_trip_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"first")
            manifest_path = root / "manifest.json"
            manifest = ProcessingManifest(manifest_path, "1.0")
            signature = manifest.file_signature(image)
            manifest.mark_complete("cam::image.jpg", signature, {"filename": "image.jpg"})
            manifest.save()

            loaded = ProcessingManifest(manifest_path, "1.0")
            self.assertEqual(loaded.cached_row("cam::image.jpg", signature)["filename"], "image.jpg")

            image.write_bytes(b"second")
            changed = loaded.file_signature(image)
            self.assertIsNone(loaded.cached_row("cam::image.jpg", changed))

            newer_pipeline = ProcessingManifest(manifest_path, "2.0")
            self.assertIsNone(newer_pipeline.cached_row("cam::image.jpg", signature))


if __name__ == "__main__":
    unittest.main()
