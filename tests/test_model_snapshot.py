import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.model_snapshot import snapshot_digest


class ModelSnapshotTests(unittest.TestCase):
    def test_snapshot_digest_hashes_files_and_symlink_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            blob = root / "blob"
            blob.write_text("weights", encoding="utf-8")
            snapshot = root / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").symlink_to(blob)

            digest = snapshot_digest(snapshot)

        self.assertEqual(digest["format"], "custom-asmr-model-snapshot-digest-v1")
        self.assertEqual(digest["snapshot_id"], "abc123")
        self.assertEqual(digest["file_count"], 2)
        self.assertEqual([item["path"] for item in digest["files"]], ["config.json", "model.safetensors"])
        self.assertEqual(digest["files"][1]["size_bytes"], len("weights"))
        self.assertRegex(digest["sha256"], r"^[a-f0-9]{64}$")

    def test_snapshot_digest_requires_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "contain files"):
                snapshot_digest(Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
