import base64
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.projects import ProjectStore


def sample_master() -> MasterDocument:
    return MasterDocument(
        source_language="ja",
        source_file="voice.srt",
        duration_ms=1000,
        segments=(
            Segment(
                id="seg_000001",
                start_ms=0,
                end_ms=1000,
                channel="MIX",
                kind="speech",
                text="ねえ",
            ),
        ),
    )


class ProjectStoreTests(unittest.TestCase):
    def test_create_from_master_persists_loadable_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProjectStore(Path(tmpdir))

            created = store.create_from_master(sample_master())
            loaded = store.load_project(created["project_id"])

            self.assertEqual(loaded["master"]["segments"][0]["text"], "ねえ")
            self.assertEqual(loaded["metadata"]["source_file"], "voice.srt")
            self.assertFalse(loaded["metadata"]["has_audio"])

    def test_create_from_audio_persists_audio_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProjectStore(Path(tmpdir))
            created = store.create_from_audio(
                "voice.wav",
                "audio/wav",
                base64.b64encode(b"audio").decode("ascii"),
            )

            audio_bytes, mime_type = store.read_audio(created["project_id"])

            self.assertEqual(audio_bytes, b"audio")
            self.assertEqual(mime_type, "audio/wav")
            self.assertTrue(created["metadata"]["has_audio"])

    def test_project_id_cannot_escape_store_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProjectStore(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "invalid project_id"):
                store.load_project("../outside")


if __name__ == "__main__":
    unittest.main()
