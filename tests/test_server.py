import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.server import handle_api_request


class ServerApiTests(unittest.TestCase):
    def post_json(self, path, payload, project_store=None):
        status, content_type, body = handle_api_request(
            path,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            project_store=project_store,
        )
        self.assertEqual(content_type, "application/json; charset=utf-8")
        return status, json.loads(body.decode("utf-8"))

    def test_srt_to_json_route_returns_master_json(self):
        status, response = self.post_json(
            "/api/srt-to-json",
            {
                "content": "1\n00:00:01,000 --> 00:00:02,000\nねえ\n",
                "source_file": "voice.wav",
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["format"], "custom-asmr-master-v1")
        self.assertEqual(response["segments"][0]["text"], "ねえ")
        self.assertEqual(response["segments"][0]["channel"], "MIX")

    def test_json_to_srt_route_uses_translated_text(self):
        master = {
            "format": "custom-asmr-master-v1",
            "source_language": "ja",
            "audio": {"source_file": "voice.wav", "duration_ms": 2000},
            "segments": [
                {
                    "id": "seg_000001",
                    "start_ms": 1000,
                    "end_ms": 2000,
                    "channel": "L",
                    "kind": "speech",
                    "text": "ねえ",
                    "needs_review": False,
                }
            ],
        }
        translated = {
            "format": "custom-asmr-translated-v1",
            "source_language": "ja",
            "target_language": "ko",
            "items": [{"id": "seg_000001", "text": "저기"}],
        }

        status, response = self.post_json("/api/json-to-srt", {"master": master, "translated": translated})

        self.assertEqual(status, 200)
        self.assertEqual(response["content"], "1\n00:00:01,000 --> 00:00:02,000\n저기\n")

    def test_api_errors_are_visible(self):
        status, response = self.post_json("/api/srt-to-json", {"content": 123})

        self.assertEqual(status, 400)
        self.assertIn("content must be a string", response["error"])

    def test_import_srt_route_persists_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProjectStore(Path(tmpdir))

            status, response = self.post_json(
                "/api/projects/import-srt",
                {
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nねえ\n",
                    "source_file": "voice.srt",
                },
                project_store=store,
            )
            load_status, loaded = self.post_json(
                "/api/projects/load",
                {"project_id": response["project_id"]},
                project_store=store,
            )

            self.assertEqual(status, 200)
            self.assertEqual(load_status, 200)
            self.assertEqual(loaded["master"]["segments"][0]["text"], "ねえ")


if __name__ == "__main__":
    unittest.main()
