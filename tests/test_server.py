import json
import base64
import io
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.server import handle_api_request
from custom_asmr_srt_stack.models import Segment


class ServerApiTests(unittest.TestCase):
    def post_json(self, path, payload, project_store=None, transcribe_audio_func=None):
        kwargs = {"project_store": project_store}
        if transcribe_audio_func is not None:
            kwargs["transcribe_audio_func"] = transcribe_audio_func
        status, content_type, body = handle_api_request(
            path,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            **kwargs,
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

    def test_model_validate_route_checks_endpoint_contract(self):
        status, response = self.post_json(
            "/api/model/validate",
            {
                "model": {
                    "adapter": "gemini",
                    "endpoint_url": "https://generativelanguage.googleapis.com",
                    "model_id": "gemini-2.5-pro",
                }
            },
        )

        self.assertEqual(status, 200)
        self.assertTrue(response["ok"])
        self.assertEqual(response["adapter"], "gemini")

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

    def test_analyze_audio_route_persists_wav_channels(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(1000)
            wav.writeframes(b"\x00\x00\x01\x00")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProjectStore(Path(tmpdir))
            upload_status, upload = self.post_json(
                "/api/projects/upload-audio",
                {
                    "file_name": "voice.wav",
                    "mime_type": "audio/wav",
                    "content_base64": base64.b64encode(output.getvalue()).decode("ascii"),
                },
                project_store=store,
            )
            analyze_status, analyzed = self.post_json(
                "/api/projects/analyze-audio",
                {"project_id": upload["project_id"]},
                project_store=store,
            )

            self.assertEqual(upload_status, 200)
            self.assertEqual(analyze_status, 200)
            self.assertEqual(analyzed["metadata"]["audio_info"]["duration_ms"], 2)
            self.assertEqual(set(analyzed["metadata"]["channels"]), {"MIX"})
            self.assertEqual(analyzed["metadata"]["normalized_audio_file"], "audio/normalized.wav")

    def test_transcribe_route_uses_analyzed_left_and_right_channels(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(1000)
            wav.writeframes(struct.pack("<hhhh", 100, 300, 200, 400))

        calls = []

        def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
            calls.append((endpoint.model_id, len(audio_bytes), mime_type, channel, source_language))
            text = "左" if channel == "L" else "右"
            return (
                Segment(
                    id="ignored",
                    start_ms=0 if channel == "L" else 1,
                    end_ms=1 if channel == "L" else 2,
                    channel="MIX",
                    kind="speech",
                    text=text,
                ),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProjectStore(Path(tmpdir))
            _, upload = self.post_json(
                "/api/projects/upload-audio",
                {
                    "file_name": "voice.wav",
                    "mime_type": "audio/wav",
                    "content_base64": base64.b64encode(output.getvalue()).decode("ascii"),
                },
                project_store=store,
            )
            self.post_json(
                "/api/projects/analyze-audio",
                {"project_id": upload["project_id"]},
                project_store=store,
            )
            status, response = self.post_json(
                "/api/projects/transcribe",
                {
                    "project_id": upload["project_id"],
                    "source_language": "ja",
                    "model": {
                        "adapter": "openai-compatible",
                        "endpoint_url": "http://localhost:8000/v1",
                        "model_id": "gemma-4-e4b",
                    },
                },
                project_store=store,
                transcribe_audio_func=fake_transcribe,
            )

            self.assertEqual(status, 200)
            self.assertEqual([call[3] for call in calls], ["L", "R"])
            self.assertEqual(
                [(segment["id"], segment["channel"], segment["text"]) for segment in response["master"]["segments"]],
                [("seg_000001", "L", "左"), ("seg_000002", "R", "右")],
            )


if __name__ == "__main__":
    unittest.main()
