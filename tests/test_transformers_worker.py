import base64
import io
import json
import struct
import unittest
import wave

from custom_asmr_srt_stack.transformers_worker import (
    TransformersRuntime,
    clean_transcription_text,
    response_for_line,
)


def mono_wav_bytes(duration_ms: int = 2) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(struct.pack("<h", 100) * duration_ms)
    return output.getvalue()


class FakeRuntime(TransformersRuntime):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def generate_text(self, model_id, audio_bytes):
        del model_id, audio_bytes
        return self.text


class TransformersWorkerTests(unittest.TestCase):
    def test_response_for_line_wraps_generated_text_as_a_clip_segment(self):
        request = {
            "type": "transcribe",
            "model_id": "google/gemma-4-E4B-it",
            "channel": "R",
            "audio_base64": base64.b64encode(mono_wav_bytes(duration_ms=7)).decode("ascii"),
        }

        response = response_for_line(FakeRuntime("ねえ"), json.dumps(request))

        self.assertTrue(response["ok"])
        self.assertEqual(
            response["segments"],
            [
                {
                    "start_ms": 0,
                    "end_ms": 7,
                    "channel": "R",
                    "kind": "speech",
                    "text": "ねえ",
                    "needs_review": True,
                }
            ],
        )

    def test_response_for_line_returns_empty_segments_for_empty_text(self):
        request = {
            "type": "transcribe",
            "model_id": "google/gemma-4-E4B-it",
            "channel": "MIX",
            "audio_base64": base64.b64encode(mono_wav_bytes()).decode("ascii"),
        }

        response = response_for_line(FakeRuntime("  "), json.dumps(request))

        self.assertEqual(response, {"ok": True, "segments": []})

    def test_response_for_line_reports_invalid_requests(self):
        response = response_for_line(FakeRuntime("ねえ"), json.dumps({"type": "unknown"}))

        self.assertFalse(response["ok"])
        self.assertIn("unsupported request type", response["error"])

    def test_clean_transcription_text_removes_common_prefix(self):
        self.assertEqual(clean_transcription_text("Transcription: ねえ"), "ねえ")


if __name__ == "__main__":
    unittest.main()
