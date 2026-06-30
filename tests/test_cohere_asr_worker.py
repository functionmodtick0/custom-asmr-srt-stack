import base64
import io
import json
import socket
import struct
import tempfile
import unittest
import wave
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import custom_asmr_srt_stack.cohere_asr_worker as cohere_asr_worker
from custom_asmr_srt_stack.cohere_asr_worker import (
    CohereAsrResult,
    CohereAsrRuntime,
    cohere_checked_model_path,
    cohere_language,
    cohere_model_kwargs,
    disable_python_network_if_requested,
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


class FakeRuntime(CohereAsrRuntime):
    def __init__(self, result: CohereAsrResult) -> None:
        super().__init__()
        self.result = result
        self.calls = []

    def generate_result(self, model_id, audio_bytes, source_language, duration_ms):
        self.calls.append((model_id, len(audio_bytes), source_language, duration_ms))
        return self.result


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    bfloat16 = object()
    float16 = object()
    float32 = object()
    cuda = FakeCuda()


class CohereAsrWorkerTests(unittest.TestCase):
    def test_response_for_line_wraps_cohere_result_as_clip_segment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "model.safetensors").write_bytes(b"fake")
            request = {
                "type": "transcribe",
                "model_id": str(model_dir),
                "channel": "MIX",
                "source_language": "ja",
                "audio_base64": base64.b64encode(mono_wav_bytes(duration_ms=9)).decode("ascii"),
            }
            runtime = FakeRuntime(CohereAsrResult("Transcription: ねえ ねえ", 0, 9))
            expected_model_dir = str(model_dir.resolve())

            response = response_for_line(runtime, json.dumps(request))

        self.assertTrue(response["ok"])
        self.assertEqual(runtime.calls[0][0], expected_model_dir)
        self.assertEqual(runtime.calls[0][2], "ja")
        self.assertEqual(runtime.calls[0][3], 9)
        self.assertEqual(
            response["segments"],
            [
                {
                    "start_ms": 0,
                    "end_ms": 9,
                    "channel": "MIX",
                    "kind": "speech",
                    "text": "ねえねえ",
                    "needs_review": True,
                }
            ],
        )

    def test_response_for_line_reports_invalid_requests(self):
        response = response_for_line(
            FakeRuntime(CohereAsrResult("ねえ", 0, 1)),
            json.dumps({"type": "unknown"}),
        )

        self.assertFalse(response["ok"])
        self.assertIn("unsupported request type", response["error"])

    def test_cohere_language_maps_japanese_source_code(self):
        self.assertEqual(cohere_language("ja"), "ja")
        self.assertEqual(cohere_language("japanese"), "ja")

    def test_model_kwargs_force_local_safetensors_loading(self):
        kwargs = cohere_model_kwargs(FakeTorch)

        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertTrue(kwargs["use_safetensors"])
        self.assertIs(kwargs["dtype"], FakeTorch.bfloat16)

    def test_checked_model_path_requires_local_safetensors_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            empty = root / "empty"
            empty.mkdir()
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model.safetensors").write_bytes(b"fake")

            self.assertEqual(cohere_checked_model_path(str(model_dir), "model_id"), str(model_dir.resolve()))
            with self.assertRaisesRegex(ValueError, "safetensors"):
                cohere_checked_model_path(str(empty), "model_id")
            with self.assertRaisesRegex(ValueError, "existing local Cohere ASR snapshot"):
                cohere_checked_model_path("CohereLabs/cohere-transcribe-03-2026", "model_id")

    def test_network_guard_blocks_python_socket_creation(self):
        original_socket = socket.socket
        original_create_connection = socket.create_connection
        try:
            with mock.patch.dict("os.environ", {"CASRT_COHERE_ASR_DISABLE_NETWORK": "1"}, clear=False):
                disable_python_network_if_requested()

            with self.assertRaisesRegex(OSError, "network access is disabled"):
                socket.socket()
            with self.assertRaisesRegex(OSError, "network access is disabled"):
                socket.create_connection(("127.0.0.1", 9), timeout=0.1)
        finally:
            socket.socket = original_socket
            socket.create_connection = original_create_connection
            cohere_asr_worker._NETWORK_DISABLED = False


if __name__ == "__main__":
    unittest.main()
