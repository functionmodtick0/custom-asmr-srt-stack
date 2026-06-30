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

import custom_asmr_srt_stack.qwen_asr_worker as qwen_asr_worker
from custom_asmr_srt_stack.qwen_asr_worker import (
    QwenAsrResult,
    QwenAsrRuntime,
    aligned_bounds_ms,
    disable_python_network_if_requested,
    qwen_checked_model_path,
    qwen_backend_kwargs,
    qwen_language,
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


class FakeRuntime(QwenAsrRuntime):
    def __init__(self, result: QwenAsrResult) -> None:
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


@dataclass(frozen=True)
class FakeAlignItem:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class FakeAlignResult:
    items: list[FakeAlignItem]


@dataclass(frozen=True)
class FakeTranscription:
    text: str
    time_stamps: FakeAlignResult | None = None


class QwenAsrWorkerTests(unittest.TestCase):
    def test_response_for_line_wraps_qwen_result_as_clip_segment(self):
        request = {
            "type": "transcribe",
            "model_id": "Qwen/Qwen3-ASR-1.7B",
            "channel": "MIX",
            "source_language": "ja",
            "audio_base64": base64.b64encode(mono_wav_bytes(duration_ms=9)).decode("ascii"),
        }
        runtime = FakeRuntime(QwenAsrResult("Transcription: ねえ ねえ", "Japanese", 1, 8, False))

        response = response_for_line(runtime, json.dumps(request))

        self.assertTrue(response["ok"])
        self.assertEqual(runtime.calls[0][0], "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(runtime.calls[0][2], "ja")
        self.assertEqual(runtime.calls[0][3], 9)
        self.assertEqual(
            response["segments"],
            [
                {
                    "start_ms": 1,
                    "end_ms": 8,
                    "channel": "MIX",
                    "kind": "speech",
                    "text": "ねえねえ",
                    "needs_review": True,
                }
            ],
        )

    def test_response_for_line_returns_empty_segments_for_empty_text(self):
        request = {
            "type": "transcribe",
            "model_id": "Qwen/Qwen3-ASR-1.7B",
            "channel": "MIX",
            "source_language": "ja",
            "audio_base64": base64.b64encode(mono_wav_bytes()).decode("ascii"),
        }

        response = response_for_line(FakeRuntime(QwenAsrResult("", "", 0, 2, False)), json.dumps(request))

        self.assertEqual(response, {"ok": True, "segments": []})

    def test_response_for_line_reports_invalid_requests(self):
        response = response_for_line(
            FakeRuntime(QwenAsrResult("ねえ", "Japanese", 0, 1, False)),
            json.dumps({"type": "unknown"}),
        )

        self.assertFalse(response["ok"])
        self.assertIn("unsupported request type", response["error"])

    def test_aligned_bounds_use_first_and_last_timestamp_items(self):
        result = FakeTranscription(
            "ねえ",
            FakeAlignResult(
                [
                    FakeAlignItem("ね", 0.12, 0.22),
                    FakeAlignItem("え", 0.24, 0.48),
                ]
            ),
        )

        self.assertEqual(aligned_bounds_ms(result, 1000), (120, 480, True))

    def test_aligned_bounds_fall_back_to_clip_when_missing(self):
        self.assertEqual(aligned_bounds_ms(FakeTranscription("ねえ"), 1000), (0, 1000, False))

    def test_qwen_language_maps_japanese_source_code(self):
        self.assertEqual(qwen_language("ja"), "Japanese")

    def test_backend_kwargs_coerce_json_dtype(self):
        with mock.patch.dict(
            "os.environ",
            {"CASRT_QWEN_ASR_BACKEND_KWARGS": '{"dtype":"float16","device_map":"cuda:0"}'},
            clear=False,
        ):
            kwargs = qwen_backend_kwargs(FakeTorch)

        self.assertIs(kwargs["dtype"], FakeTorch.float16)
        self.assertEqual(kwargs["device_map"], "cuda:0")

    def test_backend_kwargs_add_local_loading_flags_when_local_paths_are_required(self):
        with mock.patch.dict("os.environ", {"CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH": "1"}, clear=False):
            kwargs = qwen_backend_kwargs(FakeTorch)

        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])

    def test_local_model_path_requirement_rejects_repo_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            with mock.patch.dict("os.environ", {"CASRT_QWEN_ASR_REQUIRE_LOCAL_MODEL_PATH": "1"}, clear=False):
                self.assertEqual(qwen_checked_model_path(str(model_dir), "model_id"), str(model_dir.resolve()))
                with self.assertRaisesRegex(ValueError, "existing local model directory"):
                    qwen_checked_model_path("Qwen/Qwen3-ASR-1.7B", "model_id")

    def test_network_guard_blocks_python_socket_creation(self):
        original_socket = socket.socket
        original_create_connection = socket.create_connection
        try:
            with mock.patch.dict("os.environ", {"CASRT_QWEN_ASR_DISABLE_NETWORK": "1"}, clear=False):
                disable_python_network_if_requested()

            with self.assertRaisesRegex(OSError, "network access is disabled"):
                socket.socket()
            with self.assertRaisesRegex(OSError, "network access is disabled"):
                socket.create_connection(("127.0.0.1", 9), timeout=0.1)
        finally:
            socket.socket = original_socket
            socket.create_connection = original_create_connection
            qwen_asr_worker._NETWORK_DISABLED = False


if __name__ == "__main__":
    unittest.main()
