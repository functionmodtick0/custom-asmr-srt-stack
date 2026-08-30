import base64
import hashlib
import io
import json
import os
import socket
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import custom_asmr_srt_stack.qwen_hf_asr_worker as qwen_hf_asr_worker
from custom_asmr_srt_stack.qwen_hf_asr_worker import (
    QwenHfAsrResult,
    QwenHfAsrRuntime,
    checked_model_path,
    disable_python_network_if_requested,
    qwen_language,
    qwen_hf_num_beams,
    require_secure_runtime,
    response_for_line,
    validate_qwen_hf_snapshot,
)


def mono_wav_bytes(duration_ms: int = 7) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(struct.pack("<h", 100) * duration_ms)
    return output.getvalue()


def secure_env() -> dict[str, str]:
    return {
        "CASRT_LOCAL_WORKER_ENV_MODE": "offline",
        "CASRT_QWEN_HF_ASR_REQUIRE_LOCAL_MODEL_PATH": "1",
        "CASRT_QWEN_HF_ASR_LOCAL_FILES_ONLY": "1",
        "CASRT_QWEN_HF_ASR_DISABLE_NETWORK": "1",
    }


class FakeRuntime(QwenHfAsrRuntime):
    def generate_result(self, model_id: str, audio_bytes: bytes, source_language: str, duration_ms: int):
        del model_id, audio_bytes, source_language
        return QwenHfAsrResult(text="Transcription: ねえ", start_ms=0, end_ms=duration_ms)


class QwenHfAsrWorkerTests(unittest.TestCase):
    def test_response_for_line_wraps_transcription_as_clip_segment(self):
        request = {
            "type": "transcribe",
            "model_id": "/models/qwen-hf",
            "channel": "MIX",
            "source_language": "ja",
            "audio_base64": base64.b64encode(mono_wav_bytes()).decode("ascii"),
        }

        response = response_for_line(FakeRuntime(), json.dumps(request))

        self.assertTrue(response["ok"])
        self.assertEqual(
            response["segments"],
            [
                {
                    "start_ms": 0,
                    "end_ms": 7,
                    "channel": "MIX",
                    "kind": "speech",
                    "text": "ねえ",
                    "needs_review": True,
                }
            ],
        )

    def test_response_for_line_omits_traceback_on_error(self):
        response = response_for_line(FakeRuntime(), "{}")

        self.assertFalse(response["ok"])
        self.assertIn("unsupported request type", response["error"])
        self.assertNotIn("traceback", response)

    def test_secure_runtime_requires_offline_local_and_network_guards(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "CASRT_LOCAL_WORKER_ENV_MODE=offline"):
                require_secure_runtime()

        with mock.patch.dict("os.environ", secure_env(), clear=True):
            require_secure_runtime()

    def test_checked_model_path_requires_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(checked_model_path(tmpdir, "model_id"), str(Path(tmpdir).resolve()))

            with self.assertRaisesRegex(ValueError, "existing local model directory"):
                checked_model_path(str(Path(tmpdir) / "missing"), "model_id")

    def test_snapshot_preflight_accepts_native_safetensors_with_pinned_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir)
            template = "{{ messages }}"
            (snapshot / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3ASRForConditionalGeneration"],
                        "model_type": "qwen3_asr",
                    }
                ),
                encoding="utf-8",
            )
            (snapshot / "model.safetensors").write_bytes(b"safe-test-placeholder")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "chat_template.jinja").write_text(template, encoding="utf-8")
            cache = snapshot / ".cache" / "huggingface" / "download"
            cache.mkdir(parents=True)
            (cache / "config.json.metadata").write_text("metadata", encoding="utf-8")
            template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()

            with mock.patch.dict(
                os.environ,
                {"CASRT_QWEN_HF_ASR_EXPECTED_CHAT_TEMPLATE_SHA256": template_hash},
                clear=True,
            ):
                validate_qwen_hf_snapshot(snapshot)

    def test_snapshot_preflight_rejects_unsafe_files_dynamic_code_and_template_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir)
            config = {
                "architectures": ["Qwen3ASRForConditionalGeneration"],
                "model_type": "qwen3_asr",
            }
            (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"safe-test-placeholder")
            template = snapshot / "chat_template.jinja"
            template.write_text("expected", encoding="utf-8")
            expected_hash = hashlib.sha256(b"expected").hexdigest()

            (snapshot / "pytorch_model.bin").write_bytes(b"unsafe")
            with mock.patch.dict(
                os.environ,
                {"CASRT_QWEN_HF_ASR_EXPECTED_CHAT_TEMPLATE_SHA256": expected_hash},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "unsupported file"):
                    validate_qwen_hf_snapshot(snapshot)
            (snapshot / "pytorch_model.bin").unlink()

            config["text_config"] = {"auto_map": {"AutoModel": "modeling_custom.Model"}}
            (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {"CASRT_QWEN_HF_ASR_EXPECTED_CHAT_TEMPLATE_SHA256": expected_hash},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "dynamic code settings: text_config.auto_map"):
                    validate_qwen_hf_snapshot(snapshot)
            config.pop("text_config")
            (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"CASRT_QWEN_HF_ASR_EXPECTED_CHAT_TEMPLATE_SHA256": "0" * 64},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    validate_qwen_hf_snapshot(snapshot)

    def test_snapshot_preflight_validates_sharded_safetensors_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir)
            (snapshot / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3ASRForConditionalGeneration"],
                        "model_type": "qwen3_asr",
                    }
                ),
                encoding="utf-8",
            )
            shard = "model-00001-of-00001.safetensors"
            (snapshot / shard).write_bytes(b"safe-test-placeholder")
            index = {"weight_map": {"model.weight": shard}}
            (snapshot / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

            validate_qwen_hf_snapshot(snapshot)

            (snapshot / shard).unlink()
            with self.assertRaisesRegex(ValueError, "requires safetensors model weights"):
                validate_qwen_hf_snapshot(snapshot)

    def test_qwen_language_defaults_to_japanese(self):
        self.assertEqual(qwen_language("ja"), "Japanese")
        with mock.patch.dict(os.environ, {"CASRT_QWEN_HF_ASR_FORCE_LANGUAGE": "0"}):
            self.assertIsNone(qwen_language("ja"))

    def test_num_beams_defaults_to_greedy_and_accepts_explicit_beam_search(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(qwen_hf_num_beams(), 1)
        with mock.patch.dict(os.environ, {"CASRT_QWEN_HF_ASR_NUM_BEAMS": "5"}, clear=True):
            self.assertEqual(qwen_hf_num_beams(), 5)
        with mock.patch.dict(os.environ, {"CASRT_QWEN_HF_ASR_NUM_BEAMS": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                qwen_hf_num_beams()

    def test_generate_result_passes_configured_beam_count_to_model(self):
        class FakeInputs(dict):
            def to(self, _device, _dtype):
                return self

        class FakeOutput:
            def __getitem__(self, key):
                self.key = key
                return "generated-only"

        class FakeProcessor:
            def apply_transcription_request(self, **kwargs):
                self.request = kwargs
                return FakeInputs(input_ids=SimpleNamespace(shape=(1, 3)))

            def decode(self, generated_ids, **kwargs):
                self.decode_call = (generated_ids, kwargs)
                return ["確認"]

        class FakeModel:
            device = "cuda:0"
            dtype = "bfloat16"

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                return FakeOutput()

        processor = FakeProcessor()
        model = FakeModel()
        runtime = QwenHfAsrRuntime()

        with (
            mock.patch.object(runtime, "load_model", return_value=(processor, model)),
            mock.patch.dict(
                os.environ,
                {
                    "CASRT_QWEN_HF_ASR_NUM_BEAMS": "5",
                    "CASRT_QWEN_HF_ASR_MAX_NEW_TOKENS": "128",
                },
                clear=True,
            ),
        ):
            result = runtime.generate_result("/models/qwen-hf", mono_wav_bytes(), "ja", 7)

        self.assertEqual(result.text, "確認")
        self.assertEqual(processor.request["language"], "Japanese")
        self.assertEqual(model.generate_kwargs["num_beams"], 5)
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 128)
        self.assertFalse(model.generate_kwargs["do_sample"])
        self.assertEqual(processor.decode_call[0], "generated-only")
        self.assertEqual(processor.decode_call[1], {"return_format": "transcription_only"})

    def test_network_guard_blocks_python_socket_creation(self):
        original_socket = socket.socket
        original_create_connection = socket.create_connection
        try:
            with mock.patch.dict(os.environ, {"CASRT_QWEN_HF_ASR_DISABLE_NETWORK": "1"}, clear=True):
                disable_python_network_if_requested()
                with self.assertRaisesRegex(OSError, "network access is disabled"):
                    socket.socket()
                with self.assertRaisesRegex(OSError, "network access is disabled"):
                    socket.create_connection(("127.0.0.1", 1))
        finally:
            socket.socket = original_socket
            socket.create_connection = original_create_connection
            qwen_hf_asr_worker._NETWORK_DISABLED = False


if __name__ == "__main__":
    unittest.main()
