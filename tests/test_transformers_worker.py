import base64
import io
import json
import os
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

from custom_asmr_srt_stack.transformers_worker import (
    DEFAULT_MAX_NEW_TOKENS,
    TransformersRuntime,
    checked_model_path,
    clean_transcription_text,
    max_new_tokens,
    prepare_audio_for_asr,
    quantization_config,
    quantization_mode,
    require_secure_runtime,
    response_for_line,
)


def mono_wav_bytes(duration_ms: int = 2) -> bytes:
    return mono_wav_from_samples([100] * duration_ms)


def mono_wav_from_samples(samples: list[int]) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    return output.getvalue()


def read_mono_samples(audio_bytes: bytes) -> list[int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return [struct.unpack("<h", frames[index : index + 2])[0] for index in range(0, len(frames), 2)]


class FakeRuntime(TransformersRuntime):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def generate_text(self, model_id, audio_bytes):
        del model_id, audio_bytes
        return self.text


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTorch:
    bfloat16 = "bf16"

    class cuda:
        @staticmethod
        def is_available():
            return False


class FakeProcessorLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        return "processor"


class FakeModel:
    def eval(self):
        return self


class FakeModelLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        return FakeModel()


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

    def test_clean_transcription_text_compacts_japanese_spacing_and_noise(self):
        self.assertEqual(
            clean_transcription_text("ียบ みつかっ ちゃっ た 。 ねえ ねえ 、 魔女 ちゃん 、 こいつ 強い ? えっ と"),
            "みつかっちゃった。ねえねえ、魔女ちゃん、こいつ強い?えっと",
        )

    def test_clean_transcription_text_drops_non_japanese_hallucination_segments(self):
        self.assertEqual(clean_transcription_text("!"), "")
        self.assertEqual(clean_transcription_text(",yes,I know.I know."), "")
        self.assertEqual(clean_transcription_text("you and your men!"), "")
        self.assertEqual(clean_transcription_text("ねえ!"), "ねえ!")

    def test_prepare_audio_for_asr_boosts_quiet_pcm16_audio(self):
        prepared = prepare_audio_for_asr(mono_wav_from_samples([100, -100, 100]))

        self.assertEqual(read_mono_samples(prepared), [400, -400, 400])

    def test_prepare_audio_for_asr_caps_gain_at_peak_headroom(self):
        prepared = prepare_audio_for_asr(mono_wav_from_samples(([10] * 999) + [8000]))

        samples = read_mono_samples(prepared)
        self.assertGreater(samples[0], 10)
        self.assertLessEqual(max(abs(sample) for sample in samples), 23200)

    def test_quantization_config_supports_4bit_and_8bit_without_quantizing_audio_tower(self):
        four_bit = quantization_config("4bit", FakeTorch, FakeBitsAndBytesConfig)
        eight_bit = quantization_config("8bit", FakeTorch, FakeBitsAndBytesConfig)

        self.assertEqual(
            four_bit.kwargs,
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bf16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "llm_int8_skip_modules": ["lm_head", "model.audio_tower"],
            },
        )
        self.assertEqual(
            eight_bit.kwargs,
            {
                "load_in_8bit": True,
                "llm_int8_skip_modules": ["lm_head", "model.audio_tower"],
            },
        )

    def test_quantization_config_rejects_unknown_modes(self):
        with self.assertRaisesRegex(ValueError, "CASRT_TRANSFORMERS_QUANTIZATION"):
            quantization_config("int2", FakeTorch, FakeBitsAndBytesConfig)

    def test_quantization_mode_reads_normalized_environment_value(self):
        with mock.patch.dict(os.environ, {"CASRT_TRANSFORMERS_QUANTIZATION": " 8BIT "}):
            self.assertEqual(quantization_mode(), "8bit")

    def test_secure_runtime_requires_offline_local_transformers_environment(self):
        secure_env = {
            "CASRT_LOCAL_WORKER_ENV_MODE": "offline",
            "CASRT_TRANSFORMERS_REQUIRE_LOCAL_MODEL_PATH": "1",
            "CASRT_TRANSFORMERS_LOCAL_FILES_ONLY": "1",
            "CASRT_TRANSFORMERS_DISABLE_NETWORK": "1",
        }

        with mock.patch.dict(os.environ, secure_env, clear=True):
            require_secure_runtime()

        for missing_name in secure_env:
            incomplete = dict(secure_env)
            incomplete.pop(missing_name)
            with self.subTest(missing=missing_name):
                with mock.patch.dict(os.environ, incomplete, clear=True):
                    with self.assertRaisesRegex(ValueError, "local Transformers worker"):
                        require_secure_runtime()

    def test_checked_model_path_requires_existing_local_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()

            self.assertEqual(checked_model_path(str(model_dir), "model_id"), str(model_dir.resolve()))

            with self.assertRaisesRegex(ValueError, "existing local Transformers model directory"):
                checked_model_path(str(model_dir / "missing"), "model_id")

    def test_load_model_uses_local_files_without_remote_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            FakeProcessorLoader.calls.clear()
            FakeModelLoader.calls.clear()
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            secure_env = {
                "CASRT_LOCAL_WORKER_ENV_MODE": "offline",
                "CASRT_TRANSFORMERS_REQUIRE_LOCAL_MODEL_PATH": "1",
                "CASRT_TRANSFORMERS_LOCAL_FILES_ONLY": "1",
                "CASRT_TRANSFORMERS_DISABLE_NETWORK": "1",
            }
            fake_transformers = types.SimpleNamespace(
                AutoProcessor=FakeProcessorLoader,
                BitsAndBytesConfig=FakeBitsAndBytesConfig,
            )

            with (
                mock.patch.dict(os.environ, secure_env, clear=True),
                mock.patch.dict(sys.modules, {"torch": FakeTorch, "transformers": fake_transformers}),
                mock.patch(
                    "custom_asmr_srt_stack.transformers_worker.import_model_class",
                    return_value=FakeModelLoader,
                ),
                mock.patch("custom_asmr_srt_stack.transformers_worker.disable_python_network_if_requested"),
            ):
                processor, model = TransformersRuntime().load_model(str(model_dir))

        self.assertEqual(processor, "processor")
        self.assertIsInstance(model, FakeModel)
        self.assertEqual(
            FakeProcessorLoader.calls,
            [
                (
                    str(model_dir.resolve()),
                    {"local_files_only": True, "trust_remote_code": False},
                )
            ],
        )
        self.assertEqual(FakeModelLoader.calls[0][0], str(model_dir.resolve()))
        self.assertEqual(FakeModelLoader.calls[0][1]["local_files_only"], True)
        self.assertEqual(FakeModelLoader.calls[0][1]["trust_remote_code"], False)
        self.assertEqual(FakeModelLoader.calls[0][1]["use_safetensors"], True)

    def test_max_new_tokens_defaults_and_reads_environment_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(max_new_tokens(), DEFAULT_MAX_NEW_TOKENS)

        with mock.patch.dict(os.environ, {"CASRT_TRANSFORMERS_MAX_NEW_TOKENS": "128"}):
            self.assertEqual(max_new_tokens(), 128)

    def test_max_new_tokens_rejects_invalid_values(self):
        for value in ("0", "-1", "many"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"CASRT_TRANSFORMERS_MAX_NEW_TOKENS": value}):
                    with self.assertRaisesRegex(ValueError, "CASRT_TRANSFORMERS_MAX_NEW_TOKENS"):
                        max_new_tokens()


if __name__ == "__main__":
    unittest.main()
