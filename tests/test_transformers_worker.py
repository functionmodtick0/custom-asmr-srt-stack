import base64
import io
import json
import os
import struct
import unittest
import wave
from unittest import mock

from custom_asmr_srt_stack.transformers_worker import (
    DEFAULT_MAX_NEW_TOKENS,
    TransformersRuntime,
    clean_transcription_text,
    max_new_tokens,
    prepare_audio_for_asr,
    quantization_config,
    quantization_mode,
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
