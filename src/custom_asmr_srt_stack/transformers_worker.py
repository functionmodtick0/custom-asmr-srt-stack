from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
import tempfile
import traceback
import wave
from array import array
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import analyze_wav

PCM16_MAX = 32767
ASR_TARGET_RMS_DBFS = -24.0
ASR_MAX_PEAK_DBFS = -3.0
ASR_MAX_GAIN = 4.0
JAPANESE_CHAR_CLASS = r"\u3040-\u30ff\u3400-\u9fff々〆〤ヶー"
JAPANESE_OR_PUNCTUATION_CLASS = JAPANESE_CHAR_CLASS + r"。、，,.！？!?…「」『』（）【】《》〈〉"

LOCAL_TRANSCRIPTION_PROMPT = (
    "Transcribe this Japanese audio exactly. "
    "Return Japanese text only. Do not translate. "
    "Do not add timestamps, channel labels, explanations, or markdown."
)


class TransformersRuntime:
    def __init__(self) -> None:
        self._loaded: dict[str, tuple[Any, Any]] = {}

    def transcribe(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        model_id = require_string(request.get("model_id"), "model_id")
        channel = require_string(request.get("channel"), "channel")
        audio_bytes = decode_audio(request.get("audio_base64"))
        duration_ms = analyze_wav(audio_bytes).duration_ms
        if duration_ms <= 0:
            return []

        text = self.generate_text(model_id, audio_bytes)
        cleaned = clean_transcription_text(text)
        if not cleaned:
            return []
        return [
            {
                "start_ms": 0,
                "end_ms": duration_ms,
                "channel": channel,
                "kind": "speech",
                "text": cleaned,
                "needs_review": True,
            }
        ]

    def generate_text(self, model_id: str, audio_bytes: bytes) -> str:
        processor, model = self.load_model(model_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "clip.wav"
            audio_path.write_bytes(prepare_audio_for_asr(audio_bytes))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": str(audio_path)},
                        {"type": "text", "text": LOCAL_TRANSCRIPTION_PROMPT},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            input_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, input_length:]
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def load_model(self, model_id: str) -> tuple[Any, Any]:
        loaded = self._loaded.get(model_id)
        if loaded is not None:
            return loaded

        log(f"loading local Transformers model: {model_id}")
        try:
            import torch
            from transformers import AutoProcessor
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise ValueError(
                "local Transformers worker requires the local extra: uv sync --extra local"
            ) from error

        model_class = import_model_class()
        log("loading processor")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        log("loading model weights")
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        if quantization_mode() == "4bit":
            log("using 4-bit runtime quantization; skipping lm_head and audio tower")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["lm_head", "model.audio_tower"],
            )
        model = model_class.from_pretrained(
            model_id,
            **model_kwargs,
        ).eval()
        del torch

        loaded = (processor, model)
        self._loaded[model_id] = loaded
        log("model loaded")
        return loaded


def import_model_class() -> Any:
    try:
        from transformers import Gemma4ForConditionalGeneration

        return Gemma4ForConditionalGeneration
    except ImportError:
        pass

    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError as error:
        raise ValueError("installed Transformers does not expose a Gemma 4 compatible model class") from error


def decode_audio(value: Any) -> bytes:
    encoded = require_string(value, "audio_base64")
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("audio_base64 must be valid base64") from error
    if not audio_bytes:
        raise ValueError("audio_base64 must not be empty")
    return audio_bytes


def clean_transcription_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    prefixes = ("Transcription:", "Transcript:", "文字起こし:", "書き起こし:")
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = strip_non_japanese_noise_edges(text)
    text = re.sub(fr"([{JAPANESE_CHAR_CLASS}])\s+([{JAPANESE_CHAR_CLASS}])", r"\1\2", text)
    text = re.sub(fr"\s+([{JAPANESE_OR_PUNCTUATION_CLASS}])", r"\1", text)
    text = re.sub(fr"([{JAPANESE_OR_PUNCTUATION_CLASS}])\s+", r"\1", text)
    return text.strip()


def strip_non_japanese_noise_edges(text: str) -> str:
    text = re.sub(fr"^[^{JAPANESE_OR_PUNCTUATION_CLASS}]+", "", text)
    return re.sub(fr"[^{JAPANESE_OR_PUNCTUATION_CLASS}]+$", "", text)


def prepare_audio_for_asr(audio_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        params = wav.getparams()
        frames = wav.readframes(wav.getnframes())

    if params.sampwidth != 2 or not frames:
        return audio_bytes

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return audio_bytes

    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak = max(abs(sample) for sample in samples)
    if rms <= 0 or peak <= 0:
        return audio_bytes

    target_rms = dbfs_to_pcm_amplitude(ASR_TARGET_RMS_DBFS)
    max_peak = dbfs_to_pcm_amplitude(ASR_MAX_PEAK_DBFS)
    gain = min(ASR_MAX_GAIN, target_rms / rms, max_peak / peak)
    if gain <= 1.01:
        return audio_bytes

    boosted = array("h", (clip_pcm16(round(sample * gain)) for sample in samples))
    if sys.byteorder != "little":
        boosted.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams(params)
        wav.writeframes(boosted.tobytes())
    return output.getvalue()


def dbfs_to_pcm_amplitude(dbfs: float) -> float:
    return PCM16_MAX * (10 ** (dbfs / 20))


def clip_pcm16(value: int) -> int:
    return max(-32768, min(32767, value))


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def log(message: str) -> None:
    print(f"[casrt-worker] {message}", file=sys.stderr, flush=True)


def quantization_mode() -> str | None:
    value = os.environ.get("CASRT_TRANSFORMERS_QUANTIZATION", "").strip().lower()
    return value or None


def handle_request(runtime: TransformersRuntime, request: dict[str, Any]) -> dict[str, Any]:
    request_type = request.get("type")
    if request_type != "transcribe":
        raise ValueError(f"unsupported request type {request_type!r}")
    return {"ok": True, "segments": runtime.transcribe(request)}


def response_for_line(runtime: TransformersRuntime, line: str) -> dict[str, Any]:
    try:
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        response = handle_request(runtime, request)
    except Exception as error:
        detail = str(error) or error.__class__.__name__
        response = {"ok": False, "error": detail, "traceback": traceback.format_exc()}
    return response


def main() -> int:
    runtime = TransformersRuntime()
    for line in sys.stdin:
        response = response_for_line(runtime, line)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
