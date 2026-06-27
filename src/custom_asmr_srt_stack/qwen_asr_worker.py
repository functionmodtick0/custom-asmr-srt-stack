from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import analyze_wav
from custom_asmr_srt_stack.local_asr import clean_transcription_text, prepare_audio_for_asr

SOURCE_LANGUAGE_TO_QWEN = {
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "ja_jp": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
}


@dataclass(frozen=True)
class QwenAsrResult:
    text: str
    language: str
    start_ms: int
    end_ms: int
    aligned: bool


class QwenAsrRuntime:
    def __init__(self) -> None:
        self._loaded: dict[str, Any] = {}

    def transcribe(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        model_id = require_string(request.get("model_id"), "model_id")
        channel = require_string(request.get("channel"), "channel")
        source_language = require_string(request.get("source_language", "ja"), "source_language")
        audio_bytes = decode_audio(request.get("audio_base64"))
        duration_ms = analyze_wav(audio_bytes).duration_ms
        if duration_ms <= 0:
            return []

        result = self.generate_result(model_id, audio_bytes, source_language, duration_ms)
        text = clean_transcription_text(result.text)
        if not text:
            return []
        return [
            {
                "start_ms": result.start_ms,
                "end_ms": result.end_ms,
                "channel": channel,
                "kind": "speech",
                "text": text,
                "needs_review": True,
            }
        ]

    def generate_result(self, model_id: str, audio_bytes: bytes, source_language: str, duration_ms: int) -> QwenAsrResult:
        model = self.load_model(model_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "clip.wav"
            audio_path.write_bytes(prepare_audio_for_asr(audio_bytes))
            transcriptions = model.transcribe(
                audio=str(audio_path),
                context=qwen_asr_context(),
                language=qwen_language(source_language),
                return_time_stamps=bool(qwen_aligner_model_id()),
            )
        if not transcriptions:
            return QwenAsrResult(text="", language="", start_ms=0, end_ms=duration_ms, aligned=False)
        result = transcriptions[0]
        start_ms, end_ms, aligned = aligned_bounds_ms(result, duration_ms)
        return QwenAsrResult(
            text=str(getattr(result, "text", "")),
            language=str(getattr(result, "language", "")),
            start_ms=start_ms,
            end_ms=end_ms,
            aligned=aligned,
        )

    def load_model(self, model_id: str) -> Any:
        loaded = self._loaded.get(model_id)
        if loaded is not None:
            return loaded

        log(f"loading local Qwen ASR model: {model_id}")
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as error:
            raise ValueError(
                "local Qwen ASR worker requires qwen-asr in an isolated environment. "
                "Create a separate uv venv with qwen-asr installed, then set "
                "CASRT_QWEN_ASR_WORKER_COMMAND to that Python environment."
            ) from error

        backend = qwen_backend()
        kwargs = qwen_backend_kwargs(torch)
        aligner_model_id = qwen_aligner_model_id()
        aligner_kwargs = qwen_aligner_kwargs(torch) if aligner_model_id else None
        log(f"using {backend} backend")
        if backend == "vllm":
            model = Qwen3ASRModel.LLM(
                model=model_id,
                forced_aligner=aligner_model_id,
                forced_aligner_kwargs=aligner_kwargs,
                max_inference_batch_size=qwen_max_inference_batch_size(),
                max_new_tokens=qwen_max_new_tokens(),
                **kwargs,
            )
        elif backend == "transformers":
            model = Qwen3ASRModel.from_pretrained(
                model_id,
                forced_aligner=aligner_model_id,
                forced_aligner_kwargs=aligner_kwargs,
                max_inference_batch_size=qwen_max_inference_batch_size(),
                max_new_tokens=qwen_max_new_tokens(),
                **kwargs,
            )
        else:
            raise ValueError("CASRT_QWEN_ASR_BACKEND must be transformers or vllm")

        self._loaded[model_id] = model
        log("model loaded")
        return model


def aligned_bounds_ms(result: Any, duration_ms: int) -> tuple[int, int, bool]:
    timestamps = getattr(result, "time_stamps", None)
    items = getattr(timestamps, "items", None)
    if not items:
        return 0, duration_ms, False

    starts = [round(float(item.start_time) * 1000) for item in items if getattr(item, "start_time", None) is not None]
    ends = [round(float(item.end_time) * 1000) for item in items if getattr(item, "end_time", None) is not None]
    if not starts or not ends:
        return 0, duration_ms, False

    start_ms = max(0, min(duration_ms, min(starts)))
    end_ms = max(start_ms + 1, min(duration_ms, max(ends)))
    return start_ms, end_ms, True


def qwen_language(source_language: str) -> str | None:
    if not qwen_force_language():
        return None
    return SOURCE_LANGUAGE_TO_QWEN.get(source_language.strip().lower())


def qwen_force_language() -> bool:
    return os.environ.get("CASRT_QWEN_ASR_FORCE_LANGUAGE", "1").strip().lower() not in {"0", "false", "no"}


def qwen_backend() -> str:
    return os.environ.get("CASRT_QWEN_ASR_BACKEND", "transformers").strip().lower()


def qwen_backend_kwargs(torch_module: Any) -> dict[str, Any]:
    kwargs = json_object_env("CASRT_QWEN_ASR_BACKEND_KWARGS")
    if kwargs:
        return coerce_torch_dtype_kwargs(torch_module, kwargs)
    return qwen_default_transformers_kwargs(torch_module)


def qwen_default_transformers_kwargs(torch_module: Any) -> dict[str, Any]:
    dtype = torch_dtype(torch_module, os.environ.get("CASRT_QWEN_ASR_DTYPE", "bfloat16"))
    device_map = os.environ.get("CASRT_QWEN_ASR_DEVICE_MAP")
    if device_map is None:
        device_map = "cuda:0" if torch_module.cuda.is_available() else ""

    result: dict[str, Any] = {"dtype": dtype}
    if device_map.strip():
        result["device_map"] = device_map.strip()
    return result


def qwen_aligner_kwargs(torch_module: Any) -> dict[str, Any]:
    kwargs = json_object_env("CASRT_QWEN_ASR_ALIGNER_KWARGS")
    if kwargs:
        return coerce_torch_dtype_kwargs(torch_module, kwargs)
    return qwen_default_transformers_kwargs(torch_module)


def coerce_torch_dtype_kwargs(torch_module: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(kwargs)
    dtype = coerced.get("dtype")
    if isinstance(dtype, str):
        coerced["dtype"] = torch_dtype(torch_module, dtype)
    return coerced


def qwen_aligner_model_id() -> str | None:
    value = os.environ.get("CASRT_QWEN_ASR_ALIGNER_MODEL_ID", "").strip()
    return value or None


def qwen_asr_context() -> str:
    return os.environ.get("CASRT_QWEN_ASR_CONTEXT", "")


def qwen_max_inference_batch_size() -> int:
    return int(os.environ.get("CASRT_QWEN_ASR_MAX_BATCH", "32"))


def qwen_max_new_tokens() -> int:
    return int(os.environ.get("CASRT_QWEN_ASR_MAX_NEW_TOKENS", "512"))


def json_object_env(name: str) -> dict[str, Any]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def torch_dtype(torch_module: Any, value: str) -> Any:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError("CASRT_QWEN_ASR_DTYPE must be bfloat16, float16, or float32")


def decode_audio(value: Any) -> bytes:
    encoded = require_string(value, "audio_base64")
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("audio_base64 must be valid base64") from error
    if not audio_bytes:
        raise ValueError("audio_base64 must not be empty")
    return audio_bytes


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def log(message: str) -> None:
    print(f"[casrt-qwen-asr-worker] {message}", file=sys.stderr, flush=True)


def handle_request(runtime: QwenAsrRuntime, request: dict[str, Any]) -> dict[str, Any]:
    request_type = request.get("type")
    if request_type != "transcribe":
        raise ValueError(f"unsupported request type {request_type!r}")
    return {"ok": True, "segments": runtime.transcribe(request)}


def response_for_line(runtime: QwenAsrRuntime, line: str) -> dict[str, Any]:
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
    runtime = QwenAsrRuntime()
    for line in sys.stdin:
        response = response_for_line(runtime, line)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
