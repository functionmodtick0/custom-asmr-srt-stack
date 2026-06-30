from __future__ import annotations

import base64
import json
import os
import socket
import sys
import tempfile
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
TRUE_ENV_VALUES = {"1", "true", "yes"}
_NETWORK_DISABLED = False


@dataclass(frozen=True)
class QwenHfAsrResult:
    text: str
    start_ms: int
    end_ms: int


class QwenHfAsrRuntime:
    def __init__(self) -> None:
        self._loaded: dict[str, tuple[Any, Any]] = {}

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

    def generate_result(self, model_id: str, audio_bytes: bytes, source_language: str, duration_ms: int) -> QwenHfAsrResult:
        processor, model = self.load_model(model_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "clip.wav"
            audio_path.write_bytes(prepare_audio_for_asr(audio_bytes))
            language = qwen_language(source_language)
            kwargs: dict[str, Any] = {"audio": str(audio_path)}
            if language is not None:
                kwargs["language"] = language
            inputs = processor.apply_transcription_request(**kwargs)
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device, model.dtype)
            output_ids = model.generate(**inputs, max_new_tokens=qwen_hf_max_new_tokens(), do_sample=False)
            input_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, input_length:]
            text = processor.decode(generated_ids, return_format="transcription_only")[0]
        return QwenHfAsrResult(text=str(text), start_ms=0, end_ms=duration_ms)

    def load_model(self, model_id: str) -> tuple[Any, Any]:
        require_secure_runtime()
        disable_python_network_if_requested()
        checked_model_id = checked_model_path(model_id, "model_id")
        loaded = self._loaded.get(checked_model_id)
        if loaded is not None:
            return loaded

        log(f"loading local Qwen HF ASR model: {checked_model_id}")
        try:
            import torch
            from transformers import AutoModelForMultimodalLM
            from transformers import AutoProcessor
        except ImportError as error:
            raise ValueError("local Qwen HF ASR worker requires the local extra: uv sync --extra local") from error

        kwargs = default_load_kwargs(torch)
        kwargs["local_files_only"] = True
        kwargs["trust_remote_code"] = False
        kwargs["use_safetensors"] = True
        processor = AutoProcessor.from_pretrained(
            checked_model_id,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            checked_model_id,
            **kwargs,
        ).eval()
        loaded = (processor, model)
        self._loaded[checked_model_id] = loaded
        log("model loaded")
        return loaded


def default_load_kwargs(torch_module: Any) -> dict[str, Any]:
    dtype = torch_dtype(torch_module, os.environ.get("CASRT_QWEN_HF_ASR_DTYPE", "bfloat16"))
    device_map = os.environ.get("CASRT_QWEN_HF_ASR_DEVICE_MAP")
    if device_map is None:
        device_map = "cuda:0" if torch_module.cuda.is_available() else ""

    result: dict[str, Any] = {"dtype": dtype}
    if device_map.strip():
        result["device_map"] = device_map.strip()
    return result


def require_secure_runtime() -> None:
    if os.environ.get("CASRT_LOCAL_WORKER_ENV_MODE", "").strip().lower() != "offline":
        raise ValueError("CASRT_LOCAL_WORKER_ENV_MODE=offline is required for Qwen HF ASR worker")
    for name in (
        "CASRT_QWEN_HF_ASR_REQUIRE_LOCAL_MODEL_PATH",
        "CASRT_QWEN_HF_ASR_LOCAL_FILES_ONLY",
        "CASRT_QWEN_HF_ASR_DISABLE_NETWORK",
    ):
        if os.environ.get(name, "").strip().lower() not in TRUE_ENV_VALUES:
            raise ValueError(f"{name}=1 is required for Qwen HF ASR worker")


def checked_model_path(value: str, name: str) -> str:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{name} must be an existing local model directory") from error
    if not path.is_dir():
        raise ValueError(f"{name} must be an existing local model directory")
    return str(path)


def disable_python_network_if_requested() -> None:
    global _NETWORK_DISABLED
    if _NETWORK_DISABLED:
        return
    if os.environ.get("CASRT_QWEN_HF_ASR_DISABLE_NETWORK", "").strip().lower() not in TRUE_ENV_VALUES:
        return

    original_socket = socket.socket

    class BlockedSocket(original_socket):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise OSError("network access is disabled for local Qwen HF ASR worker")

    def blocked_create_connection(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("network access is disabled for local Qwen HF ASR worker")

    socket.socket = BlockedSocket
    socket.create_connection = blocked_create_connection  # type: ignore[assignment]
    _NETWORK_DISABLED = True


def qwen_language(source_language: str) -> str | None:
    if os.environ.get("CASRT_QWEN_HF_ASR_FORCE_LANGUAGE", "1").strip().lower() in {"0", "false", "no"}:
        return None
    return SOURCE_LANGUAGE_TO_QWEN.get(source_language.strip().lower())


def qwen_hf_max_new_tokens() -> int:
    raw_value = os.environ.get("CASRT_QWEN_HF_ASR_MAX_NEW_TOKENS", "256").strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError("CASRT_QWEN_HF_ASR_MAX_NEW_TOKENS must be a positive integer") from error
    if value <= 0:
        raise ValueError("CASRT_QWEN_HF_ASR_MAX_NEW_TOKENS must be a positive integer")
    return value


def torch_dtype(torch_module: Any, value: str) -> Any:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    if normalized == "auto":
        return "auto"
    raise ValueError("CASRT_QWEN_HF_ASR_DTYPE must be one of: bfloat16, float16, float32, auto")


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


def handle_request(runtime: QwenHfAsrRuntime, request: dict[str, Any]) -> dict[str, Any]:
    request_type = request.get("type")
    if request_type != "transcribe":
        raise ValueError(f"unsupported request type {request_type!r}")
    return {"ok": True, "segments": runtime.transcribe(request)}


def response_for_line(runtime: QwenHfAsrRuntime, line: str) -> dict[str, Any]:
    try:
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        return handle_request(runtime, request)
    except Exception as error:
        detail = str(error) or error.__class__.__name__
        return {"ok": False, "error": detail}


def log(message: str) -> None:
    print(f"[casrt-qwen-hf-asr-worker] {message}", file=sys.stderr, flush=True)


def main() -> int:
    runtime = QwenHfAsrRuntime()
    for line in sys.stdin:
        response = response_for_line(runtime, line)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
