from __future__ import annotations

import base64
import hashlib
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
QWEN_HF_ALLOWED_ROOT_FILES = {
    ".gitattributes",
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
QWEN_HF_CACHE_FILES = {".gitignore", "CACHEDIR.TAG"}
QWEN_HF_DYNAMIC_CONFIG_KEYS = {
    "auto_map",
    "custom_code",
    "remote_code",
    "trust_remote_code",
    "_attn_implementation_internal",
}


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
            output_ids = model.generate(
                **inputs,
                max_new_tokens=qwen_hf_max_new_tokens(),
                num_beams=qwen_hf_num_beams(),
                do_sample=False,
            )
            input_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, input_length:]
            text = processor.decode(generated_ids, return_format="transcription_only")[0]
        return QwenHfAsrResult(text=str(text), start_ms=0, end_ms=duration_ms)

    def load_model(self, model_id: str) -> tuple[Any, Any]:
        require_secure_runtime()
        disable_python_network_if_requested()
        checked_model_id = checked_model_path(model_id, "model_id")
        validate_qwen_hf_snapshot(Path(checked_model_id))
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


def validate_qwen_hf_snapshot(path: Path) -> None:
    root_files = []
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"Qwen HF ASR snapshot must not contain symlinks: {item.relative_to(path)}")
        if not item.is_file():
            continue
        relative = item.relative_to(path)
        if relative.parts[0] == ".cache":
            validate_qwen_hf_cache_file(relative)
            continue
        if len(relative.parts) != 1:
            raise ValueError(f"Qwen HF ASR snapshot file must be at the root: {relative}")
        if relative.name not in QWEN_HF_ALLOWED_ROOT_FILES and not is_safetensors_shard(relative.name):
            raise ValueError(f"Qwen HF ASR snapshot contains an unsupported file: {relative.name}")
        root_files.append(relative.name)

    if "config.json" not in root_files:
        raise ValueError("Qwen HF ASR snapshot requires config.json")
    if not any(name.endswith(".safetensors") for name in root_files):
        raise ValueError("Qwen HF ASR snapshot requires safetensors model weights")
    for name in root_files:
        if name.endswith(".safetensors") and (path / name).stat().st_size <= 0:
            raise ValueError(f"Qwen HF ASR safetensors file must not be empty: {name}")
    validate_qwen_hf_config(path / "config.json")
    validate_qwen_hf_safetensors_index(path, root_files)
    validate_qwen_hf_chat_template(path / "chat_template.jinja")


def validate_qwen_hf_cache_file(relative: Path) -> None:
    if len(relative.parts) < 2 or relative.parts[1] != "huggingface":
        raise ValueError(f"Qwen HF ASR snapshot contains an unsupported cache file: {relative}")
    if relative.name in QWEN_HF_CACHE_FILES or relative.name.endswith(".metadata"):
        return
    raise ValueError(f"Qwen HF ASR snapshot contains an unsupported cache file: {relative}")


def is_safetensors_shard(name: str) -> bool:
    return name.startswith("model-") and name.endswith(".safetensors")


def validate_qwen_hf_config(path: Path) -> None:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Qwen HF ASR config.json is invalid: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("Qwen HF ASR config.json must be a JSON object")
    if config.get("model_type") != "qwen3_asr":
        raise ValueError("Qwen HF ASR config.json model_type must be qwen3_asr")
    architectures = config.get("architectures")
    if architectures != ["Qwen3ASRForConditionalGeneration"]:
        raise ValueError("Qwen HF ASR config.json must use Qwen3ASRForConditionalGeneration")
    dynamic_keys = dynamic_config_paths(config)
    if dynamic_keys:
        raise ValueError(f"Qwen HF ASR config.json contains dynamic code settings: {', '.join(dynamic_keys)}")


def dynamic_config_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in QWEN_HF_DYNAMIC_CONFIG_KEYS and item is not None and item is not False:
                result.append(path)
            result.extend(dynamic_config_paths(item, path))
        return sorted(result)
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(dynamic_config_paths(item, f"{prefix}[{index}]"))
        return sorted(result)
    return []


def validate_qwen_hf_safetensors_index(path: Path, root_files: list[str]) -> None:
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Qwen HF ASR safetensors index is invalid: {error}") from error
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("Qwen HF ASR safetensors index requires a weight_map object")
    raw_shard_names = list(index["weight_map"].values())
    if not raw_shard_names or any(not isinstance(name, str) for name in raw_shard_names):
        raise ValueError("Qwen HF ASR safetensors index must reference model-*.safetensors shards")
    shard_names = set(raw_shard_names)
    if any(not is_safetensors_shard(name) for name in shard_names):
        raise ValueError("Qwen HF ASR safetensors index must reference model-*.safetensors shards")
    missing = sorted(name for name in shard_names if name not in root_files)
    if missing:
        raise ValueError(f"Qwen HF ASR safetensors index references missing shards: {', '.join(missing)}")


def validate_qwen_hf_chat_template(path: Path) -> None:
    if not path.is_file():
        return
    expected = os.environ.get("CASRT_QWEN_HF_ASR_EXPECTED_CHAT_TEMPLATE_SHA256", "").strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("CASRT_QWEN_HF_ASR_EXPECTED_CHAT_TEMPLATE_SHA256 must be a SHA-256 hex digest")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            "Qwen HF ASR chat_template.jinja SHA-256 mismatch: "
            f"expected {expected}, got {actual}"
        )


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


def qwen_hf_num_beams() -> int:
    raw_value = os.environ.get("CASRT_QWEN_HF_ASR_NUM_BEAMS", "1").strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError("CASRT_QWEN_HF_ASR_NUM_BEAMS must be a positive integer") from error
    if value <= 0:
        raise ValueError("CASRT_QWEN_HF_ASR_NUM_BEAMS must be a positive integer")
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
