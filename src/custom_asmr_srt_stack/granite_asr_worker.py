from __future__ import annotations

import base64
import json
import os
import socket
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from custom_asmr_srt_stack.audio import analyze_wav
from custom_asmr_srt_stack.local_asr import clean_transcription_text, prepare_audio_for_asr

SOURCE_LANGUAGE_TO_GRANITE = {
    "ja": "ja",
    "ja-jp": "ja",
    "ja_jp": "ja",
    "jpn": "ja",
    "japanese": "ja",
}
GRANITE_SAMPLE_RATE = 16_000
GRANITE_DEFAULT_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."
_NETWORK_DISABLED = False


@dataclass(frozen=True)
class GraniteAsrResult:
    text: str
    start_ms: int
    end_ms: int


class GraniteAsrRuntime:
    def __init__(self) -> None:
        self._loaded: dict[str, tuple[Any, Any]] = {}

    def transcribe(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        model_id = granite_checked_model_path(require_string(request.get("model_id"), "model_id"), "model_id")
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

    def generate_result(
        self,
        model_id: str,
        audio_bytes: bytes,
        source_language: str,
        duration_ms: int,
    ) -> GraniteAsrResult:
        if not granite_language(source_language):
            raise ValueError(f"unsupported Granite ASR source_language {source_language!r}")
        model, processor = self.load_model(model_id)
        tokenizer = processor.tokenizer
        waveform = wav_bytes_to_float_mono_16k(prepare_audio_for_asr(audio_bytes))

        import torch

        wav_tensor = torch.from_numpy(waveform).unsqueeze(0)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": granite_prompt()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        device = granite_model_device(model)
        model_inputs = processor(prompt, wav_tensor, device=device, return_tensors="pt")
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(device)
        with torch.no_grad():
            model_outputs = model.generate(
                **model_inputs,
                max_new_tokens=granite_max_new_tokens(),
                do_sample=False,
                num_beams=1,
            )
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        texts = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
        text = texts[0] if texts else ""
        return GraniteAsrResult(text=text, start_ms=0, end_ms=duration_ms)

    def load_model(self, model_id: str) -> tuple[Any, Any]:
        disable_python_network_if_requested()
        loaded = self._loaded.get(model_id)
        if loaded is not None:
            return loaded

        log(f"loading local Granite ASR model: {model_id}")
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        kwargs = granite_model_kwargs(torch)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **kwargs)
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
        model.eval()
        loaded_pair = (model, processor)
        self._loaded[model_id] = loaded_pair
        log("model loaded")
        return loaded_pair


def granite_model_kwargs(torch_module: Any) -> dict[str, Any]:
    dtype = torch_dtype(torch_module, os.environ.get("CASRT_GRANITE_ASR_DTYPE", "bfloat16"))
    device_map = os.environ.get("CASRT_GRANITE_ASR_DEVICE_MAP")
    if device_map is None:
        device_map = "cuda:0" if torch_module.cuda.is_available() else ""
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    if device_map.strip():
        kwargs["device_map"] = device_map.strip()
    return kwargs


def granite_checked_model_path(value: str, name: str) -> str:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{name} must be an existing local Granite ASR snapshot directory") from error
    if not path.is_dir():
        raise ValueError(f"{name} must be an existing local Granite ASR snapshot directory")
    if not any(path.glob("*.safetensors")) and not (path / "model.safetensors.index.json").is_file():
        raise ValueError(f"{name} must contain safetensors model weights")
    return str(path)


def granite_language(source_language: str) -> str | None:
    return SOURCE_LANGUAGE_TO_GRANITE.get(source_language.strip().lower())


def granite_prompt() -> str:
    return os.environ.get("CASRT_GRANITE_ASR_PROMPT", GRANITE_DEFAULT_PROMPT)


def granite_max_new_tokens() -> int:
    return int(os.environ.get("CASRT_GRANITE_ASR_MAX_NEW_TOKENS", "512"))


def granite_model_device(model: Any) -> str:
    device = getattr(model, "device", None)
    return str(device) if device is not None else "cpu"


def wav_bytes_to_float_mono_16k(audio_bytes: bytes) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "clip.wav"
        path.write_bytes(audio_bytes)
        import librosa

        waveform, _ = librosa.load(path, sr=GRANITE_SAMPLE_RATE, mono=True)
    return np.asarray(waveform, dtype=np.float32)


def torch_dtype(torch_module: Any, value: str) -> Any:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError("CASRT_GRANITE_ASR_DTYPE must be bfloat16, float16, or float32")


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


def disable_python_network_if_requested() -> None:
    global _NETWORK_DISABLED
    if _NETWORK_DISABLED:
        return
    if os.environ.get("CASRT_GRANITE_ASR_DISABLE_NETWORK", "").strip().lower() not in {"1", "true", "yes"}:
        return

    original_socket = socket.socket

    class BlockedSocket(original_socket):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise OSError("network access is disabled for local Granite ASR worker")

    def blocked_create_connection(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("network access is disabled for local Granite ASR worker")

    socket.socket = BlockedSocket
    socket.create_connection = blocked_create_connection  # type: ignore[assignment]
    _NETWORK_DISABLED = True


def log(message: str) -> None:
    print(f"[casrt-granite-asr-worker] {message}", file=sys.stderr, flush=True)


def handle_request(runtime: GraniteAsrRuntime, request: dict[str, Any]) -> dict[str, Any]:
    request_type = request.get("type")
    if request_type != "transcribe":
        raise ValueError(f"unsupported request type {request_type!r}")
    return {"ok": True, "segments": runtime.transcribe(request)}


def response_for_line(runtime: GraniteAsrRuntime, line: str) -> dict[str, Any]:
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
    runtime = GraniteAsrRuntime()
    for line in sys.stdin:
        response = response_for_line(runtime, line)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
