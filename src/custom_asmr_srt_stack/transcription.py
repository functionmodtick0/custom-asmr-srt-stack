from __future__ import annotations

import base64
import atexit
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable

from custom_asmr_srt_stack.models import Segment, make_segment_id, require_int, require_mapping, require_string

ADAPTERS = {"openai-compatible", "gemini", "local-transformers", "local-qwen-asr"}
LOCAL_ADAPTERS = {"local-transformers", "local-qwen-asr"}
LOCAL_TRANSFORMERS_MAX_CHUNK_MS = 30_000
LOCAL_WORKER_ENV_ALLOWLIST = {
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "PYTHONPATH",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_HOME",
    "TRANSFORMERS_CACHE",
    "USER",
    "VIRTUAL_ENV",
    "XDG_CACHE_HOME",
}
LOCAL_WORKER_ENV_PREFIXES = ("CASRT_TRANSFORMERS_", "CASRT_QWEN_ASR_")
LOCAL_WORKER_OFFLINE_ENV = {
    "CASRT_QWEN_ASR_DISABLE_NETWORK": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "TOKENIZERS_PARALLELISM": "false",
}
SENSITIVE_ENV_SUBSTRINGS = ("API_KEY", "AUTH", "PASSWORD", "SECRET", "TOKEN")


TRANSCRIPTION_PROMPT = """Transcribe this Japanese audio for subtitle editing.

Return only JSON with this shape:
{
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 1200,
      "channel": "MIX",
      "kind": "speech",
      "text": "..."
    }
  ]
}

Rules:
- Use Japanese text only. Do not translate.
- Put channel metadata in channel, never inside text.
- Allowed channel values: L, R, MIX.
- Allowed kind values: speech, breath, sfx, silence.
- Keep timestamps relative to the provided audio clip.
"""


HttpPost = Callable[[str, dict[str, str], bytes], dict[str, Any]]
LocalTranscribe = Callable[["ModelEndpoint", bytes, str, str, str], tuple[Segment, ...]]


@dataclass(frozen=True)
class ModelEndpoint:
    adapter: str
    endpoint_url: str | None
    model_id: str
    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.adapter not in ADAPTERS:
            raise ValueError(f"unsupported model adapter {self.adapter!r}")
        if self.adapter not in LOCAL_ADAPTERS and not self.endpoint_url:
            raise ValueError("endpoint_url must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")

    @classmethod
    def from_json(cls, value: Any) -> ModelEndpoint:
        data = require_mapping(value, "model endpoint")
        adapter = require_string(data.get("adapter", "openai-compatible"), "model.adapter")
        raw_endpoint_url = data.get("endpoint_url")
        endpoint_url = None if raw_endpoint_url in (None, "") else require_string(raw_endpoint_url, "model.endpoint_url")
        return cls(
            adapter=adapter,
            endpoint_url=endpoint_url,
            model_id=require_string(data.get("model_id"), "model.model_id"),
            api_key=data.get("api_key") or None,
        )


def default_http_post(url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise ValueError(f"model endpoint returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"model endpoint request failed: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"model endpoint returned invalid JSON: {error}") from error


def transcribe_audio(
    endpoint: ModelEndpoint,
    audio_bytes: bytes,
    *,
    mime_type: str,
    channel: str = "MIX",
    source_language: str = "ja",
    http_post: HttpPost = default_http_post,
    local_transcribe_func: LocalTranscribe | None = None,
) -> tuple[Segment, ...]:
    if not audio_bytes:
        raise ValueError("audio_bytes must not be empty")
    if not mime_type:
        raise ValueError("mime_type must not be empty")

    if endpoint.adapter == "local-transformers":
        transcriber = local_transcribe_func or transcribe_with_local_transformers
        return transcriber(endpoint, audio_bytes, mime_type, channel, source_language)
    if endpoint.adapter == "local-qwen-asr":
        transcriber = local_transcribe_func or transcribe_with_local_qwen_asr
        return transcriber(endpoint, audio_bytes, mime_type, channel, source_language)

    if endpoint.adapter == "openai-compatible":
        response = http_post(
            require_endpoint_url(endpoint).rstrip("/") + "/chat/completions",
            openai_headers(endpoint),
            json.dumps(
                build_openai_payload(endpoint, audio_bytes, mime_type, channel=channel, source_language=source_language)
            ).encode("utf-8"),
        )
        return parse_model_segments(extract_openai_content(response))

    response = http_post(
        build_gemini_url(endpoint),
        gemini_headers(endpoint),
        json.dumps(build_gemini_payload(endpoint, audio_bytes, mime_type, channel=channel)).encode("utf-8"),
    )
    return parse_model_segments(extract_gemini_content(response))


def openai_headers(endpoint: ModelEndpoint) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    return headers


def gemini_headers(endpoint: ModelEndpoint) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["x-goog-api-key"] = endpoint.api_key
    return headers


def build_openai_payload(
    endpoint: ModelEndpoint,
    audio_bytes: bytes,
    mime_type: str,
    *,
    channel: str,
    source_language: str,
) -> dict[str, Any]:
    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    return {
        "model": endpoint.model_id,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": TRANSCRIPTION_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"source_language={source_language}; channel={channel}",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": audio_format_from_mime_type(mime_type),
                        },
                    },
                ],
            },
        ],
    }


def build_gemini_payload(
    endpoint: ModelEndpoint,
    audio_bytes: bytes,
    mime_type: str,
    *,
    channel: str,
) -> dict[str, Any]:
    del endpoint
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"{TRANSCRIPTION_PROMPT}\nchannel={channel}"},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }


def build_gemini_url(endpoint: ModelEndpoint) -> str:
    base = require_endpoint_url(endpoint).rstrip("/")
    if ":generateContent" in base:
        return base
    return f"{base}/v1beta/models/{endpoint.model_id}:generateContent"


def require_endpoint_url(endpoint: ModelEndpoint) -> str:
    if not endpoint.endpoint_url:
        raise ValueError("endpoint_url must not be empty")
    return endpoint.endpoint_url


def adapter_max_chunk_ms(endpoint: ModelEndpoint) -> int | None:
    if endpoint.adapter == "local-transformers":
        return LOCAL_TRANSFORMERS_MAX_CHUNK_MS
    return None


def transcribe_with_local_transformers(
    endpoint: ModelEndpoint,
    audio_bytes: bytes,
    mime_type: str,
    channel: str,
    source_language: str,
) -> tuple[Segment, ...]:
    return get_transformers_worker_client().transcribe(endpoint, audio_bytes, mime_type, channel, source_language)


def transcribe_with_local_qwen_asr(
    endpoint: ModelEndpoint,
    audio_bytes: bytes,
    mime_type: str,
    channel: str,
    source_language: str,
) -> tuple[Segment, ...]:
    return get_qwen_asr_worker_client().transcribe(endpoint, audio_bytes, mime_type, channel, source_language)


class TransformersWorkerClient:
    def __init__(self, command: tuple[str, ...], *, worker_name: str = "local transformers worker") -> None:
        self.command = command
        self.worker_name = worker_name
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._request_ids = count(1)

    def transcribe(
        self,
        endpoint: ModelEndpoint,
        audio_bytes: bytes,
        mime_type: str,
        channel: str,
        source_language: str,
    ) -> tuple[Segment, ...]:
        request = {
            "request_id": next(self._request_ids),
            "type": "transcribe",
            "model_id": endpoint.model_id,
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "mime_type": mime_type,
            "channel": channel,
            "source_language": source_language,
        }
        with self._lock:
            process = self._ensure_process()
            if process.stdin is None or process.stdout is None:
                raise ValueError("local transformers worker pipes are unavailable")
            try:
                process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                process.stdin.flush()
                response_line = process.stdout.readline()
            except OSError as error:
                self.close()
                raise ValueError(f"local transformers worker communication failed: {error}") from error

        if not response_line:
            self.close()
            raise ValueError(f"{self.worker_name} exited without a response")
        return parse_worker_response(response_line, worker_name=self.worker_name)

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=local_worker_env(),
        )
        return self._process

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


_WORKER_CLIENTS: dict[tuple[str, tuple[str, ...]], TransformersWorkerClient] = {}
_WORKER_CLIENTS_LOCK = threading.Lock()


def get_transformers_worker_client() -> TransformersWorkerClient:
    command = transformers_worker_command()
    with _WORKER_CLIENTS_LOCK:
        key = ("local transformers worker", command)
        client = _WORKER_CLIENTS.get(key)
        if client is None:
            client = TransformersWorkerClient(command, worker_name="local transformers worker")
            _WORKER_CLIENTS[key] = client
        return client


def get_qwen_asr_worker_client() -> TransformersWorkerClient:
    command = qwen_asr_worker_command()
    with _WORKER_CLIENTS_LOCK:
        key = ("local Qwen ASR worker", command)
        client = _WORKER_CLIENTS.get(key)
        if client is None:
            client = TransformersWorkerClient(command, worker_name="local Qwen ASR worker")
            _WORKER_CLIENTS[key] = client
        return client


def transformers_worker_command() -> tuple[str, ...]:
    raw_command = os.environ.get("CASRT_TRANSFORMERS_WORKER_COMMAND")
    if raw_command:
        command = tuple(shlex.split(raw_command))
        if not command:
            raise ValueError("CASRT_TRANSFORMERS_WORKER_COMMAND must not be empty")
        return command
    return (sys.executable, "-m", "custom_asmr_srt_stack.transformers_worker")


def qwen_asr_worker_command() -> tuple[str, ...]:
    raw_command = os.environ.get("CASRT_QWEN_ASR_WORKER_COMMAND")
    if raw_command:
        command = tuple(shlex.split(raw_command))
        if not command:
            raise ValueError("CASRT_QWEN_ASR_WORKER_COMMAND must not be empty")
        return command
    return (sys.executable, "-m", "custom_asmr_srt_stack.qwen_asr_worker")


def local_worker_env() -> dict[str, str] | None:
    mode = os.environ.get("CASRT_LOCAL_WORKER_ENV_MODE", "inherit").strip().lower()
    if mode in {"", "inherit"}:
        return None
    if mode != "offline":
        raise ValueError("CASRT_LOCAL_WORKER_ENV_MODE must be inherit or offline")

    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in LOCAL_WORKER_ENV_ALLOWLIST or name.startswith(LOCAL_WORKER_ENV_PREFIXES):
            env[name] = value

    for name in list(env):
        if name == "TOKENIZERS_PARALLELISM":
            continue
        if any(part in name for part in SENSITIVE_ENV_SUBSTRINGS):
            env.pop(name, None)

    env.setdefault("PATH", os.defpath)
    env.update(LOCAL_WORKER_OFFLINE_ENV)
    return env


def parse_worker_response(response_line: str, *, worker_name: str = "local transformers worker") -> tuple[Segment, ...]:
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError as error:
        raise ValueError(f"local transformers worker returned invalid JSON: {error}") from error
    data = require_mapping(response, "local transformers worker response")
    if not data.get("ok"):
        error = data.get("error") or "unknown worker error"
        traceback_text = data.get("traceback")
        if isinstance(traceback_text, str) and traceback_text.strip():
            error = f"{error}\n{traceback_text.strip()}"
        raise ValueError(f"{worker_name} failed: {error}")
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("local transformers worker response segments must be an array")
    return parse_model_segments(json.dumps({"segments": segments}, ensure_ascii=False))


def close_transformers_worker_clients() -> None:
    with _WORKER_CLIENTS_LOCK:
        clients = tuple(_WORKER_CLIENTS.values())
        _WORKER_CLIENTS.clear()
    for client in clients:
        client.close()


atexit.register(close_transformers_worker_clients)


def audio_format_from_mime_type(mime_type: str) -> str:
    if "/" not in mime_type:
        raise ValueError(f"invalid mime_type {mime_type!r}")
    subtype = mime_type.split("/", 1)[1].split(";", 1)[0].lower()
    if subtype == "mpeg":
        return "mp3"
    return subtype


def extract_openai_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI-compatible response missing choices")
    message = require_mapping(require_mapping(choices[0], "choice").get("message"), "choice.message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    raise ValueError("OpenAI-compatible response message.content must be a string")


def extract_gemini_content(response: dict[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response missing candidates")
    content = require_mapping(require_mapping(candidates[0], "candidate").get("content"), "candidate.content")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Gemini response missing content parts")
    text = require_mapping(parts[0], "candidate.content.parts[0]").get("text")
    return require_string(text, "candidate.content.parts[0].text")


def parse_model_segments(content: str) -> tuple[Segment, ...]:
    data = parse_json_content(content)
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("model output segments must be an array")

    segments: list[Segment] = []
    for raw_segment in raw_segments:
        segment_data = require_mapping(raw_segment, "model segment")
        segments.append(
            Segment(
                id=make_segment_id(len(segments) + 1),
                start_ms=require_int(segment_data.get("start_ms"), "model segment.start_ms"),
                end_ms=require_int(segment_data.get("end_ms"), "model segment.end_ms"),
                channel=require_string(segment_data.get("channel"), "model segment.channel"),
                kind=require_string(segment_data.get("kind"), "model segment.kind"),
                text=require_string(segment_data.get("text"), "model segment.text"),
                needs_review=bool(segment_data.get("needs_review", False)),
            )
        )
    return tuple(segments)


def parse_json_content(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.fullmatch(r"\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*", content, re.DOTALL)
        if not match:
            raise ValueError("model output must be JSON")
        try:
            parsed = json.loads(match.group("body"))
        except json.JSONDecodeError as error:
            raise ValueError(f"model output JSON is invalid: {error}") from error
    return require_mapping(parsed, "model output")
