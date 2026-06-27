from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from custom_asmr_srt_stack.models import Segment, make_segment_id, require_int, require_mapping, require_string

ADAPTERS = {"openai-compatible", "gemini"}


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


@dataclass(frozen=True)
class ModelEndpoint:
    adapter: str
    endpoint_url: str
    model_id: str
    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.adapter not in ADAPTERS:
            raise ValueError(f"unsupported model adapter {self.adapter!r}")
        if not self.endpoint_url:
            raise ValueError("endpoint_url must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")

    @classmethod
    def from_json(cls, value: Any) -> ModelEndpoint:
        data = require_mapping(value, "model endpoint")
        return cls(
            adapter=require_string(data.get("adapter", "openai-compatible"), "model.adapter"),
            endpoint_url=require_string(data.get("endpoint_url"), "model.endpoint_url"),
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
) -> tuple[Segment, ...]:
    if not audio_bytes:
        raise ValueError("audio_bytes must not be empty")
    if not mime_type:
        raise ValueError("mime_type must not be empty")

    if endpoint.adapter == "openai-compatible":
        response = http_post(
            endpoint.endpoint_url.rstrip("/") + "/chat/completions",
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
    base = endpoint.endpoint_url.rstrip("/")
    if ":generateContent" in base:
        return base
    return f"{base}/v1beta/models/{endpoint.model_id}:generateContent"


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
