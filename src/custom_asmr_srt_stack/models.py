from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MASTER_FORMAT = "custom-asmr-master-v1"
TRANSLATION_FORMAT = "custom-asmr-translation-v1"
TRANSLATED_FORMAT = "custom-asmr-translated-v1"

CHANNELS = {"L", "R", "MIX"}
KINDS = {"speech", "breath", "sfx", "silence"}


def make_segment_id(index: int) -> str:
    if index < 1:
        raise ValueError("segment id index must be positive")
    return f"seg_{index:06d}"


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class Segment:
    id: str
    start_ms: int
    end_ms: int
    channel: str
    kind: str
    text: str
    needs_review: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("segment id must not be empty")
        if self.start_ms < 0:
            raise ValueError(f"{self.id}: start_ms must be non-negative")
        if self.end_ms <= self.start_ms:
            raise ValueError(f"{self.id}: end_ms must be greater than start_ms")
        if self.channel not in CHANNELS:
            raise ValueError(f"{self.id}: unsupported channel {self.channel!r}")
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: unsupported kind {self.kind!r}")

    @classmethod
    def from_json(cls, value: Any) -> Segment:
        data = require_mapping(value, "segment")
        return cls(
            id=require_string(data.get("id"), "segment.id"),
            start_ms=require_int(data.get("start_ms"), "segment.start_ms"),
            end_ms=require_int(data.get("end_ms"), "segment.end_ms"),
            channel=require_string(data.get("channel"), "segment.channel"),
            kind=require_string(data.get("kind"), "segment.kind"),
            text=require_string(data.get("text"), "segment.text"),
            needs_review=bool(data.get("needs_review", False)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "channel": self.channel,
            "kind": self.kind,
            "text": self.text,
            "needs_review": self.needs_review,
        }


@dataclass(frozen=True)
class MasterDocument:
    source_language: str
    source_file: str | None
    duration_ms: int | None
    segments: tuple[Segment, ...]
    format: str = MASTER_FORMAT

    def __post_init__(self) -> None:
        if self.format != MASTER_FORMAT:
            raise ValueError(f"unsupported master format {self.format!r}")
        if not self.source_language:
            raise ValueError("source_language must not be empty")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

        seen: set[str] = set()
        for segment in self.segments:
            if segment.id in seen:
                raise ValueError(f"duplicate segment id {segment.id!r}")
            seen.add(segment.id)
            if self.duration_ms is not None and segment.end_ms > self.duration_ms:
                raise ValueError(f"{segment.id}: end_ms exceeds audio duration")

    @classmethod
    def from_json(cls, value: Any) -> MasterDocument:
        data = require_mapping(value, "master document")
        audio = require_mapping(data.get("audio", {}), "audio")
        raw_segments = data.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("segments must be an array")
        duration = audio.get("duration_ms")
        return cls(
            format=require_string(data.get("format"), "format"),
            source_language=require_string(data.get("source_language"), "source_language"),
            source_file=audio.get("source_file"),
            duration_ms=None if duration is None else require_int(duration, "audio.duration_ms"),
            segments=tuple(Segment.from_json(segment) for segment in raw_segments),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "source_language": self.source_language,
            "audio": {
                "source_file": self.source_file,
                "duration_ms": self.duration_ms,
            },
            "segments": [segment.to_json() for segment in self.segments],
        }
