from __future__ import annotations

import re
from dataclasses import replace

from custom_asmr_srt_stack.models import MasterDocument, Segment, make_segment_id

TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)
CHANNEL_LABEL_RE = re.compile(r"^\[(?P<label>L|R|LR|MIX)\]\s*", re.IGNORECASE)


def parse_timestamp(value: str) -> int:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", value)
    if not match:
        raise ValueError(f"invalid SRT timestamp {value!r}")
    hours, minutes, seconds, milliseconds = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid SRT timestamp {value!r}")
    return (((hours * 60) + minutes) * 60 + seconds) * 1000 + milliseconds


def format_timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError("timestamp milliseconds must be non-negative")
    total_seconds, ms = divmod(milliseconds, 1000)
    total_minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def parse_srt(
    content: str,
    *,
    source_language: str = "ja",
    source_file: str | None = None,
) -> MasterDocument:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content.replace("\r\n", "\n")) if block.strip()]
    segments: list[Segment] = []

    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        if lines[0].strip().isdigit():
            lines = lines[1:]
        if len(lines) < 2:
            raise ValueError("SRT cue must contain timing and text")

        timing_match = TIMESTAMP_RE.match(lines[0].strip())
        if not timing_match:
            raise ValueError(f"invalid SRT cue timing {lines[0]!r}")

        channel, text = parse_srt_text_metadata(lines[1:])
        segments.append(
            Segment(
                id=make_segment_id(len(segments) + 1),
                start_ms=parse_timestamp(timing_match.group("start")),
                end_ms=parse_timestamp(timing_match.group("end")),
                channel=channel,
                kind="speech",
                text=text,
            )
        )

    duration_ms = max((segment.end_ms for segment in segments), default=None)
    return MasterDocument(
        source_language=source_language,
        source_file=source_file,
        duration_ms=duration_ms,
        segments=tuple(segments),
    )


def parse_srt_text_metadata(lines: list[str]) -> tuple[str, str]:
    text_lines = [line.rstrip() for line in lines]
    channel = "MIX"
    if text_lines:
        match = CHANNEL_LABEL_RE.match(text_lines[0].strip())
        if match:
            label = match.group("label").upper()
            if label in {"L", "R"}:
                channel = label
            text_lines[0] = CHANNEL_LABEL_RE.sub("", text_lines[0], count=1).strip()
    return channel, "\n".join(text_lines).strip()


def format_srt(
    master: MasterDocument,
    *,
    text_by_id: dict[str, str] | None = None,
    include_kinds: tuple[str, ...] = ("speech",),
) -> str:
    cues: list[str] = []
    ordered_segments = sorted(master.segments, key=lambda segment: (segment.start_ms, segment.end_ms, segment.id))

    for segment in ordered_segments:
        if segment.kind not in include_kinds:
            continue
        text = segment.text if text_by_id is None else text_by_id.get(segment.id, segment.text)
        if not text:
            continue
        checked_segment = replace(segment, text=text)
        cues.append(
            "\n".join(
                [
                    str(len(cues) + 1),
                    f"{format_timestamp(checked_segment.start_ms)} --> {format_timestamp(checked_segment.end_ms)}",
                    checked_segment.text,
                ]
            )
        )

    return "\n\n".join(cues) + ("\n" if cues else "")
