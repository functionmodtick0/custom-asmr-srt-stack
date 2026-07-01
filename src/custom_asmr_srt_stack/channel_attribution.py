from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from custom_asmr_srt_stack.audio import wav_rms_dbfs
from custom_asmr_srt_stack.models import MasterDocument, Segment

CHANNEL_ATTRIBUTION_THRESHOLD_DB = 8.0
CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS = -40.0


@dataclass(frozen=True)
class ChannelAttributionReport:
    master: MasterDocument
    segments: int
    changed_segments: int
    threshold_db: float
    diagnostics: tuple[dict[str, Any], ...]


def attribute_master_channels_by_energy(
    master: MasterDocument,
    *,
    left_audio: bytes,
    right_audio: bytes,
    threshold_db: float = CHANNEL_ATTRIBUTION_THRESHOLD_DB,
    quiet_channel_max_dbfs: float | None = CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
) -> ChannelAttributionReport:
    if threshold_db < 0:
        raise ValueError("threshold_db must be non-negative")

    attributed_segments, changed_segments, diagnostics = attribute_segments_by_energy(
        master.segments,
        left_audio=left_audio,
        right_audio=right_audio,
        threshold_db=threshold_db,
        quiet_channel_max_dbfs=quiet_channel_max_dbfs,
    )

    return ChannelAttributionReport(
        master=replace(master, segments=tuple(attributed_segments)),
        segments=len(master.segments),
        changed_segments=changed_segments,
        threshold_db=threshold_db,
        diagnostics=diagnostics,
    )


def attribute_segments_by_energy(
    segments: tuple[Segment, ...] | list[Segment],
    *,
    left_audio: bytes,
    right_audio: bytes,
    threshold_db: float,
    quiet_channel_max_dbfs: float | None = CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
) -> tuple[tuple[Segment, ...], int, tuple[dict[str, Any], ...]]:
    if threshold_db < 0:
        raise ValueError("threshold_db must be non-negative")

    attributed_segments = []
    diagnostics = []
    changed_segments = 0
    for segment in segments:
        attributed, diagnostic = attribute_segment_channel_by_energy(
            segment,
            left_audio=left_audio,
            right_audio=right_audio,
            threshold_db=threshold_db,
            quiet_channel_max_dbfs=quiet_channel_max_dbfs,
        )
        if attributed.channel != segment.channel:
            changed_segments += 1
        attributed_segments.append(attributed)
        diagnostics.append(diagnostic)
    return tuple(attributed_segments), changed_segments, tuple(diagnostics)


def attribute_segment_channel_by_energy(
    segment: Segment,
    *,
    left_audio: bytes,
    right_audio: bytes,
    threshold_db: float,
    quiet_channel_max_dbfs: float | None,
) -> tuple[Segment, dict[str, Any]]:
    base_diagnostic: dict[str, Any] = {
        "id": segment.id,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "kind": segment.kind,
        "original_channel": segment.channel,
        "threshold_db": threshold_db,
        "quiet_channel_max_dbfs": quiet_channel_max_dbfs,
    }
    if segment.kind != "speech":
        return segment, {
            **base_diagnostic,
            "attributed_channel": segment.channel,
            "reason": "skipped_non_speech",
        }

    left_db = wav_rms_dbfs(left_audio, start_ms=segment.start_ms, end_ms=segment.end_ms)
    right_db = wav_rms_dbfs(right_audio, start_ms=segment.start_ms, end_ms=segment.end_ms)
    delta_db = left_db - right_db
    diagnostic = {
        **base_diagnostic,
        "left_dbfs": left_db,
        "right_dbfs": right_db,
        "delta_db": delta_db,
        "abs_delta_db": abs(delta_db),
        "quieter_dbfs": min(left_db, right_db),
    }
    if segment.channel != "MIX":
        return segment, {
            **diagnostic,
            "attributed_channel": segment.channel,
            "reason": "skipped_existing_channel",
        }
    if delta_db >= threshold_db and channel_is_quiet_enough(right_db, quiet_channel_max_dbfs):
        attributed = replace(segment, channel="L")
        return attributed, {**diagnostic, "attributed_channel": "L", "reason": "left_dominant"}
    if -delta_db >= threshold_db and channel_is_quiet_enough(left_db, quiet_channel_max_dbfs):
        attributed = replace(segment, channel="R")
        return attributed, {**diagnostic, "attributed_channel": "R", "reason": "right_dominant"}
    reason = "below_threshold" if abs(delta_db) < threshold_db else "quieter_side_active"
    return segment, {**diagnostic, "attributed_channel": segment.channel, "reason": reason}


def channel_is_quiet_enough(dbfs: float, quiet_channel_max_dbfs: float | None) -> bool:
    return quiet_channel_max_dbfs is None or dbfs <= quiet_channel_max_dbfs


def channel_diagnostics_summary(diagnostics: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    original_channel_counts: dict[str, int] = {}
    attributed_channel_counts: dict[str, int] = {}
    speech_segments = 0
    mix_speech_segments = 0
    changed_segments = 0
    for diagnostic in diagnostics:
        reason = str(diagnostic.get("reason") or "unknown")
        original_channel = str(diagnostic.get("original_channel") or "unknown")
        attributed_channel = str(diagnostic.get("attributed_channel") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        original_channel_counts[original_channel] = original_channel_counts.get(original_channel, 0) + 1
        attributed_channel_counts[attributed_channel] = attributed_channel_counts.get(attributed_channel, 0) + 1
        if diagnostic.get("kind") == "speech":
            speech_segments += 1
            if original_channel == "MIX":
                mix_speech_segments += 1
        if original_channel != attributed_channel:
            changed_segments += 1
    segment_count = len(diagnostics)
    return {
        "segment_count": segment_count,
        "speech_segments": speech_segments,
        "mix_speech_segments": mix_speech_segments,
        "changed_segments": changed_segments,
        "changed_segment_ratio": None if segment_count == 0 else changed_segments / segment_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "original_channel_counts": dict(sorted(original_channel_counts.items())),
        "attributed_channel_counts": dict(sorted(attributed_channel_counts.items())),
    }
