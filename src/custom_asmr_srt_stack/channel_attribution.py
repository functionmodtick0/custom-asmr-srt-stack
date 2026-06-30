from __future__ import annotations

from dataclasses import dataclass, replace

from custom_asmr_srt_stack.audio import wav_rms_dbfs
from custom_asmr_srt_stack.models import MasterDocument, Segment

CHANNEL_ATTRIBUTION_THRESHOLD_DB = 6.0


@dataclass(frozen=True)
class ChannelAttributionReport:
    master: MasterDocument
    segments: int
    changed_segments: int
    threshold_db: float


def attribute_master_channels_by_energy(
    master: MasterDocument,
    *,
    left_audio: bytes,
    right_audio: bytes,
    threshold_db: float = CHANNEL_ATTRIBUTION_THRESHOLD_DB,
) -> ChannelAttributionReport:
    if threshold_db < 0:
        raise ValueError("threshold_db must be non-negative")

    attributed_segments, changed_segments = attribute_segments_by_energy(
        master.segments,
        left_audio=left_audio,
        right_audio=right_audio,
        threshold_db=threshold_db,
    )

    return ChannelAttributionReport(
        master=replace(master, segments=tuple(attributed_segments)),
        segments=len(master.segments),
        changed_segments=changed_segments,
        threshold_db=threshold_db,
    )


def attribute_segments_by_energy(
    segments: tuple[Segment, ...] | list[Segment],
    *,
    left_audio: bytes,
    right_audio: bytes,
    threshold_db: float,
) -> tuple[tuple[Segment, ...], int]:
    if threshold_db < 0:
        raise ValueError("threshold_db must be non-negative")

    attributed_segments = []
    changed_segments = 0
    for segment in segments:
        attributed = attribute_segment_channel_by_energy(
            segment,
            left_audio=left_audio,
            right_audio=right_audio,
            threshold_db=threshold_db,
        )
        if attributed.channel != segment.channel:
            changed_segments += 1
        attributed_segments.append(attributed)
    return tuple(attributed_segments), changed_segments


def attribute_segment_channel_by_energy(
    segment: Segment,
    *,
    left_audio: bytes,
    right_audio: bytes,
    threshold_db: float,
) -> Segment:
    if segment.channel != "MIX" or segment.kind != "speech":
        return segment

    left_db = wav_rms_dbfs(left_audio, start_ms=segment.start_ms, end_ms=segment.end_ms)
    right_db = wav_rms_dbfs(right_audio, start_ms=segment.start_ms, end_ms=segment.end_ms)
    if left_db - right_db >= threshold_db:
        return replace(segment, channel="L")
    if right_db - left_db >= threshold_db:
        return replace(segment, channel="R")
    return segment
