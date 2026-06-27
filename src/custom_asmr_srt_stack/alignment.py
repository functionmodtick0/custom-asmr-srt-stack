from __future__ import annotations

from dataclasses import replace
from typing import Any

from custom_asmr_srt_stack.models import MasterDocument, Segment, require_int, require_mapping, require_string

DEFAULT_LONG_SEGMENT_MS = 30_000


def apply_alignment_review_flags(
    master: MasterDocument,
    *,
    long_segment_ms: int = DEFAULT_LONG_SEGMENT_MS,
) -> MasterDocument:
    if long_segment_ms <= 0:
        raise ValueError("long_segment_ms must be positive")
    return replace(
        master,
        segments=tuple(
            replace(segment, needs_review=segment.needs_review or segment_needs_review(segment, long_segment_ms))
            for segment in master.segments
        ),
    )


def segment_needs_review(segment: Segment, long_segment_ms: int) -> bool:
    if segment.kind == "speech" and not segment.text.strip():
        return True
    return (segment.end_ms - segment.start_ms) > long_segment_ms


def merge_alignment_output(master: MasterDocument, value: Any) -> MasterDocument:
    data = require_mapping(value, "alignment output")
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("alignment output segments must be an array")

    timing_by_id: dict[str, tuple[int, int]] = {}
    for raw_segment in raw_segments:
        segment_data = require_mapping(raw_segment, "alignment segment")
        segment_id = require_string(segment_data.get("id"), "alignment segment.id")
        if segment_id in timing_by_id:
            raise ValueError(f"duplicate aligned segment id {segment_id!r}")
        timing_by_id[segment_id] = (
            require_int(segment_data.get("start_ms"), "alignment segment.start_ms"),
            require_int(segment_data.get("end_ms"), "alignment segment.end_ms"),
        )

    master_ids = {segment.id for segment in master.segments}
    aligned_ids = set(timing_by_id)
    missing_ids = sorted(master_ids - aligned_ids)
    unknown_ids = sorted(aligned_ids - master_ids)
    if missing_ids:
        raise ValueError(f"alignment output is missing ids: {', '.join(missing_ids)}")
    if unknown_ids:
        raise ValueError(f"alignment output contains unknown ids: {', '.join(unknown_ids)}")

    return apply_alignment_review_flags(
        replace(
            master,
            segments=tuple(
                replace(segment, start_ms=timing_by_id[segment.id][0], end_ms=timing_by_id[segment.id][1])
                for segment in master.segments
            ),
        )
    )
