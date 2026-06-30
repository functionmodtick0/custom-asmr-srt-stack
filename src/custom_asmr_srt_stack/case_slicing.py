from __future__ import annotations

from dataclasses import replace

from custom_asmr_srt_stack.models import MasterDocument, make_segment_id


def slice_master_document(master: MasterDocument, *, start_ms: int, end_ms: int) -> MasterDocument:
    if start_ms < 0:
        raise ValueError("start_ms must be non-negative")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")

    sliced_segments = []
    for segment in master.segments:
        overlap_start = max(segment.start_ms, start_ms)
        overlap_end = min(segment.end_ms, end_ms)
        if overlap_end <= overlap_start:
            continue
        clipped = segment.start_ms < start_ms or segment.end_ms > end_ms
        sliced_segments.append(
            replace(
                segment,
                start_ms=overlap_start - start_ms,
                end_ms=overlap_end - start_ms,
                needs_review=segment.needs_review or clipped,
            )
        )

    return replace(
        master,
        duration_ms=end_ms - start_ms,
        segments=tuple(
            replace(segment, id=make_segment_id(index + 1))
            for index, segment in enumerate(
                sorted(sliced_segments, key=lambda item: (item.start_ms, item.end_ms, item.channel, item.text))
            )
        ),
    )
