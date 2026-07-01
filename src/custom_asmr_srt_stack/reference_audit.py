from __future__ import annotations

from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.case_batch import load_review_case_index, resolve_plan_path
from custom_asmr_srt_stack.evaluation import load_transcript_document, overlap_ms, speech_segments
from custom_asmr_srt_stack.models import MasterDocument, Segment

REFERENCE_AUDIT_FORMAT = "custom-asmr-reference-audit-v1"
REFERENCE_AUDIT_SUITE_FORMAT = "custom-asmr-reference-audit-suite-v1"
DEFAULT_REFERENCE_AUDIT_OVERLAP_MIN_MS = 1
DEFAULT_REFERENCE_AUDIT_LONG_SEGMENT_MS = 30000
DEFAULT_REFERENCE_AUDIT_HIGH_SPEECH_COVERAGE_RATIO = 0.95


def audit_master_reference(
    master: MasterDocument,
    *,
    case_id: str | None = None,
    reference: str | None = None,
    overlap_min_ms: int = DEFAULT_REFERENCE_AUDIT_OVERLAP_MIN_MS,
    long_segment_ms: int = DEFAULT_REFERENCE_AUDIT_LONG_SEGMENT_MS,
    high_speech_coverage_ratio: float = DEFAULT_REFERENCE_AUDIT_HIGH_SPEECH_COVERAGE_RATIO,
) -> dict[str, Any]:
    if overlap_min_ms < 1:
        raise ValueError("overlap_min_ms must be at least 1")
    if long_segment_ms < 1:
        raise ValueError("long_segment_ms must be at least 1")
    if not 0.0 <= high_speech_coverage_ratio <= 1.0:
        raise ValueError("high_speech_coverage_ratio must be between 0 and 1")

    speech = tuple(sorted(speech_segments(master), key=lambda item: (item.start_ms, item.end_ms, item.id)))
    overlap_pairs = reference_overlap_pairs(speech, min_overlap_ms=overlap_min_ms)
    same_channel_overlap_pairs = [item for item in overlap_pairs if item["same_channel"]]
    cross_channel_overlap_pairs = [item for item in overlap_pairs if not item["same_channel"]]
    exact_boundary_overlap_pairs = [item for item in overlap_pairs if item["exact_boundary"]]
    long_segments = [
        {
            "segment_id": segment.id,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "duration_ms": segment.end_ms - segment.start_ms,
            "channel": segment.channel,
            "needs_review": segment.needs_review,
        }
        for segment in speech
        if segment.end_ms - segment.start_ms >= long_segment_ms
    ]
    review_segments = [
        {
            "segment_id": segment.id,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "duration_ms": segment.end_ms - segment.start_ms,
            "channel": segment.channel,
        }
        for segment in speech
        if segment.needs_review
    ]
    duration_ms = master.duration_ms
    speech_union_duration_ms = interval_union_duration_ms(
        [(segment.start_ms, segment.end_ms) for segment in speech]
    )
    speech_coverage_ratio = (
        None if duration_ms is None or duration_ms == 0 else speech_union_duration_ms / duration_ms
    )
    channel_counts = {channel: sum(1 for segment in speech if segment.channel == channel) for channel in ("L", "R", "MIX")}

    flags = reference_audit_flags(
        review_count=len(review_segments),
        same_channel_overlap_count=len(same_channel_overlap_pairs),
        exact_boundary_overlap_count=len(exact_boundary_overlap_pairs),
        long_segment_count=len(long_segments),
        speech_coverage_ratio=speech_coverage_ratio,
        high_speech_coverage_ratio=high_speech_coverage_ratio,
    )
    report = {
        "format": REFERENCE_AUDIT_FORMAT,
        "case_id": case_id,
        "reference": reference,
        "thresholds": {
            "overlap_min_ms": overlap_min_ms,
            "long_segment_ms": long_segment_ms,
            "high_speech_coverage_ratio": high_speech_coverage_ratio,
        },
        "audio_duration_ms": duration_ms,
        "segment_count": len(master.segments),
        "speech_segment_count": len(speech),
        "review_count": len(review_segments),
        "channel_counts": channel_counts,
        "speech_union_duration_ms": speech_union_duration_ms,
        "speech_coverage_ratio": speech_coverage_ratio,
        "overlap_pair_count": len(overlap_pairs),
        "same_channel_overlap_pair_count": len(same_channel_overlap_pairs),
        "cross_channel_overlap_pair_count": len(cross_channel_overlap_pairs),
        "exact_boundary_overlap_pair_count": len(exact_boundary_overlap_pairs),
        "pair_overlap_duration_ms": sum(item["overlap_ms"] for item in overlap_pairs),
        "long_segment_count": len(long_segments),
        "max_segment_duration_ms": max((segment.end_ms - segment.start_ms for segment in speech), default=0),
        "flag_count": len(flags),
        "flags": flags,
        "review_segments": review_segments,
        "long_segments": long_segments,
        "overlap_pairs": overlap_pairs,
    }
    return report


def audit_review_case_references(
    case_index_file: Path,
    *,
    source_language: str = "ja",
    overlap_min_ms: int = DEFAULT_REFERENCE_AUDIT_OVERLAP_MIN_MS,
    long_segment_ms: int = DEFAULT_REFERENCE_AUDIT_LONG_SEGMENT_MS,
    high_speech_coverage_ratio: float = DEFAULT_REFERENCE_AUDIT_HIGH_SPEECH_COVERAGE_RATIO,
) -> dict[str, Any]:
    case_index = load_review_case_index(case_index_file)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review case index items must be an array")

    base_dir = case_index_file.parent
    cases = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        case_id = require_non_empty_string(raw_item.get("id"), f"review case index item {index}.id")
        reference = require_non_empty_string(raw_item.get("reference"), f"review case index item {index}.reference")
        reference_path = resolve_plan_path(base_dir, reference)
        if not reference_path.is_file():
            raise ValueError(f"review case reference file is missing: {reference}")
        master = load_transcript_document(reference_path, source_language=source_language)
        cases.append(
            audit_master_reference(
                master,
                case_id=case_id,
                reference=reference,
                overlap_min_ms=overlap_min_ms,
                long_segment_ms=long_segment_ms,
                high_speech_coverage_ratio=high_speech_coverage_ratio,
            )
        )

    return {
        "format": REFERENCE_AUDIT_SUITE_FORMAT,
        "case_index": str(case_index_file),
        "case_count": len(cases),
        "summary": aggregate_reference_audits(cases),
        "cases": cases,
    }


def reference_overlap_pairs(segments: tuple[Segment, ...], *, min_overlap_ms: int) -> list[dict[str, Any]]:
    pairs = []
    for index, left in enumerate(segments):
        for right in segments[index + 1 :]:
            overlap = overlap_ms(left, right)
            if overlap < min_overlap_ms:
                continue
            pairs.append(
                {
                    "left_segment_id": left.id,
                    "right_segment_id": right.id,
                    "start_ms": max(left.start_ms, right.start_ms),
                    "end_ms": min(left.end_ms, right.end_ms),
                    "overlap_ms": overlap,
                    "left_channel": left.channel,
                    "right_channel": right.channel,
                    "same_channel": left.channel == right.channel,
                    "exact_boundary": left.start_ms == right.start_ms and left.end_ms == right.end_ms,
                }
            )
    return pairs


def aggregate_reference_audits(reports: list[dict[str, Any]]) -> dict[str, Any]:
    audio_duration_ms = sum_optional_int(report.get("audio_duration_ms") for report in reports)
    speech_union_duration_ms = sum(int(report["speech_union_duration_ms"]) for report in reports)
    channel_counts = {
        channel: sum(int(report["channel_counts"].get(channel, 0)) for report in reports)
        for channel in ("L", "R", "MIX")
    }
    flag_type_counts: dict[str, int] = {}
    for report in reports:
        for flag in report["flags"]:
            flag_type = flag["type"]
            flag_type_counts[flag_type] = flag_type_counts.get(flag_type, 0) + 1
    return {
        "audio_duration_ms": audio_duration_ms,
        "segment_count": sum(int(report["segment_count"]) for report in reports),
        "speech_segment_count": sum(int(report["speech_segment_count"]) for report in reports),
        "review_count": sum(int(report["review_count"]) for report in reports),
        "channel_counts": channel_counts,
        "speech_union_duration_ms": speech_union_duration_ms,
        "speech_coverage_ratio": None
        if audio_duration_ms is None or audio_duration_ms == 0
        else speech_union_duration_ms / audio_duration_ms,
        "overlap_pair_count": sum(int(report["overlap_pair_count"]) for report in reports),
        "same_channel_overlap_pair_count": sum(
            int(report["same_channel_overlap_pair_count"]) for report in reports
        ),
        "cross_channel_overlap_pair_count": sum(
            int(report["cross_channel_overlap_pair_count"]) for report in reports
        ),
        "exact_boundary_overlap_pair_count": sum(
            int(report["exact_boundary_overlap_pair_count"]) for report in reports
        ),
        "pair_overlap_duration_ms": sum(int(report["pair_overlap_duration_ms"]) for report in reports),
        "long_segment_count": sum(int(report["long_segment_count"]) for report in reports),
        "max_segment_duration_ms": max((int(report["max_segment_duration_ms"]) for report in reports), default=0),
        "flag_type_counts": dict(sorted(flag_type_counts.items())),
        "flagged_case_count": sum(1 for report in reports if report["flag_count"] > 0),
    }


def reference_audit_flags(
    *,
    review_count: int,
    same_channel_overlap_count: int,
    exact_boundary_overlap_count: int,
    long_segment_count: int,
    speech_coverage_ratio: float | None,
    high_speech_coverage_ratio: float,
) -> list[dict[str, Any]]:
    flags = []
    if review_count:
        flags.append({"type": "review_flag_segments", "count": review_count})
    if same_channel_overlap_count:
        flags.append({"type": "same_channel_overlap", "count": same_channel_overlap_count})
    if exact_boundary_overlap_count:
        flags.append({"type": "exact_boundary_overlap", "count": exact_boundary_overlap_count})
    if long_segment_count:
        flags.append({"type": "long_segment", "count": long_segment_count})
    if speech_coverage_ratio is not None and speech_coverage_ratio >= high_speech_coverage_ratio:
        flags.append({"type": "near_full_speech_coverage", "ratio": speech_coverage_ratio})
    return flags


def interval_union_duration_ms(intervals: list[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start_ms, end_ms in sorted(intervals):
        if not merged or start_ms > merged[-1][1]:
            merged.append([start_ms, end_ms])
        else:
            merged[-1][1] = max(merged[-1][1], end_ms)
    return sum(end_ms - start_ms for start_ms, end_ms in merged)


def sum_optional_int(values: Any) -> int | None:
    total = 0
    for value in values:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("expected integer duration")
        total += value
    return total


def require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
