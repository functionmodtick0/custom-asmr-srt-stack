from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import normalize_audio_to_wav, split_wav_channels, wav_rms_dbfs
from custom_asmr_srt_stack.case_batch import load_review_case_index, resolve_plan_path
from custom_asmr_srt_stack.channel_attribution import (
    CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
    CHANNEL_ATTRIBUTION_THRESHOLD_DB,
    channel_is_quiet_enough,
)
from custom_asmr_srt_stack.evaluation import load_transcript_document, speech_segments
from custom_asmr_srt_stack.models import MasterDocument, Segment

REFERENCE_CHANNEL_AUDIT_FORMAT = "custom-asmr-reference-channel-audit-v1"
REFERENCE_CHANNEL_AUDIT_SUITE_FORMAT = "custom-asmr-reference-channel-audit-suite-v1"
REFERENCE_CHANNEL_REVIEW_EFFORT_FORMAT = "custom-asmr-review-effort-v1"
REFERENCE_CHANNEL_REVIEW_CLIP_MAX_MS = 5000
REFERENCE_CHANNEL_REVIEW_CLIP_WINDOW_MS = 1000
REFERENCE_CHANNEL_REVIEW_CLIP_HOP_MS = 1000
REFERENCE_CHANNEL_REVIEW_REASON_PRIORITY = {
    "reference-channel-energy-mismatch": 3500.0,
    "reference-channel-energy-uncertain": 2500.0,
}


def audit_review_case_channels(
    case_index_file: Path,
    *,
    source_language: str = "ja",
    threshold_db: float = CHANNEL_ATTRIBUTION_THRESHOLD_DB,
    quiet_channel_max_dbfs: float | None = CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
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
        audio = require_non_empty_string(raw_item.get("audio"), f"review case index item {index}.audio")
        reference_path = resolve_plan_path(base_dir, reference)
        audio_path = resolve_plan_path(base_dir, audio)
        if not reference_path.is_file():
            raise ValueError(f"review case reference file is missing: {reference}")
        if not audio_path.is_file():
            raise ValueError(f"review case audio file is missing: {audio}")

        normalized_audio = normalize_audio_to_wav(
            audio_path.read_bytes(),
            file_name=audio_path.name,
            mime_type=mimetypes.guess_type(audio_path.name)[0],
        )
        _info, channels = split_wav_channels(normalized_audio)
        if not {"L", "R"}.issubset(channels):
            raise ValueError(f"review case channel audit requires stereo audio: {audio}")
        master = load_transcript_document(reference_path, source_language=source_language)
        cases.append(
            audit_master_reference_channels(
                master,
                left_audio=channels["L"],
                right_audio=channels["R"],
                case_id=case_id,
                reference=reference,
                audio=audio,
                threshold_db=threshold_db,
                quiet_channel_max_dbfs=quiet_channel_max_dbfs,
            )
        )

    return {
        "format": REFERENCE_CHANNEL_AUDIT_SUITE_FORMAT,
        "case_index": str(case_index_file),
        "case_count": len(cases),
        "thresholds": {
            "threshold_db": threshold_db,
            "quiet_channel_max_dbfs": quiet_channel_max_dbfs,
        },
        "summary": aggregate_reference_channel_audits(cases),
        "cases": cases,
    }


def audit_master_reference_channels(
    master: MasterDocument,
    *,
    left_audio: bytes,
    right_audio: bytes,
    case_id: str | None = None,
    reference: str | None = None,
    audio: str | None = None,
    threshold_db: float = CHANNEL_ATTRIBUTION_THRESHOLD_DB,
    quiet_channel_max_dbfs: float | None = CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
) -> dict[str, Any]:
    if threshold_db < 0:
        raise ValueError("threshold_db must be non-negative")
    speech = tuple(sorted(speech_segments(master), key=lambda item: (item.start_ms, item.end_ms, item.id)))
    eligible = [segment for segment in speech if segment.channel in {"L", "R"}]
    items = [
        reference_channel_energy_item(
            segment,
            left_audio=left_audio,
            right_audio=right_audio,
            threshold_db=threshold_db,
            quiet_channel_max_dbfs=quiet_channel_max_dbfs,
        )
        for segment in eligible
    ]
    return {
        "format": REFERENCE_CHANNEL_AUDIT_FORMAT,
        "case_id": case_id,
        "reference": reference,
        "audio": audio,
        "thresholds": {
            "threshold_db": threshold_db,
            "quiet_channel_max_dbfs": quiet_channel_max_dbfs,
        },
        "speech_segment_count": len(speech),
        "eligible_reference_channel_count": len(eligible),
        "reference_mix_segment_count": sum(1 for segment in speech if segment.channel == "MIX"),
        "summary": reference_channel_audit_summary(items),
        "items": items,
    }


def reference_channel_energy_item(
    segment: Segment,
    *,
    left_audio: bytes,
    right_audio: bytes,
    threshold_db: float,
    quiet_channel_max_dbfs: float | None,
) -> dict[str, Any]:
    left_db = wav_rms_dbfs(left_audio, start_ms=segment.start_ms, end_ms=segment.end_ms)
    right_db = wav_rms_dbfs(right_audio, start_ms=segment.start_ms, end_ms=segment.end_ms)
    delta_db = left_db - right_db
    if delta_db >= threshold_db and channel_is_quiet_enough(right_db, quiet_channel_max_dbfs):
        energy_channel = "L"
        reason = "left_dominant"
    elif -delta_db >= threshold_db and channel_is_quiet_enough(left_db, quiet_channel_max_dbfs):
        energy_channel = "R"
        reason = "right_dominant"
    else:
        energy_channel = "MIX"
        reason = "below_threshold" if abs(delta_db) < threshold_db else "quieter_side_active"

    if energy_channel == "MIX":
        status = "uncertain"
    elif energy_channel == segment.channel:
        status = "match"
    else:
        status = "mismatch"
    evidence_window = reference_channel_review_clip_window(
        segment,
        left_audio=left_audio,
        right_audio=right_audio,
        energy_channel=energy_channel,
        quiet_channel_max_dbfs=quiet_channel_max_dbfs,
    )
    return {
        "segment_id": segment.id,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "duration_ms": segment.end_ms - segment.start_ms,
        "review_clip_start_ms": evidence_window["start_ms"],
        "review_clip_end_ms": evidence_window["end_ms"],
        "review_clip_left_dbfs": evidence_window["left_dbfs"],
        "review_clip_right_dbfs": evidence_window["right_dbfs"],
        "review_clip_delta_db": evidence_window["delta_db"],
        "reference_channel": segment.channel,
        "energy_channel": energy_channel,
        "status": status,
        "reason": reason,
        "left_dbfs": left_db,
        "right_dbfs": right_db,
        "delta_db": delta_db,
        "abs_delta_db": abs(delta_db),
        "quieter_dbfs": min(left_db, right_db),
        "needs_review": segment.needs_review,
    }


def reference_channel_review_clip_window(
    segment: Segment,
    *,
    left_audio: bytes,
    right_audio: bytes,
    energy_channel: str,
    quiet_channel_max_dbfs: float | None,
) -> dict[str, int | float]:
    duration_ms = segment.end_ms - segment.start_ms
    if duration_ms <= REFERENCE_CHANNEL_REVIEW_CLIP_MAX_MS:
        return reference_channel_window_metrics(
            segment.start_ms,
            segment.end_ms,
            left_audio=left_audio,
            right_audio=right_audio,
        )

    probe_ms = min(REFERENCE_CHANNEL_REVIEW_CLIP_WINDOW_MS, duration_ms)
    starts = list(range(segment.start_ms, segment.end_ms - probe_ms + 1, REFERENCE_CHANNEL_REVIEW_CLIP_HOP_MS))
    last_start = segment.end_ms - probe_ms
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    best_metrics = None
    best_score = None
    for start_ms in starts:
        metrics = reference_channel_window_metrics(
            start_ms,
            start_ms + probe_ms,
            left_audio=left_audio,
            right_audio=right_audio,
        )
        score = reference_channel_evidence_score(
            energy_channel,
            left_db=float(metrics["left_dbfs"]),
            right_db=float(metrics["right_dbfs"]),
            delta_db=float(metrics["delta_db"]),
            quiet_channel_max_dbfs=quiet_channel_max_dbfs,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = metrics

    assert best_metrics is not None
    center_ms = (int(best_metrics["start_ms"]) + int(best_metrics["end_ms"])) // 2
    clip_start_ms = center_ms - (REFERENCE_CHANNEL_REVIEW_CLIP_MAX_MS // 2)
    clip_end_ms = clip_start_ms + REFERENCE_CHANNEL_REVIEW_CLIP_MAX_MS
    if clip_start_ms < segment.start_ms:
        clip_start_ms = segment.start_ms
        clip_end_ms = clip_start_ms + REFERENCE_CHANNEL_REVIEW_CLIP_MAX_MS
    if clip_end_ms > segment.end_ms:
        clip_end_ms = segment.end_ms
        clip_start_ms = clip_end_ms - REFERENCE_CHANNEL_REVIEW_CLIP_MAX_MS
    return reference_channel_window_metrics(
        clip_start_ms,
        clip_end_ms,
        left_audio=left_audio,
        right_audio=right_audio,
    )


def reference_channel_window_metrics(
    start_ms: int,
    end_ms: int,
    *,
    left_audio: bytes,
    right_audio: bytes,
) -> dict[str, int | float]:
    left_db = wav_rms_dbfs(left_audio, start_ms=start_ms, end_ms=end_ms)
    right_db = wav_rms_dbfs(right_audio, start_ms=start_ms, end_ms=end_ms)
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "left_dbfs": left_db,
        "right_dbfs": right_db,
        "delta_db": left_db - right_db,
    }


def reference_channel_evidence_score(
    energy_channel: str,
    *,
    left_db: float,
    right_db: float,
    delta_db: float,
    quiet_channel_max_dbfs: float | None,
) -> float:
    if energy_channel == "L":
        quiet_bonus = 100.0 if channel_is_quiet_enough(right_db, quiet_channel_max_dbfs) else 0.0
        return quiet_bonus + delta_db
    if energy_channel == "R":
        quiet_bonus = 100.0 if channel_is_quiet_enough(left_db, quiet_channel_max_dbfs) else 0.0
        return quiet_bonus - delta_db
    return -abs(delta_db)


def aggregate_reference_channel_audits(cases: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    speech_segment_count = 0
    eligible_reference_channel_count = 0
    reference_mix_segment_count = 0
    for case in cases:
        speech_segment_count += require_int(case.get("speech_segment_count"), "case speech_segment_count")
        eligible_reference_channel_count += require_int(
            case.get("eligible_reference_channel_count"),
            "case eligible_reference_channel_count",
        )
        reference_mix_segment_count += require_int(
            case.get("reference_mix_segment_count"),
            "case reference_mix_segment_count",
        )
        raw_items = case.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("reference channel audit case items must be an array")
        items.extend(item for item in raw_items if isinstance(item, dict))

    summary = reference_channel_audit_summary(items)
    return {
        "speech_segment_count": speech_segment_count,
        "eligible_reference_channel_count": eligible_reference_channel_count,
        "reference_mix_segment_count": reference_mix_segment_count,
        **summary,
    }


def reference_channel_audit_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    reference_channel_counts: dict[str, int] = {}
    energy_channel_counts: dict[str, int] = {}
    for item in items:
        increment(status_counts, str(item.get("status") or "unknown"))
        increment(reason_counts, str(item.get("reason") or "unknown"))
        increment(reference_channel_counts, str(item.get("reference_channel") or "unknown"))
        increment(energy_channel_counts, str(item.get("energy_channel") or "unknown"))

    eligible_count = len(items)
    match_count = status_counts.get("match", 0)
    mismatch_count = status_counts.get("mismatch", 0)
    uncertain_count = status_counts.get("uncertain", 0)
    energy_labeled_count = match_count + mismatch_count
    return {
        "eligible_count": eligible_count,
        "energy_labeled_count": energy_labeled_count,
        "energy_uncertain_count": uncertain_count,
        "match_count": match_count,
        "mismatch_count": mismatch_count,
        "match_ratio": None if energy_labeled_count == 0 else match_count / energy_labeled_count,
        "mismatch_ratio": None if energy_labeled_count == 0 else mismatch_count / energy_labeled_count,
        "energy_labeled_ratio": None if eligible_count == 0 else energy_labeled_count / eligible_count,
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "reference_channel_counts": dict(sorted(reference_channel_counts.items())),
        "energy_channel_counts": dict(sorted(energy_channel_counts.items())),
    }


def reference_channel_audit_review_effort_report(
    audit_report: dict[str, Any],
    *,
    source_report: str | None = None,
    source_case_index: str | None = None,
) -> dict[str, Any]:
    if audit_report.get("format") != REFERENCE_CHANNEL_AUDIT_SUITE_FORMAT:
        raise ValueError(f"reference channel audit review effort input must be {REFERENCE_CHANNEL_AUDIT_SUITE_FORMAT}")
    raw_cases = audit_report.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("reference channel audit cases must be an array")
    items = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("reference channel audit case must be an object")
        case_id = require_non_empty_string(raw_case.get("case_id"), "reference channel audit case_id")
        raw_items = raw_case.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("reference channel audit items must be an array")
        for raw_item in raw_items:
            if not isinstance(raw_item, dict) or raw_item.get("status") == "match":
                continue
            items.append(reference_channel_review_item(case_id, raw_item))

    sorted_items = sorted(items, key=reference_channel_review_item_sort_key)
    for index, item in enumerate(sorted_items, start=1):
        item["priority_rank"] = index
    result = {
        "format": REFERENCE_CHANNEL_REVIEW_EFFORT_FORMAT,
        "sort": "priority_score_desc",
        "item_count": len(sorted_items),
        "reason_counts": reference_channel_review_reason_counts(sorted_items),
        "items": sorted_items,
    }
    if source_report is not None:
        result["source_report"] = source_report
    if source_case_index is not None:
        result["source_case_index"] = source_case_index
    return result


def reference_channel_review_item(case_id: str, item: dict[str, Any]) -> dict[str, Any]:
    status = require_non_empty_string(item.get("status"), "reference channel audit item status")
    if status == "mismatch":
        reason = "reference-channel-energy-mismatch"
    elif status == "uncertain":
        reason = "reference-channel-energy-uncertain"
    else:
        raise ValueError(f"unsupported reference channel audit review status: {status!r}")
    result = {
        "case_id": case_id,
        "reference_id": require_non_empty_string(item.get("segment_id"), "reference channel segment_id"),
        "candidate_id": None,
        "start_ms": require_int(item.get("start_ms"), "reference channel start_ms"),
        "end_ms": require_int(item.get("end_ms"), "reference channel end_ms"),
        "reasons": [reason],
        "reference_text": "",
        "candidate_text": "",
        "reference_channel": item.get("reference_channel"),
        "candidate_channel": item.get("energy_channel"),
        "left_dbfs": item.get("left_dbfs"),
        "right_dbfs": item.get("right_dbfs"),
        "delta_db": item.get("delta_db"),
        "review_clip_start_ms": item.get("review_clip_start_ms"),
        "review_clip_end_ms": item.get("review_clip_end_ms"),
        "review_clip_left_dbfs": item.get("review_clip_left_dbfs"),
        "review_clip_right_dbfs": item.get("review_clip_right_dbfs"),
        "review_clip_delta_db": item.get("review_clip_delta_db"),
    }
    result["priority_score"] = reference_channel_review_priority_score(result)
    return result


def reference_channel_review_priority_score(item: dict[str, Any]) -> float:
    reasons = item.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        return 0.0
    reason_score = max(REFERENCE_CHANNEL_REVIEW_REASON_PRIORITY.get(str(reason), 0.0) for reason in reasons)
    start_ms = require_int(item.get("start_ms"), "review item start_ms")
    end_ms = require_int(item.get("end_ms"), "review item end_ms")
    duration_ms = max(0, end_ms - start_ms)
    return reason_score + (duration_ms / 1000.0)


def reference_channel_review_item_sort_key(item: dict[str, Any]) -> tuple[float, str, int, str]:
    return (
        -float(item.get("priority_score") or 0.0),
        str(item.get("case_id") or ""),
        int(item.get("start_ms") or 0),
        str(item.get("reference_id") or ""),
    )


def reference_channel_review_reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        reasons = item.get("reasons")
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            increment(counts, str(reason))
    return dict(sorted(counts.items()))


def increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value
