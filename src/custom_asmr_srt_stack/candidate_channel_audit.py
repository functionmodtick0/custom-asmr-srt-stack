from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import normalize_audio_to_wav, split_wav_channels, wav_rms_dbfs
from custom_asmr_srt_stack.channel_attribution import (
    CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
    CHANNEL_ATTRIBUTION_THRESHOLD_DB,
    channel_is_quiet_enough,
)
from custom_asmr_srt_stack.evaluation import (
    load_transcript_document,
    resolve_manifest_path,
    speech_segments,
    validate_eval_manifest,
)
from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.review_pack import load_audio_by_case

CANDIDATE_CHANNEL_AUDIT_FORMAT = "custom-asmr-candidate-channel-audit-v1"
CANDIDATE_CHANNEL_AUDIT_SUITE_FORMAT = "custom-asmr-candidate-channel-audit-suite-v1"


def audit_candidate_channels_manifest(
    manifest_file: Path,
    *,
    audio_map_file: Path,
    source_language: str = "ja",
    threshold_db: float = CHANNEL_ATTRIBUTION_THRESHOLD_DB,
    quiet_channel_max_dbfs: float | None = CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
) -> dict[str, Any]:
    if threshold_db < 0:
        raise ValueError("threshold_db must be non-negative")
    manifest = load_eval_manifest(manifest_file)
    cases = validate_eval_manifest(manifest)
    manifest_base_dir = manifest_file.parent
    audio_by_case = load_audio_by_case(audio_file=None, audio_map_file=audio_map_file)

    case_reports = []
    for case in cases:
        case_id = case["id"]
        audio_path = audio_by_case.get(case_id)
        if audio_path is None:
            raise ValueError(f"audio map is missing case_id {case_id!r}")
        candidate_path = resolve_manifest_path(manifest_base_dir, case["candidate"])
        if not audio_path.is_file():
            raise ValueError(f"candidate channel audit audio file does not exist: {audio_path}")
        if not candidate_path.is_file():
            raise ValueError(f"candidate channel audit candidate file does not exist: {candidate_path}")

        normalized_audio = normalize_audio_to_wav(
            audio_path.read_bytes(),
            file_name=audio_path.name,
            mime_type=mimetypes.guess_type(audio_path.name)[0],
        )
        _info, channels = split_wav_channels(normalized_audio)
        if not {"L", "R"}.issubset(channels):
            raise ValueError(f"candidate channel audit case {case_id!r} requires stereo audio")
        candidate = load_transcript_document(candidate_path, source_language=source_language)
        case_reports.append(
            audit_master_candidate_channels(
                candidate,
                left_audio=channels["L"],
                right_audio=channels["R"],
                case_id=case_id,
                candidate=case["candidate"],
                candidate_id=case.get("candidate_id"),
                audio=str(audio_path),
                threshold_db=threshold_db,
                quiet_channel_max_dbfs=quiet_channel_max_dbfs,
            )
        )

    return {
        "format": CANDIDATE_CHANNEL_AUDIT_SUITE_FORMAT,
        "manifest": str(manifest_file),
        "audio_map": str(audio_map_file),
        "case_count": len(case_reports),
        "thresholds": {
            "threshold_db": threshold_db,
            "quiet_channel_max_dbfs": quiet_channel_max_dbfs,
        },
        "summary": aggregate_candidate_channel_audits(case_reports),
        "cases": case_reports,
    }


def load_eval_manifest(manifest_file: Path) -> dict[str, Any]:
    import json

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("eval manifest must be a JSON object")
    return data


def audit_master_candidate_channels(
    master: MasterDocument,
    *,
    left_audio: bytes,
    right_audio: bytes,
    case_id: str | None = None,
    candidate: str | None = None,
    candidate_id: str | None = None,
    audio: str | None = None,
    threshold_db: float = CHANNEL_ATTRIBUTION_THRESHOLD_DB,
    quiet_channel_max_dbfs: float | None = CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
) -> dict[str, Any]:
    if threshold_db < 0:
        raise ValueError("threshold_db must be non-negative")
    speech = tuple(sorted(speech_segments(master), key=lambda item: (item.start_ms, item.end_ms, item.id)))
    items = [
        candidate_channel_energy_item(
            segment,
            left_audio=left_audio,
            right_audio=right_audio,
            threshold_db=threshold_db,
            quiet_channel_max_dbfs=quiet_channel_max_dbfs,
        )
        for segment in speech
    ]
    return {
        "format": CANDIDATE_CHANNEL_AUDIT_FORMAT,
        "case_id": case_id,
        "candidate": candidate,
        "candidate_id": candidate_id,
        "audio": audio,
        "thresholds": {
            "threshold_db": threshold_db,
            "quiet_channel_max_dbfs": quiet_channel_max_dbfs,
        },
        "speech_segment_count": len(speech),
        "summary": candidate_channel_audit_summary(items),
        "items": items,
    }


def candidate_channel_energy_item(
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
        status = "mix_match" if segment.channel == "MIX" else "over_attribution"
    elif segment.channel == energy_channel:
        status = "match"
    elif segment.channel == "MIX":
        status = "missed_attribution"
    else:
        status = "wrong_side"
    return {
        "segment_id": segment.id,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "duration_ms": segment.end_ms - segment.start_ms,
        "candidate_channel": segment.channel,
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


def aggregate_candidate_channel_audits(cases: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    speech_segment_count = 0
    for case in cases:
        speech_segment_count += require_int(case.get("speech_segment_count"), "case speech_segment_count")
        raw_items = case.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("candidate channel audit case items must be an array")
        items.extend(item for item in raw_items if isinstance(item, dict))

    return {
        "speech_segment_count": speech_segment_count,
        **candidate_channel_audit_summary(items),
    }


def candidate_channel_audit_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    candidate_channel_counts: dict[str, int] = {}
    energy_channel_counts: dict[str, int] = {}
    for item in items:
        increment(status_counts, str(item.get("status") or "unknown"))
        increment(reason_counts, str(item.get("reason") or "unknown"))
        increment(candidate_channel_counts, str(item.get("candidate_channel") or "unknown"))
        increment(energy_channel_counts, str(item.get("energy_channel") or "unknown"))

    speech_segment_count = len(items)
    match_count = status_counts.get("match", 0)
    missed_attribution_count = status_counts.get("missed_attribution", 0)
    wrong_side_count = status_counts.get("wrong_side", 0)
    mix_match_count = status_counts.get("mix_match", 0)
    over_attribution_count = status_counts.get("over_attribution", 0)
    energy_labeled_count = match_count + missed_attribution_count + wrong_side_count
    energy_uncertain_count = mix_match_count + over_attribution_count
    return {
        "speech_segment_count": speech_segment_count,
        "energy_labeled_count": energy_labeled_count,
        "energy_uncertain_count": energy_uncertain_count,
        "match_count": match_count,
        "missed_attribution_count": missed_attribution_count,
        "wrong_side_count": wrong_side_count,
        "mix_match_count": mix_match_count,
        "over_attribution_count": over_attribution_count,
        "energy_labeled_match_ratio": ratio(match_count, energy_labeled_count),
        "energy_labeled_mix_ratio": ratio(missed_attribution_count, energy_labeled_count),
        "energy_labeled_wrong_side_ratio": ratio(wrong_side_count, energy_labeled_count),
        "over_attribution_ratio": ratio(over_attribution_count, energy_uncertain_count),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "candidate_channel_counts": dict(sorted(candidate_channel_counts.items())),
        "energy_channel_counts": dict(sorted(energy_channel_counts.items())),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value
