from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.srt import parse_srt

EVAL_FORMAT = "custom-asmr-eval-v1"
EVAL_MANIFEST_FORMAT = "custom-asmr-eval-manifest-v1"
EVAL_SUITE_FORMAT = "custom-asmr-eval-suite-v1"
REVIEW_EFFORT_FORMAT = "custom-asmr-review-effort-v1"
EVAL_COMPARISON_FORMAT = "custom-asmr-eval-comparison-v1"
EVAL_CHANNELS = ("L", "R", "MIX")
REVIEW_EFFORT_TIMING_THRESHOLD_MS = 500
JAPANESE_RELAXED_REMOVED_CHARS = "ー〜～"


def load_transcript_document(path: Path, *, source_language: str = "ja") -> MasterDocument:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".srt":
        return parse_srt(content, source_language=source_language, source_file=path.name)
    return MasterDocument.from_json(json.loads(content))


def evaluate_transcripts(reference: MasterDocument, candidate: MasterDocument) -> dict[str, Any]:
    reference_speech = speech_segments(reference)
    candidate_speech = speech_segments(candidate)
    raw_reference_text = "".join(segment.text for segment in reference_speech)
    raw_candidate_text = "".join(segment.text for segment in candidate_speech)
    strict_text = text_error_summary(raw_reference_text, raw_candidate_text, mode="strict")
    practical_text = text_error_summary(raw_reference_text, raw_candidate_text, mode="practical")
    japanese_relaxed_text = text_error_summary(raw_reference_text, raw_candidate_text, mode="japanese-relaxed")
    paired = list(zip(reference_speech, candidate_speech))
    time_aligned_paired = time_aligned_segment_pairs(reference_speech, candidate_speech)
    timing_errors = timing_error_summary(paired)
    time_aligned_timing_errors = time_aligned_timing_summary(
        reference_speech,
        candidate_speech,
        time_aligned_paired,
    )
    channel_summary = channel_accuracy_summary(paired)
    time_aligned_channel_summary = channel_accuracy_summary(time_aligned_paired)
    review_count = sum(1 for segment in candidate.segments if segment.needs_review)
    review_effort = review_effort_summary(reference_speech, candidate_speech, time_aligned_paired)

    return {
        "format": EVAL_FORMAT,
        "reference_segments": len(reference_speech),
        "candidate_segments": len(candidate_speech),
        "text": strict_text,
        "text_practical": practical_text,
        "text_japanese_relaxed": japanese_relaxed_text,
        "timing": timing_errors,
        "timing_time_aligned": time_aligned_timing_errors,
        "channel": channel_summary,
        "channel_time_aligned": time_aligned_channel_summary,
        "review": {
            "candidate_review_count": review_count,
            "candidate_review_ratio": review_count / max(1, len(candidate.segments)),
        },
        "review_effort": review_effort,
    }


def evaluate_manifest(manifest_path: Path, *, source_language: str = "ja") -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = validate_eval_manifest(manifest)
    base_dir = manifest_path.parent
    manifest_reference_type = optional_manifest_string(manifest, "reference_type") or "unspecified"
    manifest_reference_notes = optional_manifest_string(manifest, "reference_notes")
    evaluated_cases = []
    reports = []

    for case in cases:
        case_id = case["id"]
        reference_path = resolve_manifest_path(base_dir, case["reference"])
        candidate_path = resolve_manifest_path(base_dir, case["candidate"])
        reference = load_transcript_document(reference_path, source_language=source_language)
        candidate = load_transcript_document(candidate_path, source_language=source_language)
        report = evaluate_transcripts(reference, candidate)
        reports.append(report)
        evaluated_cases.append(
            {
                "id": case_id,
                "candidate_id": case.get("candidate_id") or Path(case["candidate"]).stem,
                "reference_type": case.get("reference_type") or manifest_reference_type,
                "reference": case["reference"],
                "candidate": case["candidate"],
                "report": report,
            }
        )
        if case.get("reference_notes") or manifest_reference_notes:
            evaluated_cases[-1]["reference_notes"] = case.get("reference_notes") or manifest_reference_notes

    result = {
        "format": EVAL_SUITE_FORMAT,
        "manifest_format": EVAL_MANIFEST_FORMAT,
        "manifest": str(manifest_path),
        "reference_type": manifest_reference_type,
        "case_count": len(evaluated_cases),
        "cases": evaluated_cases,
        "summary": aggregate_eval_reports(reports),
    }
    if manifest_reference_notes:
        result["reference_notes"] = manifest_reference_notes
    return result


def review_effort_items_report(report: dict[str, Any], *, source_report: str | None = None) -> dict[str, Any]:
    report_format = report.get("format")
    if report_format == EVAL_FORMAT:
        items = normalize_review_effort_items(
            extract_report_review_effort_items(report),
            case_id=None,
            candidate_id=None,
            reference_type=None,
        )
    elif report_format == EVAL_SUITE_FORMAT:
        items = []
        cases = report.get("cases")
        if not isinstance(cases, list):
            raise ValueError("eval suite cases must be an array")
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("eval suite case must be an object")
            case_report = case.get("report")
            if not isinstance(case_report, dict):
                raise ValueError("eval suite case report must be an object")
            items.extend(
                normalize_review_effort_items(
                    extract_report_review_effort_items(case_report),
                    case_id=optional_report_string(case, "id"),
                    candidate_id=optional_report_string(case, "candidate_id"),
                    reference_type=optional_report_string(case, "reference_type"),
                )
            )
    else:
        raise ValueError(f"unsupported eval report format: {report_format!r}")

    result = {
        "format": REVIEW_EFFORT_FORMAT,
        "item_count": len(items),
        "reason_counts": review_effort_reason_counts(items),
        "items": items,
    }
    if source_report is not None:
        result["source_report"] = source_report
    return result


def compare_eval_reports(report_files: list[Path]) -> dict[str, Any]:
    if not report_files:
        raise ValueError("eval comparison requires at least one report")
    items = [eval_comparison_item(path, json.loads(path.read_text(encoding="utf-8"))) for path in report_files]
    ranked_items = sorted(
        items,
        key=lambda item: (
            item["segments_needing_edit_ratio"],
            item["practical_cer"],
            -(item["time_aligned_500ms_ratio"] if item["time_aligned_500ms_ratio"] is not None else -1.0),
            -(item["channel_time_aligned_accuracy"] if item["channel_time_aligned_accuracy"] is not None else -1.0),
            item["label"],
        ),
    )
    return {
        "format": EVAL_COMPARISON_FORMAT,
        "report_count": len(ranked_items),
        "ranked_by": [
            "segments_needing_edit_ratio",
            "practical_cer",
            "time_aligned_500ms_ratio desc",
            "channel_time_aligned_accuracy desc",
        ],
        "items": ranked_items,
    }


def eval_comparison_item(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_format = report.get("format")
    if report_format == EVAL_FORMAT:
        metrics = report
        case_count = 1
        reference_type = None
    elif report_format == EVAL_SUITE_FORMAT:
        metrics = report.get("summary")
        if not isinstance(metrics, dict):
            raise ValueError(f"{path}: eval suite summary must be an object")
        case_count = report.get("case_count")
        if not isinstance(case_count, int):
            raise ValueError(f"{path}: eval suite case_count must be an integer")
        reference_type = report.get("reference_type")
        if reference_type is not None and not isinstance(reference_type, str):
            raise ValueError(f"{path}: eval suite reference_type must be a string")
    else:
        raise ValueError(f"{path}: unsupported eval report format {report_format!r}")

    text_practical = require_report_mapping(metrics, "text_practical", path)
    text_japanese_relaxed = optional_report_mapping(metrics, "text_japanese_relaxed", path)
    timing_time_aligned = require_report_mapping(metrics, "timing_time_aligned", path)
    channel_time_aligned = require_report_mapping(metrics, "channel_time_aligned", path)
    review = optional_report_mapping(metrics, "review", path)
    review_effort = require_report_mapping(metrics, "review_effort", path)
    item = {
        "label": path.stem,
        "report": str(path),
        "report_format": report_format,
        "case_count": case_count,
        "practical_cer": require_report_number(text_practical, "cer", path),
        "japanese_relaxed_cer": None
        if text_japanese_relaxed is None
        else require_report_number(text_japanese_relaxed, "cer", path),
        "time_aligned_500ms_ratio": optional_report_number(timing_time_aligned, "within_500ms_ratio", path),
        "channel_time_aligned_accuracy": optional_report_number(channel_time_aligned, "accuracy", path),
        "channel_time_aligned_mix_ratio": require_report_number(channel_time_aligned, "candidate_mix_ratio", path),
        "candidate_review_ratio": None if review is None else require_report_number(review, "candidate_review_ratio", path),
        "segments_needing_edit": require_report_number(review_effort, "segments_needing_edit", path),
        "segments_needing_edit_ratio": require_report_number(review_effort, "segments_needing_edit_ratio", path),
    }
    if reference_type is not None:
        item["reference_type"] = reference_type
    return item


def require_report_mapping(metrics: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = metrics.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: eval report {key} must be an object")
    return value


def optional_report_mapping(metrics: dict[str, Any], key: str, path: Path) -> dict[str, Any] | None:
    value = metrics.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: eval report {key} must be an object or null")
    return value


def require_report_number(metrics: dict[str, Any], key: str, path: Path) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: eval report {key} must be a number")
    return float(value)


def optional_report_number(metrics: dict[str, Any], key: str, path: Path) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: eval report {key} must be a number or null")
    return float(value)


def extract_report_review_effort_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    review_effort = report.get("review_effort")
    if not isinstance(review_effort, dict):
        raise ValueError("eval report review_effort must be an object")
    items = review_effort.get("items", [])
    if not isinstance(items, list):
        raise ValueError("eval report review_effort.items must be an array")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("eval report review_effort.items entries must be objects")
    return items


def normalize_review_effort_items(
    items: list[dict[str, Any]],
    *,
    case_id: str | None,
    candidate_id: str | None,
    reference_type: str | None,
) -> list[dict[str, Any]]:
    normalized = []
    for item in items:
        normalized_item = dict(item)
        if case_id is not None:
            normalized_item["case_id"] = case_id
        if candidate_id is not None:
            normalized_item["case_candidate_id"] = candidate_id
        if reference_type is not None:
            normalized_item["reference_type"] = reference_type
        start_ms = normalized_item.get("start_ms")
        end_ms = normalized_item.get("end_ms")
        if isinstance(start_ms, int) and isinstance(end_ms, int):
            normalized_item["duration_ms"] = max(0, end_ms - start_ms)
        reference_start_ms = normalized_item.get("reference_start_ms")
        candidate_start_ms = normalized_item.get("candidate_start_ms")
        if isinstance(reference_start_ms, int) and isinstance(candidate_start_ms, int):
            normalized_item["start_delta_ms"] = candidate_start_ms - reference_start_ms
        reference_end_ms = normalized_item.get("reference_end_ms")
        candidate_end_ms = normalized_item.get("candidate_end_ms")
        if isinstance(reference_end_ms, int) and isinstance(candidate_end_ms, int):
            normalized_item["end_delta_ms"] = candidate_end_ms - reference_end_ms
        normalized.append(normalized_item)
    return normalized


def optional_report_string(report: dict[str, Any], key: str) -> str | None:
    value = report.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"eval report {key} must be a string")
    return value


def review_effort_reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        reasons = item.get("reasons")
        if not isinstance(reasons, list):
            raise ValueError("review effort item reasons must be an array")
        for reason in reasons:
            if not isinstance(reason, str):
                raise ValueError("review effort item reasons must be strings")
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def validate_eval_manifest(manifest: Any) -> list[dict[str, str]]:
    if not isinstance(manifest, dict):
        raise ValueError("eval manifest must be a JSON object")
    if manifest.get("format") != EVAL_MANIFEST_FORMAT:
        raise ValueError(f"eval manifest format must be {EVAL_MANIFEST_FORMAT}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("eval manifest cases must be a non-empty array")

    normalized_cases = []
    seen_ids = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"eval manifest case {index} must be an object")
        case_id = require_manifest_string(case, "id", index)
        if case_id in seen_ids:
            raise ValueError(f"eval manifest case id is duplicated: {case_id}")
        seen_ids.add(case_id)
        normalized = {
            "id": case_id,
            "reference": require_manifest_string(case, "reference", index),
            "candidate": require_manifest_string(case, "candidate", index),
        }
        candidate_id = case.get("candidate_id")
        if candidate_id is not None:
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(f"eval manifest case {index} candidate_id must be a non-empty string")
            normalized["candidate_id"] = candidate_id
        reference_type = case.get("reference_type")
        if reference_type is not None:
            if not isinstance(reference_type, str) or not reference_type:
                raise ValueError(f"eval manifest case {index} reference_type must be a non-empty string")
            normalized["reference_type"] = reference_type
        reference_notes = case.get("reference_notes")
        if reference_notes is not None:
            if not isinstance(reference_notes, str) or not reference_notes:
                raise ValueError(f"eval manifest case {index} reference_notes must be a non-empty string")
            normalized["reference_notes"] = reference_notes
        normalized_cases.append(normalized)
    return normalized_cases


def optional_manifest_string(manifest: dict[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"eval manifest {key} must be a non-empty string")
    return value


def require_manifest_string(case: dict[str, Any], key: str, index: int) -> str:
    value = case.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"eval manifest case {index} {key} must be a non-empty string")
    return value


def resolve_manifest_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def aggregate_eval_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reference_segments": sum(report["reference_segments"] for report in reports),
        "candidate_segments": sum(report["candidate_segments"] for report in reports),
        "text": aggregate_text_reports(reports, "text"),
        "text_practical": aggregate_text_reports(reports, "text_practical"),
        "text_japanese_relaxed": aggregate_text_reports(reports, "text_japanese_relaxed"),
        "timing": aggregate_timing_reports(reports, "timing"),
        "timing_time_aligned": aggregate_timing_reports(reports, "timing_time_aligned"),
        "channel": aggregate_channel_reports(reports, "channel"),
        "channel_time_aligned": aggregate_channel_reports(reports, "channel_time_aligned"),
        "review": aggregate_review_reports(reports),
        "review_effort": aggregate_review_effort_reports(reports),
    }


def aggregate_text_reports(reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    edit_distance = sum(report[key]["edit_distance"] for report in reports)
    reference_characters = sum(report[key]["reference_characters"] for report in reports)
    candidate_characters = sum(report[key]["candidate_characters"] for report in reports)
    return {
        "mode": reports[0][key]["mode"],
        "cer": 0.0
        if reference_characters == 0 and candidate_characters == 0
        else edit_distance / max(1, reference_characters),
        "edit_distance": edit_distance,
        "reference_characters": reference_characters,
        "candidate_characters": candidate_characters,
    }


def aggregate_timing_reports(reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    paired_segments = sum(report[key]["paired_segments"] for report in reports)
    boundary_samples = sum(report[key]["boundary_samples"] for report in reports)
    timed_reports = [report for report in reports if report[key]["paired_segments"] > 0]
    reference_segments = sum(report[key].get("reference_segments", 0) for report in reports)
    candidate_segments = sum(report[key].get("candidate_segments", 0) for report in reports)
    if paired_segments == 0:
        summary = {
            "paired_segments": 0,
            "boundary_samples": 0,
            "mean_start_error_ms": None,
            "mean_end_error_ms": None,
            "mean_boundary_error_ms": None,
            "max_boundary_error_ms": None,
            "within_250ms_count": 0,
            "within_250ms_ratio": None,
            "within_500ms_count": 0,
            "within_500ms_ratio": None,
        }
        if key == "timing_time_aligned":
            summary.update(
                {
                    "reference_segments": reference_segments,
                    "candidate_segments": candidate_segments,
                    "matched_reference_segments": 0,
                    "reference_match_ratio": None if reference_segments == 0 else 0.0,
                }
            )
        return summary
    start_error_sum = sum(
        report[key]["mean_start_error_ms"] * report[key]["paired_segments"] for report in timed_reports
    )
    end_error_sum = sum(
        report[key]["mean_end_error_ms"] * report[key]["paired_segments"] for report in timed_reports
    )
    boundary_error_sum = sum(
        report[key]["mean_boundary_error_ms"] * report[key]["paired_segments"] for report in timed_reports
    )
    within_250ms_count = sum(report[key]["within_250ms_count"] for report in reports)
    within_500ms_count = sum(report[key]["within_500ms_count"] for report in reports)
    summary = {
        "paired_segments": paired_segments,
        "boundary_samples": boundary_samples,
        "mean_start_error_ms": start_error_sum / paired_segments,
        "mean_end_error_ms": end_error_sum / paired_segments,
        "mean_boundary_error_ms": boundary_error_sum / paired_segments,
        "max_boundary_error_ms": max(report[key]["max_boundary_error_ms"] for report in timed_reports),
        "within_250ms_count": within_250ms_count,
        "within_250ms_ratio": within_250ms_count / max(1, boundary_samples),
        "within_500ms_count": within_500ms_count,
        "within_500ms_ratio": within_500ms_count / max(1, boundary_samples),
    }
    if key == "timing_time_aligned":
        summary.update(
            {
                "reference_segments": reference_segments,
                "candidate_segments": candidate_segments,
                "matched_reference_segments": paired_segments,
                "reference_match_ratio": paired_segments / max(1, reference_segments),
            }
        )
    return summary


def aggregate_channel_reports(reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    paired_segments = sum(report[key]["paired_segments"] for report in reports)
    comparable_segments = sum(report[key]["comparable_segments"] for report in reports)
    candidate_mix_segments = sum(report[key]["candidate_mix_segments"] for report in reports)
    confusion = empty_channel_confusion()
    for report in reports:
        for reference_channel in EVAL_CHANNELS:
            for candidate_channel in EVAL_CHANNELS:
                confusion[reference_channel][candidate_channel] += report[key]["confusion"][reference_channel][
                    candidate_channel
                ]

    if comparable_segments == 0:
        return {
            "paired_segments": paired_segments,
            "comparable_segments": 0,
            "accuracy": None,
            "confusion": confusion,
            "candidate_mix_segments": candidate_mix_segments,
            "candidate_mix_ratio": candidate_mix_segments / max(1, paired_segments),
        }
    correct_segments = sum(
        report[key]["accuracy"] * report[key]["comparable_segments"]
        for report in reports
        if report[key]["accuracy"] is not None
    )
    return {
        "paired_segments": paired_segments,
        "comparable_segments": comparable_segments,
        "accuracy": correct_segments / comparable_segments,
        "confusion": confusion,
        "candidate_mix_segments": candidate_mix_segments,
        "candidate_mix_ratio": candidate_mix_segments / max(1, paired_segments),
    }


def aggregate_review_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    review_count = sum(report["review"]["candidate_review_count"] for report in reports)
    candidate_segments = sum(report["candidate_segments"] for report in reports)
    return {
        "candidate_review_count": review_count,
        "candidate_review_ratio": review_count / max(1, candidate_segments),
    }


def aggregate_review_effort_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    threshold = reports[0]["review_effort"]["timing_threshold_ms"]
    reference_segments = sum(report["review_effort"]["reference_segments"] for report in reports)
    candidate_segments = sum(report["review_effort"]["candidate_segments"] for report in reports)
    matched_reference_segments = sum(report["review_effort"]["matched_reference_segments"] for report in reports)
    text_edit_segments = sum(report["review_effort"]["text_edit_segments"] for report in reports)
    channel_edit_segments = sum(report["review_effort"]["channel_edit_segments"] for report in reports)
    timing_edit_segments = sum(report["review_effort"]["timing_edit_segments"] for report in reports)
    missing_reference_segments = sum(report["review_effort"]["missing_reference_segments"] for report in reports)
    extra_candidate_segments = sum(report["review_effort"]["extra_candidate_segments"] for report in reports)
    segments_needing_edit = sum(report["review_effort"]["segments_needing_edit"] for report in reports)
    return {
        "timing_threshold_ms": threshold,
        "reference_segments": reference_segments,
        "candidate_segments": candidate_segments,
        "matched_reference_segments": matched_reference_segments,
        "text_edit_segments": text_edit_segments,
        "channel_edit_segments": channel_edit_segments,
        "timing_edit_segments": timing_edit_segments,
        "missing_reference_segments": missing_reference_segments,
        "extra_candidate_segments": extra_candidate_segments,
        "segments_needing_edit": segments_needing_edit,
        "segments_needing_edit_ratio": segments_needing_edit / max(1, reference_segments + extra_candidate_segments),
    }


def speech_segments(master: MasterDocument) -> tuple[Segment, ...]:
    return tuple(segment for segment in master.segments if segment.kind == "speech" and segment.text)


def time_aligned_segment_pairs(
    reference_segments: tuple[Segment, ...],
    candidate_segments: tuple[Segment, ...],
) -> list[tuple[Segment, Segment]]:
    pairs: list[tuple[Segment, Segment]] = []
    for reference in reference_segments:
        best_candidate = None
        best_overlap = 0
        for candidate in candidate_segments:
            current_overlap = overlap_ms(reference, candidate)
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_candidate = candidate
        if best_candidate is not None:
            pairs.append((reference, best_candidate))
    return pairs


def overlap_ms(reference: Segment, candidate: Segment) -> int:
    return max(0, min(reference.end_ms, candidate.end_ms) - max(reference.start_ms, candidate.start_ms))


def time_aligned_timing_summary(
    reference_segments: tuple[Segment, ...],
    candidate_segments: tuple[Segment, ...],
    paired_segments: list[tuple[Segment, Segment]],
) -> dict[str, Any]:
    summary = timing_error_summary(paired_segments)
    summary.update(
        {
            "reference_segments": len(reference_segments),
            "candidate_segments": len(candidate_segments),
            "matched_reference_segments": len(paired_segments),
            "reference_match_ratio": None
            if not reference_segments
            else len(paired_segments) / len(reference_segments),
        }
    )
    return summary


def text_error_summary(reference_text: str, candidate_text: str, *, mode: str) -> dict[str, Any]:
    normalized_reference = normalize_for_cer(reference_text, mode=mode)
    normalized_candidate = normalize_for_cer(candidate_text, mode=mode)
    distance = levenshtein_distance(normalized_reference, normalized_candidate)
    reference_chars = len(normalized_reference)
    return {
        "mode": mode,
        "cer": 0.0 if reference_chars == 0 and len(normalized_candidate) == 0 else distance / max(1, reference_chars),
        "edit_distance": distance,
        "reference_characters": reference_chars,
        "candidate_characters": len(normalized_candidate),
    }


def review_effort_summary(
    reference_segments: tuple[Segment, ...],
    candidate_segments: tuple[Segment, ...],
    paired_segments: list[tuple[Segment, Segment]],
) -> dict[str, Any]:
    reference_segments_needing_edit: set[str] = set()
    matched_reference_ids = {reference.id for reference, _ in paired_segments}
    items: list[dict[str, Any]] = []

    text_edit_segments = 0
    channel_edit_segments = 0
    timing_edit_segments = 0
    for reference, candidate in paired_segments:
        reasons: list[str] = []
        if normalize_for_cer(reference.text, mode="practical") != normalize_for_cer(candidate.text, mode="practical"):
            text_edit_segments += 1
            reference_segments_needing_edit.add(reference.id)
            reasons.append("text")
        if reference.channel != candidate.channel:
            channel_edit_segments += 1
            reference_segments_needing_edit.add(reference.id)
            reasons.append("channel")
        if (
            abs(reference.start_ms - candidate.start_ms) > REVIEW_EFFORT_TIMING_THRESHOLD_MS
            or abs(reference.end_ms - candidate.end_ms) > REVIEW_EFFORT_TIMING_THRESHOLD_MS
        ):
            timing_edit_segments += 1
            reference_segments_needing_edit.add(reference.id)
            reasons.append("timing")
        if reasons:
            items.append(review_effort_pair_item(reference, candidate, reasons=reasons))

    missing_reference_segments = len(reference_segments) - len(matched_reference_ids)
    for reference in reference_segments:
        if reference.id not in matched_reference_ids:
            items.append(
                {
                    "reference_id": reference.id,
                    "candidate_id": None,
                    "start_ms": reference.start_ms,
                    "end_ms": reference.end_ms,
                    "reasons": ["missing_reference"],
                    "reference_text": reference.text,
                    "candidate_text": "",
                    "reference_channel": reference.channel,
                    "candidate_channel": None,
                }
            )

    extra_candidate_segments = 0
    for candidate in candidate_segments:
        if all(overlap_ms(reference, candidate) == 0 for reference in reference_segments):
            extra_candidate_segments += 1
            items.append(
                {
                    "reference_id": None,
                    "candidate_id": candidate.id,
                    "start_ms": candidate.start_ms,
                    "end_ms": candidate.end_ms,
                    "reasons": ["extra_candidate"],
                    "reference_text": "",
                    "candidate_text": candidate.text,
                    "reference_channel": None,
                    "candidate_channel": candidate.channel,
                }
            )
    segments_needing_edit = (
        len(reference_segments_needing_edit) + missing_reference_segments + extra_candidate_segments
    )
    return {
        "timing_threshold_ms": REVIEW_EFFORT_TIMING_THRESHOLD_MS,
        "reference_segments": len(reference_segments),
        "candidate_segments": len(candidate_segments),
        "matched_reference_segments": len(paired_segments),
        "text_edit_segments": text_edit_segments,
        "channel_edit_segments": channel_edit_segments,
        "timing_edit_segments": timing_edit_segments,
        "missing_reference_segments": missing_reference_segments,
        "extra_candidate_segments": extra_candidate_segments,
        "segments_needing_edit": segments_needing_edit,
        "segments_needing_edit_ratio": segments_needing_edit / max(1, len(reference_segments) + extra_candidate_segments),
        "items": items,
    }


def review_effort_pair_item(reference: Segment, candidate: Segment, *, reasons: list[str]) -> dict[str, Any]:
    return {
        "reference_id": reference.id,
        "candidate_id": candidate.id,
        "start_ms": min(reference.start_ms, candidate.start_ms),
        "end_ms": max(reference.end_ms, candidate.end_ms),
        "reasons": reasons,
        "reference_text": reference.text,
        "candidate_text": candidate.text,
        "reference_channel": reference.channel,
        "candidate_channel": candidate.channel,
        "reference_start_ms": reference.start_ms,
        "reference_end_ms": reference.end_ms,
        "candidate_start_ms": candidate.start_ms,
        "candidate_end_ms": candidate.end_ms,
    }


def normalize_for_cer(text: str, *, mode: str = "strict") -> str:
    if mode == "strict":
        return re.sub(r"\s+", "", text)
    if mode not in {"practical", "japanese-relaxed"}:
        raise ValueError("CER normalization mode must be strict, practical, or japanese-relaxed")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", "", normalized)
    if mode == "japanese-relaxed":
        normalized = normalized.translate(str.maketrans("", "", JAPANESE_RELAXED_REMOVED_CHARS))
    return "".join(character for character in normalized if is_practical_cer_character(character))


def is_practical_cer_character(character: str) -> bool:
    category = unicodedata.category(character)
    if category.startswith("P") or category.startswith("S"):
        return False
    return True


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            deletion = previous[right_index] + 1
            insertion = current[right_index - 1] + 1
            substitution = previous[right_index - 1] + (0 if left_char == right_char else 1)
            current.append(min(deletion, insertion, substitution))
        previous = current
    return previous[-1]


def timing_error_summary(paired_segments: list[tuple[Segment, Segment]]) -> dict[str, Any]:
    if not paired_segments:
        return {
            "paired_segments": 0,
            "boundary_samples": 0,
            "mean_start_error_ms": None,
            "mean_end_error_ms": None,
            "mean_boundary_error_ms": None,
            "max_boundary_error_ms": None,
            "within_250ms_count": 0,
            "within_250ms_ratio": None,
            "within_500ms_count": 0,
            "within_500ms_ratio": None,
        }

    start_errors = [abs(reference.start_ms - candidate.start_ms) for reference, candidate in paired_segments]
    end_errors = [abs(reference.end_ms - candidate.end_ms) for reference, candidate in paired_segments]
    boundary_errors = start_errors + end_errors
    within_250ms_count = sum(1 for error in boundary_errors if error <= 250)
    within_500ms_count = sum(1 for error in boundary_errors if error <= 500)
    return {
        "paired_segments": len(paired_segments),
        "boundary_samples": len(boundary_errors),
        "mean_start_error_ms": mean(start_errors),
        "mean_end_error_ms": mean(end_errors),
        "mean_boundary_error_ms": mean(boundary_errors),
        "max_boundary_error_ms": max(boundary_errors),
        "within_250ms_count": within_250ms_count,
        "within_250ms_ratio": within_250ms_count / len(boundary_errors),
        "within_500ms_count": within_500ms_count,
        "within_500ms_ratio": within_500ms_count / len(boundary_errors),
    }


def channel_accuracy_summary(paired_segments: list[tuple[Segment, Segment]]) -> dict[str, Any]:
    confusion = empty_channel_confusion()
    candidate_mix_segments = 0
    for reference, candidate in paired_segments:
        confusion[reference.channel][candidate.channel] += 1
        if candidate.channel == "MIX":
            candidate_mix_segments += 1

    comparable = [
        (reference, candidate)
        for reference, candidate in paired_segments
        if reference.channel in {"L", "R"} and candidate.channel in {"L", "R"}
    ]
    if not comparable:
        return {
            "paired_segments": len(paired_segments),
            "comparable_segments": 0,
            "accuracy": None,
            "confusion": confusion,
            "candidate_mix_segments": candidate_mix_segments,
            "candidate_mix_ratio": candidate_mix_segments / max(1, len(paired_segments)),
        }
    correct = sum(1 for reference, candidate in comparable if reference.channel == candidate.channel)
    return {
        "paired_segments": len(paired_segments),
        "comparable_segments": len(comparable),
        "accuracy": correct / len(comparable),
        "confusion": confusion,
        "candidate_mix_segments": candidate_mix_segments,
        "candidate_mix_ratio": candidate_mix_segments / max(1, len(paired_segments)),
    }


def empty_channel_confusion() -> dict[str, dict[str, int]]:
    return {reference: {candidate: 0 for candidate in EVAL_CHANNELS} for reference in EVAL_CHANNELS}


def mean(values: list[int]) -> float:
    return sum(values) / len(values)
