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
EVAL_CHANNELS = ("L", "R", "MIX")


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

    return {
        "format": EVAL_FORMAT,
        "reference_segments": len(reference_speech),
        "candidate_segments": len(candidate_speech),
        "text": strict_text,
        "text_practical": practical_text,
        "timing": timing_errors,
        "timing_time_aligned": time_aligned_timing_errors,
        "channel": channel_summary,
        "channel_time_aligned": time_aligned_channel_summary,
        "review": {
            "candidate_review_count": review_count,
            "candidate_review_ratio": review_count / max(1, len(candidate.segments)),
        },
    }


def evaluate_manifest(manifest_path: Path, *, source_language: str = "ja") -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = validate_eval_manifest(manifest)
    base_dir = manifest_path.parent
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
                "reference": case["reference"],
                "candidate": case["candidate"],
                "report": report,
            }
        )

    return {
        "format": EVAL_SUITE_FORMAT,
        "manifest_format": EVAL_MANIFEST_FORMAT,
        "manifest": str(manifest_path),
        "case_count": len(evaluated_cases),
        "cases": evaluated_cases,
        "summary": aggregate_eval_reports(reports),
    }


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
        normalized_cases.append(normalized)
    return normalized_cases


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
        "timing": aggregate_timing_reports(reports, "timing"),
        "timing_time_aligned": aggregate_timing_reports(reports, "timing_time_aligned"),
        "channel": aggregate_channel_reports(reports, "channel"),
        "channel_time_aligned": aggregate_channel_reports(reports, "channel_time_aligned"),
        "review": aggregate_review_reports(reports),
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


def normalize_for_cer(text: str, *, mode: str = "strict") -> str:
    if mode == "strict":
        return re.sub(r"\s+", "", text)
    if mode != "practical":
        raise ValueError("CER normalization mode must be strict or practical")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", "", normalized)
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
