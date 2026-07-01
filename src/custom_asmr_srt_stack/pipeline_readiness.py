from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.evaluation import EVAL_COMPARISON_FORMAT
from custom_asmr_srt_stack.reference_audit import REFERENCE_AUDIT_FORMAT, REFERENCE_AUDIT_SUITE_FORMAT
from custom_asmr_srt_stack.vad import VAD_COVERAGE_COMPARISON_FORMAT

PIPELINE_READINESS_FORMAT = "custom-asmr-pipeline-readiness-v1"
PIPELINE_STAGE_ORDER = (
    "reference",
    "vad_chunking",
    "alignment",
    "channel_attribution",
    "text_asr",
)
ASR_ONLY_STAGE_ORDER = (
    "reference",
    "vad_chunking",
    "alignment",
    "channel_attribution",
)


def build_pipeline_readiness(
    *,
    reference_audit_file: Path | None = None,
    vad_comparison_file: Path | None = None,
    eval_comparison_file: Path | None = None,
) -> dict[str, Any]:
    stages = {
        "reference": reference_stage(reference_audit_file),
        "vad_chunking": vad_stage(vad_comparison_file),
    }
    stages.update(eval_stages(eval_comparison_file))
    asr_only_blocking_stages = [
        stage for stage in ASR_ONLY_STAGE_ORDER if stages[stage]["status"] == "fail"
    ]
    asr_only_unknown_stages = [
        stage for stage in ASR_ONLY_STAGE_ORDER if stages[stage]["status"] == "unknown"
    ]
    quality_blocking_stages = [stage for stage in PIPELINE_STAGE_ORDER if stages[stage]["status"] == "fail"]
    unknown_stages = [stage for stage in PIPELINE_STAGE_ORDER if stages[stage]["status"] == "unknown"]
    asr_only_ready = not asr_only_blocking_stages and not asr_only_unknown_stages
    production_ready = asr_only_ready and stages["text_asr"]["status"] == "pass"
    return {
        "format": PIPELINE_READINESS_FORMAT,
        "summary": {
            "asr_only_ready": asr_only_ready,
            "production_ready": production_ready,
            "next_stage": next_stage(stages),
            "asr_only_blocking_stages": asr_only_blocking_stages,
            "asr_only_unknown_stages": asr_only_unknown_stages,
            "quality_blocking_stages": quality_blocking_stages,
            "unknown_stages": unknown_stages,
        },
        "stage_order": list(PIPELINE_STAGE_ORDER),
        "asr_only_stage_order": list(ASR_ONLY_STAGE_ORDER),
        "stages": {stage: stages[stage] for stage in PIPELINE_STAGE_ORDER},
    }


def reference_stage(path: Path | None) -> dict[str, Any]:
    if path is None:
        return unknown_stage("reference", "reference audit report was not provided")
    report = read_json_report(path)
    report_format = report.get("format")
    if report_format == REFERENCE_AUDIT_SUITE_FORMAT:
        metrics = require_mapping(report.get("summary"), f"{path}: reference audit summary")
    elif report_format == REFERENCE_AUDIT_FORMAT:
        metrics = report
    else:
        raise ValueError(f"{path}: reference audit report format must be {REFERENCE_AUDIT_SUITE_FORMAT}")

    reasons = []
    for key, label in (
        ("review_count", "reference review flags remain"),
        ("same_channel_overlap_pair_count", "same-channel reference overlaps remain"),
        ("exact_boundary_overlap_pair_count", "exact-boundary reference overlaps remain"),
        ("long_segment_count", "long reference segments remain"),
    ):
        count = require_int(metrics.get(key), f"{path}: reference audit {key}")
        if count > 0:
            reasons.append(f"{label}: {count}")

    warnings = []
    flag_type_counts = metrics.get("flag_type_counts")
    if isinstance(flag_type_counts, dict) and int(flag_type_counts.get("near_full_speech_coverage", 0)) > 0:
        warnings.append(f"near-full speech coverage cases: {flag_type_counts['near_full_speech_coverage']}")

    return stage_report(
        "reference",
        reasons=reasons,
        warnings=warnings,
        metrics={
            "report": str(path),
            "segment_count": optional_int(metrics.get("segment_count"), f"{path}: reference audit segment_count"),
            "review_count": require_int(metrics.get("review_count"), f"{path}: reference audit review_count"),
            "same_channel_overlap_pair_count": require_int(
                metrics.get("same_channel_overlap_pair_count"),
                f"{path}: reference audit same_channel_overlap_pair_count",
            ),
            "exact_boundary_overlap_pair_count": require_int(
                metrics.get("exact_boundary_overlap_pair_count"),
                f"{path}: reference audit exact_boundary_overlap_pair_count",
            ),
            "long_segment_count": require_int(
                metrics.get("long_segment_count"),
                f"{path}: reference audit long_segment_count",
            ),
            "speech_coverage_ratio": optional_number(
                metrics.get("speech_coverage_ratio"),
                f"{path}: reference audit speech_coverage_ratio",
            ),
        },
    )


def vad_stage(path: Path | None) -> dict[str, Any]:
    if path is None:
        return unknown_stage("vad_chunking", "VAD coverage comparison report was not provided")
    report = read_json_report(path)
    if report.get("format") != VAD_COVERAGE_COMPARISON_FORMAT:
        raise ValueError(f"{path}: VAD report format must be {VAD_COVERAGE_COMPARISON_FORMAT}")
    items = require_non_empty_mapping_list(report.get("items"), f"{path}: VAD comparison items")
    gated = "quality_gate" in report
    if gated:
        for index, item in enumerate(items):
            if not isinstance(item.get("gate_passed"), bool):
                raise ValueError(f"{path}: gated VAD comparison item {index} gate_passed must be a boolean")
    passing_items = [item for item in items if bool(item.get("gate_passed", True))]
    chosen = passing_items[0] if passing_items else items[0]

    reasons = []
    if gated and not passing_items:
        reasons.append("no VAD candidate passes the configured coverage gate")
    missed_reference_duration_ms = require_int(
        chosen.get("missed_reference_duration_ms"),
        f"{path}: VAD chosen missed_reference_duration_ms",
    )
    if not gated and missed_reference_duration_ms > 0:
        reasons.append(f"chosen VAD candidate misses reference speech: {missed_reference_duration_ms}ms")
    if chosen.get("gate_passed") is False:
        for failure in require_string_list(chosen.get("gate_failures"), f"{path}: VAD gate_failures"):
            reasons.append(f"chosen VAD candidate gate failure: {failure}")

    return stage_report(
        "vad_chunking",
        reasons=reasons,
        metrics={
            "report": str(path),
            "chosen_label": require_string(chosen.get("label"), f"{path}: VAD chosen label"),
            "gated": gated,
            "missed_reference_duration_ms": missed_reference_duration_ms,
            "extra_detected_duration_ms": require_int(
                chosen.get("extra_detected_duration_ms"),
                f"{path}: VAD chosen extra_detected_duration_ms",
            ),
            "reference_recall": optional_number(
                chosen.get("reference_recall"),
                f"{path}: VAD chosen reference_recall",
            ),
            "detected_precision": optional_number(
                chosen.get("detected_precision"),
                f"{path}: VAD chosen detected_precision",
            ),
            "detected_max_interval_ms": optional_number(
                chosen.get("detected_max_interval_ms"),
                f"{path}: VAD chosen detected_max_interval_ms",
            ),
        },
    )


def eval_stages(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {
            "alignment": unknown_stage("alignment", "eval comparison report was not provided"),
            "channel_attribution": unknown_stage(
                "channel_attribution",
                "eval comparison report was not provided",
            ),
            "text_asr": unknown_stage("text_asr", "eval comparison report was not provided"),
        }
    report = read_json_report(path)
    if report.get("format") != EVAL_COMPARISON_FORMAT:
        raise ValueError(f"{path}: eval report format must be {EVAL_COMPARISON_FORMAT}")
    items = require_non_empty_mapping_list(report.get("items"), f"{path}: eval comparison items")
    best = items[0]
    label = require_string(best.get("label"), f"{path}: eval best label")

    timing_ratio = require_number(best.get("timing_edit_segment_ratio"), f"{path}: timing_edit_segment_ratio")
    alignment_reasons = []
    if timing_ratio > 0.0:
        alignment_reasons.append(f"best candidate still needs timing edits: {timing_ratio:.4f}")

    channel_ratio = require_number(best.get("channel_edit_segment_ratio"), f"{path}: channel_edit_segment_ratio")
    channel_reasons = []
    if channel_ratio > 0.0:
        channel_reasons.append(f"best candidate still needs channel edits: {channel_ratio:.4f}")

    text_ratio = require_number(best.get("text_edit_segment_ratio"), f"{path}: text_edit_segment_ratio")
    edit_ratio = require_number(best.get("segments_needing_edit_ratio"), f"{path}: segments_needing_edit_ratio")
    text_reasons = []
    if text_ratio > 0.0:
        text_reasons.append(f"best candidate still needs text edits: {text_ratio:.4f}")
    if edit_ratio > 0.0:
        text_reasons.append(f"best candidate still has segments needing edit: {edit_ratio:.4f}")

    return {
        "alignment": stage_report(
            "alignment",
            reasons=alignment_reasons,
            metrics={
                "report": str(path),
                "best_label": label,
                "timing_edit_segment_ratio": timing_ratio,
                "time_aligned_500ms_ratio": optional_number(
                    best.get("time_aligned_500ms_ratio"),
                    f"{path}: time_aligned_500ms_ratio",
                ),
            },
        ),
        "channel_attribution": stage_report(
            "channel_attribution",
            reasons=channel_reasons,
            metrics={
                "report": str(path),
                "best_label": label,
                "channel_edit_segment_ratio": channel_ratio,
                "channel_time_aligned_accuracy": optional_number(
                    best.get("channel_time_aligned_accuracy"),
                    f"{path}: channel_time_aligned_accuracy",
                ),
                "channel_time_aligned_mix_ratio": require_number(
                    best.get("channel_time_aligned_mix_ratio"),
                    f"{path}: channel_time_aligned_mix_ratio",
                ),
            },
        ),
        "text_asr": stage_report(
            "text_asr",
            reasons=text_reasons,
            metrics={
                "report": str(path),
                "best_label": label,
                "text_edit_segment_ratio": text_ratio,
                "segments_needing_edit_ratio": edit_ratio,
                "practical_cer": require_number(best.get("practical_cer"), f"{path}: practical_cer"),
                "dominant_review_effort_reason": optional_string(best.get("dominant_review_effort_reason")),
                "dominant_review_effort_ratio": optional_number(
                    best.get("dominant_review_effort_ratio"),
                    f"{path}: dominant_review_effort_ratio",
                ),
            },
        ),
    }


def next_stage(stages: dict[str, dict[str, Any]]) -> str | None:
    for stage in PIPELINE_STAGE_ORDER:
        if stages[stage]["status"] == "fail":
            return stage
    for stage in PIPELINE_STAGE_ORDER:
        if stages[stage]["status"] == "unknown":
            return stage
    return None


def stage_report(
    stage: str,
    *,
    reasons: list[str],
    metrics: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "fail" if reasons else "pass",
        "reasons": reasons,
        "warnings": [] if warnings is None else warnings,
        "metrics": metrics,
    }


def unknown_stage(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "unknown",
        "reasons": [reason],
        "warnings": [],
        "metrics": {},
    }


def read_json_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    return data


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_non_empty_mapping_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be an object")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return require_int(value, label)


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return require_number(value, label)


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional string value must be a string or null")
    return value


def require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{label} entries must be strings")
    return value
