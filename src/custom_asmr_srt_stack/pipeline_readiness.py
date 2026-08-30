from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.candidate_channel_audit import CANDIDATE_CHANNEL_AUDIT_SUITE_FORMAT
from custom_asmr_srt_stack.channel_reference_audit import REFERENCE_CHANNEL_AUDIT_SUITE_FORMAT
from custom_asmr_srt_stack.evaluation import (
    EVAL_COMPARISON_FORMAT,
    EVAL_FORMAT,
    EVAL_SUITE_FORMAT,
    eval_comparison_item,
)
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
VAD_DOWNSTREAM_METRICS = (
    ("practical_cer", "practical CER", "lower"),
    ("channel_aware_practical_cer", "channel-aware practical CER", "lower"),
    ("time_aligned_500ms_ratio", "time-aligned 500ms ratio", "higher"),
    ("channel_aware_time_aligned_500ms_ratio", "same-channel time-aligned 500ms ratio", "higher"),
    ("channel_time_aligned_accuracy", "channel time-aligned accuracy", "higher"),
    ("channel_time_aligned_mix_ratio", "channel time-aligned MIX ratio", "lower"),
    ("segments_needing_edit_ratio", "segments needing edit ratio", "lower"),
    ("candidate_review_ratio", "candidate review ratio", "lower"),
)


def build_pipeline_readiness(
    *,
    reference_audit_file: Path | None = None,
    reference_channel_audit_file: Path | None = None,
    vad_comparison_file: Path | None = None,
    vad_baseline_eval_file: Path | None = None,
    vad_candidate_eval_file: Path | None = None,
    eval_comparison_file: Path | None = None,
    alignment_comparison_file: Path | None = None,
    channel_comparison_file: Path | None = None,
    candidate_channel_audit_file: Path | None = None,
    quality_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_type = readiness_reference_type(eval_comparison_file, alignment_comparison_file, channel_comparison_file)
    stages = {
        "reference": reference_stage(
            reference_audit_file,
            reference_channel_audit_file=reference_channel_audit_file,
            quality_gate=quality_gate,
            reference_type=reference_type,
        ),
        "vad_chunking": vad_stage(
            vad_comparison_file,
            baseline_eval_file=vad_baseline_eval_file,
            candidate_eval_file=vad_candidate_eval_file,
            require_downstream=bool(quality_gate and quality_gate.get("preset") == "local-asmr-v1"),
        ),
    }
    stages.update(eval_stages(eval_comparison_file, quality_gate=quality_gate))
    if alignment_comparison_file is not None:
        label, best = best_eval_comparison_item(alignment_comparison_file)
        stages["alignment"] = alignment_stage_from_item(
            alignment_comparison_file,
            label,
            best,
            quality_gate=quality_gate,
        )
    if candidate_channel_audit_file is not None:
        stages["channel_attribution"] = candidate_channel_energy_stage(
            candidate_channel_audit_file,
            quality_gate=quality_gate,
        )
    elif channel_comparison_file is not None:
        stages["channel_attribution"] = channel_stage(channel_comparison_file, quality_gate=quality_gate)
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
    report = {
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
    if quality_gate:
        report["quality_gate"] = quality_gate
    return report


def reference_stage(
    path: Path | None,
    *,
    reference_channel_audit_file: Path | None = None,
    quality_gate: dict[str, Any] | None = None,
    reference_type: str | None = None,
) -> dict[str, Any]:
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
    review_count = require_int(metrics.get("review_count"), f"{path}: reference audit review_count")
    content_unreviewed_count = optional_int(
        metrics.get("content_unreviewed_count"),
        f"{path}: reference audit content_unreviewed_count",
    )
    same_channel_overlap_count = require_int(
        metrics.get("same_channel_overlap_pair_count"),
        f"{path}: reference audit same_channel_overlap_pair_count",
    )
    exact_boundary_blocking_count = reference_exact_boundary_blocking_count(path, metrics)
    long_segment_count = require_int(metrics.get("long_segment_count"), f"{path}: reference audit long_segment_count")
    for count, label in (
        (review_count, "reference review flags remain"),
        (content_unreviewed_count or 0, "reference content review evidence missing"),
        (same_channel_overlap_count, "same-channel reference overlaps remain"),
        (exact_boundary_blocking_count, "same-channel exact-boundary reference overlaps remain"),
        (long_segment_count, "long reference segments remain"),
    ):
        if count > 0:
            reasons.append(f"{label}: {count}")
    gate = readiness_stage_gate(quality_gate, "required_reference_type")
    if gate:
        required_reference_type = gate["required_reference_type"]
        if reference_type is None:
            reasons.append("reference_type is unavailable")
        elif reference_type != required_reference_type:
            reasons.append(f"reference_type {reference_type!r} != {required_reference_type!r}")
    channel_metrics = reference_channel_audit_metrics(reference_channel_audit_file)
    if channel_metrics is not None:
        mismatch_count = channel_metrics["unresolved_mismatch_count"]
        uncertain_count = channel_metrics["unresolved_uncertain_count"]
        if mismatch_count > 0:
            reasons.append(f"unreviewed reference channel labels conflict with energy: {mismatch_count}")
        if uncertain_count > 0:
            reasons.append(f"unreviewed reference channel labels have uncertain energy evidence: {uncertain_count}")

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
            "review_count": review_count,
            "content_unreviewed_count": content_unreviewed_count,
            "same_channel_overlap_pair_count": same_channel_overlap_count,
            "exact_boundary_overlap_pair_count": require_int(
                metrics.get("exact_boundary_overlap_pair_count"),
                f"{path}: reference audit exact_boundary_overlap_pair_count",
            ),
            "exact_boundary_same_channel_overlap_pair_count": exact_boundary_blocking_count,
            "exact_boundary_cross_channel_overlap_pair_count": optional_int(
                metrics.get("exact_boundary_cross_channel_overlap_pair_count"),
                f"{path}: reference audit exact_boundary_cross_channel_overlap_pair_count",
            ),
            "long_segment_count": long_segment_count,
            "speech_coverage_ratio": optional_number(
                metrics.get("speech_coverage_ratio"),
                f"{path}: reference audit speech_coverage_ratio",
            ),
            "reference_type": reference_type,
            **({} if channel_metrics is None else {"channel_audit": channel_metrics}),
        },
        quality_gate=gate,
    )


def reference_channel_audit_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    report = read_json_report(path)
    if report.get("format") != REFERENCE_CHANNEL_AUDIT_SUITE_FORMAT:
        raise ValueError(f"{path}: reference channel audit report format must be {REFERENCE_CHANNEL_AUDIT_SUITE_FORMAT}")
    summary = require_mapping(report.get("summary"), f"{path}: reference channel audit summary")
    mismatch_count = require_int(
        summary.get("mismatch_count"),
        f"{path}: reference channel audit mismatch_count",
    )
    energy_uncertain_count = require_int(
        summary.get("energy_uncertain_count"),
        f"{path}: reference channel audit energy_uncertain_count",
    )
    unresolved_mismatch_count = summary.get("unresolved_mismatch_count", mismatch_count)
    unresolved_uncertain_count = summary.get("unresolved_uncertain_count", energy_uncertain_count)
    return {
        "report": str(path),
        "eligible_reference_channel_count": require_int(
            summary.get("eligible_reference_channel_count"),
            f"{path}: reference channel audit eligible_reference_channel_count",
        ),
        "energy_labeled_count": require_int(
            summary.get("energy_labeled_count"),
            f"{path}: reference channel audit energy_labeled_count",
        ),
        "energy_uncertain_count": energy_uncertain_count,
        "match_count": require_int(summary.get("match_count"), f"{path}: reference channel audit match_count"),
        "mismatch_count": mismatch_count,
        "channel_reviewed_count": optional_int(
            summary.get("channel_reviewed_count"),
            f"{path}: reference channel audit channel_reviewed_count",
        ),
        "reviewed_exception_count": optional_int(
            summary.get("reviewed_exception_count"),
            f"{path}: reference channel audit reviewed_exception_count",
        ),
        "unresolved_mismatch_count": require_int(
            unresolved_mismatch_count,
            f"{path}: reference channel audit unresolved_mismatch_count",
        ),
        "unresolved_uncertain_count": require_int(
            unresolved_uncertain_count,
            f"{path}: reference channel audit unresolved_uncertain_count",
        ),
        "match_ratio": optional_number(summary.get("match_ratio"), f"{path}: reference channel audit match_ratio"),
        "energy_labeled_ratio": optional_number(
            summary.get("energy_labeled_ratio"),
            f"{path}: reference channel audit energy_labeled_ratio",
        ),
    }


def reference_exact_boundary_blocking_count(path: Path, metrics: dict[str, Any]) -> int:
    same_channel_count = metrics.get("exact_boundary_same_channel_overlap_pair_count")
    if same_channel_count is not None:
        return require_int(
            same_channel_count,
            f"{path}: reference audit exact_boundary_same_channel_overlap_pair_count",
        )
    return require_int(
        metrics.get("exact_boundary_overlap_pair_count"),
        f"{path}: reference audit exact_boundary_overlap_pair_count",
    )


def vad_stage(
    path: Path | None,
    *,
    baseline_eval_file: Path | None = None,
    candidate_eval_file: Path | None = None,
    require_downstream: bool = False,
) -> dict[str, Any]:
    if (baseline_eval_file is None) != (candidate_eval_file is None):
        raise ValueError("VAD baseline and candidate eval reports must be provided together")
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

    warnings = []
    downstream_validation = None
    if baseline_eval_file is None:
        message = "VAD downstream ASR validation reports were not provided"
        if require_downstream:
            reasons.append(message)
        else:
            warnings.append(message + "; coverage alone does not promote the VAD candidate")
    else:
        downstream_validation = vad_downstream_validation(
            baseline_eval_file,
            candidate_eval_file,
        )
        reasons.extend(downstream_validation["regressions"])

    metrics = {
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
    }
    if downstream_validation is not None:
        metrics["downstream_validation"] = downstream_validation

    return stage_report(
        "vad_chunking",
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        quality_gate={"requires_downstream_no_regression": True} if require_downstream else None,
    )


def vad_downstream_validation(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline_report = read_json_report(baseline_path)
    candidate_report = read_json_report(candidate_path)
    validate_vad_downstream_scope(baseline_path, baseline_report, candidate_path, candidate_report)
    baseline = eval_comparison_item(baseline_path, baseline_report)
    candidate = eval_comparison_item(candidate_path, candidate_report)
    regressions = []
    metrics = {}
    tolerance = 1e-12
    for key, label, direction in VAD_DOWNSTREAM_METRICS:
        baseline_value = baseline.get(key)
        candidate_value = candidate.get(key)
        metrics[key] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": None
            if baseline_value is None or candidate_value is None
            else candidate_value - baseline_value,
            "better": direction,
        }
        if baseline_value is None and candidate_value is None:
            continue
        if baseline_value is None or candidate_value is None:
            regressions.append(f"downstream {label} availability differs between baseline and candidate")
            continue
        regressed = (
            candidate_value > baseline_value + tolerance
            if direction == "lower"
            else candidate_value < baseline_value - tolerance
        )
        if regressed:
            regressions.append(
                f"downstream {label} regressed: {baseline_value:.4f} -> {candidate_value:.4f}"
            )
    return {
        "baseline_report": str(baseline_path),
        "candidate_report": str(candidate_path),
        "case_count": baseline["case_count"],
        "passed": not regressions,
        "regressions": regressions,
        "metrics": metrics,
    }


def validate_vad_downstream_scope(
    baseline_path: Path,
    baseline: dict[str, Any],
    candidate_path: Path,
    candidate: dict[str, Any],
) -> None:
    baseline_format = baseline.get("format")
    candidate_format = candidate.get("format")
    if baseline_format != candidate_format:
        raise ValueError("VAD downstream baseline and candidate eval report formats must match")
    if baseline_format == EVAL_FORMAT:
        return
    if baseline_format != EVAL_SUITE_FORMAT:
        raise ValueError(f"{baseline_path}: unsupported VAD downstream eval format {baseline_format!r}")
    baseline_ids = vad_downstream_case_ids(baseline_path, baseline)
    candidate_ids = vad_downstream_case_ids(candidate_path, candidate)
    if baseline_ids != candidate_ids:
        raise ValueError("VAD downstream baseline and candidate eval reports must cover the same case ids")
    if baseline.get("reference_type") != candidate.get("reference_type"):
        raise ValueError("VAD downstream baseline and candidate reference_type must match")


def vad_downstream_case_ids(path: Path, report: dict[str, Any]) -> tuple[str, ...]:
    cases = require_non_empty_mapping_list(report.get("cases"), f"{path}: eval suite cases")
    return tuple(require_string(case.get("id"), f"{path}: eval suite case id") for case in cases)


def eval_stages(path: Path | None, *, quality_gate: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {
            "alignment": unknown_stage("alignment", "eval comparison report was not provided"),
            "channel_attribution": unknown_stage(
                "channel_attribution",
                "eval comparison report was not provided",
            ),
            "text_asr": unknown_stage("text_asr", "eval comparison report was not provided"),
        }
    label, best = best_eval_comparison_item(path)

    return {
        "alignment": alignment_stage_from_item(path, label, best, quality_gate=quality_gate),
        "channel_attribution": channel_stage(path, quality_gate=quality_gate),
        "text_asr": text_stage_from_item(path, label, best, quality_gate=quality_gate),
    }


def alignment_stage_from_item(
    path: Path,
    label: str,
    best: dict[str, Any],
    *,
    quality_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    timing_ratio = require_number(best.get("timing_edit_segment_ratio"), f"{path}: timing_edit_segment_ratio")
    time_aligned_500ms_ratio = optional_number(
        best.get("time_aligned_500ms_ratio"),
        f"{path}: time_aligned_500ms_ratio",
    )
    gate = readiness_stage_gate(quality_gate, "min_time_aligned_500ms_ratio")
    reasons = []
    if gate:
        min_ratio = gate["min_time_aligned_500ms_ratio"]
        if time_aligned_500ms_ratio is None:
            reasons.append("time-aligned 500ms ratio is unavailable")
        elif time_aligned_500ms_ratio < min_ratio:
            reasons.append(f"time-aligned 500ms ratio {time_aligned_500ms_ratio:.4f} < {min_ratio:.4f}")
    elif timing_ratio > 0.0:
        reasons.append(f"best candidate still needs timing edits: {timing_ratio:.4f}")
    return stage_report(
        "alignment",
        reasons=reasons,
        metrics={
            "report": str(path),
            "best_label": label,
            "timing_edit_segment_ratio": timing_ratio,
            "time_aligned_500ms_ratio": time_aligned_500ms_ratio,
        },
        quality_gate=gate,
    )


def candidate_channel_energy_stage(path: Path, *, quality_gate: dict[str, Any] | None = None) -> dict[str, Any]:
    report = read_json_report(path)
    if report.get("format") != CANDIDATE_CHANNEL_AUDIT_SUITE_FORMAT:
        raise ValueError(f"{path}: candidate channel audit format must be {CANDIDATE_CHANNEL_AUDIT_SUITE_FORMAT}")
    summary = require_mapping(report.get("summary"), f"{path}: candidate channel audit summary")
    energy_labeled_count = require_int(
        summary.get("energy_labeled_count"),
        f"{path}: candidate channel audit energy_labeled_count",
    )
    match_ratio = optional_number(
        summary.get("energy_labeled_match_ratio"),
        f"{path}: candidate channel audit energy_labeled_match_ratio",
    )
    mix_ratio = optional_number(
        summary.get("energy_labeled_mix_ratio"),
        f"{path}: candidate channel audit energy_labeled_mix_ratio",
    )
    gate = readiness_stage_gate(
        quality_gate,
        "min_channel_time_aligned_accuracy",
        "max_channel_time_aligned_mix_ratio",
    )
    reasons = []
    warnings = ["channel attribution is judged by stereo energy proxy, not human-reviewed reference labels"]
    if energy_labeled_count == 0:
        reasons.append("candidate channel energy audit has no energy-labeled speech segments")
    if gate:
        min_accuracy = gate.get("min_channel_time_aligned_accuracy")
        if min_accuracy is not None:
            if match_ratio is None:
                reasons.append("energy-labeled candidate channel match ratio is unavailable")
            elif match_ratio < min_accuracy:
                reasons.append(
                    f"energy-labeled candidate channel match ratio {match_ratio:.4f} < {min_accuracy:.4f}"
                )
        max_mix_ratio = gate.get("max_channel_time_aligned_mix_ratio")
        if max_mix_ratio is not None:
            if mix_ratio is None:
                reasons.append("energy-labeled candidate MIX ratio is unavailable")
            elif mix_ratio > max_mix_ratio:
                reasons.append(f"energy-labeled candidate MIX ratio {mix_ratio:.4f} > {max_mix_ratio:.4f}")
    elif match_ratio is None or match_ratio < 1.0 or (mix_ratio is not None and mix_ratio > 0.0):
        reasons.append("candidate channel labels do not fully match stereo energy proxy")
    return stage_report(
        "channel_attribution",
        reasons=reasons,
        warnings=warnings,
        metrics={
            "report": str(path),
            "source": "candidate_channel_energy_audit",
            "speech_segment_count": optional_int(
                summary.get("speech_segment_count"),
                f"{path}: candidate channel audit speech_segment_count",
            ),
            "energy_labeled_count": energy_labeled_count,
            "energy_uncertain_count": require_int(
                summary.get("energy_uncertain_count"),
                f"{path}: candidate channel audit energy_uncertain_count",
            ),
            "match_count": require_int(summary.get("match_count"), f"{path}: candidate channel audit match_count"),
            "missed_attribution_count": require_int(
                summary.get("missed_attribution_count"),
                f"{path}: candidate channel audit missed_attribution_count",
            ),
            "wrong_side_count": require_int(
                summary.get("wrong_side_count"),
                f"{path}: candidate channel audit wrong_side_count",
            ),
            "over_attribution_count": require_int(
                summary.get("over_attribution_count"),
                f"{path}: candidate channel audit over_attribution_count",
            ),
            "energy_labeled_match_ratio": match_ratio,
            "energy_labeled_mix_ratio": mix_ratio,
            "energy_labeled_wrong_side_ratio": optional_number(
                summary.get("energy_labeled_wrong_side_ratio"),
                f"{path}: candidate channel audit energy_labeled_wrong_side_ratio",
            ),
            "over_attribution_ratio": optional_number(
                summary.get("over_attribution_ratio"),
                f"{path}: candidate channel audit over_attribution_ratio",
            ),
        },
        quality_gate=gate,
    )


def channel_stage(path: Path, *, quality_gate: dict[str, Any] | None = None) -> dict[str, Any]:
    label, best = best_eval_comparison_item(path)
    channel_ratio = require_number(best.get("channel_edit_segment_ratio"), f"{path}: channel_edit_segment_ratio")
    channel_accuracy = optional_number(
        best.get("channel_time_aligned_accuracy"),
        f"{path}: channel_time_aligned_accuracy",
    )
    mix_ratio = require_number(
        best.get("channel_time_aligned_mix_ratio"),
        f"{path}: channel_time_aligned_mix_ratio",
    )
    gate = readiness_stage_gate(
        quality_gate,
        "min_channel_time_aligned_accuracy",
        "max_channel_time_aligned_mix_ratio",
    )
    channel_reasons = []
    if gate:
        min_accuracy = gate.get("min_channel_time_aligned_accuracy")
        if min_accuracy is not None:
            if channel_accuracy is None:
                channel_reasons.append("channel time-aligned accuracy is unavailable")
            elif channel_accuracy < min_accuracy:
                channel_reasons.append(
                    f"channel time-aligned accuracy {channel_accuracy:.4f} < {min_accuracy:.4f}"
                )
        max_mix_ratio = gate.get("max_channel_time_aligned_mix_ratio")
        if max_mix_ratio is not None and mix_ratio > max_mix_ratio:
            channel_reasons.append(f"channel time-aligned MIX ratio {mix_ratio:.4f} > {max_mix_ratio:.4f}")
    elif channel_ratio > 0.0:
        channel_reasons.append(f"best candidate still needs channel edits: {channel_ratio:.4f}")
    return stage_report(
        "channel_attribution",
        reasons=channel_reasons,
        metrics={
            "report": str(path),
            "best_label": label,
            "channel_edit_segment_ratio": channel_ratio,
            "channel_time_aligned_accuracy": channel_accuracy,
            "channel_time_aligned_mix_ratio": mix_ratio,
        },
        quality_gate=gate,
    )


def text_stage_from_item(
    path: Path,
    label: str,
    best: dict[str, Any],
    *,
    quality_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    text_ratio = require_number(best.get("text_edit_segment_ratio"), f"{path}: text_edit_segment_ratio")
    edit_ratio = require_number(best.get("segments_needing_edit_ratio"), f"{path}: segments_needing_edit_ratio")
    practical_cer = require_number(best.get("practical_cer"), f"{path}: practical_cer")
    candidate_review_ratio = optional_number(best.get("candidate_review_ratio"), f"{path}: candidate_review_ratio")
    gate = readiness_stage_gate(
        quality_gate,
        "max_practical_cer",
        "max_segments_needing_edit_ratio",
        "max_candidate_review_ratio",
    )
    reasons = []
    if gate:
        max_practical_cer = gate.get("max_practical_cer")
        if max_practical_cer is not None and practical_cer > max_practical_cer:
            reasons.append(f"practical CER {practical_cer:.4f} > {max_practical_cer:.4f}")
        max_edit_ratio = gate.get("max_segments_needing_edit_ratio")
        if max_edit_ratio is not None and edit_ratio > max_edit_ratio:
            reasons.append(f"segments needing edit ratio {edit_ratio:.4f} > {max_edit_ratio:.4f}")
        max_candidate_review_ratio = gate.get("max_candidate_review_ratio")
        if max_candidate_review_ratio is not None:
            if candidate_review_ratio is None:
                reasons.append("candidate review ratio is unavailable")
            elif candidate_review_ratio > max_candidate_review_ratio:
                reasons.append(
                    f"candidate review ratio {candidate_review_ratio:.4f} > {max_candidate_review_ratio:.4f}"
                )
    else:
        if text_ratio > 0.0:
            reasons.append(f"best candidate still needs text edits: {text_ratio:.4f}")
        if edit_ratio > 0.0:
            reasons.append(f"best candidate still has segments needing edit: {edit_ratio:.4f}")
    return stage_report(
        "text_asr",
        reasons=reasons,
        metrics={
            "report": str(path),
            "best_label": label,
            "text_edit_segment_ratio": text_ratio,
            "segments_needing_edit_ratio": edit_ratio,
            "practical_cer": practical_cer,
            "candidate_review_ratio": candidate_review_ratio,
            "dominant_review_effort_reason": optional_string(best.get("dominant_review_effort_reason")),
            "dominant_review_effort_ratio": optional_number(
                best.get("dominant_review_effort_ratio"),
                f"{path}: dominant_review_effort_ratio",
            ),
            "reference_type": optional_string(best.get("reference_type")),
        },
        quality_gate=gate,
    )


def best_eval_comparison_item(path: Path) -> tuple[str, dict[str, Any]]:
    report = read_json_report(path)
    if report.get("format") != EVAL_COMPARISON_FORMAT:
        raise ValueError(f"{path}: eval report format must be {EVAL_COMPARISON_FORMAT}")
    items = require_non_empty_mapping_list(report.get("items"), f"{path}: eval comparison items")
    best = items[0]
    label = require_string(best.get("label"), f"{path}: eval best label")
    return label, best


def readiness_reference_type(*paths: Path | None) -> str | None:
    for path in paths:
        if path is None:
            continue
        _label, best = best_eval_comparison_item(path)
        return optional_string(best.get("reference_type"))
    return None


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
    quality_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "stage": stage,
        "status": "fail" if reasons else "pass",
        "reasons": reasons,
        "warnings": [] if warnings is None else warnings,
        "metrics": metrics,
    }
    if quality_gate:
        report["quality_gate"] = quality_gate
    return report


def readiness_stage_gate(quality_gate: dict[str, Any] | None, *keys: str) -> dict[str, Any]:
    if not quality_gate:
        return {}
    gate = {key: quality_gate[key] for key in keys if quality_gate.get(key) is not None}
    return gate


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
