from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import analyze_wav
from custom_asmr_srt_stack.models import MasterDocument, require_int, require_mapping

DEFAULT_VAD_TIMEOUT_SECONDS = 300.0
VAD_COVERAGE_FORMAT = "custom-asmr-vad-coverage-v1"
VAD_COVERAGE_SUITE_FORMAT = "custom-asmr-vad-coverage-suite-v1"
VAD_COVERAGE_COMPARISON_FORMAT = "custom-asmr-vad-coverage-comparison-v1"


def run_vad_command(audio_bytes: bytes, *, command: list[str]) -> tuple[dict[str, int], ...]:
    if not command:
        raise ValueError("VAD command must not be empty")
    audio_info = analyze_wav(audio_bytes)
    timeout_seconds = vad_timeout_seconds()
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file = Path(tmpdir) / "audio.wav"
        audio_file.write_bytes(audio_bytes)
        request = {
            "audio_file": str(audio_file),
            "audio_info": audio_info.to_json(),
        }
        try:
            result = subprocess.run(
                command,
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
                env=vad_subprocess_env(tmpdir),
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"VAD command timed out after {timeout_seconds:g}s") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown VAD error"
        raise ValueError(f"VAD command failed: {detail}")
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"VAD command returned invalid JSON: {error}") from error
    return parse_vad_intervals(output, duration_ms=audio_info.duration_ms)


def vad_timeout_seconds() -> float:
    raw_timeout = os.environ.get("CASRT_VAD_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return DEFAULT_VAD_TIMEOUT_SECONDS
    try:
        timeout = float(raw_timeout)
    except ValueError as error:
        raise ValueError("CASRT_VAD_TIMEOUT_SECONDS must be a number") from error
    if timeout <= 0:
        raise ValueError("CASRT_VAD_TIMEOUT_SECONDS must be positive")
    return timeout


def vad_subprocess_env(tmpdir: str) -> dict[str, str]:
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": tmpdir,
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    path = os.environ.get("PATH")
    if path is not None:
        env["PATH"] = path
    return env


def parse_vad_intervals(value: Any, *, duration_ms: int) -> tuple[dict[str, int], ...]:
    if duration_ms < 0:
        raise ValueError("VAD duration_ms must be non-negative")
    data = require_mapping(value, "VAD output")
    raw_intervals = data.get("intervals")
    if not isinstance(raw_intervals, list):
        raise ValueError("VAD output intervals must be an array")

    intervals = []
    previous_end_ms = 0
    for index, raw_interval in enumerate(raw_intervals):
        interval = require_mapping(raw_interval, "VAD interval")
        start_ms = require_int(interval.get("start_ms"), "VAD interval.start_ms")
        end_ms = require_int(interval.get("end_ms"), "VAD interval.end_ms")
        if start_ms < 0:
            raise ValueError("VAD interval.start_ms must be non-negative")
        if end_ms <= start_ms:
            raise ValueError("VAD interval.end_ms must be greater than start_ms")
        if end_ms > duration_ms:
            raise ValueError("VAD interval.end_ms must not exceed audio duration")
        if index > 0 and start_ms < previous_end_ms:
            raise ValueError("VAD intervals must be sorted and non-overlapping")
        intervals.append({"index": len(intervals), "start_ms": start_ms, "end_ms": end_ms})
        previous_end_ms = end_ms
    return tuple(intervals)


def vad_coverage_report(
    *,
    reference: MasterDocument,
    intervals: tuple[dict[str, int], ...] | list[dict[str, int]],
    audio_duration_ms: int,
    source: str,
) -> dict[str, Any]:
    if audio_duration_ms < 0:
        raise ValueError("VAD coverage audio_duration_ms must be non-negative")
    reference_intervals = merged_intervals(
        [
            {"start_ms": segment.start_ms, "end_ms": segment.end_ms}
            for segment in reference.segments
            if segment.kind == "speech" and segment.text
        ],
        duration_ms=audio_duration_ms,
    )
    detected_intervals = merged_intervals(intervals, duration_ms=audio_duration_ms)
    reference_duration_ms = interval_duration_ms(reference_intervals)
    detected_duration_ms = interval_duration_ms(detected_intervals)
    detected_duration_stats = interval_duration_stats(detected_intervals)
    overlap_duration_ms = interval_overlap_duration_ms(reference_intervals, detected_intervals)
    missed_reference_duration_ms = max(0, reference_duration_ms - overlap_duration_ms)
    extra_detected_duration_ms = max(0, detected_duration_ms - overlap_duration_ms)
    missed_reference_intervals = interval_difference(reference_intervals, detected_intervals)
    extra_detected_intervals = interval_difference(detected_intervals, reference_intervals)
    return {
        "format": VAD_COVERAGE_FORMAT,
        "source": source,
        "audio_duration_ms": audio_duration_ms,
        "reference_segment_count": sum(1 for segment in reference.segments if segment.kind == "speech" and segment.text),
        "reference_interval_count": len(reference_intervals),
        "detected_interval_count": len(detected_intervals),
        "detected_max_interval_ms": detected_duration_stats["max_interval_ms"],
        "detected_mean_interval_ms": detected_duration_stats["mean_interval_ms"],
        "reference_speech_duration_ms": reference_duration_ms,
        "detected_speech_duration_ms": detected_duration_ms,
        "overlap_duration_ms": overlap_duration_ms,
        "missed_reference_duration_ms": missed_reference_duration_ms,
        "extra_detected_duration_ms": extra_detected_duration_ms,
        "missed_reference_intervals": missed_reference_intervals,
        "extra_detected_intervals": extra_detected_intervals,
        "reference_recall": None if reference_duration_ms == 0 else overlap_duration_ms / reference_duration_ms,
        "detected_precision": None if detected_duration_ms == 0 else overlap_duration_ms / detected_duration_ms,
    }


def aggregate_vad_coverage_reports(reports: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
    totals = {
        "case_count": 0,
        "audio_duration_ms": 0,
        "reference_segment_count": 0,
        "reference_interval_count": 0,
        "detected_interval_count": 0,
        "detected_max_interval_ms": 0,
        "reference_speech_duration_ms": 0,
        "detected_speech_duration_ms": 0,
        "overlap_duration_ms": 0,
        "missed_reference_duration_ms": 0,
        "extra_detected_duration_ms": 0,
    }
    for index, raw_report in enumerate(reports):
        report = require_mapping(raw_report, "VAD coverage report")
        if report.get("format") != VAD_COVERAGE_FORMAT:
            raise ValueError(f"VAD coverage report {index} format must be {VAD_COVERAGE_FORMAT}")
        totals["case_count"] += 1
        for key in (
            "audio_duration_ms",
            "reference_segment_count",
            "reference_interval_count",
            "detected_interval_count",
            "reference_speech_duration_ms",
            "detected_speech_duration_ms",
            "overlap_duration_ms",
            "missed_reference_duration_ms",
            "extra_detected_duration_ms",
        ):
            totals[key] += require_int(report.get(key), f"VAD coverage report {index}.{key}")
        detected_max_interval_ms = optional_number(
            report.get("detected_max_interval_ms"),
            f"VAD coverage report {index}.detected_max_interval_ms",
        )
        if detected_max_interval_ms is not None:
            totals["detected_max_interval_ms"] = max(totals["detected_max_interval_ms"], int(detected_max_interval_ms))
    reference_duration_ms = totals["reference_speech_duration_ms"]
    detected_duration_ms = totals["detected_speech_duration_ms"]
    detected_interval_count = totals["detected_interval_count"]
    overlap_duration_ms = totals["overlap_duration_ms"]
    return {
        **totals,
        "detected_max_interval_ms": None if detected_interval_count == 0 else totals["detected_max_interval_ms"],
        "detected_mean_interval_ms": None if detected_interval_count == 0 else detected_duration_ms / detected_interval_count,
        "reference_recall": None if reference_duration_ms == 0 else overlap_duration_ms / reference_duration_ms,
        "detected_precision": None if detected_duration_ms == 0 else overlap_duration_ms / detected_duration_ms,
    }


def compare_vad_coverage_reports(report_files: list[Path]) -> dict[str, Any]:
    if not report_files:
        raise ValueError("VAD coverage comparison requires at least one report")
    items = [vad_coverage_comparison_item(path, json.loads(path.read_text(encoding="utf-8"))) for path in report_files]
    ranked_items = sorted(
        items,
        key=lambda item: (
            item["missed_reference_duration_ms"],
            item["extra_detected_duration_ms"],
            -(item["reference_recall"] if item["reference_recall"] is not None else -1.0),
            -(item["detected_precision"] if item["detected_precision"] is not None else -1.0),
            item["label"],
        ),
    )
    return {
        "format": VAD_COVERAGE_COMPARISON_FORMAT,
        "report_count": len(ranked_items),
        "ranked_by": [
            "missed_reference_duration_ms",
            "extra_detected_duration_ms",
            "reference_recall desc",
            "detected_precision desc",
        ],
        "items": ranked_items,
    }


def vad_coverage_comparison_item(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_format = report.get("format")
    if report_format == VAD_COVERAGE_FORMAT:
        metrics = report
        case_count = 1
    elif report_format == VAD_COVERAGE_SUITE_FORMAT:
        metrics = require_mapping(report.get("summary"), "VAD coverage suite summary")
        case_count = require_int(report.get("case_count"), "VAD coverage suite case_count")
    else:
        raise ValueError(f"{path}: unsupported VAD coverage report format {report_format!r}")
    source = report.get("source")
    if not isinstance(source, str) or not source:
        source = "unspecified"
    return {
        "label": path.stem,
        "report": str(path),
        "report_format": report_format,
        "source": source,
        "case_count": case_count,
        "audio_duration_ms": require_int(metrics.get("audio_duration_ms"), "VAD coverage audio_duration_ms"),
        "reference_segment_count": require_int(
            metrics.get("reference_segment_count"),
            "VAD coverage reference_segment_count",
        ),
        "reference_interval_count": require_int(
            metrics.get("reference_interval_count"),
            "VAD coverage reference_interval_count",
        ),
        "detected_interval_count": require_int(
            metrics.get("detected_interval_count"),
            "VAD coverage detected_interval_count",
        ),
        "detected_max_interval_ms": optional_number(
            metrics.get("detected_max_interval_ms"),
            "VAD coverage detected_max_interval_ms",
        ),
        "detected_mean_interval_ms": optional_number(
            metrics.get("detected_mean_interval_ms"),
            "VAD coverage detected_mean_interval_ms",
        ),
        "reference_speech_duration_ms": require_int(
            metrics.get("reference_speech_duration_ms"),
            "VAD coverage reference_speech_duration_ms",
        ),
        "detected_speech_duration_ms": require_int(
            metrics.get("detected_speech_duration_ms"),
            "VAD coverage detected_speech_duration_ms",
        ),
        "overlap_duration_ms": require_int(metrics.get("overlap_duration_ms"), "VAD coverage overlap_duration_ms"),
        "missed_reference_duration_ms": require_int(
            metrics.get("missed_reference_duration_ms"),
            "VAD coverage missed_reference_duration_ms",
        ),
        "extra_detected_duration_ms": require_int(
            metrics.get("extra_detected_duration_ms"),
            "VAD coverage extra_detected_duration_ms",
        ),
        "reference_recall": optional_number(metrics.get("reference_recall"), "VAD coverage reference_recall"),
        "detected_precision": optional_number(metrics.get("detected_precision"), "VAD coverage detected_precision"),
        "missed_reference_interval_count": vad_coverage_interval_count(report, "missed_reference_intervals"),
        "extra_detected_interval_count": vad_coverage_interval_count(report, "extra_detected_intervals"),
    }


def optional_number(value: Any, label: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a number or null")
    return value


def vad_coverage_interval_count(report: dict[str, Any], key: str) -> int | None:
    if report.get("format") == VAD_COVERAGE_FORMAT:
        intervals = report.get(key)
        if intervals is None:
            return None
        if not isinstance(intervals, list):
            raise ValueError(f"VAD coverage {key} must be an array")
        return len(intervals)
    raw_cases = report.get("cases")
    if raw_cases is None:
        return None
    if not isinstance(raw_cases, list):
        raise ValueError("VAD coverage suite cases must be an array")
    total = 0
    for index, raw_case in enumerate(raw_cases):
        case = require_mapping(raw_case, "VAD coverage suite case")
        case_report = require_mapping(case.get("report"), "VAD coverage suite case report")
        intervals = case_report.get(key)
        if intervals is None:
            return None
        if not isinstance(intervals, list):
            raise ValueError(f"VAD coverage suite case {index} {key} must be an array")
        total += len(intervals)
    return total


def merged_intervals(intervals: list[dict[str, int]] | tuple[dict[str, int], ...], *, duration_ms: int) -> tuple[dict[str, int], ...]:
    if duration_ms < 0:
        raise ValueError("VAD coverage duration_ms must be non-negative")
    normalized = []
    for raw_interval in intervals:
        interval = require_mapping(raw_interval, "VAD coverage interval")
        start_ms = require_int(interval.get("start_ms"), "VAD coverage interval.start_ms")
        end_ms = require_int(interval.get("end_ms"), "VAD coverage interval.end_ms")
        if start_ms < 0:
            raise ValueError("VAD coverage interval.start_ms must be non-negative")
        if end_ms <= start_ms:
            raise ValueError("VAD coverage interval.end_ms must be greater than start_ms")
        if end_ms > duration_ms:
            raise ValueError("VAD coverage interval.end_ms must not exceed audio duration")
        normalized.append({"start_ms": start_ms, "end_ms": end_ms})
    normalized.sort(key=lambda item: (item["start_ms"], item["end_ms"]))
    if not normalized:
        return ()
    merged: list[dict[str, int]] = []
    for interval in normalized:
        if not merged or interval["start_ms"] > merged[-1]["end_ms"]:
            merged.append({"index": len(merged), "start_ms": interval["start_ms"], "end_ms": interval["end_ms"]})
            continue
        merged[-1]["end_ms"] = max(merged[-1]["end_ms"], interval["end_ms"])
    return tuple({"index": index, "start_ms": item["start_ms"], "end_ms": item["end_ms"]} for index, item in enumerate(merged))


def interval_duration_ms(intervals: tuple[dict[str, int], ...]) -> int:
    return sum(interval["end_ms"] - interval["start_ms"] for interval in intervals)


def interval_duration_stats(intervals: tuple[dict[str, int], ...]) -> dict[str, float | int | None]:
    if not intervals:
        return {"max_interval_ms": None, "mean_interval_ms": None}
    duration_ms = interval_duration_ms(intervals)
    return {
        "max_interval_ms": max(interval["end_ms"] - interval["start_ms"] for interval in intervals),
        "mean_interval_ms": duration_ms / len(intervals),
    }


def interval_difference(
    primary_intervals: tuple[dict[str, int], ...],
    subtract_intervals: tuple[dict[str, int], ...],
) -> tuple[dict[str, int], ...]:
    fragments = []
    subtract_index = 0
    for primary in primary_intervals:
        primary_start_ms = primary["start_ms"]
        primary_end_ms = primary["end_ms"]
        cursor_ms = primary_start_ms
        while subtract_index < len(subtract_intervals) and subtract_intervals[subtract_index]["end_ms"] <= cursor_ms:
            subtract_index += 1
        blocker_index = subtract_index
        while blocker_index < len(subtract_intervals) and subtract_intervals[blocker_index]["start_ms"] < primary_end_ms:
            blocker = subtract_intervals[blocker_index]
            if blocker["start_ms"] > cursor_ms:
                fragments.append(
                    {
                        "start_ms": cursor_ms,
                        "end_ms": min(blocker["start_ms"], primary_end_ms),
                    }
                )
            cursor_ms = max(cursor_ms, blocker["end_ms"])
            if cursor_ms >= primary_end_ms:
                break
            blocker_index += 1
        if cursor_ms < primary_end_ms:
            fragments.append({"start_ms": cursor_ms, "end_ms": primary_end_ms})
    return tuple(
        {
            "index": index,
            "start_ms": fragment["start_ms"],
            "end_ms": fragment["end_ms"],
            "duration_ms": fragment["end_ms"] - fragment["start_ms"],
        }
        for index, fragment in enumerate(fragments)
        if fragment["end_ms"] > fragment["start_ms"]
    )


def interval_overlap_duration_ms(
    left_intervals: tuple[dict[str, int], ...],
    right_intervals: tuple[dict[str, int], ...],
) -> int:
    left_index = 0
    right_index = 0
    overlap = 0
    while left_index < len(left_intervals) and right_index < len(right_intervals):
        left = left_intervals[left_index]
        right = right_intervals[right_index]
        overlap += max(0, min(left["end_ms"], right["end_ms"]) - max(left["start_ms"], right["start_ms"]))
        if left["end_ms"] <= right["end_ms"]:
            left_index += 1
        else:
            right_index += 1
    return overlap
