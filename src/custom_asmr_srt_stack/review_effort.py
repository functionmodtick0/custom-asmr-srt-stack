from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.evaluation import REVIEW_EFFORT_FORMAT


def merge_review_effort_reports(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one review-effort report is required")

    reports = [load_review_effort_report(path) for path in paths]
    source_case_index = merged_source_case_index(paths, reports)
    source_reports = [str(path) for path in paths]
    items_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}

    for path, report in zip(paths, reports):
        items = require_review_effort_items(path, report)
        for index, item in enumerate(items):
            normalized = normalized_review_item(path, index, item)
            key = review_item_merge_key(normalized)
            existing = items_by_key.get(key)
            if existing is None:
                merged = dict(normalized)
                merged["source_reports"] = [str(path)]
                merged["merged_input_count"] = 1
                items_by_key[key] = merged
            else:
                merge_review_item(existing, normalized, source_report=str(path))

    merged_items = sorted(items_by_key.values(), key=review_item_sort_key)
    for rank, item in enumerate(merged_items, start=1):
        item["priority_rank"] = rank
    case_summaries = review_case_summaries(merged_items)

    result: dict[str, Any] = {
        "format": REVIEW_EFFORT_FORMAT,
        "sort": "priority_score_desc",
        "source_reports": source_reports,
        "input_report_count": len(paths),
        "input_item_count": sum(len(require_review_effort_items(path, report)) for path, report in zip(paths, reports)),
        "item_count": len(merged_items),
        "reason_counts": review_reason_counts(merged_items),
        "case_count": len(case_summaries),
        "next_case_id": case_summaries[0]["case_id"] if case_summaries else None,
        "case_summaries": case_summaries,
        "items": merged_items,
    }
    if source_case_index is not None:
        result["source_case_index"] = source_case_index
    return result


def load_review_effort_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path}: review-effort report must be a JSON object")
    if report.get("format") != REVIEW_EFFORT_FORMAT:
        raise ValueError(f"{path}: review-effort report format must be {REVIEW_EFFORT_FORMAT}")
    return report


def require_review_effort_items(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    items = report.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path}: review-effort items must be an array")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: review-effort item {index} must be an object")
    return items


def merged_source_case_index(paths: list[Path], reports: list[dict[str, Any]]) -> str | None:
    values = []
    for path, report in zip(paths, reports):
        value = report.get("source_case_index")
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}: source_case_index must be a non-empty string")
        values.append(value)
    distinct = sorted(set(values))
    if len(distinct) > 1:
        raise ValueError("review-effort reports have conflicting source_case_index values")
    return distinct[0] if distinct else None


def normalized_review_item(path: Path, index: int, item: dict[str, Any]) -> dict[str, Any]:
    reasons = item.get("reasons")
    if not isinstance(reasons, list) or not reasons or not all(isinstance(reason, str) for reason in reasons):
        raise ValueError(f"{path}: review-effort item {index} reasons must be a non-empty array of strings")
    result = dict(item)
    result["reasons"] = list(dict.fromkeys(reasons))
    priority_score = result.get("priority_score")
    if priority_score is None:
        result["priority_score"] = 0.0
    elif isinstance(priority_score, bool) or not isinstance(priority_score, (int, float)):
        raise ValueError(f"{path}: review-effort item {index} priority_score must be a number")
    else:
        result["priority_score"] = float(priority_score)
    return result


def review_item_merge_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("case_id"),
        item.get("reference_id"),
        item.get("candidate_id"),
        item.get("start_ms"),
        item.get("end_ms"),
    )


def merge_review_item(existing: dict[str, Any], incoming: dict[str, Any], *, source_report: str) -> None:
    existing["reasons"] = list(dict.fromkeys([*existing["reasons"], *incoming["reasons"]]))
    existing["priority_score"] = max(float(existing.get("priority_score") or 0.0), float(incoming["priority_score"]))
    existing["merged_input_count"] = int(existing.get("merged_input_count") or 1) + 1
    source_reports = existing.setdefault("source_reports", [])
    if isinstance(source_reports, list) and source_report not in source_reports:
        source_reports.append(source_report)

    for key, value in incoming.items():
        if key in {"reasons", "priority_score", "priority_rank", "source_reports", "merged_input_count"}:
            continue
        if key not in existing or existing[key] in (None, ""):
            existing[key] = value


def review_item_sort_key(item: dict[str, Any]) -> tuple[float, str, int, str, str]:
    start_ms = item.get("start_ms")
    return (
        -float(item.get("priority_score") or 0.0),
        str(item.get("case_id") or ""),
        start_ms if isinstance(start_ms, int) else 0,
        str(item.get("reference_id") or ""),
        str(item.get("candidate_id") or ""),
    )


def review_reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for reason in item["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def review_case_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for item in items:
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        summary = summaries.setdefault(
            case_id,
            {
                "case_id": case_id,
                "item_count": 0,
                "reason_counts": {},
                "review_duration_ms": 0,
                "first_start_ms": None,
                "last_end_ms": None,
                "top_priority_score": None,
                "top_priority_rank": None,
            },
        )
        summary["item_count"] += 1
        reasons = item.get("reasons")
        if isinstance(reasons, list):
            for reason in reasons:
                if isinstance(reason, str):
                    reason_counts = summary["reason_counts"]
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        start_ms = item.get("start_ms")
        end_ms = item.get("end_ms")
        if isinstance(start_ms, int):
            current_first = summary["first_start_ms"]
            summary["first_start_ms"] = start_ms if current_first is None else min(current_first, start_ms)
        if isinstance(end_ms, int):
            current_last = summary["last_end_ms"]
            summary["last_end_ms"] = end_ms if current_last is None else max(current_last, end_ms)
        if isinstance(start_ms, int) and isinstance(end_ms, int) and end_ms > start_ms:
            summary["review_duration_ms"] += end_ms - start_ms
        priority_score = item.get("priority_score")
        if isinstance(priority_score, (int, float)) and not isinstance(priority_score, bool):
            current_score = summary["top_priority_score"]
            if current_score is None or float(priority_score) > current_score:
                summary["top_priority_score"] = float(priority_score)
        priority_rank = item.get("priority_rank")
        if isinstance(priority_rank, int) and not isinstance(priority_rank, bool):
            current_rank = summary["top_priority_rank"]
            summary["top_priority_rank"] = priority_rank if current_rank is None else min(current_rank, priority_rank)

    result = []
    for summary in summaries.values():
        summary["reason_counts"] = dict(sorted(summary["reason_counts"].items()))
        result.append(summary)
    return sorted(
        result,
        key=lambda summary: (
            summary["top_priority_rank"] if summary["top_priority_rank"] is not None else 10**9,
            summary["case_id"],
        ),
    )
