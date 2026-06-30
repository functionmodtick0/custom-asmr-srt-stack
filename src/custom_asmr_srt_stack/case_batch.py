from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import normalize_audio_to_wav, slice_wav
from custom_asmr_srt_stack.case_slicing import slice_master_document
from custom_asmr_srt_stack.evaluation import EVAL_MANIFEST_FORMAT, load_transcript_document
from custom_asmr_srt_stack.review_pack import REVIEW_AUDIO_MAP_FORMAT

CASE_SLICE_PLAN_FORMAT = "custom-asmr-case-slice-plan-v1"
REVIEW_CASE_SET_FORMAT = "custom-asmr-review-case-set-v1"
REVIEW_CASE_STATUS_FORMAT = "custom-asmr-review-case-status-v1"


def prepare_review_cases(
    plan_file: Path,
    *,
    output_dir: Path,
    source_language: str = "ja",
) -> dict[str, Any]:
    plan = load_case_slice_plan(plan_file)
    cases = validate_case_slice_plan(plan)
    candidate_flags = [case.get("candidate") is not None for case in cases]
    if any(candidate_flags) and not all(candidate_flags):
        raise ValueError("case slice plan cannot mix candidate and non-candidate cases")

    base_dir = plan_file.parent
    source_paths = resolve_source_paths(cases, base_dir=base_dir, include_candidates=all(candidate_flags))
    for path in source_paths.values():
        if not path.is_file():
            raise ValueError(f"case slice plan source file does not exist: {path}")

    prepare_output_dir(output_dir)
    audio_dir = output_dir / "audio"
    reference_dir = output_dir / "references"
    candidate_dir = output_dir / "candidates"
    audio_dir.mkdir()
    reference_dir.mkdir()
    if all(candidate_flags):
        candidate_dir.mkdir()

    reference_type = optional_plan_string(plan, "reference_type") or "unspecified"
    reference_notes = optional_plan_string(plan, "reference_notes")
    index_items: list[dict[str, Any]] = []
    audio_map_items: list[dict[str, str]] = []
    eval_cases: list[dict[str, str]] = []

    for case in cases:
        case_id = case["id"]
        audio_path = source_paths[f"{case_id}:audio"]
        reference_path = source_paths[f"{case_id}:reference"]
        case_stem = case_file_stem(case_id)
        audio_relative = Path("audio") / f"{case_stem}.wav"
        reference_relative = Path("references") / f"{case_stem}.master.json"

        normalized_audio = normalize_audio_to_wav(
            audio_path.read_bytes(),
            file_name=audio_path.name,
            mime_type=mimetypes.guess_type(audio_path.name)[0],
        )
        sliced_audio = slice_wav(normalized_audio, start_ms=case["start_ms"], end_ms=case["end_ms"])
        (output_dir / audio_relative).write_bytes(sliced_audio)

        reference_master = load_transcript_document(reference_path, source_language=source_language)
        sliced_reference = slice_master_document(reference_master, start_ms=case["start_ms"], end_ms=case["end_ms"])
        (output_dir / reference_relative).write_text(
            json.dumps(sliced_reference.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        item = {
            "id": case_id,
            "audio": str(audio_relative),
            "reference": str(reference_relative),
            "source_audio": case["audio"],
            "source_reference": case["reference"],
            "start_ms": case["start_ms"],
            "end_ms": case["end_ms"],
            "duration_ms": case["end_ms"] - case["start_ms"],
            "segments": len(sliced_reference.segments),
            "review_count": sum(1 for segment in sliced_reference.segments if segment.needs_review),
            "reference_type": reference_type,
        }
        if reference_notes is not None:
            item["reference_notes"] = reference_notes

        audio_map_items.append({"case_id": case_id, "audio": str(audio_relative)})

        if all(candidate_flags):
            candidate_path = source_paths[f"{case_id}:candidate"]
            candidate_relative = Path("candidates") / f"{case_stem}.master.json"
            candidate_master = load_transcript_document(candidate_path, source_language=source_language)
            sliced_candidate = slice_master_document(candidate_master, start_ms=case["start_ms"], end_ms=case["end_ms"])
            (output_dir / candidate_relative).write_text(
                json.dumps(sliced_candidate.to_json(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            candidate_id = case.get("candidate_id") or Path(case["candidate"]).stem
            item["candidate"] = str(candidate_relative)
            item["source_candidate"] = case["candidate"]
            item["candidate_id"] = candidate_id
            eval_case = {
                "id": case_id,
                "reference": str(reference_relative),
                "candidate": str(candidate_relative),
                "candidate_id": candidate_id,
            }
            eval_cases.append(eval_case)

        index_items.append(item)

    audio_map = {
        "format": REVIEW_AUDIO_MAP_FORMAT,
        "items": audio_map_items,
    }
    audio_map_file = output_dir / "audio-map.json"
    audio_map_file.write_text(json.dumps(audio_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    case_index = {
        "format": REVIEW_CASE_SET_FORMAT,
        "source_plan": str(plan_file),
        "reference_type": reference_type,
        "case_count": len(index_items),
        "items": index_items,
    }
    if reference_notes is not None:
        case_index["reference_notes"] = reference_notes
    case_index_file = output_dir / "case-index.json"
    case_index_file.write_text(json.dumps(case_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "format": REVIEW_CASE_SET_FORMAT,
        "output": str(output_dir),
        "case_count": len(index_items),
        "review_count": sum(item["review_count"] for item in index_items),
        "audio_map": str(audio_map_file),
        "case_index": str(case_index_file),
    }
    if eval_cases:
        eval_manifest = {
            "format": EVAL_MANIFEST_FORMAT,
            "reference_type": reference_type,
            "cases": eval_cases,
        }
        if reference_notes is not None:
            eval_manifest["reference_notes"] = reference_notes
        eval_manifest_file = output_dir / "eval-manifest.json"
        eval_manifest_file.write_text(
            json.dumps(eval_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["eval_manifest"] = str(eval_manifest_file)
    return result


def load_case_slice_plan(plan_file: Path) -> dict[str, Any]:
    data = json.loads(plan_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("case slice plan must be a JSON object")
    return data


def review_case_status(case_index_file: Path, *, source_language: str = "ja") -> dict[str, Any]:
    case_index = load_review_case_index(case_index_file)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review case index items must be an array")

    base_dir = case_index_file.parent
    default_reference_type = case_index.get("reference_type")
    if default_reference_type is not None and not isinstance(default_reference_type, str):
        raise ValueError("review case index reference_type must be a string")

    items: list[dict[str, Any]] = []
    reference_type_counts: dict[str, int] = {}
    missing_file_count = 0
    reference_review_count = 0
    candidate_case_count = 0
    cases_needing_review: list[str] = []
    cases_with_issues: list[str] = []

    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        item = review_case_status_item(
            raw_item,
            index=index,
            base_dir=base_dir,
            default_reference_type=default_reference_type,
            source_language=source_language,
        )
        items.append(item)
        reference_type_counts[item["reference_type"]] = reference_type_counts.get(item["reference_type"], 0) + 1
        missing_file_count += item["missing_file_count"]
        reference_review_count += item["reference_review_count"]
        if item.get("candidate") is not None:
            candidate_case_count += 1
        if item["reference_review_count"] > 0:
            cases_needing_review.append(item["id"])
        if item["issues"]:
            cases_with_issues.append(item["id"])

    return {
        "format": REVIEW_CASE_STATUS_FORMAT,
        "case_index": str(case_index_file),
        "case_count": len(items),
        "candidate_case_count": candidate_case_count,
        "reference_type_counts": reference_type_counts,
        "missing_file_count": missing_file_count,
        "cases_with_issues": cases_with_issues,
        "case_issue_count": len(cases_with_issues),
        "reference_review_count": reference_review_count,
        "cases_needing_review": cases_needing_review,
        "ok": missing_file_count == 0 and not cases_with_issues,
        "items": items,
    }


def load_review_case_index(case_index_file: Path) -> dict[str, Any]:
    data = json.loads(case_index_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("review case index must be a JSON object")
    if data.get("format") != REVIEW_CASE_SET_FORMAT:
        raise ValueError(f"review case index format must be {REVIEW_CASE_SET_FORMAT}")
    return data


def review_case_status_item(
    raw_item: dict[str, Any],
    *,
    index: int,
    base_dir: Path,
    default_reference_type: str | None,
    source_language: str,
) -> dict[str, Any]:
    case_id = require_index_string(raw_item, "id", index)
    audio = file_reference_status(require_index_string(raw_item, "audio", index), base_dir=base_dir)
    reference = file_reference_status(require_index_string(raw_item, "reference", index), base_dir=base_dir)
    reference_type = raw_item.get("reference_type", default_reference_type) or "unspecified"
    if not isinstance(reference_type, str):
        raise ValueError(f"review case index item {index} reference_type must be a string")

    issues: list[str] = []
    missing_file_count = 0
    for label, file_status in (("audio", audio), ("reference", reference)):
        if not file_status["exists"]:
            missing_file_count += 1
            issues.append(f"{label} file is missing: {file_status['path']}")

    reference_segments = 0
    reference_review_count = 0
    if reference["exists"]:
        reference_counts, reference_issue = transcript_counts(
            Path(reference["resolved_path"]),
            source_language=source_language,
        )
        reference_segments = reference_counts["segments"]
        reference_review_count = reference_counts["review_count"]
        if reference_issue is not None:
            issues.append(f"reference {reference_issue}")
        else:
            index_segments = raw_item.get("segments")
            if isinstance(index_segments, int) and index_segments != reference_segments:
                issues.append(f"reference segment count {reference_segments} != index segments {index_segments}")
            index_review_count = raw_item.get("review_count")
            if isinstance(index_review_count, int) and index_review_count != reference_review_count:
                issues.append(
                    f"reference review count {reference_review_count} != index review_count {index_review_count}"
                )

    candidate = None
    candidate_segments = None
    candidate_review_count = None
    candidate_value = raw_item.get("candidate")
    if candidate_value is not None:
        if not isinstance(candidate_value, str) or not candidate_value:
            raise ValueError(f"review case index item {index} candidate must be a non-empty string")
        candidate = file_reference_status(candidate_value, base_dir=base_dir)
        if not candidate["exists"]:
            missing_file_count += 1
            issues.append(f"candidate file is missing: {candidate['path']}")
        else:
            candidate_counts, candidate_issue = transcript_counts(
                Path(candidate["resolved_path"]),
                source_language=source_language,
            )
            candidate_segments = candidate_counts["segments"]
            candidate_review_count = candidate_counts["review_count"]
            if candidate_issue is not None:
                issues.append(f"candidate {candidate_issue}")

    return {
        "id": case_id,
        "reference_type": reference_type,
        "audio": audio,
        "reference": reference,
        "candidate": candidate,
        "candidate_id": raw_item.get("candidate_id"),
        "index_segments": raw_item.get("segments"),
        "index_review_count": raw_item.get("review_count"),
        "reference_segments": reference_segments,
        "reference_review_count": reference_review_count,
        "candidate_segments": candidate_segments,
        "candidate_review_count": candidate_review_count,
        "missing_file_count": missing_file_count,
        "issues": issues,
    }


def file_reference_status(path_value: str, *, base_dir: Path) -> dict[str, Any]:
    path = resolve_plan_path(base_dir, path_value)
    return {
        "path": path_value,
        "resolved_path": str(path),
        "exists": path.is_file(),
    }


def transcript_counts(path: Path, *, source_language: str) -> tuple[dict[str, int], str | None]:
    try:
        master = load_transcript_document(path, source_language=source_language)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"segments": 0, "review_count": 0}, f"cannot be loaded: {error}"
    return {
        "segments": len(master.segments),
        "review_count": sum(1 for segment in master.segments if segment.needs_review),
    }, None


def validate_case_slice_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("format") != CASE_SLICE_PLAN_FORMAT:
        raise ValueError(f"case slice plan format must be {CASE_SLICE_PLAN_FORMAT}")
    optional_plan_string(plan, "reference_type")
    optional_plan_string(plan, "reference_notes")
    raw_cases = plan.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case slice plan cases must be a non-empty array")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case slice plan case {index} must be an object")
        case_id = require_case_string(raw_case, "id", index)
        if case_id in seen_ids:
            raise ValueError(f"case slice plan case id is duplicated: {case_id}")
        seen_ids.add(case_id)
        case_file_stem(case_id)
        start_ms = require_case_int(raw_case, "start_ms", index)
        end_ms = require_case_int(raw_case, "end_ms", index)
        if end_ms <= start_ms:
            raise ValueError(f"case slice plan case {index} end_ms must be greater than start_ms")
        normalized = {
            "id": case_id,
            "audio": require_case_string(raw_case, "audio", index),
            "reference": require_case_string(raw_case, "reference", index),
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        candidate = raw_case.get("candidate")
        if candidate is not None:
            if not isinstance(candidate, str) or not candidate:
                raise ValueError(f"case slice plan case {index} candidate must be a non-empty string")
            normalized["candidate"] = candidate
        candidate_id = raw_case.get("candidate_id")
        if candidate_id is not None:
            if "candidate" not in normalized:
                raise ValueError(f"case slice plan case {index} candidate_id requires candidate")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(f"case slice plan case {index} candidate_id must be a non-empty string")
            normalized["candidate_id"] = candidate_id
        cases.append(normalized)
    return cases


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("review cases output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_source_paths(
    cases: list[dict[str, Any]],
    *,
    base_dir: Path,
    include_candidates: bool,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for case in cases:
        case_id = case["id"]
        paths[f"{case_id}:audio"] = resolve_plan_path(base_dir, case["audio"])
        paths[f"{case_id}:reference"] = resolve_plan_path(base_dir, case["reference"])
        if include_candidates:
            paths[f"{case_id}:candidate"] = resolve_plan_path(base_dir, case["candidate"])
    return paths


def optional_plan_string(plan: dict[str, Any], key: str) -> str | None:
    value = plan.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"case slice plan {key} must be a non-empty string")
    return value


def require_case_string(case: dict[str, Any], key: str, index: int) -> str:
    value = case.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"case slice plan case {index} {key} must be a non-empty string")
    return value


def require_case_int(case: dict[str, Any], key: str, index: int) -> int:
    value = case.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"case slice plan case {index} {key} must be an integer")
    if value < 0:
        raise ValueError(f"case slice plan case {index} {key} must be non-negative")
    return value


def require_index_string(item: dict[str, Any], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"review case index item {index} {key} must be a non-empty string")
    return value


def resolve_plan_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def case_file_stem(case_id: str) -> str:
    if case_id in {".", ".."}:
        raise ValueError("case slice plan case id must be a safe file name")
    if Path(case_id).name != case_id:
        raise ValueError("case slice plan case id must not contain path separators")
    return case_id
