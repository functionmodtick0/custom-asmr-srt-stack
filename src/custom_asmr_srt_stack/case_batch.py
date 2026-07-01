from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.alignment import alignment_diagnostics, run_alignment_command
from custom_asmr_srt_stack.audio import analyze_wav, normalize_audio_to_wav, slice_wav
from custom_asmr_srt_stack.case_slicing import slice_master_document
from custom_asmr_srt_stack.evaluation import EVAL_MANIFEST_FORMAT, load_transcript_document
from custom_asmr_srt_stack.models import MasterDocument
from custom_asmr_srt_stack.review_pack import (
    DEFAULT_REVIEW_CONTEXT_MS,
    REVIEW_AUDIO_MAP_FORMAT,
    REVIEW_PACK_FORMAT,
    sanitize_clip_name,
)

CASE_SLICE_PLAN_FORMAT = "custom-asmr-case-slice-plan-v1"
CASE_CANDIDATE_ATTACH_PLAN_FORMAT = "custom-asmr-case-candidate-attach-plan-v1"
CASE_CANDIDATE_ATTACH_PLAN_BUILD_FORMAT = "custom-asmr-case-candidate-attach-plan-build-v1"
REVIEW_CASE_SET_FORMAT = "custom-asmr-review-case-set-v1"
REVIEW_CASE_STATUS_FORMAT = "custom-asmr-review-case-status-v1"
REVIEW_CASE_REFERENCE_SAVE_FORMAT = "custom-asmr-review-case-reference-save-v1"
REVIEW_CASE_CANDIDATE_ATTACH_FORMAT = "custom-asmr-review-case-candidate-attach-v1"
REVIEW_CASE_CANDIDATE_ALIGN_FORMAT = "custom-asmr-review-case-candidate-align-v1"
EVAL_MANIFEST_BUILD_FORMAT = "custom-asmr-eval-manifest-build-v1"
CASE_REFERENCE_FREEZE_FORMAT = "custom-asmr-case-reference-freeze-v1"


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
    reference_review_duration_ms = 0
    reference_review_clear_case_count = 0
    candidate_case_count = 0
    candidate_review_count = 0
    candidate_review_duration_ms = 0
    candidate_review_clear_case_count = 0
    cases_missing_candidate: list[str] = []
    cases_needing_review: list[str] = []
    cases_with_candidate_review: list[str] = []
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
        reference_review_duration_ms += item["reference_review_duration_ms"]
        if item.get("candidate") is not None:
            candidate_case_count += 1
            item_candidate_review_count = item.get("candidate_review_count")
            if isinstance(item_candidate_review_count, int):
                candidate_review_count += item_candidate_review_count
                candidate_review_duration_ms += item["candidate_review_duration_ms"]
                if item_candidate_review_count > 0:
                    cases_with_candidate_review.append(item["id"])
                elif candidate_loaded_without_issues(item):
                    candidate_review_clear_case_count += 1
        else:
            cases_missing_candidate.append(item["id"])
        if item["reference_review_count"] > 0:
            cases_needing_review.append(item["id"])
        elif reference_loaded_without_issues(item):
            reference_review_clear_case_count += 1
        if item["issues"]:
            cases_with_issues.append(item["id"])

    return {
        "format": REVIEW_CASE_STATUS_FORMAT,
        "case_index": str(case_index_file),
        "case_count": len(items),
        "candidate_case_count": candidate_case_count,
        "missing_candidate_case_count": len(cases_missing_candidate),
        "cases_missing_candidate": cases_missing_candidate,
        "next_missing_candidate_case_id": cases_missing_candidate[0] if cases_missing_candidate else None,
        "candidate_review_count": candidate_review_count,
        "candidate_review_case_count": len(cases_with_candidate_review),
        "candidate_review_clear_case_count": candidate_review_clear_case_count,
        "cases_with_candidate_review": cases_with_candidate_review,
        "next_candidate_review_case_id": cases_with_candidate_review[0] if cases_with_candidate_review else None,
        "reference_type_counts": reference_type_counts,
        "missing_file_count": missing_file_count,
        "cases_with_issues": cases_with_issues,
        "case_issue_count": len(cases_with_issues),
        "reference_review_count": reference_review_count,
        "reference_review_duration_ms": reference_review_duration_ms,
        "reference_review_case_count": len(cases_needing_review),
        "reference_review_clear_case_count": reference_review_clear_case_count,
        "cases_needing_review": cases_needing_review,
        "next_review_case_id": cases_needing_review[0] if cases_needing_review else None,
        "candidate_review_duration_ms": candidate_review_duration_ms,
        "ok": missing_file_count == 0 and not cases_with_issues,
        "items": items,
    }


def build_review_case_pack(
    case_index_file: Path,
    *,
    output_dir: Path,
    context_ms: int = DEFAULT_REVIEW_CONTEXT_MS,
    source_language: str = "ja",
) -> dict[str, Any]:
    if context_ms < 0:
        raise ValueError("context_ms must be non-negative")
    case_index = load_review_case_index(case_index_file)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review case index items must be an array")

    base_dir = case_index_file.parent
    case_sources: list[dict[str, Any]] = []
    for item_index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {item_index} must be an object")
        case_id = require_index_string(raw_item, "id", item_index)
        audio_value = require_index_string(raw_item, "audio", item_index)
        reference_value = require_index_string(raw_item, "reference", item_index)
        audio_path = resolve_plan_path(base_dir, audio_value)
        reference_path = resolve_plan_path(base_dir, reference_value)
        if not audio_path.is_file():
            raise ValueError(f"review case audio file is missing: {audio_value}")
        if not reference_path.is_file():
            raise ValueError(f"review case reference file is missing: {reference_value}")
        audio_bytes = audio_path.read_bytes()
        audio_duration_ms = analyze_wav(audio_bytes).duration_ms
        master = load_transcript_document(reference_path, source_language=source_language)
        case_sources.append(
            {
                "case_id": case_id,
                "audio": audio_value,
                "reference": reference_value,
                "audio_bytes": audio_bytes,
                "audio_duration_ms": audio_duration_ms,
                "master": master,
            }
        )

    prepare_output_dir(output_dir)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir()

    packed_items: list[dict[str, Any]] = []
    for source in case_sources:
        case_id = source["case_id"]
        audio_value = source["audio"]
        reference_value = source["reference"]
        audio_bytes = source["audio_bytes"]
        audio_duration_ms = source["audio_duration_ms"]
        master = source["master"]
        for segment in master.segments:
            if not segment.needs_review:
                continue
            clip_start_ms = max(0, segment.start_ms - context_ms)
            clip_end_ms = min(audio_duration_ms, segment.end_ms + context_ms)
            if clip_end_ms <= clip_start_ms:
                raise ValueError(f"review case segment {case_id}/{segment.id} selects an empty audio range")
            rank = len(packed_items) + 1
            clip_file = str(Path("clips") / review_case_clip_name(rank, case_id, segment.id))
            (output_dir / clip_file).write_bytes(slice_wav(audio_bytes, start_ms=clip_start_ms, end_ms=clip_end_ms))
            packed_items.append(
                {
                    "case_id": case_id,
                    "reference_id": segment.id,
                    "candidate_id": None,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "reasons": ["reference-needs-review"],
                    "reference_channel": segment.channel,
                    "reference_text": segment.text,
                    "candidate_channel": None,
                    "candidate_text": "",
                    "priority_score": float(segment.end_ms - segment.start_ms),
                    "priority_rank": rank,
                    "source_case_index": str(case_index_file),
                    "audio": audio_value,
                    "reference": reference_value,
                    "clip_file": clip_file,
                    "clip_start_ms": clip_start_ms,
                    "clip_end_ms": clip_end_ms,
                    "clip_context_ms": context_ms,
                }
            )

    result = {
        "format": REVIEW_PACK_FORMAT,
        "source_case_index": str(case_index_file),
        "clip_count": len(packed_items),
        "context_ms": context_ms,
        "items": packed_items,
    }
    (output_dir / "index.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def review_case_clip_name(index: int, case_id: str, segment_id: str) -> str:
    value = f"{index:06d}__{case_id}__reference-needs-review__{segment_id}__no-cand"
    return sanitize_clip_name(value) + ".wav"


def reference_loaded_without_issues(item: dict[str, Any]) -> bool:
    reference = item.get("reference")
    if not isinstance(reference, dict) or not reference.get("exists"):
        return False
    issues = item.get("issues")
    return isinstance(issues, list) and not any(str(issue).startswith("reference ") for issue in issues)


def candidate_loaded_without_issues(item: dict[str, Any]) -> bool:
    candidate = item.get("candidate")
    if not isinstance(candidate, dict) or not candidate.get("exists"):
        return False
    issues = item.get("issues")
    return isinstance(issues, list) and not any(str(issue).startswith("candidate ") for issue in issues)


def save_review_case_reference(
    case_index_file: Path,
    *,
    case_id: str,
    master: MasterDocument,
) -> dict[str, Any]:
    if not case_id:
        raise ValueError("review case id must be a non-empty string")
    resolved_index_path = case_index_file.expanduser().resolve()
    case_index = load_review_case_index(resolved_index_path)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review case index items must be an array")
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        if raw_item.get("id") != case_id:
            continue
        reference_path = resolve_plan_path(
            resolved_index_path.parent,
            require_index_string(raw_item, "reference", index),
        )
        if not reference_path.is_file():
            raise ValueError(f"review case reference file is missing: {raw_item['reference']}")
        reference_path.write_text(json.dumps(master.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raw_item["segments"] = len(master.segments)
        raw_item["review_count"] = sum(1 for segment in master.segments if segment.needs_review)
        resolved_index_path.write_text(json.dumps(case_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "format": REVIEW_CASE_REFERENCE_SAVE_FORMAT,
            "ok": True,
            "case_id": case_id,
            "reference": str(reference_path),
            "segments": raw_item["segments"],
            "review_count": raw_item["review_count"],
        }
    raise ValueError(f"review case id is missing: {case_id}")


def attach_review_case_candidates(
    case_index_file: Path,
    plan_file: Path,
    *,
    replace: bool = False,
    source_language: str = "ja",
) -> dict[str, Any]:
    resolved_index_path = case_index_file.expanduser().resolve()
    case_index = load_review_case_index(resolved_index_path)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("review case index items must be a non-empty array")

    plan = load_case_candidate_attach_plan(plan_file)
    candidates = validate_case_candidate_attach_plan(plan)
    candidate_by_case_id = {candidate["case_id"]: candidate for candidate in candidates}
    raw_item_by_case_id: dict[str, dict[str, Any]] = {}
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        case_id = require_index_string(raw_item, "id", index)
        if case_id in raw_item_by_case_id:
            raise ValueError(f"review case index item id is duplicated: {case_id}")
        raw_item_by_case_id[case_id] = raw_item
        if raw_item.get("candidate") is not None and not replace:
            raise ValueError(f"review case {case_id} already has a candidate; use --replace to overwrite")

    missing_cases = sorted(set(raw_item_by_case_id) - set(candidate_by_case_id))
    extra_cases = sorted(set(candidate_by_case_id) - set(raw_item_by_case_id))
    if missing_cases:
        raise ValueError("candidate attach plan is missing case ids: " + ", ".join(missing_cases))
    if extra_cases:
        raise ValueError("candidate attach plan has unknown case ids: " + ", ".join(extra_cases))

    default_candidate_id = optional_attach_plan_string(plan, "candidate_id")
    base_dir = plan_file.parent
    candidates_dir = resolved_index_path.parent / "candidates"
    prepared: list[dict[str, Any]] = []
    for case_id, candidate in candidate_by_case_id.items():
        candidate_value = candidate["candidate"]
        candidate_path = resolve_plan_path(base_dir, candidate_value)
        if not candidate_path.is_file():
            raise ValueError(f"candidate file does not exist for {case_id}: {candidate_path}")
        candidate_id = candidate.get("candidate_id") or default_candidate_id or candidate_path.stem
        candidate_master = load_transcript_document(candidate_path, source_language=source_language)
        candidate_relative = Path("candidates") / f"{case_file_stem(case_id)}.master.json"
        candidate_output = resolved_index_path.parent / candidate_relative
        if candidate_output.exists() and not replace:
            raise ValueError(f"candidate output already exists for {case_id}: {candidate_output}")
        prepared.append(
            {
                "case_id": case_id,
                "candidate": candidate_value,
                "candidate_id": candidate_id,
                "candidate_master": candidate_master,
                "candidate_relative": candidate_relative,
                "candidate_output": candidate_output,
            }
        )

    candidates_dir.mkdir(exist_ok=True)
    attached_items: list[dict[str, Any]] = []
    for item in prepared:
        item["candidate_output"].write_text(
            json.dumps(item["candidate_master"].to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raw_item = raw_item_by_case_id[item["case_id"]]
        raw_item["candidate"] = str(item["candidate_relative"])
        raw_item["candidate_id"] = item["candidate_id"]
        raw_item["source_candidate"] = item["candidate"]
        attached_items.append(
            {
                "case_id": item["case_id"],
                "candidate": str(item["candidate_relative"]),
                "candidate_id": item["candidate_id"],
                "source_candidate": item["candidate"],
                "segments": len(item["candidate_master"].segments),
                "review_count": sum(1 for segment in item["candidate_master"].segments if segment.needs_review),
            }
        )

    resolved_index_path.write_text(json.dumps(case_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "format": REVIEW_CASE_CANDIDATE_ATTACH_FORMAT,
        "case_index": str(resolved_index_path),
        "plan": str(plan_file),
        "candidate_count": len(attached_items),
        "replace": replace,
        "items": attached_items,
    }


def build_case_candidate_attach_plan(
    case_index_file: Path,
    candidate_dir: Path,
    *,
    output: Path,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    if candidate_id is not None and not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    if not candidate_dir.is_dir():
        raise ValueError(f"candidate directory does not exist: {candidate_dir}")

    case_index = load_review_case_index(case_index_file)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("review case index items must be a non-empty array")

    output_parent = output.parent.expanduser().resolve()
    candidate_entries: list[dict[str, str]] = []
    missing_cases: list[str] = []
    ambiguous_cases: list[str] = []
    seen_case_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        case_id = require_index_string(raw_item, "id", index)
        if case_id in seen_case_ids:
            raise ValueError(f"review case index item id is duplicated: {case_id}")
        seen_case_ids.add(case_id)
        matches = matching_candidate_files(candidate_dir, case_id)
        if not matches:
            missing_cases.append(case_id)
            continue
        if len(matches) > 1:
            ambiguous_cases.append(f"{case_id}: " + ", ".join(path.name for path in matches))
            continue
        candidate_entries.append(
            {
                "case_id": case_id,
                "candidate": relative_path_value(matches[0], base_dir=output_parent),
            }
        )

    if missing_cases:
        raise ValueError("candidate directory is missing case ids: " + ", ".join(missing_cases))
    if ambiguous_cases:
        raise ValueError("candidate directory has ambiguous case files: " + "; ".join(ambiguous_cases))

    plan: dict[str, Any] = {
        "format": CASE_CANDIDATE_ATTACH_PLAN_FORMAT,
        "candidates": candidate_entries,
    }
    if candidate_id is not None:
        plan["candidate_id"] = candidate_id

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "format": CASE_CANDIDATE_ATTACH_PLAN_BUILD_FORMAT,
        "output": str(output),
        "case_index": str(case_index_file),
        "candidate_dir": str(candidate_dir),
        "candidate_count": len(candidate_entries),
        "candidates": candidate_entries,
    }
    if candidate_id is not None:
        report["candidate_id"] = candidate_id
    return report


def align_review_case_candidates(
    case_index_file: Path,
    *,
    output_dir: Path,
    command: list[str],
    candidate_id: str | None = None,
    source_language: str = "ja",
) -> dict[str, Any]:
    if not command:
        raise ValueError("alignment command must not be empty")
    if candidate_id is not None and not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")

    case_index = load_review_case_index(case_index_file)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("review case index items must be a non-empty array")

    base_dir = case_index_file.parent
    prepared: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        case_id = require_index_string(raw_item, "id", index)
        if case_id in seen_case_ids:
            raise ValueError(f"review case index item id is duplicated: {case_id}")
        seen_case_ids.add(case_id)

        audio_value = require_index_string(raw_item, "audio", index)
        reference_value = require_index_string(raw_item, "reference", index)
        candidate_value = require_index_string(raw_item, "candidate", index)
        audio_path = resolve_plan_path(base_dir, audio_value)
        reference_path = resolve_plan_path(base_dir, reference_value)
        candidate_path = resolve_plan_path(base_dir, candidate_value)
        for label, path in (("audio", audio_path), ("reference", reference_path), ("candidate", candidate_path)):
            if not path.is_file():
                raise ValueError(f"review case {label} file does not exist for {case_id}: {path}")
        source_candidate_id = raw_item.get("candidate_id")
        if source_candidate_id is not None and (not isinstance(source_candidate_id, str) or not source_candidate_id):
            raise ValueError(f"review case index item {index} candidate_id must be a non-empty string")
        candidate_master = load_transcript_document(candidate_path, source_language=source_language)
        aligned_master = run_alignment_command(candidate_master, audio_file=audio_path, command=command)
        selected_candidate_id = candidate_id or f"{source_candidate_id or candidate_path.stem}-aligned"
        candidate_relative = Path("candidates") / f"{case_file_stem(case_id)}.master.json"
        diagnostics_relative = Path("diagnostics") / f"{case_file_stem(case_id)}.alignment-diagnostics.json"
        prepared.append(
            {
                "case_id": case_id,
                "audio": audio_value,
                "reference": reference_value,
                "source_candidate": candidate_value,
                "source_candidate_id": source_candidate_id or candidate_path.stem,
                "candidate_id": selected_candidate_id,
                "candidate_master": candidate_master,
                "aligned_master": aligned_master,
                "candidate_relative": candidate_relative,
                "diagnostics_relative": diagnostics_relative,
                "diagnostics": alignment_diagnostics(
                    candidate_master,
                    aligned_master,
                    audio_file=audio_path,
                    input_file=candidate_path,
                    output_file=output_dir / candidate_relative,
                ),
            }
        )

    prepare_output_dir(output_dir)
    candidates_dir = output_dir / "candidates"
    diagnostics_dir = output_dir / "diagnostics"
    candidates_dir.mkdir()
    diagnostics_dir.mkdir()

    attach_candidates: list[dict[str, str]] = []
    eval_cases: list[dict[str, str]] = []
    report_items: list[dict[str, Any]] = []
    for item in prepared:
        candidate_output = output_dir / item["candidate_relative"]
        diagnostics_output = output_dir / item["diagnostics_relative"]
        candidate_output.write_text(
            json.dumps(item["aligned_master"].to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        diagnostics_output.write_text(
            json.dumps(item["diagnostics"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        attach_candidate = {
            "case_id": item["case_id"],
            "candidate": str(item["candidate_relative"]),
            "candidate_id": item["candidate_id"],
        }
        attach_candidates.append(attach_candidate)
        eval_cases.append(
            {
                "id": item["case_id"],
                "reference": relative_path_value(resolve_plan_path(base_dir, item["reference"]), base_dir=output_dir),
                "candidate": str(item["candidate_relative"]),
                "candidate_id": item["candidate_id"],
            }
        )
        report_items.append(
            {
                "case_id": item["case_id"],
                "audio": item["audio"],
                "reference": item["reference"],
                "source_candidate": item["source_candidate"],
                "source_candidate_id": item["source_candidate_id"],
                "candidate": str(item["candidate_relative"]),
                "candidate_id": item["candidate_id"],
                "diagnostics": str(item["diagnostics_relative"]),
                "segments": len(item["aligned_master"].segments),
                "changed_segments": item["diagnostics"]["changed_segments"],
                "review_count": sum(1 for segment in item["aligned_master"].segments if segment.needs_review),
                "max_boundary_delta_ms": item["diagnostics"]["max_boundary_delta_ms"],
            }
        )

    attach_plan = {
        "format": CASE_CANDIDATE_ATTACH_PLAN_FORMAT,
        "candidates": attach_candidates,
    }
    if candidate_id is not None:
        attach_plan["candidate_id"] = candidate_id
    attach_plan_file = output_dir / "attach-plan.json"
    attach_plan_file.write_text(json.dumps(attach_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "format": EVAL_MANIFEST_FORMAT,
        "reference_type": optional_index_string(case_index, "reference_type") or "unspecified",
        "cases": eval_cases,
    }
    reference_notes = optional_index_string(case_index, "reference_notes")
    if reference_notes is not None:
        manifest["reference_notes"] = reference_notes
    eval_manifest_file = output_dir / "eval-manifest.json"
    eval_manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "format": REVIEW_CASE_CANDIDATE_ALIGN_FORMAT,
        "case_index": str(case_index_file),
        "output": str(output_dir),
        "candidate_count": len(report_items),
        "attach_plan": str(attach_plan_file),
        "eval_manifest": str(eval_manifest_file),
        "items": report_items,
    }
    report_file = output_dir / "index.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def matching_candidate_files(candidate_dir: Path, case_id: str) -> list[Path]:
    stem = case_file_stem(case_id)
    candidate_names = [
        f"{stem}.master.json",
        f"{stem}.json",
        f"{stem}.srt",
    ]
    return [candidate_dir / name for name in candidate_names if (candidate_dir / name).is_file()]


def relative_path_value(path: Path, *, base_dir: Path) -> str:
    return Path(os.path.relpath(path.expanduser().resolve(), base_dir)).as_posix()


def load_case_candidate_attach_plan(plan_file: Path) -> dict[str, Any]:
    data = json.loads(plan_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("candidate attach plan must be a JSON object")
    return data


def validate_case_candidate_attach_plan(plan: dict[str, Any]) -> list[dict[str, str]]:
    if plan.get("format") != CASE_CANDIDATE_ATTACH_PLAN_FORMAT:
        raise ValueError(f"candidate attach plan format must be {CASE_CANDIDATE_ATTACH_PLAN_FORMAT}")
    optional_attach_plan_string(plan, "candidate_id")
    raw_candidates = plan.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate attach plan candidates must be a non-empty array")

    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"candidate attach plan candidate {index} must be an object")
        case_id = require_attach_candidate_string(raw_candidate, "case_id", index)
        if case_id in seen_ids:
            raise ValueError(f"candidate attach plan case_id is duplicated: {case_id}")
        seen_ids.add(case_id)
        normalized = {
            "case_id": case_id,
            "candidate": require_attach_candidate_string(raw_candidate, "candidate", index),
        }
        candidate_id = raw_candidate.get("candidate_id")
        if candidate_id is not None:
            normalized["candidate_id"] = require_attach_candidate_string(raw_candidate, "candidate_id", index)
        candidates.append(normalized)
    return candidates


def build_eval_manifest_from_case_index(
    case_index_file: Path,
    *,
    reference_type: str | None = None,
    reference_notes: str | None = None,
    fail_on_review: bool = False,
    source_language: str = "ja",
) -> dict[str, Any]:
    if reference_type is not None and not reference_type:
        raise ValueError("eval manifest reference_type must be a non-empty string")
    if reference_notes is not None and not reference_notes:
        raise ValueError("eval manifest reference_notes must be a non-empty string")
    case_index = load_review_case_index(case_index_file)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("review case index items must be a non-empty array")
    status = review_case_status(case_index_file, source_language=source_language)
    if not status["ok"]:
        raise ValueError(
            "review case index has issues: "
            f"missing_files={status['missing_file_count']} case_issues={status['case_issue_count']}"
        )
    if fail_on_review and status["reference_review_count"] > 0:
        raise ValueError(f"review case index still has reference review_count={status['reference_review_count']}")

    selected_reference_type = reference_type or optional_index_string(case_index, "reference_type") or "unspecified"
    selected_reference_notes = reference_notes
    if selected_reference_notes is None:
        selected_reference_notes = optional_index_string(case_index, "reference_notes")

    cases: list[dict[str, str]] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        case_id = require_index_string(raw_item, "id", index)
        reference = require_index_string(raw_item, "reference", index)
        candidate = require_index_string(raw_item, "candidate", index)
        candidate_id = raw_item.get("candidate_id")
        if candidate_id is not None and (not isinstance(candidate_id, str) or not candidate_id):
            raise ValueError(f"review case index item {index} candidate_id must be a non-empty string")
        case = {
            "id": case_id,
            "reference": reference,
            "candidate": candidate,
            "candidate_id": candidate_id or Path(candidate).stem,
        }
        if reference_type is None:
            item_reference_type = raw_item.get("reference_type")
            if item_reference_type is not None:
                if not isinstance(item_reference_type, str) or not item_reference_type:
                    raise ValueError(f"review case index item {index} reference_type must be a non-empty string")
                if item_reference_type != selected_reference_type:
                    case["reference_type"] = item_reference_type
            item_reference_notes = raw_item.get("reference_notes")
            if item_reference_notes is not None:
                if not isinstance(item_reference_notes, str) or not item_reference_notes:
                    raise ValueError(f"review case index item {index} reference_notes must be a non-empty string")
                if selected_reference_notes is None or item_reference_notes != selected_reference_notes:
                    case["reference_notes"] = item_reference_notes
        cases.append(case)

    manifest: dict[str, Any] = {
        "format": EVAL_MANIFEST_FORMAT,
        "reference_type": selected_reference_type,
        "cases": cases,
    }
    if selected_reference_notes is not None:
        manifest["reference_notes"] = selected_reference_notes
    return manifest


def freeze_case_references(
    case_index_file: Path,
    *,
    output_dir: Path,
    reference_type: str = "human-reviewed",
    reference_notes: str | None = None,
    fail_on_review: bool = False,
    source_language: str = "ja",
) -> dict[str, Any]:
    if not reference_type:
        raise ValueError("frozen case reference_type must be a non-empty string")
    if reference_notes is not None and not reference_notes:
        raise ValueError("frozen case reference_notes must be a non-empty string")

    case_index = load_review_case_index(case_index_file)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("review case index items must be a non-empty array")

    base_dir = case_index_file.parent
    prepared_items = []
    candidate_flags = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        case_id = require_index_string(raw_item, "id", index)
        case_file_stem(case_id)
        audio_path = resolve_plan_path(base_dir, require_index_string(raw_item, "audio", index))
        reference_path = resolve_plan_path(base_dir, require_index_string(raw_item, "reference", index))
        candidate_value = raw_item.get("candidate")
        candidate_path = None
        if candidate_value is not None:
            if not isinstance(candidate_value, str) or not candidate_value:
                raise ValueError(f"review case index item {index} candidate must be a non-empty string")
            candidate_path = resolve_plan_path(base_dir, candidate_value)
        candidate_flags.append(candidate_path is not None)

        for label, path in (("audio", audio_path), ("reference", reference_path), ("candidate", candidate_path)):
            if path is not None and not path.is_file():
                raise ValueError(f"review case index {label} file does not exist: {path}")
        reference_master = load_transcript_document(reference_path, source_language=source_language)
        candidate_id = raw_item.get("candidate_id")
        if candidate_id is not None and (not isinstance(candidate_id, str) or not candidate_id):
            raise ValueError(f"review case index item {index} candidate_id must be a non-empty string")
        prepared_items.append(
            {
                "raw": raw_item,
                "id": case_id,
                "audio_path": audio_path,
                "reference_path": reference_path,
                "reference_master": reference_master,
                "candidate_path": candidate_path,
                "candidate_id": candidate_id,
            }
        )
    if any(candidate_flags) and not all(candidate_flags):
        raise ValueError("review case index cannot mix candidate and non-candidate cases")
    reference_review_count = sum(
        1
        for item in prepared_items
        for segment in item["reference_master"].segments
        if segment.needs_review
    )
    if fail_on_review and reference_review_count > 0:
        raise ValueError(f"review case index still has reference review_count={reference_review_count}")

    prepare_output_dir(output_dir)
    reference_dir = output_dir / "references"
    reference_dir.mkdir()

    index_items: list[dict[str, Any]] = []
    audio_map_items: list[dict[str, str]] = []
    eval_cases: list[dict[str, str]] = []
    for item in prepared_items:
        case_id = item["id"]
        reference_relative = Path("references") / f"{case_file_stem(case_id)}.master.json"
        frozen_reference = freeze_master_document(item["reference_master"])
        (output_dir / reference_relative).write_text(
            json.dumps(frozen_reference.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        index_item = {
            "id": case_id,
            "audio": str(item["audio_path"]),
            "reference": str(reference_relative),
            "source_reference": str(item["reference_path"]),
            "segments": len(frozen_reference.segments),
            "review_count": 0,
            "reference_type": reference_type,
        }
        if reference_notes is not None:
            index_item["reference_notes"] = reference_notes
        for key in ("source_audio", "start_ms", "end_ms", "duration_ms"):
            if key in item["raw"]:
                index_item[key] = item["raw"][key]
        if item["candidate_path"] is not None:
            candidate_id = item["candidate_id"] or item["candidate_path"].stem
            index_item["candidate"] = str(item["candidate_path"])
            index_item["candidate_id"] = candidate_id
            eval_cases.append(
                {
                    "id": case_id,
                    "reference": str(reference_relative),
                    "candidate": str(item["candidate_path"]),
                    "candidate_id": candidate_id,
                }
            )
        index_items.append(index_item)
        audio_map_items.append({"case_id": case_id, "audio": str(item["audio_path"])})

    audio_map_file = output_dir / "audio-map.json"
    audio_map_file.write_text(
        json.dumps({"format": REVIEW_AUDIO_MAP_FORMAT, "items": audio_map_items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_case_index = {
        "format": REVIEW_CASE_SET_FORMAT,
        "source_case_index": str(case_index_file),
        "reference_type": reference_type,
        "case_count": len(index_items),
        "items": index_items,
    }
    if reference_notes is not None:
        output_case_index["reference_notes"] = reference_notes
    case_index_output = output_dir / "case-index.json"
    case_index_output.write_text(json.dumps(output_case_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "format": CASE_REFERENCE_FREEZE_FORMAT,
        "output": str(output_dir),
        "case_count": len(index_items),
        "reference_type": reference_type,
        "review_count": 0,
        "audio_map": str(audio_map_file),
        "case_index": str(case_index_output),
    }
    if reference_notes is not None:
        result["reference_notes"] = reference_notes
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


def freeze_master_document(master: MasterDocument) -> MasterDocument:
    segments = tuple(
        replace(segment, id=f"seg_{index + 1:06d}", needs_review=False)
        for index, segment in enumerate(sorted(master.segments, key=lambda item: (item.start_ms, item.end_ms, item.id)))
    )
    return replace(master, segments=segments)


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
    reference_review_duration_ms = 0
    first_review_segment = None
    if reference["exists"]:
        reference_counts, reference_issue = transcript_status(
            Path(reference["resolved_path"]),
            source_language=source_language,
        )
        reference_segments = reference_counts["segments"]
        reference_review_count = reference_counts["review_count"]
        reference_review_duration_ms = reference_counts["review_duration_ms"]
        first_review_segment = reference_counts["first_review_segment"]
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
    candidate_review_duration_ms = None
    candidate_value = raw_item.get("candidate")
    if candidate_value is not None:
        if not isinstance(candidate_value, str) or not candidate_value:
            raise ValueError(f"review case index item {index} candidate must be a non-empty string")
        candidate = file_reference_status(candidate_value, base_dir=base_dir)
        if not candidate["exists"]:
            missing_file_count += 1
            issues.append(f"candidate file is missing: {candidate['path']}")
        else:
            candidate_counts, candidate_issue = transcript_status(
                Path(candidate["resolved_path"]),
                source_language=source_language,
            )
            candidate_segments = candidate_counts["segments"]
            candidate_review_count = candidate_counts["review_count"]
            candidate_review_duration_ms = candidate_counts["review_duration_ms"]
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
        "reference_review_duration_ms": reference_review_duration_ms,
        "first_review_segment": first_review_segment,
        "candidate_segments": candidate_segments,
        "candidate_review_count": candidate_review_count,
        "candidate_review_duration_ms": candidate_review_duration_ms,
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


def transcript_status(path: Path, *, source_language: str) -> tuple[dict[str, Any], str | None]:
    try:
        master = load_transcript_document(path, source_language=source_language)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "segments": 0,
            "review_count": 0,
            "review_duration_ms": 0,
            "first_review_segment": None,
        }, f"cannot be loaded: {error}"
    first_review_segment = next((segment.to_json() for segment in master.segments if segment.needs_review), None)
    review_segments = [segment for segment in master.segments if segment.needs_review]
    return {
        "segments": len(master.segments),
        "review_count": len(review_segments),
        "review_duration_ms": sum(max(0, segment.end_ms - segment.start_ms) for segment in review_segments),
        "first_review_segment": first_review_segment,
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


def optional_attach_plan_string(plan: dict[str, Any], key: str) -> str | None:
    value = plan.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate attach plan {key} must be a non-empty string")
    return value


def require_case_string(case: dict[str, Any], key: str, index: int) -> str:
    value = case.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"case slice plan case {index} {key} must be a non-empty string")
    return value


def require_attach_candidate_string(candidate: dict[str, Any], key: str, index: int) -> str:
    value = candidate.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate attach plan candidate {index} {key} must be a non-empty string")
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


def optional_index_string(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"review case index {key} must be a non-empty string")
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
