from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import analyze_wav, slice_wav

REVIEW_PACK_FORMAT = "custom-asmr-review-pack-v1"
REVIEW_AUDIO_MAP_FORMAT = "custom-asmr-review-audio-map-v1"
DEFAULT_REVIEW_CONTEXT_MS = 500


def build_review_pack(
    review_effort_report: dict[str, Any],
    *,
    output_dir: Path,
    audio_file: Path | None = None,
    audio_map_file: Path | None = None,
    source_case_index: Path | None = None,
    context_ms: int = DEFAULT_REVIEW_CONTEXT_MS,
) -> dict[str, Any]:
    if context_ms < 0:
        raise ValueError("context_ms must be non-negative")

    items = review_effort_items(review_effort_report)
    if audio_file is not None:
        validate_single_audio_scope(items)
    validate_review_pack_item_bounds(items)
    source_case_index_value = review_pack_source_case_index(review_effort_report, source_case_index)
    audio_by_case = load_audio_by_case(
        audio_file=audio_file,
        audio_map_file=audio_map_file,
        source_case_index_value=source_case_index_value,
    )
    prepare_output_dir(output_dir)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir()

    audio_cache: dict[str, tuple[bytes, int]] = {}
    packed_items = []
    duration_summary = empty_review_pack_duration_summary()
    for index, item in enumerate(items, start=1):
        case_id = item_case_id(item)
        audio_path = audio_path_for_case(audio_by_case, case_id)
        audio_key = str(audio_path)
        if audio_key not in audio_cache:
            audio_bytes = audio_path.read_bytes()
            info = analyze_wav(audio_bytes)
            audio_cache[audio_key] = (audio_bytes, info.duration_ms)
        audio_bytes, duration_ms = audio_cache[audio_key]
        start_ms, end_ms = item_bounds(item)
        focus_start_ms, focus_end_ms = item_review_clip_bounds(item, fallback_start_ms=start_ms, fallback_end_ms=end_ms)
        clip_start_ms = max(0, focus_start_ms - context_ms)
        clip_end_ms = min(duration_ms, focus_end_ms + context_ms)
        if clip_end_ms <= clip_start_ms:
            raise ValueError(f"review item {index} selects an empty audio range")
        update_review_pack_duration_summary(
            duration_summary,
            item=item,
            start_ms=start_ms,
            end_ms=end_ms,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
            clip_start_ms=clip_start_ms,
            clip_end_ms=clip_end_ms,
        )
        clip_file = review_clip_name(index, case_id, item)
        (clips_dir / clip_file).write_bytes(slice_wav(audio_bytes, start_ms=clip_start_ms, end_ms=clip_end_ms))
        packed_item = dict(item)
        packed_item.update(
            {
                "clip_file": str(Path("clips") / clip_file),
                "clip_start_ms": clip_start_ms,
                "clip_end_ms": clip_end_ms,
                "clip_context_ms": context_ms,
            }
        )
        if source_case_index_value is not None and item.get("case_id") and item.get("reference_id"):
            packed_item["source_case_index"] = source_case_index_value
        packed_items.append(packed_item)

    result = {
        "format": REVIEW_PACK_FORMAT,
        "source_report": review_effort_report.get("source_report"),
        "clip_count": len(packed_items),
        "duration_summary": duration_summary,
        "items": packed_items,
    }
    preserve_review_effort_summary(review_effort_report, result)
    if source_case_index_value is not None:
        result["source_case_index"] = source_case_index_value
    (output_dir / "index.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def preserve_review_effort_summary(source: dict[str, Any], target: dict[str, Any]) -> None:
    case_summaries = source.get("case_summaries")
    if isinstance(case_summaries, list):
        target["case_summaries"] = case_summaries
    case_count = source.get("case_count")
    if isinstance(case_count, int) and not isinstance(case_count, bool):
        target["case_count"] = case_count
    next_case_id = source.get("next_case_id")
    if next_case_id is None or isinstance(next_case_id, str):
        target["next_case_id"] = next_case_id


def empty_review_pack_duration_summary() -> dict[str, int]:
    return {
        "source_item_duration_ms_sum": 0,
        "effective_item_duration_ms_sum": 0,
        "clip_duration_ms_sum": 0,
        "clip_duration_ms_max": 0,
        "focus_item_count": 0,
    }


def update_review_pack_duration_summary(
    summary: dict[str, int],
    *,
    item: dict[str, Any],
    start_ms: int,
    end_ms: int,
    focus_start_ms: int,
    focus_end_ms: int,
    clip_start_ms: int,
    clip_end_ms: int,
) -> None:
    clip_duration_ms = clip_end_ms - clip_start_ms
    summary["source_item_duration_ms_sum"] += end_ms - start_ms
    summary["effective_item_duration_ms_sum"] += focus_end_ms - focus_start_ms
    summary["clip_duration_ms_sum"] += clip_duration_ms
    summary["clip_duration_ms_max"] = max(summary["clip_duration_ms_max"], clip_duration_ms)
    if item_has_review_clip_bounds(item):
        summary["focus_item_count"] += 1


def item_has_review_clip_bounds(item: dict[str, Any]) -> bool:
    return item.get("review_clip_start_ms") is not None or item.get("review_clip_end_ms") is not None


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("review pack output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)


def review_effort_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("format") != "custom-asmr-review-effort-v1":
        raise ValueError("review pack input must be custom-asmr-review-effort-v1")
    items = report.get("items")
    if not isinstance(items, list):
        raise ValueError("review pack input items must be an array")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("review pack input items must be objects")
    return items


def review_pack_source_case_index(report: dict[str, Any], source_case_index: Path | None) -> str | None:
    if source_case_index is not None:
        if not source_case_index.exists():
            raise ValueError(f"source case index is missing: {source_case_index}")
        return str(source_case_index)
    value = report.get("source_case_index")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("review pack source_case_index must be a non-empty string")
    return value


def validate_single_audio_scope(items: list[dict[str, Any]]) -> None:
    case_ids = {item_case_id(item) for item in items}
    case_ids.discard(None)
    if len(case_ids) > 1:
        raise ValueError("review pack with multiple case_id values requires --audio-map")


def load_audio_by_case(
    *,
    audio_file: Path | None,
    audio_map_file: Path | None,
    source_case_index_value: str | None = None,
) -> dict[str | None, Path]:
    if audio_file is None and audio_map_file is None and source_case_index_value is None:
        raise ValueError(
            "review pack requires --audio, --audio-map, --source-case-index, or embedded source_case_index"
        )
    if audio_file is not None and audio_map_file is not None:
        raise ValueError("review pack accepts only one of --audio or --audio-map")
    if audio_file is not None:
        return {None: audio_file}
    if audio_map_file is None:
        assert source_case_index_value is not None
        return load_audio_by_source_case_index(Path(source_case_index_value))
    data = json.loads(audio_map_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("audio map must be a JSON object")
    if data.get("format") == REVIEW_AUDIO_MAP_FORMAT:
        raw_items = data.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("audio map items must be an array")
        mapping: dict[str | None, Path] = {}
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"audio map item {index} must be an object")
            case_id = item.get("case_id")
            audio = item.get("audio")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"audio map item {index} case_id must be a non-empty string")
            if not isinstance(audio, str) or not audio:
                raise ValueError(f"audio map item {index} audio must be a non-empty string")
            mapping[case_id] = resolve_audio_map_path(audio_map_file.parent, audio)
        return mapping

    mapping = {}
    for case_id, audio in data.items():
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("audio map keys must be non-empty strings")
        if not isinstance(audio, str) or not audio:
            raise ValueError("audio map values must be non-empty strings")
        mapping[case_id] = resolve_audio_map_path(audio_map_file.parent, audio)
    return mapping


def load_audio_by_source_case_index(case_index_file: Path) -> dict[str | None, Path]:
    if not case_index_file.exists():
        raise ValueError(f"source case index is missing: {case_index_file}")
    data = json.loads(case_index_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "custom-asmr-review-case-set-v1":
        raise ValueError("source case index must be custom-asmr-review-case-set-v1")
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("source case index items must be an array")
    mapping: dict[str | None, Path] = {}
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"source case index item {index} must be an object")
        case_id = item.get("id")
        audio = item.get("audio")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"source case index item {index} id must be a non-empty string")
        if not isinstance(audio, str) or not audio:
            raise ValueError(f"source case index item {index} audio must be a non-empty string")
        mapping[case_id] = resolve_audio_map_path(case_index_file.parent, audio)
    return mapping


def resolve_audio_map_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def item_case_id(item: dict[str, Any]) -> str | None:
    case_id = item.get("case_id")
    if case_id is None:
        return None
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("review item case_id must be a non-empty string")
    return case_id


def audio_path_for_case(audio_by_case: dict[str | None, Path], case_id: str | None) -> Path:
    if None in audio_by_case:
        return audio_by_case[None]
    if case_id is None:
        raise ValueError("review item without case_id requires --audio")
    try:
        return audio_by_case[case_id]
    except KeyError as error:
        raise ValueError(f"audio map is missing case_id {case_id!r}") from error


def item_bounds(item: dict[str, Any]) -> tuple[int, int]:
    start_ms = item.get("start_ms")
    end_ms = item.get("end_ms")
    if not isinstance(start_ms, int) or not isinstance(end_ms, int):
        raise ValueError("review item start_ms and end_ms must be integers")
    if end_ms <= start_ms:
        raise ValueError("review item end_ms must be greater than start_ms")
    return start_ms, end_ms


def validate_review_pack_item_bounds(items: list[dict[str, Any]]) -> None:
    for item in items:
        start_ms, end_ms = item_bounds(item)
        item_review_clip_bounds(item, fallback_start_ms=start_ms, fallback_end_ms=end_ms)


def item_review_clip_bounds(
    item: dict[str, Any],
    *,
    fallback_start_ms: int,
    fallback_end_ms: int,
) -> tuple[int, int]:
    start_ms = item.get("review_clip_start_ms")
    end_ms = item.get("review_clip_end_ms")
    if start_ms is None and end_ms is None:
        return fallback_start_ms, fallback_end_ms
    if not isinstance(start_ms, int) or not isinstance(end_ms, int):
        raise ValueError("review item review_clip_start_ms and review_clip_end_ms must both be integers")
    if start_ms < fallback_start_ms or end_ms > fallback_end_ms or end_ms <= start_ms:
        raise ValueError("review item review clip bounds must stay inside start_ms/end_ms")
    return start_ms, end_ms


def review_clip_name(index: int, case_id: str | None, item: dict[str, Any]) -> str:
    reason = "unknown"
    reasons = item.get("reasons")
    if isinstance(reasons, list) and reasons and isinstance(reasons[0], str):
        reason = reasons[0]
    reference_id = item.get("reference_id") if isinstance(item.get("reference_id"), str) else "no-ref"
    candidate_id = item.get("candidate_id") if isinstance(item.get("candidate_id"), str) else "no-cand"
    parts = [f"{index:06d}", case_id or "single", reason, reference_id, candidate_id]
    return sanitize_clip_name("__".join(parts)) + ".wav"


def sanitize_clip_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return sanitized or "clip"
