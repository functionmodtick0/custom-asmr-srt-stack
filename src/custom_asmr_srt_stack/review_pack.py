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
    context_ms: int = DEFAULT_REVIEW_CONTEXT_MS,
) -> dict[str, Any]:
    if context_ms < 0:
        raise ValueError("context_ms must be non-negative")
    if audio_file is None and audio_map_file is None:
        raise ValueError("review pack requires --audio or --audio-map")
    if audio_file is not None and audio_map_file is not None:
        raise ValueError("review pack accepts only one of --audio or --audio-map")

    items = review_effort_items(review_effort_report)
    if audio_file is not None:
        validate_single_audio_scope(items)
    audio_by_case = load_audio_by_case(audio_file=audio_file, audio_map_file=audio_map_file)
    prepare_output_dir(output_dir)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir()

    audio_cache: dict[str, tuple[bytes, int]] = {}
    packed_items = []
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
        clip_start_ms = max(0, start_ms - context_ms)
        clip_end_ms = min(duration_ms, end_ms + context_ms)
        if clip_end_ms <= clip_start_ms:
            raise ValueError(f"review item {index} selects an empty audio range")
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
        packed_items.append(packed_item)

    result = {
        "format": REVIEW_PACK_FORMAT,
        "source_report": review_effort_report.get("source_report"),
        "clip_count": len(packed_items),
        "items": packed_items,
    }
    (output_dir / "index.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


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


def validate_single_audio_scope(items: list[dict[str, Any]]) -> None:
    case_ids = {item_case_id(item) for item in items}
    case_ids.discard(None)
    if len(case_ids) > 1:
        raise ValueError("review pack with multiple case_id values requires --audio-map")


def load_audio_by_case(*, audio_file: Path | None, audio_map_file: Path | None) -> dict[str | None, Path]:
    if audio_file is not None:
        return {None: audio_file}
    assert audio_map_file is not None
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
