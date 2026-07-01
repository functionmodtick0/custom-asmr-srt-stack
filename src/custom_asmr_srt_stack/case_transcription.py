from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.case_batch import (
    case_file_stem,
    load_review_case_index,
    require_index_string,
    resolve_plan_path,
)
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.transcription import ModelEndpoint, transcribe_audio
from custom_asmr_srt_stack.workflow import analyze_project, transcribe_project

CASE_CANDIDATE_TRANSCRIPTION_FORMAT = "custom-asmr-case-candidate-transcription-v1"


def transcribe_review_case_candidates(
    case_index_file: Path,
    *,
    output_dir: Path,
    model_endpoint: ModelEndpoint,
    project_root: Path | None = None,
    source_language: str = "ja",
    transcribe_audio_func=transcribe_audio,
) -> dict[str, Any]:
    case_index = load_review_case_index(case_index_file)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("review case index items must be a non-empty array")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("candidate output directory must be empty")

    base_dir = case_index_file.parent
    prepared_items: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        case_id = require_index_string(raw_item, "id", index)
        if case_id in seen_case_ids:
            raise ValueError(f"review case index item id is duplicated: {case_id}")
        seen_case_ids.add(case_id)
        audio_value = require_index_string(raw_item, "audio", index)
        audio_path = resolve_plan_path(base_dir, audio_value)
        if not audio_path.is_file():
            raise ValueError(f"review case audio file does not exist for {case_id}: {audio_path}")
        prepared_items.append(
            {
                "case_id": case_id,
                "audio_value": audio_value,
                "audio_path": audio_path,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_project_root = project_root or output_dir / "projects"
    store = ProjectStore(selected_project_root)
    result_items: list[dict[str, Any]] = []
    for item in prepared_items:
        audio_path = item["audio_path"]
        audio_bytes = audio_path.read_bytes()
        created = store.create_from_audio(
            audio_path.name,
            mimetypes.guess_type(audio_path.name)[0] or "audio/wav",
            base64.b64encode(audio_bytes).decode("ascii"),
        )
        project_id = created["project_id"]
        analyzed = analyze_project(store, project_id)
        metadata = analyzed["metadata"]
        master = transcribe_project(
            store,
            project_id,
            model_endpoint,
            metadata,
            source_language=source_language,
            transcribe_audio_func=transcribe_audio_func,
        )
        store.save_master(project_id, master)
        candidate_relative = f"{case_file_stem(item['case_id'])}.master.json"
        candidate_output = output_dir / candidate_relative
        candidate_output.write_text(json.dumps(master.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result_items.append(
            {
                "case_id": item["case_id"],
                "audio": item["audio_value"],
                "candidate": candidate_relative,
                "project_id": project_id,
                "segments": len(master.segments),
                "review_count": sum(1 for segment in master.segments if segment.needs_review),
            }
        )

    return {
        "format": CASE_CANDIDATE_TRANSCRIPTION_FORMAT,
        "case_index": str(case_index_file),
        "output": str(output_dir),
        "project_root": str(selected_project_root),
        "adapter": model_endpoint.adapter,
        "model_id": model_endpoint.model_id,
        "candidate_count": len(result_items),
        "items": result_items,
    }
