from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.models import MasterDocument, Segment, require_int, require_mapping, require_string

DEFAULT_LONG_SEGMENT_MS = 30_000
ALIGNER_ENV_ALLOWLIST = {
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_HOME",
    "TRANSFORMERS_CACHE",
    "USER",
    "VIRTUAL_ENV",
    "XDG_CACHE_HOME",
}
ALIGNER_ENV_PREFIXES = ("CASRT_ALIGNER_", "CASRT_QWEN_ALIGNER_")
ALIGNER_ENV_BLOCKLIST = {"CASRT_ALIGNER_COMMAND"}
ALIGNER_OFFLINE_ENV = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
SENSITIVE_ENV_SUBSTRINGS = ("API_KEY", "AUTH", "PASSWORD", "SECRET", "TOKEN")


def apply_alignment_review_flags(
    master: MasterDocument,
    *,
    long_segment_ms: int = DEFAULT_LONG_SEGMENT_MS,
) -> MasterDocument:
    if long_segment_ms <= 0:
        raise ValueError("long_segment_ms must be positive")
    return replace(
        master,
        segments=tuple(
            replace(segment, needs_review=segment.needs_review or segment_needs_review(segment, long_segment_ms))
            for segment in master.segments
        ),
    )


def segment_needs_review(segment: Segment, long_segment_ms: int) -> bool:
    if segment.kind == "speech" and not segment.text.strip():
        return True
    return (segment.end_ms - segment.start_ms) > long_segment_ms


def merge_alignment_output(master: MasterDocument, value: Any) -> MasterDocument:
    data = require_mapping(value, "alignment output")
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("alignment output segments must be an array")

    timing_by_id: dict[str, tuple[int, int]] = {}
    for raw_segment in raw_segments:
        segment_data = require_mapping(raw_segment, "alignment segment")
        segment_id = require_string(segment_data.get("id"), "alignment segment.id")
        if segment_id in timing_by_id:
            raise ValueError(f"duplicate aligned segment id {segment_id!r}")
        timing_by_id[segment_id] = (
            require_int(segment_data.get("start_ms"), "alignment segment.start_ms"),
            require_int(segment_data.get("end_ms"), "alignment segment.end_ms"),
        )

    master_ids = {segment.id for segment in master.segments}
    aligned_ids = set(timing_by_id)
    missing_ids = sorted(master_ids - aligned_ids)
    unknown_ids = sorted(aligned_ids - master_ids)
    if missing_ids:
        raise ValueError(f"alignment output is missing ids: {', '.join(missing_ids)}")
    if unknown_ids:
        raise ValueError(f"alignment output contains unknown ids: {', '.join(unknown_ids)}")

    return apply_alignment_review_flags(
        replace(
            master,
            segments=tuple(
                replace(segment, start_ms=timing_by_id[segment.id][0], end_ms=timing_by_id[segment.id][1])
                for segment in master.segments
            ),
        )
    )


def run_alignment_command(master: MasterDocument, *, audio_file: Path, command: list[str]) -> MasterDocument:
    if not command:
        raise ValueError("alignment command must not be empty")
    if not audio_file.exists():
        raise ValueError("alignment audio file is missing")
    request = {
        "audio_file": str(audio_file),
        "master": master.to_json(),
    }
    result = subprocess.run(
        command,
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
        env=aligner_env(),
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown aligner error"
        raise ValueError(f"alignment command failed: {detail}")
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"alignment command returned invalid JSON: {error}") from error
    return merge_alignment_output(master, output)


def alignment_diagnostics(
    original: MasterDocument,
    aligned: MasterDocument,
    *,
    audio_file: Path,
    input_file: Path,
    output_file: Path,
) -> dict[str, Any]:
    aligned_by_id = {segment.id: segment for segment in aligned.segments}
    items = []
    changed_segments = 0
    review_flag_changes = 0
    max_boundary_delta_ms = 0
    boundary_count = 0
    total_abs_boundary_delta_ms = 0
    within_250ms_boundary_count = 0
    within_500ms_boundary_count = 0
    for segment in original.segments:
        aligned_segment = aligned_by_id.get(segment.id)
        if aligned_segment is None:
            raise ValueError(f"aligned master is missing segment id {segment.id!r}")
        start_delta = aligned_segment.start_ms - segment.start_ms
        end_delta = aligned_segment.end_ms - segment.end_ms
        duration_delta = (aligned_segment.end_ms - aligned_segment.start_ms) - (segment.end_ms - segment.start_ms)
        changed = start_delta != 0 or end_delta != 0
        review_flag_changed = aligned_segment.needs_review != segment.needs_review
        if changed:
            changed_segments += 1
        if review_flag_changed:
            review_flag_changes += 1
        max_boundary_delta_ms = max(max_boundary_delta_ms, abs(start_delta), abs(end_delta))
        for delta in (start_delta, end_delta):
            abs_delta = abs(delta)
            boundary_count += 1
            total_abs_boundary_delta_ms += abs_delta
            if abs_delta <= 250:
                within_250ms_boundary_count += 1
            if abs_delta <= 500:
                within_500ms_boundary_count += 1
        items.append(
            {
                "id": segment.id,
                "kind": segment.kind,
                "channel": segment.channel,
                "text": segment.text,
                "original_start_ms": segment.start_ms,
                "original_end_ms": segment.end_ms,
                "aligned_start_ms": aligned_segment.start_ms,
                "aligned_end_ms": aligned_segment.end_ms,
                "start_delta_ms": start_delta,
                "end_delta_ms": end_delta,
                "duration_delta_ms": duration_delta,
                "changed": changed,
                "needs_review_before": segment.needs_review,
                "needs_review_after": aligned_segment.needs_review,
                "review_flag_changed": review_flag_changed,
            }
        )

    unknown_ids = sorted({segment.id for segment in aligned.segments} - {segment.id for segment in original.segments})
    if unknown_ids:
        raise ValueError(f"aligned master contains unknown segment ids: {', '.join(unknown_ids)}")

    mean_abs_boundary_delta_ms = None if boundary_count == 0 else total_abs_boundary_delta_ms / boundary_count
    within_250ms_boundary_ratio = None if boundary_count == 0 else within_250ms_boundary_count / boundary_count
    within_500ms_boundary_ratio = None if boundary_count == 0 else within_500ms_boundary_count / boundary_count
    return {
        "format": "custom-asmr-alignment-diagnostics-v1",
        "audio": str(audio_file),
        "input": str(input_file),
        "output": str(output_file),
        "segments": len(original.segments),
        "changed_segments": changed_segments,
        "review_flag_changes": review_flag_changes,
        "max_boundary_delta_ms": max_boundary_delta_ms,
        "boundary_count": boundary_count,
        "mean_abs_boundary_delta_ms": mean_abs_boundary_delta_ms,
        "within_250ms_boundary_count": within_250ms_boundary_count,
        "within_250ms_boundary_ratio": within_250ms_boundary_ratio,
        "within_500ms_boundary_count": within_500ms_boundary_count,
        "within_500ms_boundary_ratio": within_500ms_boundary_ratio,
        "items": items,
    }


def aligner_env() -> dict[str, str] | None:
    mode = os.environ.get("CASRT_ALIGNER_ENV_MODE", "inherit").strip().lower()
    if mode in {"", "inherit"}:
        return None
    if mode != "offline":
        raise ValueError("CASRT_ALIGNER_ENV_MODE must be inherit or offline")

    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in ALIGNER_ENV_ALLOWLIST or name.startswith(ALIGNER_ENV_PREFIXES):
            env[name] = value

    for name in list(env):
        if name in ALIGNER_ENV_BLOCKLIST:
            env.pop(name, None)
            continue
        if name == "TOKENIZERS_PARALLELISM":
            continue
        if any(part in name for part in SENSITIVE_ENV_SUBSTRINGS):
            env.pop(name, None)

    env.setdefault("PATH", os.defpath)
    env.update(ALIGNER_OFFLINE_ENV)
    return env
