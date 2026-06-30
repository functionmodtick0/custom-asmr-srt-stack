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
