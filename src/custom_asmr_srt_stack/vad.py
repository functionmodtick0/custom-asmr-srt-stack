from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import analyze_wav
from custom_asmr_srt_stack.models import require_int, require_mapping

DEFAULT_VAD_TIMEOUT_SECONDS = 300.0


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
