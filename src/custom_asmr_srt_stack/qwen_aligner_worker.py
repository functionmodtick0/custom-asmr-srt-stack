from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import io
import json
import os
import socket
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import analyze_wav, slice_wav
from custom_asmr_srt_stack.models import MasterDocument, Segment, require_mapping, require_string

SOURCE_LANGUAGE_TO_QWEN = {
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "ja_jp": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
}
_NETWORK_DISABLED = False
QWEN_ASR_EXPECTED_VERSION = "0.0.6"
QWEN_ASR_EXPECTED_RECORD_SHA256 = "56454a099599cb3c86fd96347baa86269cc62e0d9eced004eeb2faa26b3a8a7c"
TRUE_ENV_VALUES = {"1", "true", "yes"}


class QwenAlignerRuntime:
    def __init__(self) -> None:
        self._loaded: dict[str, Any] = {}

    def load_aligner(self, model_id: str) -> Any:
        require_secure_runtime()
        disable_python_network_if_requested()
        checked_model_id = checked_model_path(model_id, "model_id")
        loaded = self._loaded.get(checked_model_id)
        if loaded is not None:
            return loaded

        log(f"loading local Qwen forced aligner: {checked_model_id}")
        verify_qwen_asr_package()
        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as error:
            raise ValueError(
                "Qwen aligner worker requires qwen-asr in an isolated environment. "
                "Use the qwen ASR venv and run this module through CASRT_ALIGNER_COMMAND."
            ) from error

        kwargs = apply_local_load_kwargs(default_load_kwargs(torch))
        aligner = Qwen3ForcedAligner.from_pretrained(checked_model_id, **kwargs)
        self._loaded[checked_model_id] = aligner
        log("aligner loaded")
        return aligner


def align_request(runtime: QwenAlignerRuntime, request: dict[str, Any], *, model_id: str) -> dict[str, Any]:
    audio_file = Path(require_string(request.get("audio_file"), "audio_file"))
    master = MasterDocument.from_json(require_mapping(request.get("master"), "master"))
    aligner = runtime.load_aligner(model_id)
    aligned = align_master(master, audio_file=audio_file, aligner=aligner)
    return {
        "segments": [
            {"id": segment.id, "start_ms": segment.start_ms, "end_ms": segment.end_ms}
            for segment in aligned.segments
        ]
    }


def align_master(master: MasterDocument, *, audio_file: Path, aligner: Any) -> MasterDocument:
    audio_bytes = audio_file.read_bytes()
    analyze_wav(audio_bytes)
    language = qwen_language(master.source_language)
    aligned_segments = list(master.segments)
    speech_indexes = [
        index
        for index, segment in enumerate(aligned_segments)
        if segment.kind == "speech" and segment.text.strip()
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths: list[str] = []
        texts: list[str] = []
        languages: list[str] = []
        for clip_index, segment_index in enumerate(speech_indexes):
            segment = aligned_segments[segment_index]
            clip_path = Path(tmpdir) / f"clip-{clip_index:06d}.wav"
            clip_path.write_bytes(slice_wav(audio_bytes, start_ms=segment.start_ms, end_ms=segment.end_ms))
            clip_paths.append(str(clip_path))
            texts.append(segment.text)
            languages.append(language)

        aligned_results: list[Any] = []
        for start in range(0, len(clip_paths), qwen_aligner_batch_size()):
            end = start + qwen_aligner_batch_size()
            aligned_results.extend(
                aligner.align(
                    audio=clip_paths[start:end],
                    text=texts[start:end],
                    language=languages[start:end],
                )
            )

    if len(aligned_results) != len(speech_indexes):
        raise ValueError(
            f"Qwen aligner returned {len(aligned_results)} results for {len(speech_indexes)} speech segments"
        )

    for index, result in zip(speech_indexes, aligned_results):
        segment = aligned_segments[index]
        duration_ms = segment.end_ms - segment.start_ms
        bounds = aligned_bounds_ms(result, duration_ms)
        if bounds is None:
            continue
        start_ms, end_ms = bounds
        aligned_segments[index] = replace(
            segment,
            start_ms=segment.start_ms + start_ms,
            end_ms=segment.start_ms + end_ms,
        )

    return replace(master, segments=tuple(aligned_segments))


def aligned_bounds_ms(result: Any, duration_ms: int) -> tuple[int, int] | None:
    items = getattr(result, "items", None)
    if not items:
        return None

    starts = [round(float(item.start_time) * 1000) for item in items if getattr(item, "start_time", None) is not None]
    ends = [round(float(item.end_time) * 1000) for item in items if getattr(item, "end_time", None) is not None]
    if not starts or not ends:
        return None

    start_ms = max(0, min(duration_ms, min(starts)))
    end_ms = max(start_ms + 1, min(duration_ms, max(ends)))
    if end_ms - start_ms < qwen_min_aligned_duration_ms():
        return None
    return start_ms, end_ms


def qwen_language(source_language: str) -> str:
    return SOURCE_LANGUAGE_TO_QWEN.get(source_language.strip().lower(), "Japanese")


def default_load_kwargs(torch_module: Any) -> dict[str, Any]:
    dtype = torch_dtype(torch_module, os.environ.get("CASRT_QWEN_ALIGNER_DTYPE", "bfloat16"))
    device_map = os.environ.get("CASRT_QWEN_ALIGNER_DEVICE_MAP")
    if device_map is None:
        device_map = "cuda:0" if torch_module.cuda.is_available() else ""

    result: dict[str, Any] = {"dtype": dtype}
    if device_map.strip():
        result["device_map"] = device_map.strip()
    return result


def apply_local_load_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    result = dict(kwargs)
    result["local_files_only"] = True
    result["trust_remote_code"] = False
    return result


def require_secure_runtime() -> None:
    if os.environ.get("CASRT_ALIGNER_ENV_MODE", "").strip().lower() != "offline":
        raise ValueError("CASRT_ALIGNER_ENV_MODE=offline is required for Qwen aligner worker")
    for name in (
        "CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH",
        "CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY",
        "CASRT_QWEN_ALIGNER_DISABLE_NETWORK",
    ):
        if os.environ.get(name, "").strip().lower() not in TRUE_ENV_VALUES:
            raise ValueError(f"{name}=1 is required for Qwen aligner worker")


def verify_qwen_asr_package() -> None:
    try:
        distribution = importlib_metadata.distribution("qwen-asr")
    except importlib_metadata.PackageNotFoundError as error:
        raise ValueError("qwen-asr package is not installed") from error

    if distribution.version != QWEN_ASR_EXPECTED_VERSION:
        raise ValueError(
            f"qwen-asr version {distribution.version} is not the pinned {QWEN_ASR_EXPECTED_VERSION}"
        )

    record_path = Path(distribution.locate_file(f"qwen_asr-{distribution.version}.dist-info/RECORD"))
    if not record_path.is_file():
        raise ValueError("qwen-asr RECORD file is missing")
    record_bytes = record_path.read_bytes()
    digest = hashlib.sha256(record_bytes).hexdigest()
    if digest != QWEN_ASR_EXPECTED_RECORD_SHA256:
        raise ValueError("qwen-asr RECORD hash does not match the pinned package fingerprint")
    verify_record_file_hashes(distribution, record_path)
    verify_qwen_asr_import_origin(distribution)


def verify_record_file_hashes(distribution: Any, record_path: Path) -> None:
    rows = csv.reader(io.StringIO(record_path.read_text(encoding="utf-8")))
    for row in rows:
        if len(row) < 3:
            raise ValueError("qwen-asr RECORD contains a malformed row")
        relative_path, hash_spec, raw_size = row[:3]
        if not hash_spec:
            continue
        algorithm, separator, expected_hash = hash_spec.partition("=")
        if separator != "=" or algorithm != "sha256" or not expected_hash:
            raise ValueError("qwen-asr RECORD contains an unsupported hash entry")
        installed_path = Path(distribution.locate_file(relative_path)).resolve(strict=True)
        file_bytes = installed_path.read_bytes()
        actual_hash = base64.urlsafe_b64encode(hashlib.sha256(file_bytes).digest()).decode("ascii").rstrip("=")
        if actual_hash != expected_hash:
            raise ValueError(f"qwen-asr installed file hash mismatch: {relative_path}")
        if raw_size:
            try:
                expected_size = int(raw_size)
            except ValueError as error:
                raise ValueError("qwen-asr RECORD contains a non-integer file size") from error
            if len(file_bytes) != expected_size:
                raise ValueError(f"qwen-asr installed file size mismatch: {relative_path}")


def verify_qwen_asr_import_origin(distribution: Any) -> None:
    expected_origin = Path(distribution.locate_file("qwen_asr/__init__.py")).resolve(strict=True)
    spec = importlib.util.find_spec("qwen_asr")
    if spec is None or spec.origin is None:
        raise ValueError("qwen_asr import spec is missing")
    actual_origin = Path(spec.origin).resolve(strict=True)
    if actual_origin != expected_origin:
        raise ValueError("qwen_asr import origin does not match the verified distribution")


def checked_model_path(value: str, name: str) -> str:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{name} must be an existing local model directory") from error
    if not path.is_dir():
        raise ValueError(f"{name} must be an existing local model directory")
    return str(path)


def disable_python_network_if_requested() -> None:
    global _NETWORK_DISABLED
    if _NETWORK_DISABLED:
        return
    if os.environ.get("CASRT_QWEN_ALIGNER_DISABLE_NETWORK", "").strip().lower() not in TRUE_ENV_VALUES:
        return

    original_socket = socket.socket

    class BlockedSocket(original_socket):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise OSError("network access is disabled for local Qwen aligner worker")

    def blocked_create_connection(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("network access is disabled for local Qwen aligner worker")

    socket.socket = BlockedSocket
    socket.create_connection = blocked_create_connection  # type: ignore[assignment]
    _NETWORK_DISABLED = True


def qwen_aligner_batch_size() -> int:
    value = int(os.environ.get("CASRT_QWEN_ALIGNER_BATCH_SIZE", "8"))
    if value <= 0:
        raise ValueError("CASRT_QWEN_ALIGNER_BATCH_SIZE must be positive")
    return value


def qwen_min_aligned_duration_ms() -> int:
    value = int(os.environ.get("CASRT_QWEN_ALIGNER_MIN_ALIGNED_DURATION_MS", "80"))
    if value < 0:
        raise ValueError("CASRT_QWEN_ALIGNER_MIN_ALIGNED_DURATION_MS must be non-negative")
    return value


def torch_dtype(torch_module: Any, value: str) -> Any:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError("CASRT_QWEN_ALIGNER_DTYPE must be bfloat16, float16, or float32")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align a casrt master document with Qwen3-ForcedAligner.")
    parser.add_argument("--model-id", default=os.environ.get("CASRT_QWEN_ALIGNER_MODEL_ID", ""))
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[casrt-qwen-aligner-worker] {message}", file=sys.stderr, flush=True)


def response_for_stdin(runtime: QwenAlignerRuntime, stdin_text: str, *, model_id: str) -> dict[str, Any]:
    try:
        require_secure_runtime()
        if not model_id:
            raise ValueError("--model-id or CASRT_QWEN_ALIGNER_MODEL_ID is required")
        request = json.loads(stdin_text)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        return align_request(runtime, request, model_id=model_id)
    except Exception as error:
        detail = str(error) or error.__class__.__name__
        return {"ok": False, "error": detail}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    response = response_for_stdin(QwenAlignerRuntime(), sys.stdin.read(), model_id=args.model_id)
    if response.get("ok") is False:
        print(response.get("error", "unknown aligner error"), file=sys.stderr)
        return 1
    print(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
