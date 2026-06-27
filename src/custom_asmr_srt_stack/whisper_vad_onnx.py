from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


TARGET_SAMPLE_RATE = 16_000
DEFAULT_FRAME_MS = 20
DEFAULT_CHUNK_MS = 30_000
EXPECTED_INPUT_SHAPE = [1, 80, 3000]
EXPECTED_OUTPUT_SHAPE = [1, 1500]
EXPECTED_METADATA = {
    "whisper_model_name": "openai/whisper-base",
    "frame_duration_ms": DEFAULT_FRAME_MS,
    "total_duration_ms": DEFAULT_CHUNK_MS,
    "input_shape": EXPECTED_INPUT_SHAPE,
    "output_shape": EXPECTED_OUTPUT_SHAPE,
    "export_batch_size": 1,
}


@dataclass(frozen=True)
class WhisperVadOnnxSettings:
    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech_ms: int = 250
    min_silence_ms: int = 100
    pad_ms: int = 30
    output_activation: str = "sigmoid"
    force_cpu: bool = True
    num_threads: int = 1


def detect_command_intervals(
    *,
    audio_file: Path,
    model: Path,
    metadata: Path | None = None,
    settings: WhisperVadOnnxSettings | None = None,
) -> tuple[dict[str, int], ...]:
    settings = settings or WhisperVadOnnxSettings()
    validate_settings(settings)
    audio = load_audio_mono_16k(audio_file)
    duration_ms = round(len(audio) / TARGET_SAMPLE_RATE * 1000)
    probabilities = run_onnx_frame_probabilities(
        audio,
        model=model,
        metadata=metadata,
        settings=settings,
    )
    return probabilities_to_intervals(
        probabilities,
        duration_ms=duration_ms,
        frame_ms=DEFAULT_FRAME_MS,
        threshold=settings.threshold,
        neg_threshold=settings.neg_threshold,
        min_speech_ms=settings.min_speech_ms,
        min_silence_ms=settings.min_silence_ms,
        pad_ms=settings.pad_ms,
    )


def load_audio_mono_16k(audio_file: Path) -> Any:
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError as error:
        raise ValueError(
            "whisper ASMR ONNX VAD requires optional local dependencies: "
            "librosa, numpy, soundfile"
        ) from error

    audio, sample_rate = sf.read(str(audio_file), dtype="float32", always_2d=True)
    if audio.size == 0:
        return np.array([], dtype=np.float32)
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return mono.astype(np.float32, copy=False)


def run_onnx_frame_probabilities(
    audio: Any,
    *,
    model: Path,
    metadata: Path | None,
    settings: WhisperVadOnnxSettings,
) -> list[float]:
    try:
        import numpy as np
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
    except ImportError as error:
        raise ValueError(
            "whisper ASMR ONNX VAD requires optional local dependencies: "
            "numpy, onnxruntime, transformers"
        ) from error

    if not model.exists():
        raise ValueError(f"ONNX VAD model does not exist: {model}")
    metadata_path = metadata or model.with_name("model_metadata.json")
    validate_model_files(model, metadata_path)
    model_metadata = load_metadata(metadata_path)
    validate_model_metadata(model_metadata)
    frame_ms = require_positive_int(model_metadata.get("frame_duration_ms", DEFAULT_FRAME_MS), "frame_duration_ms")
    chunk_ms = require_positive_int(model_metadata.get("total_duration_ms", DEFAULT_CHUNK_MS), "total_duration_ms")
    chunk_samples = round(TARGET_SAMPLE_RATE * (chunk_ms / 1000))
    frames_per_chunk = round(chunk_ms / frame_ms)
    if chunk_samples <= 0 or frames_per_chunk <= 0:
        raise ValueError("ONNX VAD metadata produced an invalid chunk shape")

    providers = ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.inter_op_num_threads = settings.num_threads
    options.intra_op_num_threads = settings.num_threads
    session = ort.InferenceSession(str(model), providers=providers, sess_options=options)
    validate_session_contract(session)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    feature_extractor = WhisperFeatureExtractor()

    probabilities: list[float] = []
    total_frames = math.ceil(len(audio) / TARGET_SAMPLE_RATE * 1000 / frame_ms) if len(audio) else 0
    for offset in range(0, len(audio), chunk_samples):
        chunk = audio[offset : offset + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")
        inputs = feature_extractor(chunk, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="np")
        raw_output = session.run(output_names, {input_name: inputs.input_features})[0]
        frame_values = np.asarray(raw_output[0], dtype=np.float32)[:frames_per_chunk]
        probabilities.extend(activate_outputs(frame_values, settings.output_activation))
    return probabilities[:total_frames]


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"ONNX VAD metadata does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"ONNX VAD metadata is invalid JSON: {path}") from error
    if not isinstance(data, dict):
        raise ValueError("ONNX VAD metadata must be an object")
    return data


def validate_model_files(model: Path, metadata: Path) -> None:
    if model.name != "model.onnx":
        raise ValueError("ONNX VAD model file must be named model.onnx")
    if metadata.name != "model_metadata.json":
        raise ValueError("ONNX VAD metadata file must be named model_metadata.json")
    model_path = model.resolve()
    metadata_path = metadata.resolve()
    if model_path.parent != metadata_path.parent:
        raise ValueError("ONNX VAD model and metadata must be in the same directory")
    if not model_path.parent.exists():
        raise ValueError(f"ONNX VAD model directory does not exist: {model_path.parent}")
    allowed = {model_path, metadata_path}
    unexpected = sorted(
        path.name
        for path in model_path.parent.iterdir()
        if path.resolve() not in allowed
    )
    if unexpected:
        raise ValueError("ONNX VAD model directory contains unexpected files: " + ", ".join(unexpected))


def validate_model_metadata(data: dict[str, Any]) -> None:
    for name, expected in EXPECTED_METADATA.items():
        if data.get(name) != expected:
            raise ValueError(f"ONNX VAD metadata {name} must be {expected!r}")
    opset_version = data.get("opset_version")
    if opset_version is not None and opset_version != 17:
        raise ValueError("ONNX VAD metadata opset_version must be 17")


def validate_session_contract(session: Any) -> None:
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("ONNX VAD must use CPUExecutionProvider only")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError("ONNX VAD model must have exactly one input")
    if len(outputs) != 1:
        raise ValueError("ONNX VAD model must have exactly one output")
    input_shape = list(inputs[0].shape)
    if len(input_shape) != 3 or input_shape[1:] != EXPECTED_INPUT_SHAPE[1:] or input_shape[0] not in {1, "s6"}:
        raise ValueError(f"ONNX VAD input shape must be {EXPECTED_INPUT_SHAPE!r}")
    if list(outputs[0].shape) != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(f"ONNX VAD output shape must be {EXPECTED_OUTPUT_SHAPE!r}")


def require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"ONNX VAD metadata {name} must be an integer")
    if value <= 0:
        raise ValueError(f"ONNX VAD metadata {name} must be positive")
    return value


def activate_outputs(values: Any, activation: str) -> list[float]:
    if activation == "identity":
        return [clamp_probability(float(value)) for value in flatten_values(values)]
    elif activation == "sigmoid":
        return [clamp_probability(sigmoid(float(value))) for value in flatten_values(values)]
    raise ValueError("output activation must be one of: sigmoid, identity")


def flatten_values(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list):
        return values
    return list(values)


def sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1 / (1 + exponent)
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def probabilities_to_intervals(
    probabilities: Sequence[float],
    *,
    duration_ms: int,
    frame_ms: int = DEFAULT_FRAME_MS,
    threshold: float = 0.5,
    neg_threshold: float | None = None,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    pad_ms: int = 30,
) -> tuple[dict[str, int], ...]:
    validate_postprocess_args(
        duration_ms=duration_ms,
        frame_ms=frame_ms,
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        pad_ms=pad_ms,
    )
    if not probabilities or duration_ms == 0:
        return ()
    off_threshold = neg_threshold if neg_threshold is not None else max(threshold - 0.15, 0.01)
    min_speech_frames = max(1, math.ceil(min_speech_ms / frame_ms))
    min_silence_frames = max(1, math.ceil(min_silence_ms / frame_ms))

    active = False
    start_frame = 0
    silence_start: int | None = None
    raw_ranges: list[tuple[int, int]] = []
    for index, probability in enumerate(probabilities):
        if probability < 0 or probability > 1:
            raise ValueError("VAD probabilities must be between 0 and 1")
        if not active:
            if probability >= threshold:
                active = True
                start_frame = index
                silence_start = None
            continue

        if probability < off_threshold:
            if silence_start is None:
                silence_start = index
            if index - silence_start + 1 >= min_silence_frames:
                if silence_start - start_frame >= min_speech_frames:
                    raw_ranges.append((start_frame, silence_start))
                active = False
                silence_start = None
        elif probability >= threshold:
            silence_start = None

    if active and len(probabilities) - start_frame >= min_speech_frames:
        raw_ranges.append((start_frame, len(probabilities)))

    padded: list[dict[str, int]] = []
    for start, end in raw_ranges:
        start_ms = max(0, (start * frame_ms) - pad_ms)
        end_ms = min(duration_ms, (end * frame_ms) + pad_ms)
        if end_ms <= start_ms:
            continue
        if padded and start_ms <= padded[-1]["end_ms"]:
            padded[-1]["end_ms"] = max(padded[-1]["end_ms"], end_ms)
        else:
            padded.append({"index": len(padded), "start_ms": start_ms, "end_ms": end_ms})
    return tuple(padded)


def add_rescue_intervals(
    base_intervals: Sequence[dict[str, int]],
    rescue_intervals: Sequence[dict[str, int]],
    *,
    min_rescue_ms: int,
) -> tuple[dict[str, int], ...]:
    if min_rescue_ms < 0:
        raise ValueError("min_rescue_ms must be non-negative")
    base_ranges = [(item["start_ms"], item["end_ms"]) for item in base_intervals]
    rescue_ranges = [(item["start_ms"], item["end_ms"]) for item in rescue_intervals]
    extras: list[tuple[int, int]] = []
    for rescue_range in merge_ranges(rescue_ranges):
        for gap in subtract_ranges(rescue_range, base_ranges):
            if gap[1] - gap[0] >= min_rescue_ms:
                extras.append(gap)
    return indexed_intervals(sorted(base_ranges + extras))


def subtract_ranges(target: tuple[int, int], blockers: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    pieces = [target]
    for blocker_start, blocker_end in merge_ranges(blockers):
        next_pieces: list[tuple[int, int]] = []
        for start, end in pieces:
            if blocker_end <= start or blocker_start >= end:
                next_pieces.append((start, end))
                continue
            if blocker_start > start:
                next_pieces.append((start, blocker_start))
            if blocker_end < end:
                next_pieces.append((blocker_end, end))
        pieces = next_pieces
    return pieces


def merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def indexed_intervals(ranges: Sequence[tuple[int, int]]) -> tuple[dict[str, int], ...]:
    return tuple({"index": index, "start_ms": start, "end_ms": end} for index, (start, end) in enumerate(ranges))


def validate_postprocess_args(
    *,
    duration_ms: int,
    frame_ms: int,
    threshold: float,
    neg_threshold: float | None,
    min_speech_ms: int,
    min_silence_ms: int,
    pad_ms: int,
) -> None:
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")
    if neg_threshold is not None and (neg_threshold < 0 or neg_threshold > threshold):
        raise ValueError("neg_threshold must be between 0 and threshold")
    if min_speech_ms < 0:
        raise ValueError("min_speech_ms must be non-negative")
    if min_silence_ms < 0:
        raise ValueError("min_silence_ms must be non-negative")
    if pad_ms < 0:
        raise ValueError("pad_ms must be non-negative")


def validate_settings(settings: WhisperVadOnnxSettings) -> None:
    validate_postprocess_args(
        duration_ms=0,
        frame_ms=DEFAULT_FRAME_MS,
        threshold=settings.threshold,
        neg_threshold=settings.neg_threshold,
        min_speech_ms=settings.min_speech_ms,
        min_silence_ms=settings.min_silence_ms,
        pad_ms=settings.pad_ms,
    )
    if settings.output_activation not in {"sigmoid", "identity"}:
        raise ValueError("output activation must be one of: sigmoid, identity")
    if not settings.force_cpu:
        raise ValueError("ONNX VAD must run with force_cpu enabled")
    if settings.num_threads <= 0:
        raise ValueError("num_threads must be positive")
