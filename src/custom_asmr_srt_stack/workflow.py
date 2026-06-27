from __future__ import annotations

import os
import shlex
from dataclasses import replace
from typing import Any

from custom_asmr_srt_stack.alignment import apply_alignment_review_flags, run_alignment_command
from custom_asmr_srt_stack.audio import chunk_intervals, normalize_audio_to_wav, slice_wav, split_wav_channels
from custom_asmr_srt_stack.models import MasterDocument, Segment, make_segment_id, require_int, require_mapping
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.transcription import ModelEndpoint, adapter_max_chunk_ms, transcribe_audio


def analyze_project(store: ProjectStore, project_id: str) -> dict[str, Any]:
    audio_bytes, mime_type = store.read_audio(project_id)
    project = store.load_project(project_id)
    metadata = project_metadata(project)
    normalized_wav = normalize_audio_to_wav(
        audio_bytes,
        file_name=metadata.get("source_file"),
        mime_type=mime_type,
    )
    audio_info, channel_audio = split_wav_channels(normalized_wav)
    return store.save_audio_analysis(
        project_id,
        audio_info.to_json(),
        chunk_intervals(audio_info.duration_ms),
        channel_audio,
        normalized_wav,
    )


def project_metadata(project: dict[str, Any]) -> dict[str, Any]:
    metadata = project.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("project metadata must be an object")
    return metadata


def transcribe_project(
    store: ProjectStore,
    project_id: str,
    model_endpoint: ModelEndpoint,
    metadata: dict[str, Any],
    *,
    source_language: str,
    transcribe_audio_func=transcribe_audio,
) -> MasterDocument:
    channels = metadata.get("channels")
    channel_names = transcription_channel_names(channels, model_endpoint)
    if not channel_names:
        raise ValueError("project must be analyzed before transcription")

    chunks = transcription_chunks(metadata, model_endpoint)
    raw_segments: list[Segment] = []
    for channel in channel_names:
        if isinstance(channels, dict) and channel in channels:
            channel_audio = store.read_channel_audio(project_id, channel)
            mime_type = "audio/wav"
        else:
            channel_audio, mime_type = store.read_audio(project_id)

        for chunk in chunks:
            clip = slice_wav(channel_audio, start_ms=chunk["start_ms"], end_ms=chunk["end_ms"])
            raw_segments.extend(
                replace(
                    segment,
                    channel=channel,
                    start_ms=chunk["start_ms"] + segment.start_ms,
                    end_ms=chunk["start_ms"] + segment.end_ms,
                )
                for segment in transcribe_audio_func(
                    model_endpoint,
                    clip,
                    mime_type=mime_type,
                    channel=channel,
                    source_language=source_language,
                )
            )

    segments = tuple(
        replace(segment, id=make_segment_id(index + 1))
        for index, segment in enumerate(
            sorted(raw_segments, key=lambda segment: (segment.start_ms, segment.end_ms, segment.channel, segment.text))
        )
    )
    audio_info = metadata.get("audio_info")
    duration_ms = None
    if isinstance(audio_info, dict) and audio_info.get("duration_ms") is not None:
        duration_ms = int(audio_info["duration_ms"])
    master = apply_alignment_review_flags(
        MasterDocument(
            source_language=source_language,
            source_file=metadata.get("source_file"),
            duration_ms=duration_ms,
            segments=segments,
        )
    )
    aligner_command = os.environ.get("CASRT_ALIGNER_COMMAND")
    if not aligner_command:
        return master

    normalized_audio_file = metadata.get("normalized_audio_file")
    if not isinstance(normalized_audio_file, str):
        raise ValueError("alignment requires analyzed audio with normalized_audio_file")
    return run_alignment_command(
        master,
        audio_file=store.require_project_root(project_id) / normalized_audio_file,
        command=shlex.split(aligner_command),
    )


def analysis_chunks(metadata: dict[str, Any]) -> tuple[dict[str, int], ...]:
    raw_chunks = metadata.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError("project must be analyzed before transcription")

    chunks: list[dict[str, int]] = []
    for raw_chunk in raw_chunks:
        chunk = require_mapping(raw_chunk, "metadata.chunks[]")
        start_ms = require_int(chunk.get("start_ms"), "metadata.chunks[].start_ms")
        end_ms = require_int(chunk.get("end_ms"), "metadata.chunks[].end_ms")
        if start_ms < 0:
            raise ValueError("metadata.chunks[].start_ms must be non-negative")
        if end_ms <= start_ms:
            raise ValueError("metadata.chunks[].end_ms must be greater than start_ms")
        chunks.append({"start_ms": start_ms, "end_ms": end_ms})
    return tuple(chunks)


def transcription_chunks(metadata: dict[str, Any], model_endpoint: ModelEndpoint) -> tuple[dict[str, int], ...]:
    chunks = analysis_chunks(metadata)
    max_chunk_ms = adapter_max_chunk_ms(model_endpoint)
    if max_chunk_ms is None:
        return chunks
    split_chunks: list[dict[str, int]] = []
    for chunk in chunks:
        split_chunks.extend(split_interval(chunk["start_ms"], chunk["end_ms"], max_chunk_ms))
    return tuple(split_chunks)


def transcription_channel_names(channels: Any, model_endpoint: ModelEndpoint) -> list[str]:
    if not isinstance(channels, dict):
        return []
    if model_endpoint.adapter == "local-qwen-asr" and "MIX" in channels:
        return ["MIX"]
    if {"L", "R"}.issubset(channels):
        return ["L", "R"]
    if "MIX" in channels:
        return ["MIX"]
    return []


def split_interval(start_ms: int, end_ms: int, max_chunk_ms: int) -> tuple[dict[str, int], ...]:
    if max_chunk_ms <= 0:
        raise ValueError("max_chunk_ms must be positive")
    if end_ms <= start_ms:
        raise ValueError("interval end_ms must be greater than start_ms")

    chunks: list[dict[str, int]] = []
    current = start_ms
    while current < end_ms:
        next_end = min(end_ms, current + max_chunk_ms)
        chunks.append({"start_ms": current, "end_ms": next_end})
        current = next_end
    return tuple(chunks)


def retranscribe_segment(
    store: ProjectStore,
    project_id: str,
    master: MasterDocument,
    metadata: dict[str, Any],
    *,
    segment_id: str,
    model_endpoint: ModelEndpoint,
    source_language: str,
    transcribe_audio_func=transcribe_audio,
) -> MasterDocument:
    target = next((segment for segment in master.segments if segment.id == segment_id), None)
    if target is None:
        raise ValueError("segment not found")

    channels = metadata.get("channels")
    if isinstance(channels, dict) and target.channel in channels:
        channel_audio = store.read_channel_audio(project_id, target.channel)
        mime_type = "audio/wav"
    else:
        audio_bytes, mime_type = store.read_audio(project_id)
        channel_audio = normalize_audio_to_wav(
            audio_bytes,
            file_name=metadata.get("source_file"),
            mime_type=mime_type,
        )
        mime_type = "audio/wav"

    intervals = retranscription_intervals(target, model_endpoint)
    replacement_segments: list[Segment] = []
    for interval in intervals:
        clip = slice_wav(channel_audio, start_ms=interval["start_ms"], end_ms=interval["end_ms"])
        replacement_segments.extend(
            replace(
                segment,
                channel=target.channel,
                start_ms=interval["start_ms"] + segment.start_ms,
                end_ms=interval["start_ms"] + segment.end_ms,
            )
            for segment in transcribe_audio_func(
                model_endpoint,
                clip,
                mime_type=mime_type,
                channel=target.channel,
                source_language=source_language,
            )
        )
    replacement = tuple(replacement_segments)
    if not replacement:
        raise ValueError("retranscription returned no segments")

    merged: list[Segment] = []
    for segment in master.segments:
        if segment.id == segment_id:
            merged.extend(replacement)
        else:
            merged.append(segment)

    return apply_alignment_review_flags(
        replace(
            master,
            segments=tuple(
                replace(segment, id=make_segment_id(index + 1))
                for index, segment in enumerate(
                    sorted(merged, key=lambda segment: (segment.start_ms, segment.end_ms, segment.channel, segment.text))
                )
            ),
        )
    )


def retranscription_intervals(target: Segment, model_endpoint: ModelEndpoint) -> tuple[dict[str, int], ...]:
    max_chunk_ms = adapter_max_chunk_ms(model_endpoint)
    if max_chunk_ms is None:
        return ({"start_ms": target.start_ms, "end_ms": target.end_ms},)
    return split_interval(target.start_ms, target.end_ms, max_chunk_ms)
