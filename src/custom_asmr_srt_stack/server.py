from __future__ import annotations

import json
import os
import shlex
from dataclasses import replace
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.alignment import apply_alignment_review_flags, run_alignment_command
from custom_asmr_srt_stack.audio import chunk_intervals, normalize_audio_to_wav, slice_wav, split_wav_channels
from custom_asmr_srt_stack.models import MasterDocument, Segment, make_segment_id, require_mapping, require_string
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.srt import format_srt, parse_srt
from custom_asmr_srt_stack.translation import export_translation_json, parse_translated_texts
from custom_asmr_srt_stack.transcription import ModelEndpoint, transcribe_audio

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"


def json_response(status: HTTPStatus, payload: dict[str, Any]) -> tuple[int, str, bytes]:
    return status.value, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")


def handle_api_request(
    path: str,
    raw_body: bytes,
    *,
    project_store: ProjectStore | None = None,
    transcribe_audio_func=transcribe_audio,
) -> tuple[int, str, bytes]:
    store = project_store or ProjectStore.default()
    try:
        payload = json.loads(raw_body.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")

        if path == "/api/srt-to-json":
            content = payload.get("content")
            if not isinstance(content, str):
                raise ValueError("content must be a string")
            master = parse_srt(
                content,
                source_language=str(payload.get("source_language") or "ja"),
                source_file=payload.get("source_file"),
            )
            return json_response(HTTPStatus.OK, master.to_json())

        if path == "/api/projects/import-srt":
            content = payload.get("content")
            if not isinstance(content, str):
                raise ValueError("content must be a string")
            master = parse_srt(
                content,
                source_language=str(payload.get("source_language") or "ja"),
                source_file=payload.get("source_file"),
            )
            return json_response(HTTPStatus.OK, store.create_from_master(master))

        if path == "/api/projects/import-master-json":
            master = MasterDocument.from_json(payload.get("master"))
            return json_response(HTTPStatus.OK, store.create_from_master(master))

        if path == "/api/projects/upload-audio":
            return json_response(
                HTTPStatus.OK,
                store.create_from_audio(
                    file_name=str(payload.get("file_name") or ""),
                    mime_type=str(payload.get("mime_type") or ""),
                    content_base64=str(payload.get("content_base64") or ""),
                ),
            )

        if path == "/api/projects/save-master":
            project_id = str(payload.get("project_id") or "")
            master = MasterDocument.from_json(payload.get("master"))
            return json_response(HTTPStatus.OK, store.save_master(project_id, master))

        if path == "/api/projects/load":
            project_id = str(payload.get("project_id") or "")
            return json_response(HTTPStatus.OK, store.load_project(project_id))

        if path == "/api/projects/analyze-audio":
            project_id = str(payload.get("project_id") or "")
            audio_bytes, mime_type = store.read_audio(project_id)
            project = store.load_project(project_id)
            metadata = require_mapping(project.get("metadata"), "metadata")
            normalized_wav = normalize_audio_to_wav(
                audio_bytes,
                file_name=metadata.get("source_file"),
                mime_type=mime_type,
            )
            audio_info, channel_audio = split_wav_channels(normalized_wav)
            return json_response(
                HTTPStatus.OK,
                store.save_audio_analysis(
                    project_id,
                    audio_info.to_json(),
                    chunk_intervals(audio_info.duration_ms),
                    channel_audio,
                    normalized_wav,
                ),
            )

        if path == "/api/projects/transcribe":
            project_id = str(payload.get("project_id") or "")
            model_endpoint = ModelEndpoint.from_json(payload.get("model"))
            source_language = str(payload.get("source_language") or "ja")
            project = store.load_project(project_id)
            metadata = require_mapping(project.get("metadata"), "metadata")
            master = transcribe_project(
                store,
                project_id,
                model_endpoint,
                metadata,
                source_language=source_language,
                transcribe_audio_func=transcribe_audio_func,
            )
            return json_response(HTTPStatus.OK, store.save_master(project_id, master))

        if path == "/api/projects/retranscribe-segment":
            project_id = str(payload.get("project_id") or "")
            segment_id = str(payload.get("segment_id") or "")
            model_endpoint = ModelEndpoint.from_json(payload.get("model"))
            source_language = str(payload.get("source_language") or "ja")
            project = store.load_project(project_id)
            metadata = require_mapping(project.get("metadata"), "metadata")
            master = MasterDocument.from_json(project.get("master"))
            updated = retranscribe_segment(
                store,
                project_id,
                master,
                metadata,
                segment_id=segment_id,
                model_endpoint=model_endpoint,
                source_language=source_language,
                transcribe_audio_func=transcribe_audio_func,
            )
            return json_response(HTTPStatus.OK, store.save_master(project_id, updated))

        if path == "/api/export-translation-json":
            master = MasterDocument.from_json(payload.get("master"))
            return json_response(HTTPStatus.OK, export_translation_json(master))

        if path == "/api/json-to-srt":
            master = MasterDocument.from_json(payload.get("master"))
            translated = payload.get("translated")
            text_by_id = None if translated is None else parse_translated_texts(master, translated)
            return json_response(HTTPStatus.OK, {"content": format_srt(master, text_by_id=text_by_id)})

        if path == "/api/model/validate":
            endpoint = ModelEndpoint.from_json(payload.get("model"))
            return json_response(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "adapter": endpoint.adapter,
                    "model_id": endpoint.model_id,
                },
            )

        return json_response(HTTPStatus.NOT_FOUND, {"error": "unknown API route"})
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return json_response(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {error}"})
    except ValueError as error:
        return json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def transcribe_project(
    store: ProjectStore,
    project_id: str,
    model_endpoint: ModelEndpoint,
    metadata: dict[str, Any],
    *,
    source_language: str,
    transcribe_audio_func,
) -> MasterDocument:
    channels = metadata.get("channels")
    channel_names: list[str]
    if isinstance(channels, dict) and {"L", "R"}.issubset(channels):
        channel_names = ["L", "R"]
    elif isinstance(channels, dict) and "MIX" in channels:
        channel_names = ["MIX"]
    else:
        channel_names = ["MIX"]

    raw_segments: list[Segment] = []
    for channel in channel_names:
        if isinstance(channels, dict) and channel in channels:
            audio_bytes = store.read_channel_audio(project_id, channel)
            mime_type = "audio/wav"
        else:
            audio_bytes, mime_type = store.read_audio(project_id)
        raw_segments.extend(
            replace(segment, channel=channel)
            for segment in transcribe_audio_func(
                model_endpoint,
                audio_bytes,
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


def retranscribe_segment(
    store: ProjectStore,
    project_id: str,
    master: MasterDocument,
    metadata: dict[str, Any],
    *,
    segment_id: str,
    model_endpoint: ModelEndpoint,
    source_language: str,
    transcribe_audio_func,
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

    clip = slice_wav(channel_audio, start_ms=target.start_ms, end_ms=target.end_ms)
    replacement = tuple(
        replace(
            segment,
            channel=target.channel,
            start_ms=target.start_ms + segment.start_ms,
            end_ms=target.start_ms + segment.end_ms,
        )
        for segment in transcribe_audio_func(
            model_endpoint,
            clip,
            mime_type=mime_type,
            channel=target.channel,
            source_language=source_language,
        )
    )
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


class AppRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory or str(WEB_ROOT), **kwargs)

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        status, content_type, body = handle_api_request(self.path, self.rfile.read(content_length))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "127.0.0.1", port: int = 5173) -> None:
    server = ThreadingHTTPServer((host, port), AppRequestHandler)
    print(f"Serving custom ASMR SRT stack on http://{host}:{port}", flush=True)
    server.serve_forever()
