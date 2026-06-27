from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.models import MasterDocument, require_mapping
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.srt import format_srt, parse_srt
from custom_asmr_srt_stack.translation import export_translation_json, parse_translated_texts
from custom_asmr_srt_stack.transcription import ModelEndpoint, transcribe_audio
from custom_asmr_srt_stack.workflow import analyze_project, retranscribe_segment, transcribe_project

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
            return json_response(HTTPStatus.OK, analyze_project(store, project_id))

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
