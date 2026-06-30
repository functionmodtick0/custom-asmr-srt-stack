from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from custom_asmr_srt_stack.models import MasterDocument, require_mapping, require_string
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.case_batch import REVIEW_CASE_SET_FORMAT
from custom_asmr_srt_stack.review_pack import REVIEW_PACK_FORMAT
from custom_asmr_srt_stack.srt import format_srt, parse_srt
from custom_asmr_srt_stack.translation import export_translation_json, parse_translated_texts
from custom_asmr_srt_stack.transcription import ModelEndpoint, transcribe_audio
from custom_asmr_srt_stack.workflow import analyze_project, retranscribe_segment, transcribe_project

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"


def json_response(status: HTTPStatus, payload: dict[str, Any]) -> tuple[int, str, bytes]:
    return status.value, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")


def bytes_response(status: HTTPStatus, content_type: str, body: bytes) -> tuple[int, str, bytes]:
    return status.value, content_type, body


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

        if path == "/api/review-pack/load":
            pack_path = require_string(payload.get("path"), "review pack path")
            return json_response(HTTPStatus.OK, load_review_pack_response(Path(pack_path)))

        if path == "/api/review/load":
            review_path = require_string(payload.get("path"), "review path")
            return json_response(HTTPStatus.OK, load_review_response(Path(review_path)))

        if path == "/api/review-case/load":
            case_path = require_string(payload.get("path"), "review case path")
            return json_response(HTTPStatus.OK, load_review_case_set_response(Path(case_path)))

        if path == "/api/review-case/save-reference":
            case_index_path = require_string(payload.get("case_index_path"), "case_index_path")
            case_id = require_string(payload.get("case_id"), "case_id")
            master = MasterDocument.from_json(payload.get("master"))
            return json_response(
                HTTPStatus.OK,
                save_review_case_reference_response(Path(case_index_path), case_id=case_id, master=master),
            )

        return json_response(HTTPStatus.NOT_FOUND, {"error": "unknown API route"})
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return json_response(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {error}"})
    except ValueError as error:
        return json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def handle_api_get_request(path: str) -> tuple[int, str, bytes]:
    try:
        parsed = urlparse(path)
        if parsed.path == "/api/review-pack/clip":
            query = parse_qs(parsed.query)
            index_values = query.get("index") or []
            clip_values = query.get("clip") or []
            if len(index_values) != 1 or len(clip_values) != 1:
                raise ValueError("review pack clip requires index and clip")
            index_path = review_pack_index_path(Path(index_values[0]))
            clip_path = review_pack_clip_path(index_path, clip_values[0])
            return bytes_response(HTTPStatus.OK, "audio/wav", clip_path.read_bytes())
        if parsed.path == "/api/review-case/audio":
            query = parse_qs(parsed.query)
            index_values = query.get("index") or []
            audio_values = query.get("audio") or []
            if len(index_values) != 1 or len(audio_values) != 1:
                raise ValueError("review case audio requires index and audio")
            index_path = review_case_index_path(Path(index_values[0]))
            audio_path = review_case_item_path(index_path, audio_values[0], "audio")
            return bytes_response(HTTPStatus.OK, "audio/wav", audio_path.read_bytes())
        return json_response(HTTPStatus.NOT_FOUND, {"error": "unknown API route"})
    except (OSError, ValueError) as error:
        return json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def load_review_response(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_dir():
        review_pack_index = expanded / "index.json"
        review_case_index = expanded / "case-index.json"
        if review_pack_index.is_file() and not review_case_index.is_file():
            return load_review_pack_response(review_pack_index)
        if review_case_index.is_file() and not review_pack_index.is_file():
            return load_review_case_set_response(review_case_index)
        if review_pack_index.is_file() and review_case_index.is_file():
            raise ValueError("review path contains both index.json and case-index.json")
    if expanded.name == "index.json":
        return load_review_pack_response(expanded)
    if expanded.name == "case-index.json":
        return load_review_case_set_response(expanded)
    raise ValueError("review path must be a review-pack directory, index.json, case directory, or case-index.json")


def load_review_pack_response(path: Path) -> dict[str, Any]:
    index_path = review_pack_index_path(path)
    data = json.loads(index_path.read_text(encoding="utf-8"))
    pack = require_mapping(data, "review pack")
    if pack.get("format") != REVIEW_PACK_FORMAT:
        raise ValueError(f"review pack format must be {REVIEW_PACK_FORMAT}")
    items = pack.get("items")
    if not isinstance(items, list):
        raise ValueError("review pack items must be an array")
    normalized_items = []
    for item in items:
        item_mapping = require_mapping(item, "review pack item")
        clip_file = require_string(item_mapping.get("clip_file"), "review pack item.clip_file")
        review_pack_clip_path(index_path, clip_file)
        normalized_item = dict(item_mapping)
        normalized_item["clip_url"] = review_pack_clip_url(index_path, clip_file)
        normalized_items.append(normalized_item)
    response = dict(pack)
    response["kind"] = "review-pack"
    response["index_path"] = str(index_path)
    response["pack_dir"] = str(index_path.parent)
    response["items"] = normalized_items
    return response


def load_review_case_set_response(path: Path) -> dict[str, Any]:
    case_index_path = review_case_index_path(path)
    case_index = read_review_case_index(case_index_path)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review case index items must be an array")
    normalized_items = []
    for index, item in enumerate(raw_items):
        item_mapping = require_mapping(item, f"review case item {index}")
        case_id = require_string(item_mapping.get("id"), f"review case item {index}.id")
        audio = require_string(item_mapping.get("audio"), f"review case item {index}.audio")
        reference = require_string(item_mapping.get("reference"), f"review case item {index}.reference")
        review_case_item_path(case_index_path, audio, "audio")
        reference_path = review_case_item_path(case_index_path, reference, "reference")
        normalized_item = dict(item_mapping)
        normalized_item["id"] = case_id
        normalized_item["audio_url"] = review_case_audio_url(case_index_path, audio)
        normalized_item["reference_master"] = load_review_case_reference(reference_path)
        normalized_items.append(normalized_item)
    response = dict(case_index)
    response["kind"] = "review-case-set"
    response["case_index_path"] = str(case_index_path)
    response["case_dir"] = str(case_index_path.parent)
    response["items"] = normalized_items
    return response


def save_review_case_reference_response(case_index_path: Path, *, case_id: str, master: MasterDocument) -> dict[str, Any]:
    resolved_index_path = review_case_index_path(case_index_path)
    case_index = read_review_case_index(resolved_index_path)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review case index items must be an array")
    for index, raw_item in enumerate(raw_items):
        item = require_mapping(raw_item, f"review case item {index}")
        if item.get("id") != case_id:
            continue
        reference = require_string(item.get("reference"), f"review case item {index}.reference")
        reference_path = review_case_item_path(resolved_index_path, reference, "reference")
        reference_path.write_text(json.dumps(master.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        item["segments"] = len(master.segments)
        item["review_count"] = sum(1 for segment in master.segments if segment.needs_review)
        resolved_index_path.write_text(json.dumps(case_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "ok": True,
            "case_id": case_id,
            "reference": str(reference_path),
            "segments": item["segments"],
            "review_count": item["review_count"],
        }
    raise ValueError(f"review case id is missing: {case_id}")


def review_pack_index_path(path: Path) -> Path:
    expanded = path.expanduser()
    index_path = expanded / "index.json" if expanded.is_dir() else expanded
    if index_path.name != "index.json":
        raise ValueError("review pack path must be a directory or index.json")
    if not index_path.is_file():
        raise ValueError(f"review pack index is missing: {index_path}")
    return index_path.resolve()


def review_pack_clip_path(index_path: Path, clip_file: str) -> Path:
    resolved_index = index_path.expanduser().resolve()
    base_dir = resolved_index.parent
    raw_clip_path = Path(clip_file)
    if raw_clip_path.is_absolute():
        raise ValueError("review pack clip_file must be relative")
    clip_path = (base_dir / raw_clip_path).resolve()
    try:
        clip_path.relative_to(base_dir)
    except ValueError as error:
        raise ValueError("review pack clip_file must stay inside the pack directory") from error
    if not clip_path.is_file():
        raise ValueError(f"review pack clip is missing: {clip_file}")
    return clip_path


def review_pack_clip_url(index_path: Path, clip_file: str) -> str:
    return "/api/review-pack/clip?index=" + quote(str(index_path), safe="") + "&clip=" + quote(clip_file, safe="")


def review_case_index_path(path: Path) -> Path:
    expanded = path.expanduser()
    index_path = expanded / "case-index.json" if expanded.is_dir() else expanded
    if index_path.name != "case-index.json":
        raise ValueError("review case path must be a directory or case-index.json")
    if not index_path.is_file():
        raise ValueError(f"review case index is missing: {index_path}")
    return index_path.resolve()


def read_review_case_index(index_path: Path) -> dict[str, Any]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    case_index = require_mapping(data, "review case index")
    if case_index.get("format") != REVIEW_CASE_SET_FORMAT:
        raise ValueError(f"review case index format must be {REVIEW_CASE_SET_FORMAT}")
    return case_index


def review_case_item_path(index_path: Path, path_value: str, label: str) -> Path:
    raw_path = Path(path_value).expanduser()
    resolved_path = raw_path.resolve() if raw_path.is_absolute() else (index_path.parent / raw_path).resolve()
    if not resolved_path.is_file():
        raise ValueError(f"review case {label} file is missing: {path_value}")
    return resolved_path


def review_case_audio_url(index_path: Path, audio: str) -> str:
    return "/api/review-case/audio?index=" + quote(str(index_path), safe="") + "&audio=" + quote(audio, safe="")


def load_review_case_reference(reference_path: Path) -> dict[str, Any]:
    return MasterDocument.from_json(json.loads(reference_path.read_text(encoding="utf-8"))).to_json()


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

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            status, content_type, body = handle_api_get_request(self.path)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def run_server(host: str = "127.0.0.1", port: int = 5173) -> None:
    server = ThreadingHTTPServer((host, port), AppRequestHandler)
    print(f"Serving custom ASMR SRT stack on http://{host}:{port}", flush=True)
    server.serve_forever()
