from __future__ import annotations

import base64
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.models import MasterDocument, require_mapping, require_string

PROJECT_ID_RE = re.compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class ProjectStore:
    root: Path

    @classmethod
    def default(cls) -> ProjectStore:
        return cls(Path.cwd() / ".casrt" / "projects")

    def create_from_master(self, master: MasterDocument) -> dict[str, Any]:
        project_id = uuid.uuid4().hex
        project_root = self.project_root(project_id)
        project_root.mkdir(parents=True, exist_ok=False)
        self.write_json(project_root / "master.json", master.to_json())
        metadata = {
            "project_id": project_id,
            "source_file": master.source_file,
            "source_language": master.source_language,
            "has_audio": False,
        }
        self.write_json(project_root / "project.json", metadata)
        return {"project_id": project_id, "master": master.to_json(), "metadata": metadata}

    def create_from_audio(self, file_name: str, mime_type: str, content_base64: str) -> dict[str, Any]:
        if not file_name:
            raise ValueError("file_name must not be empty")
        if not mime_type:
            raise ValueError("mime_type must not be empty")
        try:
            audio_bytes = base64.b64decode(content_base64, validate=True)
        except ValueError as error:
            raise ValueError("content_base64 must be valid base64") from error
        if not audio_bytes:
            raise ValueError("audio content must not be empty")

        project_id = uuid.uuid4().hex
        project_root = self.project_root(project_id)
        audio_root = project_root / "audio"
        audio_root.mkdir(parents=True, exist_ok=False)
        extension = safe_extension(file_name)
        audio_path = audio_root / f"original{extension}"
        audio_path.write_bytes(audio_bytes)

        metadata = {
            "project_id": project_id,
            "source_file": file_name,
            "mime_type": mime_type,
            "audio_file": str(audio_path.relative_to(project_root)),
            "has_audio": True,
        }
        self.write_json(project_root / "project.json", metadata)
        return {"project_id": project_id, "metadata": metadata}

    def save_master(self, project_id: str, master: MasterDocument) -> dict[str, Any]:
        project_root = self.require_project_root(project_id)
        self.write_json(project_root / "master.json", master.to_json())
        return {"project_id": project_id, "master": master.to_json()}

    def load_project(self, project_id: str) -> dict[str, Any]:
        project_root = self.require_project_root(project_id)
        metadata = self.read_json(project_root / "project.json")
        master_path = project_root / "master.json"
        result: dict[str, Any] = {"project_id": project_id, "metadata": metadata}
        if master_path.exists():
            result["master"] = MasterDocument.from_json(self.read_json(master_path)).to_json()
        return result

    def read_audio(self, project_id: str) -> tuple[bytes, str]:
        project = self.load_project(project_id)
        metadata = require_mapping(project.get("metadata"), "metadata")
        audio_file = require_string(metadata.get("audio_file"), "metadata.audio_file")
        mime_type = require_string(metadata.get("mime_type"), "metadata.mime_type")
        audio_path = self.require_project_root(project_id) / audio_file
        if not audio_path.exists():
            raise ValueError("project audio file is missing")
        return audio_path.read_bytes(), mime_type

    def project_root(self, project_id: str) -> Path:
        if not PROJECT_ID_RE.fullmatch(project_id):
            raise ValueError("invalid project_id")
        return self.root / project_id

    def require_project_root(self, project_id: str) -> Path:
        project_root = self.project_root(project_id)
        if not project_root.is_dir():
            raise ValueError("project not found")
        return project_root

    @staticmethod
    def write_json(path: Path, value: dict[str, Any]) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ValueError(f"{path.name} is missing")
        data = json.loads(path.read_text(encoding="utf-8"))
        return require_mapping(data, path.name)


def safe_extension(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if not suffix or len(suffix) > 12 or not re.fullmatch(r"\.[a-z0-9]+", suffix):
        return ".bin"
    return suffix
