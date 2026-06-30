from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def snapshot_digest(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("snapshot path must be an existing directory")

    files = []
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        resolved = file_path.resolve(strict=True)
        files.append(
            {
                "path": str(file_path.relative_to(root)),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )

    if not files:
        raise ValueError("snapshot path must contain files")

    digest = hashlib.sha256()
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\0")

    return {
        "format": "custom-asmr-model-snapshot-digest-v1",
        "snapshot": str(root),
        "snapshot_id": root.name,
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": files,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
