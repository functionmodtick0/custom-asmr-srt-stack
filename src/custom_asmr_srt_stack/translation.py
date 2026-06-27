from __future__ import annotations

from typing import Any

from custom_asmr_srt_stack.models import (
    TRANSLATED_FORMAT,
    TRANSLATION_FORMAT,
    MasterDocument,
    require_mapping,
    require_string,
)


def export_translation_json(master: MasterDocument) -> dict[str, Any]:
    return {
        "format": TRANSLATION_FORMAT,
        "source_language": master.source_language,
        "items": [
            {
                "id": segment.id,
                "text": segment.text,
            }
            for segment in master.segments
            if segment.kind == "speech"
        ],
    }


def parse_translated_texts(master: MasterDocument, value: Any) -> dict[str, str]:
    data = require_mapping(value, "translated document")
    document_format = require_string(data.get("format"), "format")
    if document_format != TRANSLATED_FORMAT:
        raise ValueError(f"unsupported translated format {document_format!r}")

    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be an array")

    texts: dict[str, str] = {}
    for item in items:
        item_data = require_mapping(item, "translated item")
        item_id = require_string(item_data.get("id"), "translated item.id")
        if item_id in texts:
            raise ValueError(f"duplicate translated id {item_id!r}")
        texts[item_id] = require_string(item_data.get("text"), "translated item.text")

    required_ids = {segment.id for segment in master.segments if segment.kind == "speech"}
    translated_ids = set(texts)
    missing_ids = sorted(required_ids - translated_ids)
    unknown_ids = sorted(translated_ids - required_ids)
    if missing_ids:
        raise ValueError(f"translated JSON is missing ids: {', '.join(missing_ids)}")
    if unknown_ids:
        raise ValueError(f"translated JSON contains unknown ids: {', '.join(unknown_ids)}")

    return texts
