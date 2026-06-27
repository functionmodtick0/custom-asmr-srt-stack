from __future__ import annotations

import argparse
import json
from pathlib import Path

from custom_asmr_srt_stack.models import MasterDocument
from custom_asmr_srt_stack.srt import format_srt, parse_srt
from custom_asmr_srt_stack.translation import export_translation_json, parse_translated_texts


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def srt_to_json(args: argparse.Namespace) -> None:
    master = parse_srt(
        read_text(args.input),
        source_language=args.source_language,
        source_file=args.source_file,
    )
    write_text(args.output, json.dumps(master.to_json(), ensure_ascii=False, indent=2) + "\n")


def json_to_srt(args: argparse.Namespace) -> None:
    master = MasterDocument.from_json(json.loads(read_text(args.input)))
    text_by_id = None
    if args.translated is not None:
        text_by_id = parse_translated_texts(master, json.loads(read_text(args.translated)))
    write_text(args.output, format_srt(master, text_by_id=text_by_id))


def export_translation(args: argparse.Namespace) -> None:
    master = MasterDocument.from_json(json.loads(read_text(args.input)))
    write_text(args.output, json.dumps(export_translation_json(master), ensure_ascii=False, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="casrt")
    subcommands = parser.add_subparsers(dest="command", required=True)

    import_srt = subcommands.add_parser("srt-to-json", help="Convert SRT into master JSON.")
    import_srt.add_argument("input", type=Path)
    import_srt.add_argument("-o", "--output", type=Path, required=True)
    import_srt.add_argument("--source-language", default="ja")
    import_srt.add_argument("--source-file")
    import_srt.set_defaults(func=srt_to_json)

    export_srt = subcommands.add_parser("json-to-srt", help="Convert master JSON into SRT.")
    export_srt.add_argument("input", type=Path)
    export_srt.add_argument("-o", "--output", type=Path, required=True)
    export_srt.add_argument("--translated", type=Path, help="Translated JSON to merge by segment id.")
    export_srt.set_defaults(func=json_to_srt)

    translation_export = subcommands.add_parser(
        "export-translation-json",
        help="Export clean id/text JSON for external translation tools.",
    )
    translation_export.add_argument("input", type=Path)
    translation_export.add_argument("-o", "--output", type=Path, required=True)
    translation_export.set_defaults(func=export_translation)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
