from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.models import MasterDocument
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.server import run_server
from custom_asmr_srt_stack.srt import format_srt, parse_srt
from custom_asmr_srt_stack.translation import export_translation_json, parse_translated_texts
from custom_asmr_srt_stack.workflow import analyze_project


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def store_from_args(args: argparse.Namespace) -> ProjectStore:
    if args.project_root is not None:
        return ProjectStore(args.project_root)
    return ProjectStore.default()


def emit(args: argparse.Namespace, payload: dict[str, Any], message: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(message)


def project_summary(project: dict[str, Any]) -> dict[str, Any]:
    master = project.get("master")
    segments = []
    if isinstance(master, dict) and isinstance(master.get("segments"), list):
        segments = master["segments"]
    return {
        "project_id": project["project_id"],
        "metadata": project.get("metadata", {}),
        "segment_count": len(segments),
        "review_count": sum(1 for segment in segments if isinstance(segment, dict) and segment.get("needs_review")),
    }


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


def serve(args: argparse.Namespace) -> None:
    run_server(host=args.host, port=args.port)


def project_create_audio(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    file_bytes = args.input.read_bytes()
    mime_type = args.mime_type or mimetypes.guess_type(args.input.name)[0] or "application/octet-stream"
    created = store.create_from_audio(
        args.input.name,
        mime_type,
        base64.b64encode(file_bytes).decode("ascii"),
    )
    emit(
        args,
        created,
        f"project {created['project_id']} created from audio: {args.input.name}",
    )


def project_create_srt(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    master = parse_srt(read_text(args.input), source_language=args.source_language, source_file=args.input.name)
    created = store.create_from_master(master)
    emit(
        args,
        created,
        f"project {created['project_id']} created from srt: {args.input.name}",
    )


def project_create_master(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    master = MasterDocument.from_json(json.loads(read_text(args.input)))
    created = store.create_from_master(master)
    emit(
        args,
        created,
        f"project {created['project_id']} created from master: {args.input.name}",
    )


def project_show(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    summary = project_summary(store.load_project(args.project_id))
    emit(
        args,
        summary,
        f"project {summary['project_id']}: segments={summary['segment_count']} review={summary['review_count']}",
    )


def project_analyze(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    analyzed = analyze_project(store, args.project_id)
    metadata = analyzed["metadata"]
    channels = ",".join(sorted(metadata.get("channels", {})))
    duration_ms = metadata.get("audio_info", {}).get("duration_ms")
    emit(
        args,
        analyzed,
        f"project {args.project_id} analyzed: channels={channels} duration_ms={duration_ms}",
    )


def project_export_master(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    project = store.load_project(args.project_id)
    master = MasterDocument.from_json(project.get("master"))
    write_text(args.output, json.dumps(master.to_json(), ensure_ascii=False, indent=2) + "\n")
    emit(args, {"project_id": args.project_id, "output": str(args.output)}, f"master exported: {args.output}")


def project_export_translation(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    project = store.load_project(args.project_id)
    master = MasterDocument.from_json(project.get("master"))
    write_text(args.output, json.dumps(export_translation_json(master), ensure_ascii=False, indent=2) + "\n")
    emit(args, {"project_id": args.project_id, "output": str(args.output)}, f"translation exported: {args.output}")


def project_export_srt(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    project = store.load_project(args.project_id)
    master = MasterDocument.from_json(project.get("master"))
    text_by_id = None
    if args.translated is not None:
        text_by_id = parse_translated_texts(master, json.loads(read_text(args.translated)))
    write_text(args.output, format_srt(master, text_by_id=text_by_id))
    emit(args, {"project_id": args.project_id, "output": str(args.output)}, f"srt exported: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="casrt")
    subcommands = parser.add_subparsers(dest="command", required=True)
    project_parent = argparse.ArgumentParser(add_help=False)
    project_parent.add_argument("--project-root", type=Path)
    output_parent = argparse.ArgumentParser(add_help=False)
    output_parent.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

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

    serve_web = subcommands.add_parser("serve", help="Run the local WebUI server.")
    serve_web.add_argument("--host", default="127.0.0.1")
    serve_web.add_argument("--port", type=int, default=5173)
    serve_web.set_defaults(func=serve)

    project = subcommands.add_parser("project", help="Manage transcript projects.")
    project_subcommands = project.add_subparsers(dest="project_command", required=True)

    create_audio = project_subcommands.add_parser(
        "create-audio",
        parents=[project_parent, output_parent],
        help="Create a project from an audio file.",
    )
    create_audio.add_argument("input", type=Path)
    create_audio.add_argument("--mime-type")
    create_audio.set_defaults(func=project_create_audio)

    create_srt = project_subcommands.add_parser(
        "create-srt",
        parents=[project_parent, output_parent],
        help="Create a project from an SRT file.",
    )
    create_srt.add_argument("input", type=Path)
    create_srt.add_argument("--source-language", default="ja")
    create_srt.set_defaults(func=project_create_srt)

    create_master = project_subcommands.add_parser(
        "create-master",
        parents=[project_parent, output_parent],
        help="Create a project from master JSON.",
    )
    create_master.add_argument("input", type=Path)
    create_master.set_defaults(func=project_create_master)

    show_project = project_subcommands.add_parser(
        "show",
        parents=[project_parent, output_parent],
        help="Show project status.",
    )
    show_project.add_argument("project_id")
    show_project.set_defaults(func=project_show)

    analyze_audio = project_subcommands.add_parser(
        "analyze",
        parents=[project_parent, output_parent],
        help="Normalize and split project audio.",
    )
    analyze_audio.add_argument("project_id")
    analyze_audio.set_defaults(func=project_analyze)

    export_master = project_subcommands.add_parser(
        "export-master",
        parents=[project_parent, output_parent],
        help="Export project master JSON.",
    )
    export_master.add_argument("project_id")
    export_master.add_argument("-o", "--output", type=Path, required=True)
    export_master.set_defaults(func=project_export_master)

    export_translation_project = project_subcommands.add_parser(
        "export-translation",
        parents=[project_parent, output_parent],
        help="Export project translation JSON.",
    )
    export_translation_project.add_argument("project_id")
    export_translation_project.add_argument("-o", "--output", type=Path, required=True)
    export_translation_project.set_defaults(func=project_export_translation)

    export_srt_project = project_subcommands.add_parser(
        "export-srt",
        parents=[project_parent, output_parent],
        help="Export project SRT.",
    )
    export_srt_project.add_argument("project_id")
    export_srt_project.add_argument("-o", "--output", type=Path, required=True)
    export_srt_project.add_argument("--translated", type=Path)
    export_srt_project.set_defaults(func=project_export_srt)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
