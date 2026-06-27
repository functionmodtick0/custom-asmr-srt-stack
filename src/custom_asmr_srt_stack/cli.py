from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.evaluation import evaluate_manifest, evaluate_transcripts, load_transcript_document
from custom_asmr_srt_stack.models import MasterDocument
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.server import run_server
from custom_asmr_srt_stack.srt import format_srt, parse_srt
from custom_asmr_srt_stack.translation import export_translation_json, parse_translated_texts
from custom_asmr_srt_stack.transcription import ModelEndpoint, transcribe_audio
from custom_asmr_srt_stack.workflow import analyze_project
from custom_asmr_srt_stack.workflow import retranscribe_segment as retranscribe_project_segment
from custom_asmr_srt_stack.workflow import transcribe_project


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


def model_endpoint_from_args(args: argparse.Namespace) -> ModelEndpoint:
    return ModelEndpoint(
        adapter=args.adapter,
        endpoint_url=args.endpoint_url,
        model_id=args.model_id,
        api_key=args.api_key,
    )


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


def eval_transcript(args: argparse.Namespace) -> None:
    reference = load_transcript_document(args.reference, source_language=args.source_language)
    candidate = load_transcript_document(args.candidate, source_language=args.source_language)
    report = evaluate_transcripts(reference, candidate)
    if args.output is not None:
        write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        report,
        (
            f"cer={report['text']['cer']:.4f} "
            f"practical_cer={report['text_practical']['cer']:.4f} "
            f"segments={report['candidate_segments']}/{report['reference_segments']} "
            f"timing_ms={report['timing']['mean_boundary_error_ms']} "
            f"time_aligned_timing_ms={report['timing_time_aligned']['mean_boundary_error_ms']}"
        ),
    )
    enforce_quality_gate(report, args)


def eval_manifest(args: argparse.Namespace) -> None:
    report = evaluate_manifest(args.manifest, source_language=args.source_language)
    if args.output is not None:
        write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    timing = report["summary"]["timing"]["mean_boundary_error_ms"]
    channel_accuracy = report["summary"]["channel"]["accuracy"]
    emit(
        args,
        report,
        (
            f"cases={report['case_count']} "
            f"cer={report['summary']['text']['cer']:.4f} "
            f"practical_cer={report['summary']['text_practical']['cer']:.4f} "
            f"timing_ms={timing} "
            f"time_aligned_timing_ms={report['summary']['timing_time_aligned']['mean_boundary_error_ms']} "
            f"channel_accuracy={channel_accuracy}"
        ),
    )
    enforce_quality_gate(report["summary"], args)


def enforce_quality_gate(metrics: dict[str, Any], args: argparse.Namespace) -> None:
    failures: list[str] = []
    max_practical_cer = ratio_arg(args.max_practical_cer, "--max-practical-cer")
    min_time_aligned_500ms_ratio = ratio_arg(
        args.min_time_aligned_500ms_ratio,
        "--min-time-aligned-500ms-ratio",
    )
    min_channel_time_aligned_accuracy = ratio_arg(
        args.min_channel_time_aligned_accuracy,
        "--min-channel-time-aligned-accuracy",
    )
    max_channel_time_aligned_mix_ratio = ratio_arg(
        args.max_channel_time_aligned_mix_ratio,
        "--max-channel-time-aligned-mix-ratio",
    )

    if max_practical_cer is not None:
        practical_cer = float(metrics["text_practical"]["cer"])
        if practical_cer > max_practical_cer:
            failures.append(f"practical CER {practical_cer:.4f} > {max_practical_cer:.4f}")

    if min_time_aligned_500ms_ratio is not None:
        ratio = float(metrics["timing_time_aligned"]["within_500ms_ratio"])
        if ratio < min_time_aligned_500ms_ratio:
            failures.append(f"time-aligned 500ms ratio {ratio:.4f} < {min_time_aligned_500ms_ratio:.4f}")

    if min_channel_time_aligned_accuracy is not None:
        accuracy = metrics["channel_time_aligned"]["accuracy"]
        if accuracy is None:
            failures.append("channel time-aligned accuracy is unavailable")
        elif float(accuracy) < min_channel_time_aligned_accuracy:
            failures.append(
                f"channel time-aligned accuracy {float(accuracy):.4f} < {min_channel_time_aligned_accuracy:.4f}"
            )

    if max_channel_time_aligned_mix_ratio is not None:
        mix_ratio = float(metrics["channel_time_aligned"]["candidate_mix_ratio"])
        if mix_ratio > max_channel_time_aligned_mix_ratio:
            failures.append(
                f"channel time-aligned MIX ratio {mix_ratio:.4f} > {max_channel_time_aligned_mix_ratio:.4f}"
            )

    if failures:
        raise ValueError("quality gate failed: " + "; ".join(failures))


def ratio_arg(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


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


def model_validate(args: argparse.Namespace) -> None:
    endpoint = model_endpoint_from_args(args)
    emit(
        args,
        {"ok": True, "adapter": endpoint.adapter, "model_id": endpoint.model_id},
        f"model settings valid: {endpoint.adapter} / {endpoint.model_id}",
    )


def project_transcribe(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    project = store.load_project(args.project_id)
    metadata = project.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("project metadata must be an object")
    master = transcribe_project(
        store,
        args.project_id,
        model_endpoint_from_args(args),
        metadata,
        source_language=args.source_language,
        transcribe_audio_func=transcribe_audio,
    )
    saved = store.save_master(args.project_id, master)
    review_count = sum(1 for segment in master.segments if segment.needs_review)
    emit(
        args,
        saved,
        f"project {args.project_id} transcribed: segments={len(master.segments)} review={review_count}",
    )


def project_retranscribe(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    project = store.load_project(args.project_id)
    metadata = project.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("project metadata must be an object")
    master = MasterDocument.from_json(project.get("master"))
    updated = retranscribe_project_segment(
        store,
        args.project_id,
        master,
        metadata,
        segment_id=args.segment_id,
        model_endpoint=model_endpoint_from_args(args),
        source_language=args.source_language,
        transcribe_audio_func=transcribe_audio,
    )
    saved = store.save_master(args.project_id, updated)
    emit(
        args,
        saved,
        f"project {args.project_id} retranscribed: segment={args.segment_id} segments={len(updated.segments)}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="casrt")
    subcommands = parser.add_subparsers(dest="command", required=True)
    project_parent = argparse.ArgumentParser(add_help=False)
    project_parent.add_argument("--project-root", type=Path)
    output_parent = argparse.ArgumentParser(add_help=False)
    output_parent.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    model_parent = argparse.ArgumentParser(add_help=False)
    model_parent.add_argument(
        "--adapter",
        choices=["openai-compatible", "gemini", "local-transformers", "local-qwen-asr"],
        required=True,
    )
    model_parent.add_argument("--endpoint-url")
    model_parent.add_argument("--model-id", required=True)
    model_parent.add_argument("--api-key")

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

    eval_transcript_parser = subcommands.add_parser("eval-transcript", help="Evaluate a candidate transcript.")
    eval_transcript_parser.add_argument("reference", type=Path)
    eval_transcript_parser.add_argument("candidate", type=Path)
    eval_transcript_parser.add_argument("-o", "--output", type=Path)
    eval_transcript_parser.add_argument("--source-language", default="ja")
    eval_transcript_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    add_quality_gate_args(eval_transcript_parser)
    eval_transcript_parser.set_defaults(func=eval_transcript)

    eval_manifest_parser = subcommands.add_parser("eval-manifest", help="Evaluate a transcript manifest.")
    eval_manifest_parser.add_argument("manifest", type=Path)
    eval_manifest_parser.add_argument("-o", "--output", type=Path)
    eval_manifest_parser.add_argument("--source-language", default="ja")
    eval_manifest_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    add_quality_gate_args(eval_manifest_parser)
    eval_manifest_parser.set_defaults(func=eval_manifest)

    serve_web = subcommands.add_parser("serve", help="Run the local WebUI server.")
    serve_web.add_argument("--host", default="127.0.0.1")
    serve_web.add_argument("--port", type=int, default=5173)
    serve_web.set_defaults(func=serve)

    model = subcommands.add_parser("model", help="Validate model endpoint settings.")
    model_subcommands = model.add_subparsers(dest="model_command", required=True)
    validate_model = model_subcommands.add_parser(
        "validate",
        parents=[model_parent, output_parent],
        help="Validate model endpoint settings.",
    )
    validate_model.set_defaults(func=model_validate)

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

    transcribe_audio_project = project_subcommands.add_parser(
        "transcribe",
        parents=[project_parent, model_parent, output_parent],
        help="Transcribe an analyzed project.",
    )
    transcribe_audio_project.add_argument("project_id")
    transcribe_audio_project.add_argument("--source-language", default="ja")
    transcribe_audio_project.set_defaults(func=project_transcribe)

    retranscribe_audio_segment = project_subcommands.add_parser(
        "retranscribe",
        parents=[project_parent, model_parent, output_parent],
        help="Retranscribe a selected segment.",
    )
    retranscribe_audio_segment.add_argument("project_id")
    retranscribe_audio_segment.add_argument("segment_id")
    retranscribe_audio_segment.add_argument("--source-language", default="ja")
    retranscribe_audio_segment.set_defaults(func=project_retranscribe)

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


def add_quality_gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-practical-cer",
        type=float,
        help="Fail if practical CER is above this 0..1 ratio.",
    )
    parser.add_argument(
        "--min-time-aligned-500ms-ratio",
        type=float,
        help="Fail if time-aligned 500ms boundary ratio is below this 0..1 ratio.",
    )
    parser.add_argument(
        "--min-channel-time-aligned-accuracy",
        type=float,
        help="Fail if time-aligned L/R channel accuracy is below this 0..1 ratio.",
    )
    parser.add_argument(
        "--max-channel-time-aligned-mix-ratio",
        type=float,
        help="Fail if time-aligned candidate MIX ratio is above this 0..1 ratio.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
