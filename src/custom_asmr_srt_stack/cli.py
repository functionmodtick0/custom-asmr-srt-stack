from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.alignment import run_alignment_command
from custom_asmr_srt_stack.audio import normalize_audio_to_wav, split_wav_channels
from custom_asmr_srt_stack.channel_attribution import (
    CHANNEL_ATTRIBUTION_THRESHOLD_DB,
    attribute_master_channels_by_energy,
)
from custom_asmr_srt_stack.evaluation import (
    evaluate_manifest,
    evaluate_transcripts,
    load_transcript_document,
    review_effort_items_report,
)
from custom_asmr_srt_stack.model_snapshot import snapshot_digest
from custom_asmr_srt_stack.models import MasterDocument
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.review_pack import build_review_pack
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


def freeze_reference(args: argparse.Namespace) -> None:
    master = load_transcript_document(args.input, source_language=args.source_language)
    segments = tuple(
        replace(segment, id=f"seg_{index + 1:06d}", needs_review=False)
        for index, segment in enumerate(sorted(master.segments, key=lambda item: (item.start_ms, item.end_ms, item.id)))
    )
    frozen = MasterDocument(
        source_language=master.source_language,
        source_file=master.source_file,
        duration_ms=master.duration_ms,
        segments=segments,
    )
    write_text(args.output, json.dumps(frozen.to_json(), ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        {"output": str(args.output), "segments": len(segments), "reference_type": "human-reviewed"},
        f"reference frozen: {args.output} segments={len(segments)}",
    )


def align_transcript(args: argparse.Namespace) -> None:
    command = os.environ.get("CASRT_ALIGNER_COMMAND")
    if not command:
        raise ValueError("CASRT_ALIGNER_COMMAND is required")
    master = load_transcript_document(args.input, source_language=args.source_language)
    aligned = run_alignment_command(master, audio_file=args.audio, command=shlex.split(command))
    write_text(args.output, json.dumps(aligned.to_json(), ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        {"output": str(args.output), "segments": len(aligned.segments), "aligner": "CASRT_ALIGNER_COMMAND"},
        f"transcript aligned: {args.output} segments={len(aligned.segments)}",
    )


def attribute_channels(args: argparse.Namespace) -> None:
    mime_type = mimetypes.guess_type(args.audio.name)[0]
    normalized_audio = normalize_audio_to_wav(
        args.audio.read_bytes(),
        file_name=args.audio.name,
        mime_type=mime_type,
    )
    _, channels = split_wav_channels(normalized_audio)
    if not {"L", "R"}.issubset(channels):
        raise ValueError("channel attribution requires stereo audio")
    master = load_transcript_document(args.input, source_language=args.source_language)
    report = attribute_master_channels_by_energy(
        master,
        left_audio=channels["L"],
        right_audio=channels["R"],
        threshold_db=args.threshold_db,
    )
    write_text(args.output, json.dumps(report.master.to_json(), ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        {
            "output": str(args.output),
            "segments": report.segments,
            "changed_segments": report.changed_segments,
            "threshold_db": report.threshold_db,
        },
        f"channels attributed: {args.output} changed={report.changed_segments}/{report.segments}",
    )


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
    enforce_reference_type_gate(report, args)


def review_effort(args: argparse.Namespace) -> None:
    report = review_effort_items_report(json.loads(read_text(args.report)), source_report=str(args.report))
    if args.output is not None:
        write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        report,
        (
            f"review_effort_items={report['item_count']} "
            f"reasons={json.dumps(report['reason_counts'], ensure_ascii=False, sort_keys=True)}"
        ),
    )


def review_pack(args: argparse.Namespace) -> None:
    report = build_review_pack(
        json.loads(read_text(args.review_effort)),
        output_dir=args.output,
        audio_file=args.audio,
        audio_map_file=args.audio_map,
    )
    emit(
        args,
        report,
        f"review pack: {args.output} clips={report['clip_count']}",
    )


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
    max_segments_needing_edit_ratio = ratio_arg(
        args.max_segments_needing_edit_ratio,
        "--max-segments-needing-edit-ratio",
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

    if max_segments_needing_edit_ratio is not None:
        edit_ratio = float(metrics["review_effort"]["segments_needing_edit_ratio"])
        if edit_ratio > max_segments_needing_edit_ratio:
            failures.append(
                f"segments needing edit ratio {edit_ratio:.4f} > {max_segments_needing_edit_ratio:.4f}"
            )

    if failures:
        raise ValueError("quality gate failed: " + "; ".join(failures))


def enforce_reference_type_gate(report: dict[str, Any], args: argparse.Namespace) -> None:
    required = getattr(args, "require_reference_type", None)
    if required is None:
        return

    failures = []
    for case in report.get("cases", []):
        case_type = case.get("reference_type") or "unspecified"
        if case_type != required:
            failures.append(f"{case.get('id', '<unknown>')} reference_type {case_type!r} != {required!r}")

    if failures:
        raise ValueError("reference type gate failed: " + "; ".join(failures))


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


def model_digest(args: argparse.Namespace) -> None:
    digest = snapshot_digest(args.snapshot)
    if args.output is not None:
        write_text(args.output, json.dumps(digest, ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        digest,
        f"snapshot digest: files={digest['file_count']} sha256={digest['sha256']} snapshot={digest['snapshot_id']}",
    )


def vad_whisper_asmr_onnx(args: argparse.Namespace) -> None:
    from custom_asmr_srt_stack.audio import speech_intervals_by_energy
    from custom_asmr_srt_stack.models import require_mapping
    from custom_asmr_srt_stack.whisper_vad_onnx import (
        WhisperVadOnnxSettings,
        add_rescue_intervals,
        detect_command_intervals,
    )
    from custom_asmr_srt_stack.workflow import qwen_energy_chunk_kwargs

    request = require_mapping(json.loads(sys.stdin.read()), "VAD request")
    audio_file = request.get("audio_file")
    if not isinstance(audio_file, str) or not audio_file:
        raise ValueError("VAD request audio_file must be a non-empty string")
    intervals = detect_command_intervals(
        audio_file=Path(audio_file),
        model=args.model,
        metadata=args.metadata,
        settings=WhisperVadOnnxSettings(
            threshold=args.threshold,
            neg_threshold=args.neg_threshold,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
            pad_ms=args.pad_ms,
            output_activation=args.output_activation,
            force_cpu=args.force_cpu,
            num_threads=args.num_threads,
        ),
    )
    if args.energy_rescue_min_ms is not None:
        energy_intervals = speech_intervals_by_energy(Path(audio_file).read_bytes(), **qwen_energy_chunk_kwargs())
        intervals = add_rescue_intervals(
            energy_intervals,
            intervals,
            min_rescue_ms=args.energy_rescue_min_ms,
        )
    print(json.dumps({"intervals": list(intervals)}, ensure_ascii=False))


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
        choices=[
            "openai-compatible",
            "gemini",
            "local-transformers",
            "local-qwen-asr",
            "local-qwen-hf-asr",
            "local-cohere-asr",
        ],
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

    freeze_reference_parser = subcommands.add_parser(
        "freeze-reference",
        parents=[output_parent],
        help="Freeze a reviewed SRT/master file into reference master JSON.",
    )
    freeze_reference_parser.add_argument("input", type=Path)
    freeze_reference_parser.add_argument("-o", "--output", type=Path, required=True)
    freeze_reference_parser.add_argument("--source-language", default="ja")
    freeze_reference_parser.set_defaults(func=freeze_reference)

    align_transcript_parser = subcommands.add_parser(
        "align-transcript",
        parents=[output_parent],
        help="Align an existing SRT/master transcript with CASRT_ALIGNER_COMMAND.",
    )
    align_transcript_parser.add_argument("audio", type=Path)
    align_transcript_parser.add_argument("input", type=Path)
    align_transcript_parser.add_argument("-o", "--output", type=Path, required=True)
    align_transcript_parser.add_argument("--source-language", default="ja")
    align_transcript_parser.set_defaults(func=align_transcript)

    attribute_channels_parser = subcommands.add_parser(
        "attribute-channels",
        parents=[output_parent],
        help="Apply L/R energy channel attribution to an existing SRT/master transcript.",
    )
    attribute_channels_parser.add_argument("audio", type=Path)
    attribute_channels_parser.add_argument("input", type=Path)
    attribute_channels_parser.add_argument("-o", "--output", type=Path, required=True)
    attribute_channels_parser.add_argument("--source-language", default="ja")
    attribute_channels_parser.add_argument("--threshold-db", type=float, default=CHANNEL_ATTRIBUTION_THRESHOLD_DB)
    attribute_channels_parser.set_defaults(func=attribute_channels)

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
    eval_manifest_parser.add_argument(
        "--require-reference-type",
        help="Fail if any manifest case has a different effective reference_type.",
    )
    eval_manifest_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    add_quality_gate_args(eval_manifest_parser)
    eval_manifest_parser.set_defaults(func=eval_manifest)

    review_effort_parser = subcommands.add_parser(
        "review-effort",
        help="Export review effort items from an eval-transcript or eval-manifest report.",
    )
    review_effort_parser.add_argument("report", type=Path)
    review_effort_parser.add_argument("-o", "--output", type=Path)
    review_effort_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    review_effort_parser.set_defaults(func=review_effort)

    review_pack_parser = subcommands.add_parser(
        "review-pack",
        help="Create audio clips from a review-effort JSON report.",
    )
    review_pack_parser.add_argument("review_effort", type=Path)
    review_pack_parser.add_argument("-o", "--output", type=Path, required=True)
    review_pack_parser.add_argument("--audio", type=Path, help="Single WAV file for reports without case audio mapping.")
    review_pack_parser.add_argument("--audio-map", type=Path, help="JSON map from case_id to WAV file.")
    review_pack_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    review_pack_parser.set_defaults(func=review_pack)

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
    digest_model = model_subcommands.add_parser(
        "digest",
        parents=[output_parent],
        help="Hash a local model snapshot directory for reproducible benchmark records.",
    )
    digest_model.add_argument("snapshot", type=Path)
    digest_model.add_argument("-o", "--output", type=Path)
    digest_model.set_defaults(func=model_digest)

    vad = subcommands.add_parser("vad", help="Run internal VAD command helpers.")
    vad_subcommands = vad.add_subparsers(dest="vad_command", required=True)
    whisper_asmr_onnx = vad_subcommands.add_parser(
        "whisper-asmr-onnx",
        help="Read a CASRT VAD request from stdin and run the ASMR Whisper ONNX VAD.",
    )
    whisper_asmr_onnx.add_argument("--model", type=Path, required=True)
    whisper_asmr_onnx.add_argument("--metadata", type=Path)
    whisper_asmr_onnx.add_argument("--threshold", type=float, default=0.5)
    whisper_asmr_onnx.add_argument("--neg-threshold", type=float)
    whisper_asmr_onnx.add_argument("--min-speech-ms", type=int, default=250)
    whisper_asmr_onnx.add_argument("--min-silence-ms", type=int, default=100)
    whisper_asmr_onnx.add_argument("--pad-ms", type=int, default=30)
    whisper_asmr_onnx.add_argument("--output-activation", choices=["sigmoid", "identity"], default="sigmoid")
    whisper_asmr_onnx.add_argument("--force-cpu", action="store_true", default=True)
    whisper_asmr_onnx.add_argument("--num-threads", type=int, default=1)
    whisper_asmr_onnx.add_argument(
        "--energy-rescue-min-ms",
        type=int,
        help="Add ONNX-only gaps at least this long to the internal energy intervals.",
    )
    whisper_asmr_onnx.set_defaults(func=vad_whisper_asmr_onnx)

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
    parser.add_argument(
        "--max-segments-needing-edit-ratio",
        type=float,
        help="Fail if review effort segment edit ratio is above this 0..1 ratio.",
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
