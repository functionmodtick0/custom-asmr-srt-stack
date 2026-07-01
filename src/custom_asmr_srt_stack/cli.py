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

from custom_asmr_srt_stack.alignment import alignment_diagnostics, run_alignment_command
from custom_asmr_srt_stack.audio import normalize_audio_to_wav, slice_wav, split_wav_channels
from custom_asmr_srt_stack.case_batch import (
    EVAL_MANIFEST_BUILD_FORMAT,
    align_review_case_candidates,
    attach_review_case_candidates,
    build_case_candidate_attach_plan,
    build_review_case_pack,
    build_eval_manifest_from_case_index,
    freeze_case_references as freeze_case_references_batch,
    load_review_case_index,
    prepare_review_cases,
    require_index_string,
    review_case_status,
    resolve_plan_path,
    save_review_case_reference,
)
from custom_asmr_srt_stack.case_transcription import transcribe_review_case_candidates
from custom_asmr_srt_stack.case_slicing import slice_master_document
from custom_asmr_srt_stack.channel_attribution import (
    CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
    CHANNEL_ATTRIBUTION_THRESHOLD_DB,
    attribute_master_channels_by_energy,
)
from custom_asmr_srt_stack.channel_sweep import sweep_channel_attribution
from custom_asmr_srt_stack.evaluation import (
    compare_eval_reports,
    evaluate_manifest,
    evaluate_transcripts,
    load_transcript_document,
    review_effort_items_report,
)
from custom_asmr_srt_stack.model_snapshot import snapshot_digest
from custom_asmr_srt_stack.models import MasterDocument
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.review_pack import DEFAULT_REVIEW_CONTEXT_MS, build_review_pack
from custom_asmr_srt_stack.server import run_server
from custom_asmr_srt_stack.srt import format_srt, parse_srt
from custom_asmr_srt_stack.translation import export_translation_json, parse_translated_texts
from custom_asmr_srt_stack.transcription import ModelEndpoint, transcribe_audio
from custom_asmr_srt_stack.workflow import analyze_project
from custom_asmr_srt_stack.workflow import retranscribe_segment as retranscribe_project_segment
from custom_asmr_srt_stack.workflow import transcribe_project

PRODUCT_GATE_REFERENCE_TYPE = "human-reviewed"
PRODUCT_GATE_THRESHOLDS = {
    "max_practical_cer": 0.10,
    "min_time_aligned_500ms_ratio": 0.90,
    "min_channel_time_aligned_accuracy": 0.85,
    "max_channel_time_aligned_mix_ratio": 0.50,
    "max_segments_needing_edit_ratio": 0.15,
    "max_candidate_review_ratio": 0.00,
}


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
    diagnostics_output = None
    if args.diagnostics_output is not None:
        diagnostics_output = str(args.diagnostics_output)
        write_text(
            args.diagnostics_output,
            json.dumps(
                alignment_diagnostics(
                    master,
                    aligned,
                    audio_file=args.audio,
                    input_file=args.input,
                    output_file=args.output,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    payload = {"output": str(args.output), "segments": len(aligned.segments), "aligner": "CASRT_ALIGNER_COMMAND"}
    if diagnostics_output is not None:
        payload["diagnostics_output"] = diagnostics_output
    emit(
        args,
        payload,
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
        quiet_channel_max_dbfs=args.quiet_channel_max_dbfs,
    )
    write_text(args.output, json.dumps(report.master.to_json(), ensure_ascii=False, indent=2) + "\n")
    diagnostics_output = None
    if args.diagnostics_output is not None:
        diagnostics_output = str(args.diagnostics_output)
        write_text(
            args.diagnostics_output,
            json.dumps(
                {
                    "format": "custom-asmr-channel-diagnostics-v1",
                    "audio": str(args.audio),
                    "input": str(args.input),
                    "output": str(args.output),
                    "segments": report.segments,
                    "threshold_db": report.threshold_db,
                    "quiet_channel_max_dbfs": args.quiet_channel_max_dbfs,
                    "items": list(report.diagnostics),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    payload = {
        "output": str(args.output),
        "segments": report.segments,
        "changed_segments": report.changed_segments,
        "threshold_db": report.threshold_db,
        "quiet_channel_max_dbfs": args.quiet_channel_max_dbfs,
    }
    if diagnostics_output is not None:
        payload["diagnostics_output"] = diagnostics_output
    emit(
        args,
        payload,
        f"channels attributed: {args.output} changed={report.changed_segments}/{report.segments}",
    )


def sweep_channel_attribution_command(args: argparse.Namespace) -> None:
    report = sweep_channel_attribution(
        args.manifest,
        audio_map_file=args.audio_map,
        output_dir=args.output,
        threshold_db_values=args.threshold_db or [CHANNEL_ATTRIBUTION_THRESHOLD_DB],
        quiet_channel_max_dbfs_values=args.quiet_channel_max_dbfs or [CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS],
        source_language=args.source_language,
        reset_speech_channels_to_mix=args.reset_speech_channels_to_mix,
    )
    comparison_path = Path(report["comparison"])
    comparison = json.loads(read_text(comparison_path))
    annotate_comparison_quality_gates(comparison, args)
    if "quality_gate" in comparison:
        write_text(comparison_path, json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
        report["quality_gate"] = comparison["quality_gate"]
        write_text(args.output / "index.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        report,
        f"channel attribution sweep: {args.output} settings={report['setting_count']} cases={report['case_count']}",
    )


def slice_case(args: argparse.Namespace) -> None:
    mime_type = mimetypes.guess_type(args.audio.name)[0]
    normalized_audio = normalize_audio_to_wav(
        args.audio.read_bytes(),
        file_name=args.audio.name,
        mime_type=mime_type,
    )
    sliced_audio = slice_wav(normalized_audio, start_ms=args.start_ms, end_ms=args.end_ms)
    args.audio_output.parent.mkdir(parents=True, exist_ok=True)
    args.audio_output.write_bytes(sliced_audio)

    master = load_transcript_document(args.transcript, source_language=args.source_language)
    sliced_master = slice_master_document(master, start_ms=args.start_ms, end_ms=args.end_ms)
    write_text(args.transcript_output, json.dumps(sliced_master.to_json(), ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        {
            "audio_output": str(args.audio_output),
            "transcript_output": str(args.transcript_output),
            "start_ms": args.start_ms,
            "end_ms": args.end_ms,
            "duration_ms": args.end_ms - args.start_ms,
            "segments": len(sliced_master.segments),
            "review_count": sum(1 for segment in sliced_master.segments if segment.needs_review),
        },
        (
            f"case sliced: audio={args.audio_output} transcript={args.transcript_output} "
            f"duration_ms={args.end_ms - args.start_ms} segments={len(sliced_master.segments)}"
        ),
    )


def prepare_review_cases_command(args: argparse.Namespace) -> None:
    report = prepare_review_cases(
        args.plan,
        output_dir=args.output,
        source_language=args.source_language,
    )
    emit(
        args,
        report,
        f"review cases prepared: {args.output} cases={report['case_count']} review={report['review_count']}",
    )


def review_case_status_command(args: argparse.Namespace) -> None:
    report = review_case_status(args.case_index, source_language=args.source_language)
    if args.output is not None:
        write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        report,
        (
            f"review case status: cases={report['case_count']} "
            f"review={report['reference_review_count']} "
            f"review_ms={report['reference_review_duration_ms']} "
            f"candidate_review={report['candidate_review_count']} "
            f"candidate_review_ms={report['candidate_review_duration_ms']} issues={report['case_issue_count']}"
        ),
    )
    if args.fail_on_issues and not report["ok"]:
        raise ValueError(
            "review case status failed: "
            f"missing_files={report['missing_file_count']} case_issues={report['case_issue_count']}"
        )
    if args.fail_on_review and report["reference_review_count"] > 0:
        raise ValueError(f"review case status failed: review_count={report['reference_review_count']}")
    if args.fail_on_missing_candidates and report["missing_candidate_case_count"] > 0:
        raise ValueError(
            "review case status failed: "
            f"missing_candidate_count={report['missing_candidate_case_count']}"
        )
    if args.fail_on_candidate_review and report["candidate_review_count"] > 0:
        raise ValueError(f"review case status failed: candidate_review_count={report['candidate_review_count']}")


def save_review_case_reference_command(args: argparse.Namespace) -> None:
    master = load_transcript_document(args.input, source_language=args.source_language)
    report = save_review_case_reference(args.case_index, case_id=args.case_id, master=master)
    emit(
        args,
        report,
        f"review case reference saved: {report['case_id']} segments={report['segments']} review={report['review_count']}",
    )


def attach_review_case_candidates_command(args: argparse.Namespace) -> None:
    report = attach_review_case_candidates(
        args.case_index,
        args.plan,
        replace=args.replace,
        source_language=args.source_language,
    )
    emit(
        args,
        report,
        f"review case candidates attached: cases={report['candidate_count']} replace={report['replace']}",
    )


def build_case_candidate_attach_plan_command(args: argparse.Namespace) -> None:
    report = build_case_candidate_attach_plan(
        args.case_index,
        args.candidate_dir,
        output=args.output,
        candidate_id=args.candidate_id,
    )
    emit(
        args,
        report,
        f"candidate attach plan built: {args.output} candidates={report['candidate_count']}",
    )


def transcribe_review_case_candidates_command(args: argparse.Namespace) -> None:
    report = transcribe_review_case_candidates(
        args.case_index,
        output_dir=args.output,
        model_endpoint=model_endpoint_from_args(args),
        project_root=args.project_root,
        source_language=args.source_language,
        transcribe_audio_func=transcribe_audio,
    )
    emit(
        args,
        report,
        f"review case candidates transcribed: {args.output} candidates={report['candidate_count']}",
    )


def align_review_case_candidates_command(args: argparse.Namespace) -> None:
    command = os.environ.get("CASRT_ALIGNER_COMMAND")
    if not command:
        raise ValueError("CASRT_ALIGNER_COMMAND is required")
    report = align_review_case_candidates(
        args.case_index,
        output_dir=args.output,
        command=shlex.split(command),
        candidate_id=args.candidate_id,
        source_language=args.source_language,
    )
    emit(
        args,
        report,
        f"review case candidates aligned: {args.output} candidates={report['candidate_count']}",
    )


def freeze_case_references(args: argparse.Namespace) -> None:
    report = freeze_case_references_batch(
        args.case_index,
        output_dir=args.output,
        reference_type=args.reference_type,
        reference_notes=args.reference_notes,
        fail_on_review=args.fail_on_review,
        source_language=args.source_language,
    )
    emit(
        args,
        report,
        f"case references frozen: {args.output} cases={report['case_count']} type={report['reference_type']}",
    )


def build_eval_manifest(args: argparse.Namespace) -> None:
    manifest = build_eval_manifest_from_case_index(
        args.case_index,
        reference_type=args.reference_type,
        reference_notes=args.reference_notes,
        fail_on_review=args.fail_on_review,
        source_language=args.source_language,
    )
    write_text(args.output, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    report = {
        "format": EVAL_MANIFEST_BUILD_FORMAT,
        "output": str(args.output),
        "case_index": str(args.case_index),
        "case_count": len(manifest["cases"]),
        "reference_type": manifest["reference_type"],
    }
    if "reference_notes" in manifest:
        report["reference_notes"] = manifest["reference_notes"]
    emit(
        args,
        report,
        f"eval manifest built: {args.output} cases={report['case_count']} type={report['reference_type']}",
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


def compare_evals(args: argparse.Namespace) -> None:
    report = compare_eval_reports(args.reports)
    annotate_comparison_quality_gates(report, args)
    if args.output is not None:
        write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    best = report["items"][0]
    emit(
        args,
        report,
        (
            f"best={best['label']} "
            f"review_effort={best['segments_needing_edit_ratio']:.4f} "
            f"practical_cer={best['practical_cer']:.4f}"
        ),
    )


def annotate_comparison_quality_gates(report: dict[str, Any], args: argparse.Namespace) -> None:
    thresholds = quality_gate_thresholds(args)
    required_reference_type = reference_type_requirement(args)
    gate = {key: value for key, value in thresholds.items() if value is not None}
    if required_reference_type is not None:
        gate["require_reference_type"] = required_reference_type
    if getattr(args, "product_gate", False):
        gate["preset"] = "local-asmr-v1"
    if not gate:
        return

    report["quality_gate"] = gate
    for item in report["items"]:
        failures = comparison_quality_gate_failures(
            item,
            required_reference_type=required_reference_type,
            **thresholds,
        )
        item["gate_passed"] = not failures
        item["gate_failures"] = failures


def comparison_quality_gate_failures(
    item: dict[str, Any],
    *,
    max_practical_cer: float | None,
    min_time_aligned_500ms_ratio: float | None,
    min_channel_time_aligned_accuracy: float | None,
    max_channel_time_aligned_mix_ratio: float | None,
    max_segments_needing_edit_ratio: float | None,
    max_candidate_review_ratio: float | None,
    required_reference_type: str | None,
) -> list[str]:
    failures: list[str] = []
    if required_reference_type is not None:
        reference_type = item.get("reference_type")
        if reference_type is None:
            failures.append("reference_type is unavailable")
        elif reference_type != required_reference_type:
            failures.append(f"reference_type {reference_type!r} != {required_reference_type!r}")
    if max_practical_cer is not None and item["practical_cer"] > max_practical_cer:
        failures.append(f"practical CER {item['practical_cer']:.4f} > {max_practical_cer:.4f}")
    if min_time_aligned_500ms_ratio is not None:
        ratio = item["time_aligned_500ms_ratio"]
        if ratio is None:
            failures.append("time-aligned 500ms ratio is unavailable")
        elif ratio < min_time_aligned_500ms_ratio:
            failures.append(f"time-aligned 500ms ratio {ratio:.4f} < {min_time_aligned_500ms_ratio:.4f}")
    if min_channel_time_aligned_accuracy is not None:
        accuracy = item["channel_time_aligned_accuracy"]
        if accuracy is None:
            failures.append("channel time-aligned accuracy is unavailable")
        elif accuracy < min_channel_time_aligned_accuracy:
            failures.append(
                f"channel time-aligned accuracy {accuracy:.4f} < {min_channel_time_aligned_accuracy:.4f}"
            )
    if (
        max_channel_time_aligned_mix_ratio is not None
        and item["channel_time_aligned_mix_ratio"] > max_channel_time_aligned_mix_ratio
    ):
        failures.append(
            "channel time-aligned MIX ratio "
            f"{item['channel_time_aligned_mix_ratio']:.4f} > {max_channel_time_aligned_mix_ratio:.4f}"
        )
    if (
        max_segments_needing_edit_ratio is not None
        and item["segments_needing_edit_ratio"] > max_segments_needing_edit_ratio
    ):
        failures.append(
            "segments needing edit ratio "
            f"{item['segments_needing_edit_ratio']:.4f} > {max_segments_needing_edit_ratio:.4f}"
        )
    if max_candidate_review_ratio is not None:
        candidate_review_ratio = item.get("candidate_review_ratio")
        if candidate_review_ratio is None:
            failures.append("candidate review ratio is unavailable")
        elif candidate_review_ratio > max_candidate_review_ratio:
            failures.append(
                f"candidate review ratio {candidate_review_ratio:.4f} > {max_candidate_review_ratio:.4f}"
            )
    return failures


def review_pack(args: argparse.Namespace) -> None:
    report = build_review_pack(
        json.loads(read_text(args.review_effort)),
        output_dir=args.output,
        audio_file=args.audio,
        audio_map_file=args.audio_map,
        source_case_index=args.source_case_index,
    )
    emit(
        args,
        report,
        f"review pack: {args.output} clips={report['clip_count']}",
    )


def review_case_pack(args: argparse.Namespace) -> None:
    report = build_review_case_pack(
        args.case_index,
        output_dir=args.output,
        context_ms=args.context_ms,
        source_language=args.source_language,
    )
    emit(
        args,
        report,
        f"review case pack: {args.output} clips={report['clip_count']}",
    )


def enforce_quality_gate(metrics: dict[str, Any], args: argparse.Namespace) -> None:
    failures: list[str] = []
    thresholds = quality_gate_thresholds(args)
    max_practical_cer = thresholds["max_practical_cer"]
    min_time_aligned_500ms_ratio = thresholds["min_time_aligned_500ms_ratio"]
    min_channel_time_aligned_accuracy = thresholds["min_channel_time_aligned_accuracy"]
    max_channel_time_aligned_mix_ratio = thresholds["max_channel_time_aligned_mix_ratio"]
    max_segments_needing_edit_ratio = thresholds["max_segments_needing_edit_ratio"]
    max_candidate_review_ratio = thresholds["max_candidate_review_ratio"]

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

    if max_candidate_review_ratio is not None:
        candidate_review_ratio = float(metrics["review"]["candidate_review_ratio"])
        if candidate_review_ratio > max_candidate_review_ratio:
            failures.append(
                f"candidate review ratio {candidate_review_ratio:.4f} > {max_candidate_review_ratio:.4f}"
            )

    if failures:
        raise ValueError("quality gate failed: " + "; ".join(failures))


def enforce_reference_type_gate(report: dict[str, Any], args: argparse.Namespace) -> None:
    required = reference_type_requirement(args)
    if required is None:
        return

    failures = []
    for case in report.get("cases", []):
        case_type = case.get("reference_type") or "unspecified"
        if case_type != required:
            failures.append(f"{case.get('id', '<unknown>')} reference_type {case_type!r} != {required!r}")

    if failures:
        raise ValueError("reference type gate failed: " + "; ".join(failures))


def quality_gate_thresholds(args: argparse.Namespace) -> dict[str, float | None]:
    return {
        "max_practical_cer": quality_gate_ratio(args, "max_practical_cer", "--max-practical-cer"),
        "min_time_aligned_500ms_ratio": quality_gate_ratio(
            args,
            "min_time_aligned_500ms_ratio",
            "--min-time-aligned-500ms-ratio",
        ),
        "min_channel_time_aligned_accuracy": quality_gate_ratio(
            args,
            "min_channel_time_aligned_accuracy",
            "--min-channel-time-aligned-accuracy",
        ),
        "max_channel_time_aligned_mix_ratio": quality_gate_ratio(
            args,
            "max_channel_time_aligned_mix_ratio",
            "--max-channel-time-aligned-mix-ratio",
        ),
        "max_segments_needing_edit_ratio": quality_gate_ratio(
            args,
            "max_segments_needing_edit_ratio",
            "--max-segments-needing-edit-ratio",
        ),
        "max_candidate_review_ratio": quality_gate_ratio(
            args,
            "max_candidate_review_ratio",
            "--max-candidate-review-ratio",
        ),
    }


def quality_gate_ratio(args: argparse.Namespace, name: str, option_name: str) -> float | None:
    value = getattr(args, name)
    if value is None and getattr(args, "product_gate", False):
        value = PRODUCT_GATE_THRESHOLDS[name]
    return ratio_arg(value, option_name)


def reference_type_requirement(args: argparse.Namespace) -> str | None:
    required = getattr(args, "require_reference_type", None)
    if required is not None:
        return required
    if getattr(args, "product_gate", False):
        return PRODUCT_GATE_REFERENCE_TYPE
    return None


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


def project_save_master(args: argparse.Namespace) -> None:
    store = store_from_args(args)
    master = MasterDocument.from_json(json.loads(read_text(args.input)))
    saved = store.save_master(args.project_id, master)
    payload = {
        "project_id": args.project_id,
        "segment_count": len(master.segments),
        "review_count": sum(1 for segment in master.segments if segment.needs_review),
        "master": saved["master"],
    }
    emit(
        args,
        payload,
        f"master saved: segments={payload['segment_count']} review={payload['review_count']}",
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


def vad_coverage_interval_source(
    audio_bytes: bytes,
    *,
    audio_duration_ms: int,
    intervals_path: Path | None,
    vad_adapter_command: str | None,
) -> tuple[str, tuple[dict[str, int], ...]]:
    from custom_asmr_srt_stack.audio import speech_intervals_by_energy
    from custom_asmr_srt_stack.vad import parse_vad_intervals, run_vad_command
    from custom_asmr_srt_stack.workflow import qwen_energy_chunk_kwargs, qwen_energy_max_chunk_ms, split_long_chunks

    if intervals_path is not None and vad_adapter_command is not None:
        raise ValueError("--intervals and --vad-command cannot be used together")
    if intervals_path is not None:
        return str(intervals_path), parse_vad_intervals(
            json.loads(read_text(intervals_path)),
            duration_ms=audio_duration_ms,
        )
    if vad_adapter_command is not None:
        return f"command:{vad_adapter_command}", run_vad_command(
            audio_bytes,
            command=shlex.split(vad_adapter_command),
        )

    intervals = speech_intervals_by_energy(audio_bytes, **qwen_energy_chunk_kwargs())
    max_energy_chunk_ms = qwen_energy_max_chunk_ms()
    if max_energy_chunk_ms is not None:
        intervals = split_long_chunks(intervals, max_energy_chunk_ms)
    return "energy", intervals


def vad_coverage(args: argparse.Namespace) -> None:
    from custom_asmr_srt_stack.audio import analyze_wav
    from custom_asmr_srt_stack.vad import vad_coverage_report

    audio_bytes = args.audio.read_bytes()
    audio_info = analyze_wav(audio_bytes)
    interval_source, intervals = vad_coverage_interval_source(
        audio_bytes,
        audio_duration_ms=audio_info.duration_ms,
        intervals_path=args.intervals,
        vad_adapter_command=args.vad_adapter_command,
    )

    reference = load_transcript_document(args.reference, source_language=args.source_language)
    report = vad_coverage_report(
        reference=reference,
        intervals=intervals,
        audio_duration_ms=audio_info.duration_ms,
        source=interval_source,
    )
    if args.output is not None:
        write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        report,
        (
            f"vad coverage: recall={report['reference_recall']} "
            f"precision={report['detected_precision']} intervals={report['detected_interval_count']}"
        ),
    )


def vad_coverage_cases(args: argparse.Namespace) -> None:
    from custom_asmr_srt_stack.audio import analyze_wav
    from custom_asmr_srt_stack.vad import (
        VAD_COVERAGE_SUITE_FORMAT,
        aggregate_vad_coverage_reports,
        vad_coverage_report,
    )

    case_index = load_review_case_index(args.case_index)
    raw_items = case_index.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review case index items must be an array")

    case_reports = []
    reports = []
    interval_source = f"command:{args.vad_adapter_command}" if args.vad_adapter_command is not None else "energy"
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review case index item {index} must be an object")
        case_id = require_index_string(raw_item, "id", index)
        audio_value = require_index_string(raw_item, "audio", index)
        reference_value = require_index_string(raw_item, "reference", index)
        audio_path = resolve_plan_path(args.case_index.parent, audio_value)
        reference_path = resolve_plan_path(args.case_index.parent, reference_value)
        audio_bytes = audio_path.read_bytes()
        audio_info = analyze_wav(audio_bytes)
        case_interval_source, intervals = vad_coverage_interval_source(
            audio_bytes,
            audio_duration_ms=audio_info.duration_ms,
            intervals_path=None,
            vad_adapter_command=args.vad_adapter_command,
        )
        interval_source = case_interval_source
        reference = load_transcript_document(reference_path, source_language=args.source_language)
        report = vad_coverage_report(
            reference=reference,
            intervals=intervals,
            audio_duration_ms=audio_info.duration_ms,
            source=case_interval_source,
        )
        reports.append(report)
        case_reports.append(
            {
                "id": case_id,
                "audio": audio_value,
                "reference": reference_value,
                "report": report,
            }
        )

    summary = aggregate_vad_coverage_reports(reports)
    suite_report = {
        "format": VAD_COVERAGE_SUITE_FORMAT,
        "case_index": str(args.case_index),
        "source": interval_source,
        "case_count": len(case_reports),
        "summary": summary,
        "cases": case_reports,
    }
    if args.output is not None:
        write_text(args.output, json.dumps(suite_report, ensure_ascii=False, indent=2) + "\n")
    emit(
        args,
        suite_report,
        (
            f"vad coverage cases: cases={summary['case_count']} "
            f"recall={summary['reference_recall']} precision={summary['detected_precision']}"
        ),
    )


def vad_compare_coverage(args: argparse.Namespace) -> None:
    from custom_asmr_srt_stack.vad import compare_vad_coverage_reports

    report = compare_vad_coverage_reports(args.reports)
    annotate_vad_coverage_gates(report, args)
    if args.output is not None:
        write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    best = report["items"][0]
    emit(
        args,
        report,
        (
            f"best={best['label']} "
            f"recall={best['reference_recall']} "
            f"precision={best['detected_precision']} "
            f"missed_ms={best['missed_reference_duration_ms']}"
        ),
    )


def annotate_vad_coverage_gates(report: dict[str, Any], args: argparse.Namespace) -> None:
    max_detected_interval_ms = getattr(args, "max_detected_interval_ms", None)
    if max_detected_interval_ms is None:
        return
    if max_detected_interval_ms <= 0:
        raise ValueError("--max-detected-interval-ms must be positive")
    report["quality_gate"] = {"max_detected_interval_ms": max_detected_interval_ms}
    for item in report["items"]:
        gate_failures = []
        detected_max_interval_ms = item.get("detected_max_interval_ms")
        if detected_max_interval_ms is not None and detected_max_interval_ms > max_detected_interval_ms:
            gate_failures.append(
                f"detected max interval {detected_max_interval_ms:g}ms > {max_detected_interval_ms:g}ms"
            )
        item["gate_passed"] = not gate_failures
        item["gate_failures"] = gate_failures


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
            "local-granite-asr",
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
    align_transcript_parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Write per-segment timing delta diagnostics JSON.",
    )
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
    attribute_channels_parser.add_argument(
        "--quiet-channel-max-dbfs",
        type=float,
        default=CHANNEL_ATTRIBUTION_QUIET_MAX_DBFS,
        help="Keep MIX unless the quieter side is at or below this dBFS value.",
    )
    attribute_channels_parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Write per-segment L/R energy attribution diagnostics JSON.",
    )
    attribute_channels_parser.set_defaults(func=attribute_channels)

    sweep_channel_attribution_parser = subcommands.add_parser(
        "sweep-channel-attribution",
        parents=[output_parent],
        help="Evaluate channel attribution thresholds over an eval manifest and audio map.",
    )
    sweep_channel_attribution_parser.add_argument("manifest", type=Path)
    sweep_channel_attribution_parser.add_argument("--audio-map", type=Path, required=True)
    sweep_channel_attribution_parser.add_argument("-o", "--output", type=Path, required=True)
    sweep_channel_attribution_parser.add_argument(
        "--threshold-db",
        type=float,
        action="append",
        help="Threshold to test. Repeat for multiple values; default is the product threshold.",
    )
    sweep_channel_attribution_parser.add_argument(
        "--quiet-channel-max-dbfs",
        type=float,
        action="append",
        help="Quiet-side gate to test. Repeat for multiple values; default is the product quiet-side gate.",
    )
    sweep_channel_attribution_parser.add_argument(
        "--reset-speech-channels-to-mix",
        action="store_true",
        help="Reset every speech candidate channel to MIX before applying each attribution setting.",
    )
    sweep_channel_attribution_parser.add_argument("--source-language", default="ja")
    add_quality_gate_args(sweep_channel_attribution_parser, action_verb="Annotate comparison")
    sweep_channel_attribution_parser.set_defaults(func=sweep_channel_attribution_command)

    slice_case_parser = subcommands.add_parser(
        "slice-case",
        parents=[output_parent],
        help="Slice matching audio and SRT/master transcript into a rebased eval case.",
    )
    slice_case_parser.add_argument("audio", type=Path)
    slice_case_parser.add_argument("transcript", type=Path)
    slice_case_parser.add_argument("--start-ms", type=int, required=True)
    slice_case_parser.add_argument("--end-ms", type=int, required=True)
    slice_case_parser.add_argument("--audio-output", type=Path, required=True)
    slice_case_parser.add_argument("--transcript-output", type=Path, required=True)
    slice_case_parser.add_argument("--source-language", default="ja")
    slice_case_parser.set_defaults(func=slice_case)

    prepare_review_cases_parser = subcommands.add_parser(
        "prepare-review-cases",
        parents=[output_parent],
        help="Prepare multiple sliced review/eval cases from a JSON plan.",
    )
    prepare_review_cases_parser.add_argument("plan", type=Path)
    prepare_review_cases_parser.add_argument("-o", "--output", type=Path, required=True)
    prepare_review_cases_parser.add_argument("--source-language", default="ja")
    prepare_review_cases_parser.set_defaults(func=prepare_review_cases_command)

    review_case_status_parser = subcommands.add_parser(
        "review-case-status",
        parents=[output_parent],
        help="Report prepared review case integrity and remaining reference review flags.",
    )
    review_case_status_parser.add_argument("case_index", type=Path)
    review_case_status_parser.add_argument("-o", "--output", type=Path)
    review_case_status_parser.add_argument("--source-language", default="ja")
    review_case_status_parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Return a failing exit code after emitting the report if files are missing or counts are stale.",
    )
    review_case_status_parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Return a failing exit code after emitting the report if reference segments still need review.",
    )
    review_case_status_parser.add_argument(
        "--fail-on-missing-candidates",
        action="store_true",
        help="Return a failing exit code after emitting the report if cases have no candidate transcript path.",
    )
    review_case_status_parser.add_argument(
        "--fail-on-candidate-review",
        action="store_true",
        help="Return a failing exit code after emitting the report if candidate segments still need review.",
    )
    review_case_status_parser.set_defaults(func=review_case_status_command)

    save_review_case_reference_parser = subcommands.add_parser(
        "save-review-case-reference",
        parents=[output_parent],
        help="Replace one prepared review case reference and update case-index counts.",
    )
    save_review_case_reference_parser.add_argument("case_index", type=Path)
    save_review_case_reference_parser.add_argument("case_id")
    save_review_case_reference_parser.add_argument("input", type=Path)
    save_review_case_reference_parser.add_argument("--source-language", default="ja")
    save_review_case_reference_parser.set_defaults(func=save_review_case_reference_command)

    attach_review_case_candidates_parser = subcommands.add_parser(
        "attach-review-case-candidates",
        parents=[output_parent],
        help="Attach case-local candidate transcripts to an existing prepared review case index.",
    )
    attach_review_case_candidates_parser.add_argument("case_index", type=Path)
    attach_review_case_candidates_parser.add_argument("plan", type=Path)
    attach_review_case_candidates_parser.add_argument("--source-language", default="ja")
    attach_review_case_candidates_parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite existing candidate entries and candidate files.",
    )
    attach_review_case_candidates_parser.set_defaults(func=attach_review_case_candidates_command)

    build_case_candidate_attach_plan_parser = subcommands.add_parser(
        "build-candidate-attach-plan",
        parents=[output_parent],
        help="Build a candidate attach plan by matching candidate files to review case ids.",
    )
    build_case_candidate_attach_plan_parser.add_argument("case_index", type=Path)
    build_case_candidate_attach_plan_parser.add_argument("candidate_dir", type=Path)
    build_case_candidate_attach_plan_parser.add_argument("-o", "--output", type=Path, required=True)
    build_case_candidate_attach_plan_parser.add_argument(
        "--candidate-id",
        help="Candidate id to store at the plan level for every matched case.",
    )
    build_case_candidate_attach_plan_parser.set_defaults(func=build_case_candidate_attach_plan_command)

    transcribe_review_case_candidates_parser = subcommands.add_parser(
        "transcribe-review-case-candidates",
        parents=[project_parent, model_parent, output_parent],
        help="Transcribe every prepared review case audio into case-local candidate master files.",
    )
    transcribe_review_case_candidates_parser.add_argument("case_index", type=Path)
    transcribe_review_case_candidates_parser.add_argument("-o", "--output", type=Path, required=True)
    transcribe_review_case_candidates_parser.add_argument("--source-language", default="ja")
    transcribe_review_case_candidates_parser.set_defaults(func=transcribe_review_case_candidates_command)

    align_review_case_candidates_parser = subcommands.add_parser(
        "align-review-case-candidates",
        parents=[output_parent],
        help="Align every prepared review case candidate with CASRT_ALIGNER_COMMAND.",
    )
    align_review_case_candidates_parser.add_argument("case_index", type=Path)
    align_review_case_candidates_parser.add_argument("-o", "--output", type=Path, required=True)
    align_review_case_candidates_parser.add_argument("--source-language", default="ja")
    align_review_case_candidates_parser.add_argument(
        "--candidate-id",
        help="Candidate id to store in the generated attach plan and eval manifest.",
    )
    align_review_case_candidates_parser.set_defaults(func=align_review_case_candidates_command)

    freeze_case_references_parser = subcommands.add_parser(
        "freeze-case-references",
        parents=[output_parent],
        help="Freeze every reference in a prepared case index into a new case set.",
    )
    freeze_case_references_parser.add_argument("case_index", type=Path)
    freeze_case_references_parser.add_argument("-o", "--output", type=Path, required=True)
    freeze_case_references_parser.add_argument("--source-language", default="ja")
    freeze_case_references_parser.add_argument(
        "--reference-type",
        default="human-reviewed",
        help="Reference authority for the frozen case set. Use human-reviewed only after manual review.",
    )
    freeze_case_references_parser.add_argument("--reference-notes")
    freeze_case_references_parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Fail before writing output if any reference segment still has needs_review=true.",
    )
    freeze_case_references_parser.set_defaults(func=freeze_case_references)

    build_eval_manifest_parser = subcommands.add_parser(
        "build-eval-manifest",
        parents=[output_parent],
        help="Build an eval manifest from a prepared review case index with candidates.",
    )
    build_eval_manifest_parser.add_argument("case_index", type=Path)
    build_eval_manifest_parser.add_argument("-o", "--output", type=Path, required=True)
    build_eval_manifest_parser.add_argument("--source-language", default="ja")
    build_eval_manifest_parser.add_argument(
        "--reference-type",
        help="Override the manifest reference_type, e.g. human-reviewed after manual review.",
    )
    build_eval_manifest_parser.add_argument("--reference-notes")
    build_eval_manifest_parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Fail if reference segments still have needs_review=true.",
    )
    build_eval_manifest_parser.set_defaults(func=build_eval_manifest)

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

    compare_evals_parser = subcommands.add_parser(
        "compare-evals",
        help="Rank eval-transcript/eval-manifest reports by review effort and quality metrics.",
    )
    compare_evals_parser.add_argument("reports", type=Path, nargs="+")
    compare_evals_parser.add_argument("-o", "--output", type=Path)
    compare_evals_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    add_quality_gate_args(compare_evals_parser, action_verb="Mark")
    compare_evals_parser.set_defaults(func=compare_evals)

    review_pack_parser = subcommands.add_parser(
        "review-pack",
        help="Create audio clips from a review-effort JSON report.",
    )
    review_pack_parser.add_argument("review_effort", type=Path)
    review_pack_parser.add_argument("-o", "--output", type=Path, required=True)
    review_pack_parser.add_argument("--audio", type=Path, help="Single WAV file for reports without case audio mapping.")
    review_pack_parser.add_argument("--audio-map", type=Path, help="JSON map from case_id to WAV file.")
    review_pack_parser.add_argument(
        "--source-case-index",
        type=Path,
        help="Prepared review case-index.json to attach so WebUI can open the source case from pack items.",
    )
    review_pack_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    review_pack_parser.set_defaults(func=review_pack)

    review_case_pack_parser = subcommands.add_parser(
        "review-case-pack",
        help="Create audio clips from prepared review case reference flags.",
    )
    review_case_pack_parser.add_argument("case_index", type=Path)
    review_case_pack_parser.add_argument("-o", "--output", type=Path, required=True)
    review_case_pack_parser.add_argument("--source-language", default="ja")
    review_case_pack_parser.add_argument(
        "--context-ms",
        type=int,
        default=DEFAULT_REVIEW_CONTEXT_MS,
        help="Milliseconds of audio context to include before and after each review segment.",
    )
    review_case_pack_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    review_case_pack_parser.set_defaults(func=review_case_pack)

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
    vad_coverage_parser = vad_subcommands.add_parser(
        "coverage",
        parents=[output_parent],
        help="Compare VAD/chunk intervals against a reference transcript.",
    )
    vad_coverage_parser.add_argument("audio", type=Path)
    vad_coverage_parser.add_argument("reference", type=Path)
    vad_coverage_parser.add_argument("--source-language", default="ja")
    vad_coverage_source = vad_coverage_parser.add_mutually_exclusive_group()
    vad_coverage_source.add_argument("--intervals", type=Path, help="JSON file with {intervals:[{start_ms,end_ms}]}.")
    vad_coverage_source.add_argument(
        "--vad-command",
        dest="vad_adapter_command",
        help="CASRT VAD command to run for this audio instead of built-in energy intervals.",
    )
    vad_coverage_parser.add_argument("-o", "--output", type=Path)
    vad_coverage_parser.set_defaults(func=vad_coverage)
    vad_coverage_cases_parser = vad_subcommands.add_parser(
        "coverage-cases",
        parents=[output_parent],
        help="Compare VAD/chunk intervals against every reference in a prepared review case set.",
    )
    vad_coverage_cases_parser.add_argument("case_index", type=Path)
    vad_coverage_cases_parser.add_argument("--source-language", default="ja")
    vad_coverage_cases_parser.add_argument(
        "--vad-command",
        dest="vad_adapter_command",
        help="CASRT VAD command to run for each case instead of built-in energy intervals.",
    )
    vad_coverage_cases_parser.add_argument("-o", "--output", type=Path)
    vad_coverage_cases_parser.set_defaults(func=vad_coverage_cases)
    vad_compare_coverage_parser = vad_subcommands.add_parser(
        "compare-coverage",
        parents=[output_parent],
        help="Compare VAD coverage reports and rank candidates by missed speech, then extra detection.",
    )
    vad_compare_coverage_parser.add_argument("reports", type=Path, nargs="+")
    vad_compare_coverage_parser.add_argument("-o", "--output", type=Path)
    vad_compare_coverage_parser.add_argument(
        "--max-detected-interval-ms",
        type=int,
        help="Mark candidates with a detected chunk longer than this as gate failures.",
    )
    vad_compare_coverage_parser.set_defaults(func=vad_compare_coverage)

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

    save_master = project_subcommands.add_parser(
        "save-master",
        parents=[project_parent, output_parent],
        help="Replace a project master JSON.",
    )
    save_master.add_argument("project_id")
    save_master.add_argument("input", type=Path)
    save_master.set_defaults(func=project_save_master)

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


def add_quality_gate_args(parser: argparse.ArgumentParser, *, action_verb: str = "Fail") -> None:
    parser.add_argument(
        "--product-gate",
        action="store_true",
        help=(
            f"{action_verb} with the documented local ASMR product gate "
            "(human-reviewed reference where applicable, practical CER <= 0.10, "
            "time-aligned 500ms >= 0.90, L/R channel accuracy >= 0.85, "
            "candidate MIX <= 0.50, review effort <= 0.15, candidate review ratio = 0)."
        ),
    )
    parser.add_argument(
        "--max-practical-cer",
        type=float,
        help=f"{action_verb} if practical CER is above this 0..1 ratio.",
    )
    parser.add_argument(
        "--min-time-aligned-500ms-ratio",
        type=float,
        help=f"{action_verb} if time-aligned 500ms boundary ratio is below this 0..1 ratio.",
    )
    parser.add_argument(
        "--min-channel-time-aligned-accuracy",
        type=float,
        help=f"{action_verb} if time-aligned L/R channel accuracy is below this 0..1 ratio.",
    )
    parser.add_argument(
        "--max-channel-time-aligned-mix-ratio",
        type=float,
        help=f"{action_verb} if time-aligned candidate MIX ratio is above this 0..1 ratio.",
    )
    parser.add_argument(
        "--max-segments-needing-edit-ratio",
        type=float,
        help=f"{action_verb} if review effort segment edit ratio is above this 0..1 ratio.",
    )
    parser.add_argument(
        "--max-candidate-review-ratio",
        type=float,
        help=f"{action_verb} if unresolved candidate needs_review ratio is above this 0..1 ratio.",
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
