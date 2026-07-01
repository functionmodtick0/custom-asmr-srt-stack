from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.audio import normalize_audio_to_wav, split_wav_channels
from custom_asmr_srt_stack.channel_attribution import attribute_master_channels_by_energy, channel_diagnostics_summary
from custom_asmr_srt_stack.evaluation import (
    EVAL_MANIFEST_FORMAT,
    compare_eval_reports,
    evaluate_manifest,
    load_transcript_document,
    optional_manifest_string,
    resolve_manifest_path,
    validate_eval_manifest,
)
from custom_asmr_srt_stack.models import MasterDocument
from custom_asmr_srt_stack.review_pack import load_audio_by_case

CHANNEL_ATTRIBUTION_SWEEP_FORMAT = "custom-asmr-channel-attribution-sweep-v1"


def sweep_channel_attribution(
    manifest_file: Path,
    *,
    audio_map_file: Path,
    output_dir: Path,
    threshold_db_values: list[float],
    quiet_channel_max_dbfs_values: list[float],
    source_language: str = "ja",
    reset_speech_channels_to_mix: bool = False,
) -> dict[str, Any]:
    if not threshold_db_values:
        raise ValueError("channel attribution sweep requires at least one threshold")
    if not quiet_channel_max_dbfs_values:
        raise ValueError("channel attribution sweep requires at least one quiet-channel max dBFS value")
    for threshold_db in threshold_db_values:
        if threshold_db < 0:
            raise ValueError("channel attribution sweep threshold values must be non-negative")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    cases = validate_eval_manifest(manifest)
    manifest_base_dir = manifest_file.parent
    audio_by_case = load_audio_by_case(audio_file=None, audio_map_file=audio_map_file)
    validate_sweep_source_files(cases, manifest_base_dir=manifest_base_dir, audio_by_case=audio_by_case)
    prepare_output_dir(output_dir)

    source_reference_type = optional_manifest_string(manifest, "reference_type") or "unspecified"
    source_reference_notes = optional_manifest_string(manifest, "reference_notes")
    items: list[dict[str, Any]] = []
    report_files: list[Path] = []

    for threshold_db in threshold_db_values:
        for quiet_channel_max_dbfs in quiet_channel_max_dbfs_values:
            setting_id = channel_sweep_setting_id(
                threshold_db=threshold_db,
                quiet_channel_max_dbfs=quiet_channel_max_dbfs,
            )
            setting_dir = output_dir / setting_id
            candidate_dir = setting_dir / "candidates"
            candidate_dir.mkdir(parents=True)
            setting_cases = []
            changed_segments = 0
            total_segments = 0
            setting_diagnostics: list[dict[str, Any]] = []

            for case in cases:
                case_id = case["id"]
                audio_path = audio_path_for_case(audio_by_case, case_id)
                normalized_audio = normalize_audio_to_wav(
                    audio_path.read_bytes(),
                    file_name=audio_path.name,
                    mime_type=mimetypes.guess_type(audio_path.name)[0],
                )
                _, channels = split_wav_channels(normalized_audio)
                if not {"L", "R"}.issubset(channels):
                    raise ValueError(f"channel attribution sweep case {case_id!r} requires stereo audio")
                candidate_path = resolve_manifest_path(manifest_base_dir, case["candidate"])
                candidate = load_transcript_document(candidate_path, source_language=source_language)
                if reset_speech_channels_to_mix:
                    candidate = reset_speech_channels(candidate)
                attributed = attribute_master_channels_by_energy(
                    candidate,
                    left_audio=channels["L"],
                    right_audio=channels["R"],
                    threshold_db=threshold_db,
                    quiet_channel_max_dbfs=quiet_channel_max_dbfs,
                )
                changed_segments += attributed.changed_segments
                total_segments += attributed.segments
                setting_diagnostics.extend(attributed.diagnostics)
                candidate_relative = Path("candidates") / f"{safe_case_file_stem(case_id)}.master.json"
                (setting_dir / candidate_relative).write_text(
                    json.dumps(attributed.master.to_json(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                setting_case = {
                    "id": case_id,
                    "reference": relative_path_value(
                        resolve_manifest_path(manifest_base_dir, case["reference"]),
                        base_dir=setting_dir,
                    ),
                    "candidate": str(candidate_relative),
                    "candidate_id": f"{case.get('candidate_id') or Path(case['candidate']).stem}-{setting_id}",
                    "reference_type": case.get("reference_type") or source_reference_type,
                }
                if case.get("reference_notes") or source_reference_notes:
                    setting_case["reference_notes"] = case.get("reference_notes") or source_reference_notes
                setting_cases.append(setting_case)

            setting_manifest = {
                "format": EVAL_MANIFEST_FORMAT,
                "reference_type": source_reference_type,
                "cases": setting_cases,
            }
            if source_reference_notes is not None:
                setting_manifest["reference_notes"] = source_reference_notes
            setting_manifest_file = setting_dir / f"{setting_id}.eval-manifest.json"
            setting_manifest_file.write_text(
                json.dumps(setting_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            eval_report = evaluate_manifest(setting_manifest_file, source_language=source_language)
            eval_report_file = setting_dir / f"{setting_id}.eval-report.json"
            eval_report_file.write_text(json.dumps(eval_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report_files.append(eval_report_file)
            items.append(
                {
                    "setting_id": setting_id,
                    "threshold_db": threshold_db,
                    "quiet_channel_max_dbfs": quiet_channel_max_dbfs,
                    "reset_speech_channels_to_mix": reset_speech_channels_to_mix,
                    "changed_segments": changed_segments,
                    "total_segments": total_segments,
                    **channel_diagnostics_summary(setting_diagnostics),
                    "eval_manifest": str(setting_manifest_file),
                    "eval_report": str(eval_report_file),
                }
            )

    comparison = compare_eval_reports(report_files)
    comparison_file = output_dir / "comparison.json"
    comparison_file.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "format": CHANNEL_ATTRIBUTION_SWEEP_FORMAT,
        "manifest": str(manifest_file),
        "audio_map": str(audio_map_file),
        "output": str(output_dir),
        "case_count": len(cases),
        "setting_count": len(items),
        "reset_speech_channels_to_mix": reset_speech_channels_to_mix,
        "items": items,
        "comparison": str(comparison_file),
    }
    (output_dir / "index.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("channel attribution sweep output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)


def reset_speech_channels(master: MasterDocument) -> MasterDocument:
    return replace(
        master,
        segments=tuple(
            replace(segment, channel="MIX") if segment.kind == "speech" else segment for segment in master.segments
        ),
    )


def relative_path_value(path: Path, *, base_dir: Path) -> str:
    return Path(os.path.relpath(path.expanduser().resolve(), base_dir.expanduser().resolve())).as_posix()


def validate_sweep_source_files(
    cases: list[dict[str, str]],
    *,
    manifest_base_dir: Path,
    audio_by_case: dict[str | None, Path],
) -> None:
    for case in cases:
        case_id = case["id"]
        audio_path = audio_path_for_case(audio_by_case, case_id)
        if not audio_path.is_file():
            raise ValueError(f"channel attribution sweep audio file does not exist: {audio_path}")
        candidate_path = resolve_manifest_path(manifest_base_dir, case["candidate"])
        if not candidate_path.is_file():
            raise ValueError(f"channel attribution sweep candidate file does not exist: {candidate_path}")
        reference_path = resolve_manifest_path(manifest_base_dir, case["reference"])
        if not reference_path.is_file():
            raise ValueError(f"channel attribution sweep reference file does not exist: {reference_path}")


def audio_path_for_case(audio_by_case: dict[str | None, Path], case_id: str) -> Path:
    try:
        return audio_by_case[case_id]
    except KeyError as error:
        raise ValueError(f"audio map is missing case_id {case_id!r}") from error


def channel_sweep_setting_id(*, threshold_db: float, quiet_channel_max_dbfs: float) -> str:
    return f"th{safe_float_id(threshold_db)}_quiet{safe_float_id(quiet_channel_max_dbfs)}"


def safe_float_id(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def safe_case_file_stem(case_id: str) -> str:
    if case_id in {".", ".."} or Path(case_id).name != case_id:
        raise ValueError("channel attribution sweep case id must be a safe file name")
    return case_id
