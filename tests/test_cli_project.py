import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from custom_asmr_srt_stack.audio import analyze_wav
from custom_asmr_srt_stack.cli import main
from custom_asmr_srt_stack.models import MasterDocument, Segment


def run_cli(argv):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result = main(argv)
    return result, stdout.getvalue()


def run_cli_with_stderr(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = main(argv)
    return result, stdout.getvalue(), stderr.getvalue()


def write_mono_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(struct.pack("<hh", 100, 200))


def write_stereo_samples(path: Path, samples: list[tuple[int, int]]) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        frames = bytearray()
        for left, right in samples:
            frames.extend(struct.pack("<hh", left, right))
        wav.writeframes(bytes(frames))


class ProjectCliTests(unittest.TestCase):
    def test_create_srt_show_and_export_project_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            srt_path = root / "input.srt"
            project_root = root / "projects"
            master_out = root / "master.json"
            translation_out = root / "translation.json"
            srt_out = root / "out.srt"
            srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")

            result, output = run_cli(
                [
                    "project",
                    "create-srt",
                    "--project-root",
                    str(project_root),
                    "--json",
                    str(srt_path),
                ]
            )
            project_id = json.loads(output)["project_id"]

            self.assertEqual(result, 0)
            self.assertTrue((project_root / project_id / "master.json").exists())

            _, show_output = run_cli(["project", "show", "--project-root", str(project_root), "--json", project_id])
            self.assertEqual(json.loads(show_output)["segment_count"], 1)

            export_master_result, _ = run_cli(
                ["project", "export-master", "--project-root", str(project_root), project_id, "-o", str(master_out)]
            )
            export_translation_result, _ = run_cli(
                [
                    "project",
                    "export-translation",
                    "--project-root",
                    str(project_root),
                    project_id,
                    "-o",
                    str(translation_out),
                ]
            )
            export_srt_result, _ = run_cli(
                ["project", "export-srt", "--project-root", str(project_root), project_id, "-o", str(srt_out)]
            )

            self.assertEqual(export_master_result, 0)
            self.assertEqual(export_translation_result, 0)
            self.assertEqual(export_srt_result, 0)

            self.assertEqual(json.loads(master_out.read_text(encoding="utf-8"))["segments"][0]["text"], "ねえ")
            self.assertEqual(json.loads(translation_out.read_text(encoding="utf-8"))["items"][0]["text"], "ねえ")
            self.assertIn("ねえ", srt_out.read_text(encoding="utf-8"))

    def test_save_master_replaces_project_master_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_root = root / "projects"
            srt_path = root / "input.srt"
            edited_master = root / "edited.master.json"
            srt_out = root / "edited.srt"
            srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            _, output = run_cli(
                [
                    "project",
                    "create-srt",
                    "--project-root",
                    str(project_root),
                    "--json",
                    str(srt_path),
                ]
            )
            project_id = json.loads(output)["project_id"]
            master = json.loads((project_root / project_id / "master.json").read_text(encoding="utf-8"))
            master["segments"][0].update(
                {
                    "start_ms": 1100,
                    "end_ms": 1900,
                    "channel": "L",
                    "text": "直した",
                    "needs_review": True,
                }
            )
            edited_master.write_text(json.dumps(master), encoding="utf-8")

            result, save_output = run_cli(
                [
                    "project",
                    "save-master",
                    "--project-root",
                    str(project_root),
                    "--json",
                    project_id,
                    str(edited_master),
                ]
            )

            self.assertEqual(result, 0)
            saved = json.loads(save_output)
            self.assertEqual(saved["segment_count"], 1)
            self.assertEqual(saved["review_count"], 1)
            self.assertEqual(saved["master"]["segments"][0]["channel"], "L")
            _, show_output = run_cli(["project", "show", "--project-root", str(project_root), "--json", project_id])
            self.assertEqual(json.loads(show_output)["review_count"], 1)
            run_cli(["project", "export-srt", "--project-root", str(project_root), project_id, "-o", str(srt_out)])
            exported_srt = srt_out.read_text(encoding="utf-8")
            self.assertIn("00:00:01,100 --> 00:00:01,900", exported_srt)
            self.assertIn("直した", exported_srt)

    def test_create_audio_and_analyze_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "voice.wav"
            project_root = root / "projects"
            write_mono_wav(audio_path)

            _, output = run_cli(
                [
                    "project",
                    "create-audio",
                    "--project-root",
                    str(project_root),
                    "--json",
                    str(audio_path),
                ]
            )
            project_id = json.loads(output)["project_id"]

            result, analyze_output = run_cli(
                ["project", "analyze", "--project-root", str(project_root), "--json", project_id]
            )

            self.assertEqual(result, 0)
            metadata = json.loads(analyze_output)["metadata"]
            self.assertEqual(set(metadata["channels"]), {"MIX"})
            self.assertEqual(metadata["audio_info"]["duration_ms"], 2)

    def test_model_validate_outputs_contract(self):
        result, output = run_cli(
            [
                "model",
                "validate",
                "--adapter",
                "openai-compatible",
                "--endpoint-url",
                "http://localhost:8000/v1",
                "--model-id",
                "gemma-4-e4b",
                "--json",
            ]
        )

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output)["model_id"], "gemma-4-e4b")

    def test_model_validate_accepts_local_transformers_without_endpoint_url(self):
        result, output = run_cli(
            [
                "model",
                "validate",
                "--adapter",
                "local-transformers",
                "--model-id",
                "google/gemma-4-E4B-it",
                "--json",
            ]
        )

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output)["adapter"], "local-transformers")

    def test_model_validate_accepts_local_cohere_without_endpoint_url(self):
        result, output = run_cli(
            [
                "model",
                "validate",
                "--adapter",
                "local-cohere-asr",
                "--model-id",
                "/models/cohere-transcribe-03-2026",
                "--json",
            ]
        )

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output)["adapter"], "local-cohere-asr")

    def test_model_validate_accepts_local_granite_without_endpoint_url(self):
        result, output = run_cli(
            [
                "model",
                "validate",
                "--adapter",
                "local-granite-asr",
                "--model-id",
                "/models/granite-speech-4.1-2b",
                "--json",
            ]
        )

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output)["adapter"], "local-granite-asr")

    def test_model_validate_accepts_local_qwen_hf_without_endpoint_url(self):
        result, output = run_cli(
            [
                "model",
                "validate",
                "--adapter",
                "local-qwen-hf-asr",
                "--model-id",
                "/models/qwen3-asr-1.7b-hf",
                "--json",
            ]
        )

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output)["adapter"], "local-qwen-hf-asr")

    def test_model_digest_outputs_snapshot_hash_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            report_path = root / "digest.json"

            result, output = run_cli(
                [
                    "model",
                    "digest",
                    str(snapshot),
                    "-o",
                    str(report_path),
                    "--json",
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["snapshot_id"], "snapshot")
            self.assertEqual(report["file_count"], 1)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["sha256"], report["sha256"])

    def test_vad_coverage_outputs_interval_recall_and_precision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "front.wav"
            reference = root / "reference.srt"
            intervals = root / "intervals.json"
            report_path = root / "vad-coverage.json"
            write_mono_wav(audio)
            reference.write_text("1\n00:00:00,000 --> 00:00:00,002\nねえ\n", encoding="utf-8")
            intervals.write_text(json.dumps({"intervals": [{"start_ms": 0, "end_ms": 1}]}), encoding="utf-8")

            result, output = run_cli(
                [
                    "vad",
                    "coverage",
                    "--json",
                    "-o",
                    str(report_path),
                    "--intervals",
                    str(intervals),
                    str(audio),
                    str(reference),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-vad-coverage-v1")
            self.assertEqual(report["source"], str(intervals))
            self.assertEqual(report["reference_speech_duration_ms"], 2)
            self.assertEqual(report["detected_speech_duration_ms"], 1)
            self.assertEqual(report["detected_max_interval_ms"], 1)
            self.assertEqual(report["detected_mean_interval_ms"], 1)
            self.assertEqual(report["overlap_duration_ms"], 1)
            self.assertEqual(report["reference_recall"], 0.5)
            self.assertEqual(report["detected_precision"], 1.0)
            self.assertEqual(
                report["missed_reference_intervals"],
                [{"index": 0, "start_ms": 1, "end_ms": 2, "duration_ms": 1}],
            )
            self.assertEqual(report["extra_detected_intervals"], [])
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["reference_recall"], 0.5)

    def test_vad_coverage_cases_outputs_aggregate_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "front.wav"
            reference = root / "reference.srt"
            case_index = root / "case-index.json"
            report_path = root / "vad-coverage-suite.json"
            vad_script = root / "vad.py"
            write_mono_wav(audio)
            reference.write_text("1\n00:00:00,000 --> 00:00:00,002\nねえ\n", encoding="utf-8")
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [{"id": "front", "audio": "front.wav", "reference": "reference.srt"}],
                    }
                ),
                encoding="utf-8",
            )
            vad_script.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "request = json.loads(sys.stdin.read())",
                        "assert request['audio_info']['duration_ms'] == 2",
                        "print(json.dumps({'intervals': [{'start_ms': 0, 'end_ms': 1}]}))",
                    ]
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "vad",
                    "coverage-cases",
                    "--json",
                    "-o",
                    str(report_path),
                    "--vad-command",
                    f"{sys.executable} {vad_script}",
                    str(case_index),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-vad-coverage-suite-v1")
            self.assertEqual(report["case_count"], 1)
            self.assertEqual(report["source"], f"command:{sys.executable} {vad_script}")
            self.assertEqual(report["summary"]["reference_speech_duration_ms"], 2)
            self.assertEqual(report["summary"]["detected_speech_duration_ms"], 1)
            self.assertEqual(report["summary"]["detected_max_interval_ms"], 1)
            self.assertEqual(report["summary"]["detected_mean_interval_ms"], 1)
            self.assertEqual(report["summary"]["overlap_duration_ms"], 1)
            self.assertEqual(report["summary"]["reference_recall"], 0.5)
            self.assertEqual(report["summary"]["detected_precision"], 1.0)
            self.assertEqual(report["cases"][0]["id"], "front")
            self.assertEqual(
                report["cases"][0]["report"]["missed_reference_intervals"],
                [{"index": 0, "start_ms": 1, "end_ms": 2, "duration_ms": 1}],
            )
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["case_count"], 1)

    def test_vad_coverage_cases_records_energy_option_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "front.wav"
            reference = root / "reference.srt"
            case_index = root / "case-index.json"
            write_mono_wav(audio)
            reference.write_text("1\n00:00:00,000 --> 00:00:00,002\nねえ\n", encoding="utf-8")
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [{"id": "front", "audio": "front.wav", "reference": "reference.srt"}],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "vad",
                    "coverage-cases",
                    "--json",
                    "--energy-threshold-dbfs",
                    "-80",
                    "--energy-min-speech-ms",
                    "0",
                    "--energy-pad-ms",
                    "0",
                    "--energy-max-chunk-ms",
                    "1",
                    str(case_index),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["source"], "energy")
            self.assertEqual(report["source_settings"]["threshold_dbfs"], -80.0)
            self.assertEqual(report["source_settings"]["min_speech_ms"], 0)
            self.assertEqual(report["source_settings"]["pad_ms"], 0)
            self.assertEqual(report["source_settings"]["max_chunk_ms"], 1)
            self.assertEqual(report["summary"]["detected_interval_count"], 2)
            self.assertEqual(report["summary"]["detected_max_interval_ms"], 1)
            self.assertEqual(report["summary"]["reference_recall"], 1.0)

    def test_vad_compare_coverage_ranks_reports_by_missed_speech(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            low_miss = root / "low-miss.json"
            high_miss = root / "high-miss.json"
            comparison_path = root / "vad-comparison.json"
            low_miss.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-suite-v1",
                        "source": "low-miss",
                        "case_count": 1,
                        "summary": {
                            "case_count": 1,
                            "audio_duration_ms": 100,
                            "reference_segment_count": 1,
                            "reference_interval_count": 1,
                            "detected_interval_count": 1,
                            "detected_max_interval_ms": 110,
                            "detected_mean_interval_ms": 110,
                            "reference_speech_duration_ms": 100,
                            "detected_speech_duration_ms": 110,
                            "overlap_duration_ms": 100,
                            "missed_reference_duration_ms": 0,
                            "extra_detected_duration_ms": 10,
                            "reference_recall": 1.0,
                            "detected_precision": 100 / 110,
                        },
                        "cases": [
                            {
                                "id": "case",
                                "report": {
                                    "missed_reference_intervals": [],
                                    "extra_detected_intervals": [{"start_ms": 100, "end_ms": 110}],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            high_miss.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-suite-v1",
                        "source": "high-miss",
                        "case_count": 1,
                        "summary": {
                            "case_count": 1,
                            "audio_duration_ms": 100,
                            "reference_segment_count": 1,
                            "reference_interval_count": 1,
                            "detected_interval_count": 1,
                            "detected_max_interval_ms": 50,
                            "detected_mean_interval_ms": 50,
                            "reference_speech_duration_ms": 100,
                            "detected_speech_duration_ms": 50,
                            "overlap_duration_ms": 50,
                            "missed_reference_duration_ms": 50,
                            "extra_detected_duration_ms": 0,
                            "reference_recall": 0.5,
                            "detected_precision": 1.0,
                        },
                        "cases": [
                            {
                                "id": "case",
                                "report": {
                                    "missed_reference_intervals": [{"start_ms": 50, "end_ms": 100}],
                                    "extra_detected_intervals": [],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "vad",
                    "compare-coverage",
                    "--json",
                    "-o",
                    str(comparison_path),
                    "--max-detected-interval-ms",
                    "100",
                    str(high_miss),
                    str(low_miss),
                ]
            )

            self.assertEqual(result, 0)
            comparison = json.loads(output)
            self.assertEqual(comparison["format"], "custom-asmr-vad-coverage-comparison-v1")
            self.assertEqual([item["label"] for item in comparison["items"]], ["low-miss", "high-miss"])
            self.assertEqual(comparison["quality_gate"], {"max_detected_interval_ms": 100})
            self.assertEqual(comparison["items"][0]["missed_reference_duration_ms"], 0)
            self.assertEqual(comparison["items"][0]["detected_max_interval_ms"], 110)
            self.assertEqual(comparison["items"][0]["detected_mean_interval_ms"], 110)
            self.assertEqual(comparison["items"][0]["extra_detected_interval_count"], 1)
            self.assertFalse(comparison["items"][0]["gate_passed"])
            self.assertIn("detected max interval", comparison["items"][0]["gate_failures"][0])
            self.assertEqual(comparison["items"][1]["missed_reference_interval_count"], 1)
            self.assertTrue(comparison["items"][1]["gate_passed"])
            self.assertEqual(
                json.loads(comparison_path.read_text(encoding="utf-8"))["items"][0]["label"],
                "low-miss",
            )

    def test_vad_compare_coverage_can_fail_on_chunk_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            long_chunk = root / "full-audio.json"
            short_chunk = root / "chunked.json"
            comparison_path = root / "comparison.json"
            long_chunk.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-v1",
                        "source": "full-audio",
                        "audio_duration_ms": 120000,
                        "reference_segment_count": 1,
                        "reference_interval_count": 1,
                        "detected_interval_count": 1,
                        "detected_max_interval_ms": 120000,
                        "detected_mean_interval_ms": 120000,
                        "reference_speech_duration_ms": 120000,
                        "detected_speech_duration_ms": 120000,
                        "overlap_duration_ms": 120000,
                        "missed_reference_duration_ms": 0,
                        "extra_detected_duration_ms": 0,
                        "missed_reference_intervals": [],
                        "extra_detected_intervals": [],
                        "reference_recall": 1.0,
                        "detected_precision": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            short_chunk.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-v1",
                        "source": "chunked",
                        "audio_duration_ms": 120000,
                        "reference_segment_count": 1,
                        "reference_interval_count": 1,
                        "detected_interval_count": 4,
                        "detected_max_interval_ms": 30000,
                        "detected_mean_interval_ms": 29500,
                        "reference_speech_duration_ms": 120000,
                        "detected_speech_duration_ms": 118000,
                        "overlap_duration_ms": 118000,
                        "missed_reference_duration_ms": 2000,
                        "extra_detected_duration_ms": 0,
                        "missed_reference_intervals": [{"start_ms": 118000, "end_ms": 120000}],
                        "extra_detected_intervals": [],
                        "reference_recall": 118000 / 120000,
                        "detected_precision": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "vad",
                    "compare-coverage",
                    "--json",
                    "-o",
                    str(comparison_path),
                    "--max-detected-interval-ms",
                    "60000",
                    "--fail-on-gate",
                    str(long_chunk),
                    str(short_chunk),
                ]
            )

            self.assertEqual(result, 1)
            comparison = json.loads(output)
            self.assertEqual(comparison["items"][0]["label"], "full-audio")
            self.assertFalse(comparison["items"][0]["gate_passed"])
            self.assertTrue(comparison["items"][1]["gate_passed"])
            self.assertIn("VAD coverage gate failed for 1 report", error)
            self.assertIn("full-audio", error)
            self.assertEqual(
                json.loads(comparison_path.read_text(encoding="utf-8"))["items"][0]["label"],
                "full-audio",
            )

    def test_vad_compare_coverage_marks_recall_miss_and_precision_gates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            low_precision = root / "low-precision.json"
            low_recall = root / "low-recall.json"
            comparison_path = root / "comparison.json"
            low_precision.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-v1",
                        "source": "low-precision",
                        "audio_duration_ms": 1000,
                        "reference_segment_count": 1,
                        "reference_interval_count": 1,
                        "detected_interval_count": 2,
                        "detected_max_interval_ms": 700,
                        "detected_mean_interval_ms": 550,
                        "reference_speech_duration_ms": 1000,
                        "detected_speech_duration_ms": 1100,
                        "overlap_duration_ms": 1000,
                        "missed_reference_duration_ms": 0,
                        "extra_detected_duration_ms": 100,
                        "missed_reference_intervals": [],
                        "extra_detected_intervals": [{"start_ms": 1000, "end_ms": 1100}],
                        "reference_recall": 1.0,
                        "detected_precision": 1000 / 1100,
                    }
                ),
                encoding="utf-8",
            )
            low_recall.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-v1",
                        "source": "low-recall",
                        "audio_duration_ms": 1000,
                        "reference_segment_count": 1,
                        "reference_interval_count": 1,
                        "detected_interval_count": 1,
                        "detected_max_interval_ms": 900,
                        "detected_mean_interval_ms": 900,
                        "reference_speech_duration_ms": 1000,
                        "detected_speech_duration_ms": 900,
                        "overlap_duration_ms": 900,
                        "missed_reference_duration_ms": 100,
                        "extra_detected_duration_ms": 0,
                        "missed_reference_intervals": [{"start_ms": 900, "end_ms": 1000}],
                        "extra_detected_intervals": [],
                        "reference_recall": 0.9,
                        "detected_precision": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "vad",
                    "compare-coverage",
                    "--json",
                    "-o",
                    str(comparison_path),
                    "--max-missed-reference-ms",
                    "50",
                    "--min-reference-recall",
                    "0.95",
                    "--min-detected-precision",
                    "0.95",
                    "--fail-on-gate",
                    str(low_recall),
                    str(low_precision),
                ]
            )

            self.assertEqual(result, 1)
            comparison = json.loads(output)
            self.assertEqual(
                comparison["quality_gate"],
                {
                    "max_missed_reference_ms": 50,
                    "min_reference_recall": 0.95,
                    "min_detected_precision": 0.95,
                },
            )
            self.assertEqual([item["label"] for item in comparison["items"]], ["low-precision", "low-recall"])
            self.assertFalse(comparison["items"][0]["gate_passed"])
            self.assertIn("detected precision", comparison["items"][0]["gate_failures"][0])
            self.assertFalse(comparison["items"][1]["gate_passed"])
            self.assertEqual(len(comparison["items"][1]["gate_failures"]), 2)
            self.assertIn("missed reference duration", comparison["items"][1]["gate_failures"][0])
            self.assertIn("reference recall", comparison["items"][1]["gate_failures"][1])
            self.assertIn("VAD coverage gate failed for 2 report", error)
            self.assertEqual(
                json.loads(comparison_path.read_text(encoding="utf-8"))["quality_gate"]["min_reference_recall"],
                0.95,
            )

    def test_vad_compare_coverage_fail_on_gate_requires_gate_option(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "coverage.json"
            report_path.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-v1",
                        "source": "energy",
                        "audio_duration_ms": 100,
                        "reference_segment_count": 1,
                        "reference_interval_count": 1,
                        "detected_interval_count": 1,
                        "detected_max_interval_ms": 50,
                        "detected_mean_interval_ms": 50,
                        "reference_speech_duration_ms": 100,
                        "detected_speech_duration_ms": 50,
                        "overlap_duration_ms": 50,
                        "missed_reference_duration_ms": 50,
                        "extra_detected_duration_ms": 0,
                        "missed_reference_intervals": [{"start_ms": 50, "end_ms": 100}],
                        "extra_detected_intervals": [],
                        "reference_recall": 0.5,
                        "detected_precision": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                ["vad", "compare-coverage", "--json", "--fail-on-gate", str(report_path)]
            )

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output)["format"], "custom-asmr-vad-coverage-comparison-v1")
            self.assertIn("--fail-on-gate requires", error)

    def test_eval_transcript_outputs_json_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            report_path = root / "report.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,100 --> 00:00:02,200\nね\n", encoding="utf-8")

            result, output = run_cli(
                [
                    "eval-transcript",
                    "--json",
                    "-o",
                    str(report_path),
                    str(reference),
                    str(candidate),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-eval-v1")
            self.assertEqual(report["text"]["edit_distance"], 1)
            self.assertEqual(report["text_practical"]["mode"], "practical")
            self.assertTrue(report_path.exists())

    def test_freeze_reference_sorts_segments_and_clears_review_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "reviewed.master.json"
            frozen_path = root / "reference.master.json"
            source.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "sample.wav", "duration_ms": 3000},
                        "segments": [
                            {
                                "id": "draft-b",
                                "start_ms": 1600,
                                "end_ms": 2100,
                                "channel": "R",
                                "kind": "speech",
                                "text": "あと",
                                "needs_review": True,
                            },
                            {
                                "id": "draft-a",
                                "start_ms": 500,
                                "end_ms": 1000,
                                "channel": "L",
                                "kind": "speech",
                                "text": "ねえ",
                                "needs_review": True,
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result, output = run_cli(["freeze-reference", str(source), "-o", str(frozen_path), "--json"])

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["reference_type"], "human-reviewed")
            self.assertEqual(report["segments"], 2)
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            self.assertEqual([segment["id"] for segment in frozen["segments"]], ["seg_000001", "seg_000002"])
            self.assertEqual([segment["text"] for segment in frozen["segments"]], ["ねえ", "あと"])
            self.assertFalse(any(segment["needs_review"] for segment in frozen["segments"]))

    def test_freeze_reference_accepts_reviewed_srt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "reviewed.srt"
            frozen_path = root / "reference.master.json"
            source.write_text(
                "2\n00:00:01,500 --> 00:00:02,000\n[R] あと\n\n"
                "1\n00:00:00,500 --> 00:00:01,000\n[L] ねえ\n",
                encoding="utf-8",
            )

            result, _ = run_cli(["freeze-reference", str(source), "-o", str(frozen_path)])

            self.assertEqual(result, 0)
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            self.assertEqual([segment["channel"] for segment in frozen["segments"]], ["L", "R"])
            self.assertEqual([segment["id"] for segment in frozen["segments"]], ["seg_000001", "seg_000002"])

    def test_align_transcript_runs_configured_aligner_command(self):
        script = (
            "import json,sys;"
            "json.load(sys.stdin);"
            "print(json.dumps({'segments':[{'id':'seg_000001','start_ms':120,'end_ms':900}]}))"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            source = root / "candidate.srt"
            aligned_path = root / "aligned.master.json"
            audio.write_bytes(b"audio")
            source.write_text("1\n00:00:00,000 --> 00:00:01,000\nねえ\n", encoding="utf-8")

            with mock.patch.dict("os.environ", {"CASRT_ALIGNER_COMMAND": f"{sys.executable} -c {json.dumps(script)}"}):
                result, output = run_cli(
                    [
                        "align-transcript",
                        str(audio),
                        str(source),
                        "-o",
                        str(aligned_path),
                        "--json",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output)["segments"], 1)
            aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
            self.assertEqual(aligned["segments"][0]["start_ms"], 120)
            self.assertEqual(aligned["segments"][0]["end_ms"], 900)

    def test_align_transcript_writes_diagnostics_output(self):
        script = (
            "import json,sys;"
            "json.load(sys.stdin);"
            "print(json.dumps({'segments':[{'id':'seg_000001','start_ms':120,'end_ms':40100}]}))"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            source = root / "candidate.master.json"
            aligned_path = root / "aligned.master.json"
            diagnostics_path = root / "alignment-diagnostics.json"
            audio.write_bytes(b"audio")
            source.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "audio.wav", "duration_ms": 50000},
                        "segments": [
                            {
                                "id": "seg_000001",
                                "start_ms": 0,
                                "end_ms": 1000,
                                "channel": "MIX",
                                "kind": "speech",
                                "text": "長い",
                                "needs_review": False,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with mock.patch.dict("os.environ", {"CASRT_ALIGNER_COMMAND": f"{sys.executable} -c {json.dumps(script)}"}):
                result, output = run_cli(
                    [
                        "align-transcript",
                        str(audio),
                        str(source),
                        "-o",
                        str(aligned_path),
                        "--diagnostics-output",
                        str(diagnostics_path),
                        "--json",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output)["diagnostics_output"], str(diagnostics_path))
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["format"], "custom-asmr-alignment-diagnostics-v1")
            self.assertEqual(diagnostics["changed_segments"], 1)
            self.assertEqual(diagnostics["review_flag_changes"], 1)
            self.assertEqual(diagnostics["max_boundary_delta_ms"], 39100)
            self.assertEqual(diagnostics["boundary_count"], 2)
            self.assertEqual(diagnostics["mean_abs_boundary_delta_ms"], 19610)
            self.assertEqual(diagnostics["within_250ms_boundary_count"], 1)
            self.assertEqual(diagnostics["within_250ms_boundary_ratio"], 0.5)
            self.assertEqual(diagnostics["within_500ms_boundary_count"], 1)
            self.assertEqual(diagnostics["within_500ms_boundary_ratio"], 0.5)
            item = diagnostics["items"][0]
            self.assertEqual(item["original_start_ms"], 0)
            self.assertEqual(item["original_end_ms"], 1000)
            self.assertEqual(item["aligned_start_ms"], 120)
            self.assertEqual(item["aligned_end_ms"], 40100)
            self.assertEqual(item["start_delta_ms"], 120)
            self.assertEqual(item["end_delta_ms"], 39100)
            self.assertTrue(item["needs_review_after"])

    def test_align_transcript_requires_configured_aligner_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            source = root / "candidate.srt"
            aligned_path = root / "aligned.master.json"
            audio.write_bytes(b"audio")
            source.write_text("1\n00:00:00,000 --> 00:00:01,000\nねえ\n", encoding="utf-8")

            with mock.patch.dict("os.environ", {}, clear=True):
                result, _, error = run_cli_with_stderr(
                    ["align-transcript", str(audio), str(source), "-o", str(aligned_path)]
                )

            self.assertEqual(result, 1)
            self.assertIn("CASRT_ALIGNER_COMMAND is required", error)

    def test_attribute_channels_relabels_mix_speech_from_stereo_energy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "stereo.wav"
            source = root / "candidate.srt"
            output_path = root / "attributed.master.json"
            write_stereo_samples(audio, [(6000, 100)] * 1000 + [(100, 6000)] * 1000)
            source.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n左\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\n右\n",
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "attribute-channels",
                    "--json",
                    str(audio),
                    str(source),
                    "-o",
                    str(output_path),
                ]
            )

            self.assertEqual(result, 0)
            payload = json.loads(output)
            self.assertEqual(payload["changed_segments"], 2)
            self.assertEqual(payload["reason_counts"], {"left_dominant": 1, "right_dominant": 1})
            self.assertEqual(payload["attributed_channel_counts"], {"L": 1, "R": 1})
            attributed = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([segment["channel"] for segment in attributed["segments"]], ["L", "R"])

    def test_attribute_channels_fails_for_mono_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "mono.wav"
            source = root / "candidate.srt"
            output_path = root / "attributed.master.json"
            write_mono_wav(audio)
            source.write_text("1\n00:00:00,000 --> 00:00:00,001\n声\n", encoding="utf-8")

            result, _, error = run_cli_with_stderr(
                [
                    "attribute-channels",
                    str(audio),
                    str(source),
                    "-o",
                    str(output_path),
                ]
            )

            self.assertEqual(result, 1)
            self.assertIn("requires stereo audio", error)
            self.assertFalse(output_path.exists())

    def test_attribute_channels_keeps_mix_when_quieter_side_is_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "active-both.wav"
            source = root / "candidate.srt"
            output_path = root / "attributed.master.json"
            write_stereo_samples(audio, [(6000, 2000)] * 1000)
            source.write_text("1\n00:00:00,000 --> 00:00:01,000\n両方\n", encoding="utf-8")

            result, output = run_cli(
                [
                    "attribute-channels",
                    "--json",
                    str(audio),
                    str(source),
                    "-o",
                    str(output_path),
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output)["changed_segments"], 0)
            attributed = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(attributed["segments"][0]["channel"], "MIX")

    def test_attribute_channels_can_disable_quiet_side_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "active-both.wav"
            source = root / "candidate.srt"
            output_path = root / "attributed.master.json"
            write_stereo_samples(audio, [(6000, 2000)] * 1000)
            source.write_text("1\n00:00:00,000 --> 00:00:01,000\n左寄り\n", encoding="utf-8")

            result, output = run_cli(
                [
                    "attribute-channels",
                    "--json",
                    "--quiet-channel-max-dbfs",
                    "none",
                    str(audio),
                    str(source),
                    "-o",
                    str(output_path),
                ]
            )

            self.assertEqual(result, 0)
            payload = json.loads(output)
            self.assertIsNone(payload["quiet_channel_max_dbfs"])
            self.assertEqual(payload["changed_segments"], 1)
            self.assertEqual(payload["reason_counts"], {"left_dominant": 1})
            attributed = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(attributed["segments"][0]["channel"], "L")

    def test_attribute_channels_writes_diagnostics_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "stereo.wav"
            source = root / "candidate.srt"
            output_path = root / "attributed.master.json"
            diagnostics_path = root / "channel-diagnostics.json"
            write_stereo_samples(
                audio,
                [(6000, 100)] * 1000 + [(6000, 2000)] * 1000 + [(100, 6000)] * 1000,
            )
            source.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n左\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\n両方\n\n"
                "3\n00:00:02,000 --> 00:00:03,000\n右\n",
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "attribute-channels",
                    "--json",
                    str(audio),
                    str(source),
                    "-o",
                    str(output_path),
                    "--diagnostics-output",
                    str(diagnostics_path),
                ]
            )

            self.assertEqual(result, 0)
            payload = json.loads(output)
            self.assertEqual(payload["diagnostics_output"], str(diagnostics_path))
            self.assertEqual(
                payload["reason_counts"],
                {"left_dominant": 1, "quieter_side_active": 1, "right_dominant": 1},
            )
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["format"], "custom-asmr-channel-diagnostics-v1")
            self.assertEqual(
                diagnostics["reason_counts"],
                {"left_dominant": 1, "quieter_side_active": 1, "right_dominant": 1},
            )
            self.assertEqual(diagnostics["attributed_channel_counts"], {"L": 1, "MIX": 1, "R": 1})
            self.assertEqual([item["reason"] for item in diagnostics["items"]], [
                "left_dominant",
                "quieter_side_active",
                "right_dominant",
            ])
            self.assertEqual([item["attributed_channel"] for item in diagnostics["items"]], ["L", "MIX", "R"])
            self.assertGreater(diagnostics["items"][0]["left_dbfs"], diagnostics["items"][0]["right_dbfs"])

    def test_sweep_channel_attribution_writes_setting_reports_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "stereo.wav"
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            audio_map = root / "audio-map.json"
            manifest = root / "manifest.json"
            output_dir = root / "sweep"
            write_stereo_samples(audio, [(6000, 100)] * 1000 + [(100, 6000)] * 1000)
            reference.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n[L] 左\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\n[R] 右\n",
                encoding="utf-8",
            )
            candidate.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n左\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\n右\n",
                encoding="utf-8",
            )
            audio_map.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-audio-map-v1",
                        "items": [{"case_id": "front-a", "audio": "stereo.wav"}],
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "reference_type": "human-reviewed",
                        "cases": [
                            {
                                "id": "front-a",
                                "reference": "reference.srt",
                                "candidate": "candidate.srt",
                                "candidate_id": "mix-draft",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "sweep-channel-attribution",
                    "--json",
                    str(manifest),
                    "--audio-map",
                    str(audio_map),
                    "-o",
                    str(output_dir),
                    "--threshold-db",
                    "3",
                    "--threshold-db",
                    "50",
                    "--product-gate",
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-channel-attribution-sweep-v1")
            self.assertEqual(report["setting_count"], 2)
            self.assertEqual(report["quality_gate"]["preset"], "local-asmr-v1")
            self.assertEqual([item["changed_segments"] for item in report["items"]], [2, 0])
            self.assertEqual(report["items"][0]["reason_counts"], {"left_dominant": 1, "right_dominant": 1})
            self.assertEqual(report["items"][1]["reason_counts"], {"below_threshold": 2})
            self.assertEqual(report["items"][1]["attributed_channel_counts"], {"MIX": 2})
            comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
            index = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["quality_gate"]["require_reference_type"], "human-reviewed")
            self.assertEqual(index["quality_gate"], comparison["quality_gate"])
            self.assertEqual(comparison["items"][0]["label"], "th3_quietm40.eval-report")
            self.assertEqual(comparison["items"][0]["segments_needing_edit"], 0.0)
            self.assertTrue(comparison["items"][0]["gate_passed"])
            self.assertEqual(comparison["items"][1]["segments_needing_edit"], 2.0)
            self.assertFalse(comparison["items"][1]["gate_passed"])
            self.assertTrue(
                any("channel time-aligned MIX ratio" in failure for failure in comparison["items"][1]["gate_failures"])
            )
            setting_manifest = json.loads(
                (output_dir / "th3_quietm40" / "th3_quietm40.eval-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(setting_manifest["cases"][0]["reference"], "../../reference.srt")
            self.assertTrue((output_dir / "th3_quietm40" / "candidates" / "front-a.master.json").exists())

    def test_sweep_channel_attribution_rejects_missing_sources_before_output_side_effects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "stereo.wav"
            reference = root / "reference.srt"
            audio_map = root / "audio-map.json"
            manifest = root / "manifest.json"
            output_dir = root / "sweep"
            write_stereo_samples(audio, [(6000, 100)] * 1000)
            reference.write_text("1\n00:00:00,000 --> 00:00:01,000\n[L] 左\n", encoding="utf-8")
            audio_map.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-audio-map-v1",
                        "items": [{"case_id": "front-a", "audio": "stereo.wav"}],
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "cases": [
                            {
                                "id": "front-a",
                                "reference": "reference.srt",
                                "candidate": "missing.srt",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                [
                    "sweep-channel-attribution",
                    str(manifest),
                    "--audio-map",
                    str(audio_map),
                    "-o",
                    str(output_dir),
                ]
            )

            self.assertEqual(result, 1)
            self.assertIn("candidate file does not exist", error)
            self.assertFalse(output_dir.exists())

    def test_sweep_channel_attribution_can_reset_existing_speech_channels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "stereo.wav"
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            audio_map = root / "audio-map.json"
            manifest = root / "manifest.json"
            output_dir = root / "sweep"
            write_stereo_samples(audio, [(100, 6000)] * 1000)
            reference.write_text("1\n00:00:00,000 --> 00:00:01,000\n[R] 右\n", encoding="utf-8")
            candidate.write_text("1\n00:00:00,000 --> 00:00:01,000\n[L] 右\n", encoding="utf-8")
            audio_map.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-audio-map-v1",
                        "items": [{"case_id": "front-a", "audio": "stereo.wav"}],
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "reference_type": "human-reviewed",
                        "cases": [
                            {
                                "id": "front-a",
                                "reference": "reference.srt",
                                "candidate": "candidate.srt",
                                "candidate_id": "already-attributed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "sweep-channel-attribution",
                    "--json",
                    str(manifest),
                    "--audio-map",
                    str(audio_map),
                    "-o",
                    str(output_dir),
                    "--threshold-db",
                    "3",
                    "--reset-speech-channels-to-mix",
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertTrue(report["reset_speech_channels_to_mix"])
            self.assertTrue(report["items"][0]["reset_speech_channels_to_mix"])
            self.assertEqual(report["items"][0]["changed_segments"], 1)
            attributed = json.loads(
                (output_dir / "th3_quietm40" / "candidates" / "front-a.master.json").read_text(encoding="utf-8")
            )
            self.assertEqual(attributed["segments"][0]["channel"], "R")
            comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["items"][0]["segments_needing_edit"], 0.0)

    def test_sweep_channel_attribution_can_test_without_quiet_side_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "stereo.wav"
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            audio_map = root / "audio-map.json"
            manifest = root / "manifest.json"
            output_dir = root / "sweep"
            write_stereo_samples(audio, [(6000, 2000)] * 1000)
            reference.write_text("1\n00:00:00,000 --> 00:00:01,000\n[L] 左寄り\n", encoding="utf-8")
            candidate.write_text("1\n00:00:00,000 --> 00:00:01,000\n左寄り\n", encoding="utf-8")
            audio_map.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-audio-map-v1",
                        "items": [{"case_id": "front-a", "audio": "stereo.wav"}],
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "reference_type": "human-reviewed",
                        "cases": [{"id": "front-a", "reference": "reference.srt", "candidate": "candidate.srt"}],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "sweep-channel-attribution",
                    "--json",
                    str(manifest),
                    "--audio-map",
                    str(audio_map),
                    "-o",
                    str(output_dir),
                    "--threshold-db",
                    "8",
                    "--quiet-channel-max-dbfs",
                    "none",
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["items"][0]["setting_id"], "th8_quietnone")
            self.assertIsNone(report["items"][0]["quiet_channel_max_dbfs"])
            self.assertEqual(report["items"][0]["changed_segments"], 1)
            self.assertEqual(report["items"][0]["reason_counts"], {"left_dominant": 1})
            self.assertTrue((output_dir / "th8_quietnone" / "candidates" / "front-a.master.json").exists())

    def test_slice_case_writes_rebased_audio_and_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            transcript = root / "source.srt"
            audio_output = root / "case.wav"
            transcript_output = root / "case.master.json"
            write_stereo_samples(audio, [(100, 200), (300, 400), (500, 600), (700, 800)])
            transcript.write_text(
                "1\n00:00:00,000 --> 00:00:00,002\n前半\n\n"
                "2\n00:00:00,001 --> 00:00:00,003\n中央\n\n"
                "3\n00:00:00,002 --> 00:00:00,004\n後半\n",
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "slice-case",
                    "--json",
                    str(audio),
                    str(transcript),
                    "--start-ms",
                    "1",
                    "--end-ms",
                    "3",
                    "--audio-output",
                    str(audio_output),
                    "--transcript-output",
                    str(transcript_output),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["duration_ms"], 2)
            self.assertEqual(report["segments"], 3)
            self.assertEqual(report["review_count"], 2)
            self.assertEqual(analyze_wav(audio_output.read_bytes()).duration_ms, 2)
            sliced = json.loads(transcript_output.read_text(encoding="utf-8"))
            self.assertEqual(sliced["audio"]["duration_ms"], 2)
            self.assertEqual(
                [
                    (segment["id"], segment["start_ms"], segment["end_ms"], segment["text"], segment["needs_review"])
                    for segment in sliced["segments"]
                ],
                [
                    ("seg_000001", 0, 1, "前半", True),
                    ("seg_000002", 0, 2, "中央", False),
                    ("seg_000003", 1, 2, "後半", True),
                ],
            )

    def test_prepare_review_cases_writes_case_outputs_and_eval_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            plan = root / "plan.json"
            output_dir = root / "cases"
            write_stereo_samples(audio, [(100, 200), (300, 400), (500, 600), (700, 800)])
            reference.write_text(
                "1\n00:00:00,000 --> 00:00:00,002\n前半\n\n"
                "2\n00:00:00,001 --> 00:00:00,003\n中央\n",
                encoding="utf-8",
            )
            candidate.write_text("1\n00:00:00,001 --> 00:00:00,003\n候補\n", encoding="utf-8")
            plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-slice-plan-v1",
                        "reference_type": "pseudo-gold",
                        "reference_notes": "stable-ts draft",
                        "cases": [
                            {
                                "id": "front-a",
                                "audio": "source.wav",
                                "reference": "reference.srt",
                                "candidate": "candidate.srt",
                                "candidate_id": "draft-candidate",
                                "start_ms": 1,
                                "end_ms": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(["prepare-review-cases", "--json", str(plan), "-o", str(output_dir)])

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["case_count"], 1)
            self.assertEqual(report["review_count"], 1)
            self.assertTrue((output_dir / "audio" / "front-a.wav").exists())
            self.assertEqual(analyze_wav((output_dir / "audio" / "front-a.wav").read_bytes()).duration_ms, 2)
            audio_map = json.loads((output_dir / "audio-map.json").read_text(encoding="utf-8"))
            self.assertEqual(audio_map["items"], [{"case_id": "front-a", "audio": "audio/front-a.wav"}])
            case_index = json.loads((output_dir / "case-index.json").read_text(encoding="utf-8"))
            self.assertEqual(case_index["reference_type"], "pseudo-gold")
            self.assertEqual(case_index["items"][0]["reference_notes"], "stable-ts draft")
            self.assertEqual(case_index["items"][0]["candidate_id"], "draft-candidate")
            sliced_reference = json.loads(
                (output_dir / "references" / "front-a.master.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sliced_reference["audio"]["duration_ms"], 2)
            self.assertEqual(sliced_reference["segments"][0]["needs_review"], True)
            eval_manifest = json.loads((output_dir / "eval-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_manifest["cases"][0]["reference"], "references/front-a.master.json")
            self.assertEqual(eval_manifest["cases"][0]["candidate"], "candidates/front-a.master.json")

    def test_prepare_review_cases_rejects_mixed_candidate_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            plan = root / "plan.json"
            output_dir = root / "cases"
            write_stereo_samples(audio, [(100, 200), (300, 400)])
            reference.write_text("1\n00:00:00,000 --> 00:00:00,002\n参照\n", encoding="utf-8")
            candidate.write_text("1\n00:00:00,000 --> 00:00:00,002\n候補\n", encoding="utf-8")
            plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-slice-plan-v1",
                        "cases": [
                            {
                                "id": "with-candidate",
                                "audio": "source.wav",
                                "reference": "reference.srt",
                                "candidate": "candidate.srt",
                                "start_ms": 0,
                                "end_ms": 2,
                            },
                            {
                                "id": "without-candidate",
                                "audio": "source.wav",
                                "reference": "reference.srt",
                                "start_ms": 0,
                                "end_ms": 2,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                ["prepare-review-cases", str(plan), "-o", str(output_dir)]
            )

            self.assertEqual(result, 1)
            self.assertIn("cannot mix candidate and non-candidate cases", error)
            self.assertFalse(output_dir.exists())

    def test_prepare_review_cases_rejects_missing_source_before_output_side_effects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan = root / "plan.json"
            output_dir = root / "cases"
            plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-slice-plan-v1",
                        "cases": [
                            {
                                "id": "missing-source",
                                "audio": "missing.wav",
                                "reference": "missing.srt",
                                "start_ms": 0,
                                "end_ms": 1000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                ["prepare-review-cases", str(plan), "-o", str(output_dir)]
            )

            self.assertEqual(result, 1)
            self.assertIn("source file does not exist", error)
            self.assertFalse(output_dir.exists())

    def test_review_case_status_reports_prepared_case_integrity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            reference = root / "reference.srt"
            plan = root / "plan.json"
            output_dir = root / "cases"
            status_path = root / "status.json"
            write_stereo_samples(audio, [(100, 200), (300, 400), (500, 600), (700, 800)])
            reference.write_text(
                "1\n00:00:00,000 --> 00:00:00,002\n前半\n\n"
                "2\n00:00:00,001 --> 00:00:00,003\n中央\n",
                encoding="utf-8",
            )
            plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-slice-plan-v1",
                        "reference_type": "pseudo-gold",
                        "cases": [
                            {
                                "id": "front-a",
                                "audio": "source.wav",
                                "reference": "reference.srt",
                                "start_ms": 1,
                                "end_ms": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["prepare-review-cases", str(plan), "-o", str(output_dir)])

            result, output = run_cli(
                [
                    "review-case-status",
                    "--json",
                    "-o",
                    str(status_path),
                    str(output_dir / "case-index.json"),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-review-case-status-v1")
            self.assertTrue(report["ok"])
            self.assertEqual(report["case_count"], 1)
            self.assertEqual(report["candidate_case_count"], 0)
            self.assertEqual(report["missing_candidate_case_count"], 1)
            self.assertEqual(report["cases_missing_candidate"], ["front-a"])
            self.assertEqual(report["next_missing_candidate_case_id"], "front-a")
            self.assertEqual(report["candidate_review_count"], 0)
            self.assertEqual(report["candidate_review_case_count"], 0)
            self.assertEqual(report["candidate_review_clear_case_count"], 0)
            self.assertEqual(report["cases_with_candidate_review"], [])
            self.assertIsNone(report["next_candidate_review_case_id"])
            self.assertEqual(report["reference_type_counts"], {"pseudo-gold": 1})
            self.assertEqual(report["reference_review_count"], 1)
            self.assertEqual(report["reference_review_duration_ms"], 1)
            self.assertEqual(report["reference_review_case_count"], 1)
            self.assertEqual(report["reference_review_clear_case_count"], 0)
            self.assertEqual(report["cases_needing_review"], ["front-a"])
            self.assertEqual(report["next_review_case_id"], "front-a")
            self.assertEqual(report["items"][0]["reference_segments"], 2)
            self.assertEqual(report["items"][0]["reference_review_count"], 1)
            self.assertEqual(report["items"][0]["reference_review_duration_ms"], 1)
            self.assertEqual(
                {
                    key: report["items"][0]["first_review_segment"][key]
                    for key in ("start_ms", "end_ms", "channel", "text", "needs_review")
                },
                {
                    "start_ms": 0,
                    "end_ms": 1,
                    "channel": "MIX",
                    "text": "前半",
                    "needs_review": True,
                },
            )
            self.assertEqual(json.loads(status_path.read_text(encoding="utf-8"))["case_count"], 1)

            missing_candidate_status_path = root / "status.missing-candidates.json"
            fail_result, fail_output, fail_error = run_cli_with_stderr(
                [
                    "review-case-status",
                    "--json",
                    "-o",
                    str(missing_candidate_status_path),
                    "--fail-on-missing-candidates",
                    str(output_dir / "case-index.json"),
                ]
            )

            self.assertEqual(fail_result, 1)
            fail_report = json.loads(fail_output)
            self.assertTrue(fail_report["ok"])
            self.assertEqual(fail_report["missing_candidate_case_count"], 1)
            self.assertEqual(fail_report["cases_missing_candidate"], ["front-a"])
            self.assertEqual(
                json.loads(missing_candidate_status_path.read_text(encoding="utf-8"))[
                    "missing_candidate_case_count"
                ],
                1,
            )
            self.assertIn("missing_candidate_count=1", fail_error)

    def test_audit_review_case_references_reports_overlap_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            references = root / "references"
            references.mkdir()
            master = MasterDocument(
                source_language="ja",
                source_file="voice.wav",
                duration_ms=2000,
                segments=(
                    Segment("seg_000001", 0, 1000, "L", "speech", "あ"),
                    Segment("seg_000002", 500, 1200, "L", "speech", "い"),
                ),
            )
            (references / "front.master.json").write_text(
                json.dumps(master.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            case_index = root / "case-index.json"
            output_path = root / "audit.json"
            review_effort_path = root / "audit-review-effort.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "reference_type": "pseudo-gold",
                        "items": [{"id": "front", "reference": "references/front.master.json"}],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "audit-review-case-references",
                    "--json",
                    "-o",
                    str(output_path),
                    "--review-effort-output",
                    str(review_effort_path),
                    str(case_index),
                ]
            )
            saved_report = json.loads(output_path.read_text(encoding="utf-8"))
            review_effort_report = json.loads(review_effort_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        report = json.loads(output)
        self.assertEqual(report["format"], "custom-asmr-reference-audit-suite-v1")
        self.assertEqual(report["review_effort_output"], str(review_effort_path))
        self.assertEqual(saved_report["format"], "custom-asmr-reference-audit-suite-v1")
        self.assertEqual(saved_report["review_effort_output"], str(review_effort_path))
        self.assertEqual(review_effort_report["format"], "custom-asmr-review-effort-v1")
        self.assertEqual(review_effort_report["reason_counts"], {"reference-same-channel-overlap": 1})
        self.assertEqual(report["summary"]["overlap_pair_count"], 1)
        self.assertEqual(report["summary"]["same_channel_overlap_pair_count"], 1)
        self.assertEqual(report["summary"]["flag_type_counts"], {"same_channel_overlap": 1})

    def test_audit_review_case_references_can_fail_after_writing_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            references = root / "references"
            references.mkdir()
            master = MasterDocument(
                source_language="ja",
                source_file="voice.wav",
                duration_ms=2000,
                segments=(
                    Segment("seg_000001", 0, 1000, "L", "speech", "あ"),
                    Segment("seg_000002", 500, 1200, "L", "speech", "い"),
                ),
            )
            (references / "front.master.json").write_text(
                json.dumps(master.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            case_index = root / "case-index.json"
            output_path = root / "audit.json"
            review_effort_path = root / "audit-review-effort.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [{"id": "front", "reference": "references/front.master.json"}],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "audit-review-case-references",
                    "--json",
                    "--fail-on-audit",
                    "-o",
                    str(output_path),
                    "--review-effort-output",
                    str(review_effort_path),
                    str(case_index),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output)["format"], "custom-asmr-reference-audit-suite-v1")
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["summary"]["overlap_pair_count"], 1)
            self.assertEqual(
                json.loads(review_effort_path.read_text(encoding="utf-8"))["reason_counts"],
                {"reference-same-channel-overlap": 1},
            )
            self.assertIn("reference_audit_item_count=1", error)
            self.assertIn("reference-same-channel-overlap", error)

    def test_audit_review_case_channels_reports_energy_mismatch_and_uncertain_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio"
            references = root / "references"
            audio_dir.mkdir()
            references.mkdir()
            audio = audio_dir / "front.wav"
            write_stereo_samples(
                audio,
                [(6000, 100)] * 1000 + [(100, 6000)] * 1000 + [(6000, 6000)] * 1000,
            )
            master = MasterDocument(
                source_language="ja",
                source_file="front.wav",
                duration_ms=3000,
                segments=(
                    Segment("seg_000001", 0, 1000, "L", "speech", "あ"),
                    Segment("seg_000002", 1000, 2000, "L", "speech", "い"),
                    Segment("seg_000003", 2000, 3000, "R", "speech", "う"),
                ),
            )
            (references / "front.master.json").write_text(
                json.dumps(master.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            case_index = root / "case-index.json"
            output_path = root / "channel-audit.json"
            review_effort_path = root / "channel-audit-review-effort.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "reference_type": "pseudo-gold",
                        "items": [
                            {
                                "id": "front",
                                "audio": "audio/front.wav",
                                "reference": "references/front.master.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "audit-review-case-channels",
                    "--json",
                    "--threshold-db",
                    "3",
                    "--quiet-channel-max-dbfs",
                    "none",
                    "-o",
                    str(output_path),
                    "--review-effort-output",
                    str(review_effort_path),
                    str(case_index),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            saved_report = json.loads(output_path.read_text(encoding="utf-8"))
            review_effort = json.loads(review_effort_path.read_text(encoding="utf-8"))
            self.assertEqual(report["format"], "custom-asmr-reference-channel-audit-suite-v1")
            self.assertEqual(saved_report["format"], "custom-asmr-reference-channel-audit-suite-v1")
            self.assertEqual(report["summary"]["match_count"], 1)
            self.assertEqual(report["summary"]["mismatch_count"], 1)
            self.assertEqual(report["summary"]["energy_uncertain_count"], 1)
            self.assertEqual(report["summary"]["energy_channel_counts"], {"L": 1, "MIX": 1, "R": 1})
            self.assertEqual(
                review_effort["reason_counts"],
                {
                    "reference-channel-energy-mismatch": 1,
                    "reference-channel-energy-uncertain": 1,
                },
            )
            self.assertEqual(review_effort["items"][0]["reference_text"], "")

    def test_audit_review_case_channels_can_fail_after_writing_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio"
            references = root / "references"
            audio_dir.mkdir()
            references.mkdir()
            audio = audio_dir / "front.wav"
            write_stereo_samples(audio, [(100, 6000)] * 1000)
            master = MasterDocument(
                source_language="ja",
                source_file="front.wav",
                duration_ms=1000,
                segments=(Segment("seg_000001", 0, 1000, "L", "speech", "あ"),),
            )
            (references / "front.master.json").write_text(
                json.dumps(master.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            case_index = root / "case-index.json"
            output_path = root / "channel-audit.json"
            review_effort_path = root / "channel-audit-review-effort.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front",
                                "audio": "audio/front.wav",
                                "reference": "references/front.master.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "audit-review-case-channels",
                    "--json",
                    "--fail-on-audit",
                    "--threshold-db",
                    "3",
                    "--quiet-channel-max-dbfs",
                    "none",
                    "-o",
                    str(output_path),
                    "--review-effort-output",
                    str(review_effort_path),
                    str(case_index),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output)["format"], "custom-asmr-reference-channel-audit-suite-v1")
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["summary"]["mismatch_count"], 1)
            self.assertEqual(
                json.loads(review_effort_path.read_text(encoding="utf-8"))["reason_counts"],
                {"reference-channel-energy-mismatch": 1},
            )
            self.assertIn("reference_channel_audit_item_count=1", error)
            self.assertIn("reference-channel-energy-mismatch", error)

    def test_review_case_status_can_fail_after_reporting_candidate_review_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio"
            reference_dir = root / "references"
            candidate_dir = root / "candidates"
            audio_dir.mkdir()
            reference_dir.mkdir()
            candidate_dir.mkdir()
            (audio_dir / "front.wav").write_bytes(b"RIFFcaseWAVE")
            reference = {
                "format": "custom-asmr-master-v1",
                "source_language": "ja",
                "audio": {"source_file": "front.wav", "duration_ms": 2},
                "segments": [
                    {
                        "id": "seg_000001",
                        "start_ms": 0,
                        "end_ms": 2,
                        "channel": "MIX",
                        "kind": "speech",
                        "text": "参照",
                        "needs_review": False,
                    }
                ],
            }
            candidate = {
                "format": "custom-asmr-master-v1",
                "source_language": "ja",
                "audio": {"source_file": "front.wav", "duration_ms": 2},
                "segments": [
                    {
                        "id": "seg_000001",
                        "start_ms": 0,
                        "end_ms": 2,
                        "channel": "MIX",
                        "kind": "speech",
                        "text": "候補",
                        "needs_review": True,
                    }
                ],
            }
            (reference_dir / "front.master.json").write_text(json.dumps(reference), encoding="utf-8")
            (candidate_dir / "front.master.json").write_text(json.dumps(candidate), encoding="utf-8")
            case_index = root / "case-index.json"
            status_path = root / "status.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "audio/front.wav",
                                "reference": "references/front.master.json",
                                "candidate": "candidates/front.master.json",
                                "segments": 1,
                                "review_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "review-case-status",
                    "--json",
                    "-o",
                    str(status_path),
                    "--fail-on-candidate-review",
                    str(case_index),
                ]
            )

            self.assertEqual(result, 1)
            report = json.loads(output)
            self.assertTrue(report["ok"])
            self.assertEqual(report["candidate_review_count"], 1)
            self.assertEqual(report["candidate_review_duration_ms"], 2)
            self.assertEqual(report["candidate_review_case_count"], 1)
            self.assertEqual(report["candidate_review_clear_case_count"], 0)
            self.assertEqual(report["cases_with_candidate_review"], ["front-a"])
            self.assertEqual(report["next_candidate_review_case_id"], "front-a")
            self.assertEqual(report["items"][0]["candidate_review_duration_ms"], 2)
            self.assertEqual(json.loads(status_path.read_text(encoding="utf-8"))["candidate_review_count"], 1)
            self.assertIn("candidate_review_count=1", error)

    def test_review_case_status_can_fail_after_reporting_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_index = root / "case-index.json"
            status_path = root / "status.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "reference_type": "human-reviewed",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "missing.wav",
                                "reference": "missing.master.json",
                                "segments": 1,
                                "review_count": 0,
                                "reference_type": "human-reviewed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "review-case-status",
                    "--json",
                    "-o",
                    str(status_path),
                    "--fail-on-issues",
                    str(case_index),
                ]
            )

            self.assertEqual(result, 1)
            report = json.loads(output)
            self.assertFalse(report["ok"])
            self.assertEqual(report["missing_file_count"], 2)
            self.assertEqual(report["reference_review_case_count"], 0)
            self.assertEqual(report["reference_review_clear_case_count"], 0)
            self.assertIsNone(report["next_review_case_id"])
            self.assertIsNone(report["items"][0]["first_review_segment"])
            self.assertIn("audio file is missing", report["items"][0]["issues"][0])
            self.assertEqual(json.loads(status_path.read_text(encoding="utf-8"))["missing_file_count"], 2)
            self.assertIn("review case status failed", error)

    def test_save_review_case_reference_updates_reference_and_index_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "cases"
            reference_dir = case_dir / "references"
            audio_dir = case_dir / "audio"
            reference_dir.mkdir(parents=True)
            audio_dir.mkdir()
            (audio_dir / "front.wav").write_bytes(b"RIFFcaseWAVE")
            reference_path = reference_dir / "front.master.json"
            reference_path.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "front.wav", "duration_ms": 2000},
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )
            case_index = case_dir / "case-index.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "reference_type": "pseudo-gold",
                        "items": [
                            {
                                "id": "front",
                                "audio": "audio/front.wav",
                                "reference": "references/front.master.json",
                                "segments": 0,
                                "review_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            edited = root / "edited.master.json"
            edited.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "front.wav", "duration_ms": 2000},
                        "segments": [
                            {
                                "id": "seg_000001",
                                "start_ms": 0,
                                "end_ms": 1000,
                                "channel": "L",
                                "kind": "speech",
                                "text": "修正",
                                "needs_review": True,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "save-review-case-reference",
                    "--json",
                    str(case_index),
                    "front",
                    str(edited),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            saved_reference = json.loads(reference_path.read_text(encoding="utf-8"))
            saved_index = json.loads(case_index.read_text(encoding="utf-8"))
            self.assertEqual(report["format"], "custom-asmr-review-case-reference-save-v1")
            self.assertEqual(report["segments"], 1)
            self.assertEqual(report["review_count"], 1)
            self.assertEqual(saved_reference["segments"][0]["text"], "修正")
            self.assertEqual(saved_index["items"][0]["segments"], 1)
            self.assertEqual(saved_index["items"][0]["review_count"], 1)

    def test_save_review_case_reference_rejects_missing_reference_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_index = root / "case-index.json"
            edited = root / "edited.master.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front",
                                "audio": "front.wav",
                                "reference": "missing.master.json",
                                "segments": 0,
                                "review_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            edited.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "front.wav", "duration_ms": 2000},
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "save-review-case-reference",
                    "--json",
                    str(case_index),
                    "front",
                    str(edited),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(output, "")
            self.assertIn("review case reference file is missing", error)
            self.assertFalse((root / "missing.master.json").exists())

    def test_attach_review_case_candidates_updates_index_and_unblocks_manifest_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            plan = root / "plan.json"
            attach_plan = root / "attach-candidates.json"
            output_dir = root / "cases"
            manifest_path = root / "manifest.json"
            write_stereo_samples(audio, [(100, 200), (300, 400), (500, 600), (700, 800)])
            reference.write_text("1\n00:00:00,000 --> 00:00:00,002\n参照\n", encoding="utf-8")
            candidate.write_text("1\n00:00:00,000 --> 00:00:00,002\n候補\n", encoding="utf-8")
            plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-slice-plan-v1",
                        "reference_type": "pseudo-gold",
                        "cases": [
                            {
                                "id": "front-a",
                                "audio": "source.wav",
                                "reference": "reference.srt",
                                "start_ms": 0,
                                "end_ms": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            attach_plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-candidate-attach-plan-v1",
                        "candidate_id": "local-candidate",
                        "candidates": [{"case_id": "front-a", "candidate": "candidate.srt"}],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["prepare-review-cases", str(plan), "-o", str(output_dir)])

            result, output = run_cli(
                [
                    "attach-review-case-candidates",
                    "--json",
                    str(output_dir / "case-index.json"),
                    str(attach_plan),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-review-case-candidate-attach-v1")
            self.assertEqual(report["candidate_count"], 1)
            self.assertEqual(report["items"][0]["candidate"], "candidates/front-a.master.json")
            self.assertEqual(report["items"][0]["candidate_id"], "local-candidate")
            candidate_master = json.loads((output_dir / "candidates" / "front-a.master.json").read_text(encoding="utf-8"))
            self.assertEqual(candidate_master["segments"][0]["text"], "候補")
            case_index = json.loads((output_dir / "case-index.json").read_text(encoding="utf-8"))
            self.assertEqual(case_index["items"][0]["candidate"], "candidates/front-a.master.json")
            self.assertEqual(case_index["items"][0]["candidate_id"], "local-candidate")

            status_result, status_output = run_cli(
                [
                    "review-case-status",
                    "--json",
                    "--fail-on-missing-candidates",
                    "--fail-on-candidate-review",
                    str(output_dir / "case-index.json"),
                ]
            )
            self.assertEqual(status_result, 0)
            status = json.loads(status_output)
            self.assertEqual(status["candidate_case_count"], 1)
            self.assertEqual(status["missing_candidate_case_count"], 0)
            self.assertEqual(status["cases_missing_candidate"], [])
            self.assertIsNone(status["next_missing_candidate_case_id"])
            self.assertEqual(status["candidate_review_count"], 0)
            self.assertEqual(status["candidate_review_case_count"], 0)
            self.assertEqual(status["candidate_review_clear_case_count"], 1)
            self.assertEqual(status["cases_with_candidate_review"], [])
            self.assertIsNone(status["next_candidate_review_case_id"])

            manifest_result, manifest_output = run_cli(
                [
                    "build-eval-manifest",
                    "--json",
                    str(output_dir / "case-index.json"),
                    "-o",
                    str(manifest_path),
                ]
            )
            self.assertEqual(manifest_result, 0)
            self.assertEqual(json.loads(manifest_output)["case_count"], 1)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["cases"][0]["candidate"], "candidates/front-a.master.json")
            self.assertEqual(manifest["cases"][0]["candidate_id"], "local-candidate")

    def test_align_review_case_candidates_writes_aligned_candidates_and_eval_manifest(self):
        script = (
            "import json,sys;"
            "request=json.load(sys.stdin);"
            "print(json.dumps({'segments':["
            "{'id':segment['id'],'start_ms':segment['start_ms']+10,'end_ms':segment['end_ms']+20}"
            " for segment in request['master']['segments']]}))"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "cases"
            audio_dir = case_dir / "audio"
            reference_dir = case_dir / "references"
            candidate_dir = case_dir / "candidates"
            output_dir = root / "aligned"
            audio_dir.mkdir(parents=True)
            reference_dir.mkdir()
            candidate_dir.mkdir()
            write_stereo_samples(audio_dir / "front-a.wav", [(100, 200), (300, 400)])
            reference = {
                "format": "custom-asmr-master-v1",
                "source_language": "ja",
                "audio": {"source_file": "front-a.wav", "duration_ms": 200},
                "segments": [
                    {
                        "id": "seg_000001",
                        "start_ms": 0,
                        "end_ms": 100,
                        "channel": "L",
                        "kind": "speech",
                        "text": "参照",
                        "needs_review": False,
                    }
                ],
            }
            candidate = {
                "format": "custom-asmr-master-v1",
                "source_language": "ja",
                "audio": {"source_file": "front-a.wav", "duration_ms": 200},
                "segments": [
                    {
                        "id": "seg_000001",
                        "start_ms": 0,
                        "end_ms": 100,
                        "channel": "MIX",
                        "kind": "speech",
                        "text": "候補",
                        "needs_review": False,
                    }
                ],
            }
            (reference_dir / "front-a.master.json").write_text(json.dumps(reference), encoding="utf-8")
            (candidate_dir / "front-a.master.json").write_text(json.dumps(candidate), encoding="utf-8")
            case_index = case_dir / "case-index.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "reference_type": "pseudo-gold",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "audio/front-a.wav",
                                "reference": "references/front-a.master.json",
                                "candidate": "candidates/front-a.master.json",
                                "candidate_id": "base-candidate",
                                "segments": 1,
                                "review_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict("os.environ", {"CASRT_ALIGNER_COMMAND": f"{sys.executable} -c {json.dumps(script)}"}):
                result, output = run_cli(
                    [
                        "align-review-case-candidates",
                        "--json",
                        str(case_index),
                        "-o",
                        str(output_dir),
                    ]
                )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-review-case-candidate-align-v1")
            self.assertEqual(report["candidate_count"], 1)
            self.assertEqual(report["segments"], 1)
            self.assertEqual(report["changed_segments"], 1)
            self.assertEqual(report["boundary_count"], 2)
            self.assertEqual(report["mean_abs_boundary_delta_ms"], 15)
            self.assertEqual(report["within_250ms_boundary_ratio"], 1.0)
            self.assertEqual(report["within_500ms_boundary_ratio"], 1.0)
            self.assertEqual(report["items"][0]["candidate_id"], "base-candidate-aligned")
            self.assertEqual(report["items"][0]["changed_segments"], 1)
            self.assertEqual(report["items"][0]["boundary_count"], 2)
            self.assertEqual(report["items"][0]["mean_abs_boundary_delta_ms"], 15)
            aligned = json.loads((output_dir / "candidates" / "front-a.master.json").read_text(encoding="utf-8"))
            self.assertEqual(aligned["segments"][0]["start_ms"], 10)
            self.assertEqual(aligned["segments"][0]["end_ms"], 120)
            diagnostics = json.loads(
                (output_dir / "diagnostics" / "front-a.alignment-diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostics["changed_segments"], 1)
            self.assertEqual(diagnostics["boundary_count"], 2)
            self.assertEqual(diagnostics["mean_abs_boundary_delta_ms"], 15)
            self.assertEqual(diagnostics["within_250ms_boundary_ratio"], 1.0)
            attach_plan = json.loads((output_dir / "attach-plan.json").read_text(encoding="utf-8"))
            self.assertEqual(attach_plan["candidates"][0]["candidate"], "candidates/front-a.master.json")
            self.assertEqual(attach_plan["candidates"][0]["candidate_id"], "base-candidate-aligned")
            manifest = json.loads((output_dir / "eval-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["cases"][0]["candidate"], "candidates/front-a.master.json")
            self.assertEqual(manifest["cases"][0]["candidate_id"], "base-candidate-aligned")
            self.assertEqual(manifest["cases"][0]["reference"], "../cases/references/front-a.master.json")

    def test_align_review_case_candidates_requires_configured_aligner_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_index = root / "case-index.json"
            output_dir = root / "aligned"
            case_index.write_text(
                json.dumps({"format": "custom-asmr-review-case-set-v1", "items": []}),
                encoding="utf-8",
            )

            with mock.patch.dict("os.environ", {}, clear=True):
                result, _, error = run_cli_with_stderr(
                    ["align-review-case-candidates", str(case_index), "-o", str(output_dir)]
                )

            self.assertEqual(result, 1)
            self.assertIn("CASRT_ALIGNER_COMMAND is required", error)
            self.assertFalse(output_dir.exists())

    def test_build_candidate_attach_plan_matches_case_files_and_feeds_attach(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "cases"
            candidate_dir = root / "candidate-outputs"
            plan_dir = root / "plans"
            case_dir.mkdir()
            candidate_dir.mkdir()
            case_index = case_dir / "case-index.json"
            output_plan = plan_dir / "attach-candidates.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "audio/front-a.wav",
                                "reference": "references/front-a.master.json",
                                "segments": 1,
                                "review_count": 0,
                            },
                            {
                                "id": "front-b",
                                "audio": "audio/front-b.wav",
                                "reference": "references/front-b.master.json",
                                "segments": 1,
                                "review_count": 0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (candidate_dir / "front-a.srt").write_text(
                "1\n00:00:00,000 --> 00:00:00,002\n候補A\n",
                encoding="utf-8",
            )
            (candidate_dir / "front-b.master.json").write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "front-b.wav", "duration_ms": 2},
                        "segments": [
                            {
                                "id": "seg_000001",
                                "start_ms": 0,
                                "end_ms": 2,
                                "channel": "MIX",
                                "kind": "speech",
                                "text": "候補B",
                                "needs_review": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "build-candidate-attach-plan",
                    "--json",
                    str(case_index),
                    str(candidate_dir),
                    "-o",
                    str(output_plan),
                    "--candidate-id",
                    "local-candidate",
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-case-candidate-attach-plan-build-v1")
            self.assertEqual(report["candidate_count"], 2)
            plan = json.loads(output_plan.read_text(encoding="utf-8"))
            self.assertEqual(plan["format"], "custom-asmr-case-candidate-attach-plan-v1")
            self.assertEqual(plan["candidate_id"], "local-candidate")
            self.assertEqual(
                plan["candidates"],
                [
                    {"case_id": "front-a", "candidate": "../candidate-outputs/front-a.srt"},
                    {"case_id": "front-b", "candidate": "../candidate-outputs/front-b.master.json"},
                ],
            )

            attach_result, attach_output = run_cli(
                [
                    "attach-review-case-candidates",
                    "--json",
                    str(case_index),
                    str(output_plan),
                ]
            )

            self.assertEqual(attach_result, 0)
            attach_report = json.loads(attach_output)
            self.assertEqual(attach_report["candidate_count"], 2)
            updated_case_index = json.loads(case_index.read_text(encoding="utf-8"))
            self.assertEqual(updated_case_index["items"][0]["candidate"], "candidates/front-a.master.json")
            self.assertEqual(updated_case_index["items"][1]["candidate"], "candidates/front-b.master.json")

    def test_build_candidate_attach_plan_rejects_ambiguous_case_files_without_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_dir = root / "candidate-outputs"
            candidate_dir.mkdir()
            case_index = root / "case-index.json"
            output_plan = root / "attach-candidates.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "audio/front-a.wav",
                                "reference": "references/front-a.master.json",
                                "segments": 1,
                                "review_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (candidate_dir / "front-a.srt").write_text(
                "1\n00:00:00,000 --> 00:00:00,002\n候補\n",
                encoding="utf-8",
            )
            (candidate_dir / "front-a.json").write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "front-a.wav", "duration_ms": 2},
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "build-candidate-attach-plan",
                    "--json",
                    str(case_index),
                    str(candidate_dir),
                    "-o",
                    str(output_plan),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(output, "")
            self.assertIn("ambiguous case files", error)
            self.assertFalse(output_plan.exists())

    def test_transcribe_review_case_candidates_creates_case_named_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "cases"
            audio_dir = case_dir / "audio"
            output_dir = root / "candidate-outputs"
            project_root = root / "projects"
            audio_dir.mkdir(parents=True)
            write_stereo_samples(audio_dir / "front-a.wav", [(100, 200), (300, 400)])
            write_stereo_samples(audio_dir / "front-b.wav", [(500, 600), (700, 800)])
            case_index = case_dir / "case-index.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "audio/front-a.wav",
                                "reference": "references/front-a.master.json",
                                "segments": 0,
                                "review_count": 0,
                            },
                            {
                                "id": "front-b",
                                "audio": "audio/front-b.wav",
                                "reference": "references/front-b.master.json",
                                "segments": 0,
                                "review_count": 0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                del endpoint, mime_type, source_language
                calls.append((channel, analyze_wav(audio_bytes).duration_ms))
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=analyze_wav(audio_bytes).duration_ms,
                        channel=channel,
                        kind="speech",
                        text=f"{channel}候補",
                    ),
                )

            with mock.patch("custom_asmr_srt_stack.cli.transcribe_audio", side_effect=fake_transcribe):
                result, output = run_cli(
                    [
                        "transcribe-review-case-candidates",
                        "--project-root",
                        str(project_root),
                        "--adapter",
                        "openai-compatible",
                        "--endpoint-url",
                        "http://localhost:8000/v1",
                        "--model-id",
                        "local-test-model",
                        "--json",
                        str(case_index),
                        "-o",
                        str(output_dir),
                    ]
                )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-case-candidate-transcription-v1")
            self.assertEqual(report["candidate_count"], 2)
            self.assertEqual([item["case_id"] for item in report["items"]], ["front-a", "front-b"])
            self.assertTrue((output_dir / "front-a.master.json").is_file())
            self.assertTrue((output_dir / "front-b.master.json").is_file())
            front_a = json.loads((output_dir / "front-a.master.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [(segment["channel"], segment["text"]) for segment in front_a["segments"]],
                [("L", "L候補"), ("R", "R候補")],
            )
            self.assertEqual(calls, [("L", 2), ("R", 2), ("L", 2), ("R", 2)])
            self.assertEqual(len(list(project_root.iterdir())), 2)

    def test_transcribe_review_case_candidates_rejects_missing_audio_before_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_index = root / "case-index.json"
            output_dir = root / "candidate-outputs"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "audio/missing.wav",
                                "reference": "references/front-a.master.json",
                                "segments": 0,
                                "review_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "transcribe-review-case-candidates",
                    "--adapter",
                    "openai-compatible",
                    "--endpoint-url",
                    "http://localhost:8000/v1",
                    "--model-id",
                    "local-test-model",
                    "--json",
                    str(case_index),
                    "-o",
                    str(output_dir),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(output, "")
            self.assertIn("review case audio file does not exist", error)
            self.assertFalse(output_dir.exists())

    def test_attach_review_case_candidates_rejects_incomplete_plan_before_side_effects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "cases"
            audio_dir = case_dir / "audio"
            reference_dir = case_dir / "references"
            audio_dir.mkdir(parents=True)
            reference_dir.mkdir()
            write_stereo_samples(audio_dir / "front-a.wav", [(100, 200), (300, 400)])
            (reference_dir / "front-a.master.json").write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "front-a.wav", "duration_ms": 2},
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )
            case_index = case_dir / "case-index.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "audio/front-a.wav",
                                "reference": "references/front-a.master.json",
                                "segments": 0,
                                "review_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            attach_plan = root / "attach-candidates.json"
            attach_plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-candidate-attach-plan-v1",
                        "candidates": [{"case_id": "front-b", "candidate": "missing.srt"}],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                ["attach-review-case-candidates", str(case_index), str(attach_plan)]
            )

            self.assertEqual(result, 1)
            self.assertIn("missing case ids: front-a", error)
            self.assertFalse((case_dir / "candidates").exists())
            unchanged = json.loads(case_index.read_text(encoding="utf-8"))
            self.assertNotIn("candidate", unchanged["items"][0])

    def test_attach_review_case_candidates_requires_replace_for_existing_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "cases"
            audio_dir = case_dir / "audio"
            reference_dir = case_dir / "references"
            candidate_dir = case_dir / "candidates"
            audio_dir.mkdir(parents=True)
            reference_dir.mkdir()
            candidate_dir.mkdir()
            write_stereo_samples(audio_dir / "front-a.wav", [(100, 200), (300, 400)])
            (reference_dir / "front-a.master.json").write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "front-a.wav", "duration_ms": 2},
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )
            (candidate_dir / "front-a.master.json").write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "front-a.wav", "duration_ms": 2},
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )
            new_candidate = root / "new-candidate.srt"
            new_candidate.write_text("1\n00:00:00,000 --> 00:00:00,001\n新規\n", encoding="utf-8")
            case_index = case_dir / "case-index.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "audio/front-a.wav",
                                "reference": "references/front-a.master.json",
                                "candidate": "candidates/front-a.master.json",
                                "candidate_id": "old",
                                "segments": 0,
                                "review_count": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            attach_plan = root / "attach-candidates.json"
            attach_plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-candidate-attach-plan-v1",
                        "candidate_id": "new",
                        "candidates": [{"case_id": "front-a", "candidate": "new-candidate.srt"}],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                ["attach-review-case-candidates", str(case_index), str(attach_plan)]
            )

            self.assertEqual(result, 1)
            self.assertIn("already has a candidate", error)
            unchanged = json.loads(case_index.read_text(encoding="utf-8"))
            self.assertEqual(unchanged["items"][0]["candidate_id"], "old")

            replace_result, replace_output = run_cli(
                [
                    "attach-review-case-candidates",
                    "--json",
                    "--replace",
                    str(case_index),
                    str(attach_plan),
                ]
            )

            self.assertEqual(replace_result, 0)
            self.assertTrue(json.loads(replace_output)["replace"])
            replaced_index = json.loads(case_index.read_text(encoding="utf-8"))
            self.assertEqual(replaced_index["items"][0]["candidate_id"], "new")
            replaced_candidate = json.loads((candidate_dir / "front-a.master.json").read_text(encoding="utf-8"))
            self.assertEqual(replaced_candidate["segments"][0]["text"], "新規")

    def test_freeze_case_references_can_fail_on_remaining_review_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            reference = root / "reference.srt"
            plan = root / "plan.json"
            prepared_dir = root / "cases"
            frozen_dir = root / "frozen"
            write_stereo_samples(audio, [(100, 200), (300, 400), (500, 600), (700, 800)])
            reference.write_text(
                "1\n00:00:00,000 --> 00:00:00,002\n前半\n\n"
                "2\n00:00:00,001 --> 00:00:00,003\n中央\n",
                encoding="utf-8",
            )
            plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-slice-plan-v1",
                        "reference_type": "pseudo-gold",
                        "cases": [
                            {
                                "id": "front-a",
                                "audio": "source.wav",
                                "reference": "reference.srt",
                                "start_ms": 1,
                                "end_ms": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["prepare-review-cases", str(plan), "-o", str(prepared_dir)])

            result, output, error = run_cli_with_stderr(
                [
                    "freeze-case-references",
                    "--fail-on-review",
                    str(prepared_dir / "case-index.json"),
                    "-o",
                    str(frozen_dir),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(output, "")
            self.assertIn("reference review_count=1", error)
            self.assertFalse(frozen_dir.exists())

    def test_freeze_case_references_can_fail_on_reference_audit_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "cases"
            audio_dir = case_dir / "audio"
            reference_dir = case_dir / "references"
            audio_dir.mkdir(parents=True)
            reference_dir.mkdir()
            (audio_dir / "front.wav").write_bytes(b"RIFFcaseWAVE")
            reference = MasterDocument(
                source_language="ja",
                source_file="front.wav",
                duration_ms=2000,
                segments=(
                    Segment("seg_000001", 0, 1000, "L", "speech", "あ"),
                    Segment("seg_000002", 500, 1500, "L", "speech", "い"),
                ),
            )
            (reference_dir / "front.master.json").write_text(
                json.dumps(reference.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            case_index = case_dir / "case-index.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front",
                                "audio": "audio/front.wav",
                                "reference": "references/front.master.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            frozen_dir = root / "frozen"

            result, output, error = run_cli_with_stderr(
                [
                    "freeze-case-references",
                    "--fail-on-reference-audit",
                    str(case_index),
                    "-o",
                    str(frozen_dir),
                ]
            )

        self.assertEqual(result, 1)
        self.assertEqual(output, "")
        self.assertIn("reference_audit_item_count=1", error)
        self.assertIn("reference-same-channel-overlap", error)
        self.assertFalse(frozen_dir.exists())

    def test_freeze_case_references_writes_clean_case_set_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            plan = root / "plan.json"
            prepared_dir = root / "cases"
            frozen_dir = root / "frozen"
            status_path = root / "frozen-status.json"
            write_stereo_samples(audio, [(100, 200), (300, 400), (500, 600), (700, 800)])
            reference.write_text(
                "2\n00:00:00,002 --> 00:00:00,004\n後半\n\n"
                "1\n00:00:00,000 --> 00:00:00,002\n前半\n",
                encoding="utf-8",
            )
            candidate.write_text("1\n00:00:00,001 --> 00:00:00,003\n候補\n", encoding="utf-8")
            plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-slice-plan-v1",
                        "reference_type": "pseudo-gold",
                        "cases": [
                            {
                                "id": "front-a",
                                "audio": "source.wav",
                                "reference": "reference.srt",
                                "candidate": "candidate.srt",
                                "candidate_id": "draft-candidate",
                                "start_ms": 1,
                                "end_ms": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["prepare-review-cases", str(plan), "-o", str(prepared_dir)])

            result, output = run_cli(
                [
                    "freeze-case-references",
                    "--json",
                    "--reference-type",
                    "human-reviewed",
                    "--reference-notes",
                    "manual pass complete",
                    str(prepared_dir / "case-index.json"),
                    "-o",
                    str(frozen_dir),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-case-reference-freeze-v1")
            self.assertEqual(report["reference_type"], "human-reviewed")
            self.assertEqual(report["review_count"], 0)
            frozen_reference = json.loads((frozen_dir / "references" / "front-a.master.json").read_text(encoding="utf-8"))
            self.assertEqual([segment["id"] for segment in frozen_reference["segments"]], [
                "seg_000001",
                "seg_000002",
            ])
            self.assertFalse(any(segment["needs_review"] for segment in frozen_reference["segments"]))
            case_index = json.loads((frozen_dir / "case-index.json").read_text(encoding="utf-8"))
            self.assertEqual(case_index["reference_type"], "human-reviewed")
            self.assertEqual(case_index["items"][0]["review_count"], 0)
            self.assertEqual(case_index["items"][0]["candidate"], str(prepared_dir / "candidates" / "front-a.master.json"))
            eval_manifest = json.loads((frozen_dir / "eval-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_manifest["reference_type"], "human-reviewed")
            self.assertEqual(eval_manifest["cases"][0]["candidate_id"], "draft-candidate")

            status_result, status_output = run_cli(
                [
                    "review-case-status",
                    "--json",
                    "--fail-on-review",
                    "-o",
                    str(status_path),
                    str(frozen_dir / "case-index.json"),
                ]
            )
            self.assertEqual(status_result, 0)
            self.assertEqual(json.loads(status_output)["reference_review_count"], 0)

    def test_build_eval_manifest_can_fail_on_reference_audit_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "cases"
            audio_dir = case_dir / "audio"
            reference_dir = case_dir / "references"
            candidate_dir = case_dir / "candidates"
            audio_dir.mkdir(parents=True)
            reference_dir.mkdir()
            candidate_dir.mkdir()
            (audio_dir / "front.wav").write_bytes(b"RIFFcaseWAVE")
            reference = MasterDocument(
                source_language="ja",
                source_file="front.wav",
                duration_ms=2000,
                segments=(
                    Segment("seg_000001", 0, 1000, "L", "speech", "あ"),
                    Segment("seg_000002", 500, 1500, "L", "speech", "い"),
                ),
            )
            candidate = MasterDocument(
                source_language="ja",
                source_file="front.wav",
                duration_ms=2000,
                segments=(Segment("seg_000001", 0, 1500, "MIX", "speech", "あい"),),
            )
            (reference_dir / "front.master.json").write_text(
                json.dumps(reference.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            (candidate_dir / "front.master.json").write_text(
                json.dumps(candidate.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            case_index = case_dir / "case-index.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front",
                                "audio": "audio/front.wav",
                                "reference": "references/front.master.json",
                                "candidate": "candidates/front.master.json",
                                "candidate_id": "draft",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "eval-manifest.json"

            result, output, error = run_cli_with_stderr(
                [
                    "build-eval-manifest",
                    "--fail-on-reference-audit",
                    str(case_index),
                    "-o",
                    str(manifest),
                ]
            )

        self.assertEqual(result, 1)
        self.assertEqual(output, "")
        self.assertIn("reference_audit_item_count=1", error)
        self.assertFalse(manifest.exists())

    def test_freeze_case_references_rejects_missing_sources_before_output_side_effects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_index = root / "case-index.json"
            output_dir = root / "frozen"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "missing.wav",
                                "reference": "missing.master.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                ["freeze-case-references", str(case_index), "-o", str(output_dir)]
            )

            self.assertEqual(result, 1)
            self.assertIn("audio file does not exist", error)
            self.assertFalse(output_dir.exists())

    def test_build_eval_manifest_from_prepared_candidate_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            plan = root / "plan.json"
            output_dir = root / "cases"
            manifest_path = root / "human-reviewed-manifest.json"
            write_stereo_samples(audio, [(100, 200), (300, 400), (500, 600), (700, 800)])
            reference.write_text("1\n00:00:00,000 --> 00:00:00,002\n参照\n", encoding="utf-8")
            candidate.write_text("1\n00:00:00,000 --> 00:00:00,002\n候補\n", encoding="utf-8")
            plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-slice-plan-v1",
                        "reference_type": "pseudo-gold",
                        "cases": [
                            {
                                "id": "front-a",
                                "audio": "source.wav",
                                "reference": "reference.srt",
                                "candidate": "candidate.srt",
                                "candidate_id": "draft-candidate",
                                "start_ms": 0,
                                "end_ms": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["prepare-review-cases", str(plan), "-o", str(output_dir)])

            result, output = run_cli(
                [
                    "build-eval-manifest",
                    "--json",
                    "--reference-type",
                    "human-reviewed",
                    "--reference-notes",
                    "manual pass complete",
                    str(output_dir / "case-index.json"),
                    "-o",
                    str(manifest_path),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-eval-manifest-build-v1")
            self.assertEqual(report["case_count"], 1)
            self.assertEqual(report["reference_type"], "human-reviewed")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "custom-asmr-eval-manifest-v1")
            self.assertEqual(manifest["reference_type"], "human-reviewed")
            self.assertEqual(manifest["reference_notes"], "manual pass complete")
            self.assertEqual(
                manifest["cases"][0],
                {
                    "id": "front-a",
                    "reference": "references/front-a.master.json",
                    "candidate": "candidates/front-a.master.json",
                    "candidate_id": "draft-candidate",
                },
            )

    def test_build_eval_manifest_can_require_clean_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            plan = root / "plan.json"
            output_dir = root / "cases"
            manifest_path = root / "manifest.json"
            write_stereo_samples(audio, [(100, 200), (300, 400), (500, 600), (700, 800)])
            reference.write_text("1\n00:00:00,000 --> 00:00:00,002\n参照\n", encoding="utf-8")
            candidate.write_text("1\n00:00:00,000 --> 00:00:00,002\n候補\n", encoding="utf-8")
            plan.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-case-slice-plan-v1",
                        "cases": [
                            {
                                "id": "front-a",
                                "audio": "source.wav",
                                "reference": "reference.srt",
                                "candidate": "candidate.srt",
                                "start_ms": 1,
                                "end_ms": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["prepare-review-cases", str(plan), "-o", str(output_dir)])

            result, _, error = run_cli_with_stderr(
                [
                    "build-eval-manifest",
                    "--fail-on-review",
                    str(output_dir / "case-index.json"),
                    "-o",
                    str(manifest_path),
                ]
            )

            self.assertEqual(result, 1)
            self.assertIn("reference review_count=1", error)
            self.assertFalse(manifest_path.exists())

    def test_eval_transcript_quality_gate_fails_after_emitting_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            report_path = root / "report.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,100 --> 00:00:02,200\nね\n", encoding="utf-8")

            result, output, error = run_cli_with_stderr(
                [
                    "eval-transcript",
                    "--json",
                    "-o",
                    str(report_path),
                    "--max-practical-cer",
                    "0.10",
                    str(reference),
                    str(candidate),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output)["text_practical"]["edit_distance"], 1)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["text_practical"]["edit_distance"], 1)
            self.assertIn("quality gate failed", error)
            self.assertIn("practical CER", error)

    def test_eval_transcript_quality_gate_fails_when_channel_accuracy_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,100 --> 00:00:02,200\nねえ\n", encoding="utf-8")

            result, _, error = run_cli_with_stderr(
                [
                    "eval-transcript",
                    "--min-channel-time-aligned-accuracy",
                    "0.90",
                    str(reference),
                    str(candidate),
                ]
            )

            self.assertEqual(result, 1)
            self.assertIn("channel time-aligned accuracy is unavailable", error)

    def test_eval_transcript_quality_gate_fails_when_candidate_mix_ratio_is_too_high(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\n[L] ねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,100 --> 00:00:02,200\n[LR] ねえ\n", encoding="utf-8")

            result, _, error = run_cli_with_stderr(
                [
                    "eval-transcript",
                    "--max-channel-time-aligned-mix-ratio",
                    "0.50",
                    str(reference),
                    str(candidate),
                ]
            )

            self.assertEqual(result, 1)
            self.assertIn("channel time-aligned MIX ratio", error)

    def test_eval_transcript_quality_gate_fails_when_review_effort_is_too_high(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,000 --> 00:00:02,000\nね\n", encoding="utf-8")

            result, output, error = run_cli_with_stderr(
                [
                    "eval-transcript",
                    "--json",
                    "--max-segments-needing-edit-ratio",
                    "0.0",
                    str(reference),
                    str(candidate),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output)["review_effort"]["segments_needing_edit"], 1)
            self.assertIn("segments needing edit ratio", error)

    def test_eval_transcript_quality_gate_fails_when_candidate_review_ratio_is_too_high(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.master.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "candidate.wav", "duration_ms": 3000},
                        "segments": [
                            {
                                "id": "seg_000001",
                                "start_ms": 1000,
                                "end_ms": 2000,
                                "channel": "MIX",
                                "kind": "speech",
                                "text": "ねえ",
                                "needs_review": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "eval-transcript",
                    "--json",
                    "--max-candidate-review-ratio",
                    "0.0",
                    str(reference),
                    str(candidate),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output)["review"]["candidate_review_ratio"], 1.0)
            self.assertIn("candidate review ratio", error)

    def test_eval_transcript_product_gate_applies_documented_thresholds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.master.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\n[L] ねえ\n", encoding="utf-8")
            candidate.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "candidate.wav", "duration_ms": 3000},
                        "segments": [
                            {
                                "id": "seg_000001",
                                "start_ms": 1000,
                                "end_ms": 2000,
                                "channel": "L",
                                "kind": "speech",
                                "text": "ねえ",
                                "needs_review": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "eval-transcript",
                    "--json",
                    "--product-gate",
                    str(reference),
                    str(candidate),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output)["review_effort"]["segments_needing_edit"], 0)
            self.assertIn("candidate review ratio 1.0000 > 0.0000", error)

    def test_eval_manifest_outputs_aggregated_json_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            manifest = root / "gold.json"
            report_path = root / "report.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,100 --> 00:00:02,200\nね\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "cases": [
                            {
                                "id": "sample",
                                "reference": "reference.srt",
                                "candidate": "candidate.srt",
                                "candidate_id": "qwen-energy",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "eval-manifest",
                    "--json",
                    "-o",
                    str(report_path),
                    str(manifest),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-eval-suite-v1")
            self.assertEqual(report["case_count"], 1)
            self.assertEqual(report["summary"]["text"]["edit_distance"], 1)
            self.assertEqual(report["cases"][0]["candidate_id"], "qwen-energy")
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["summary"]["text"]["edit_distance"], 1)

    def test_eval_manifest_product_gate_requires_human_reviewed_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            manifest = root / "gold.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\n[L] ねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,000 --> 00:00:02,000\n[L] ねえ\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "cases": [{"id": "sample", "reference": "reference.srt", "candidate": "candidate.srt"}],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "eval-manifest",
                    "--json",
                    "--product-gate",
                    str(manifest),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output)["summary"]["review_effort"]["segments_needing_edit"], 0)
            self.assertIn("reference type gate failed", error)
            self.assertIn("human-reviewed", error)

    def test_compare_evals_ranks_reports_by_review_effort(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate_good = root / "candidate-good.srt"
            candidate_bad = root / "candidate-bad.srt"
            report_good = root / "report-good.json"
            report_bad = root / "report-bad.json"
            comparison_path = root / "comparison.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate_good.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate_bad.write_text("1\n00:00:01,000 --> 00:00:02,000\nね\n", encoding="utf-8")
            run_cli(["eval-transcript", "-o", str(report_good), str(reference), str(candidate_good)])
            run_cli(["eval-transcript", "-o", str(report_bad), str(reference), str(candidate_bad)])

            result, output = run_cli(
                [
                    "compare-evals",
                    "--json",
                    "-o",
                    str(comparison_path),
                    "--max-practical-cer",
                    "0.10",
                    str(report_bad),
                    str(report_good),
                ]
            )

            self.assertEqual(result, 0)
            comparison = json.loads(output)
            self.assertEqual(comparison["format"], "custom-asmr-eval-comparison-v1")
            self.assertEqual([item["label"] for item in comparison["items"]], ["report-good", "report-bad"])
            self.assertEqual(comparison["items"][0]["segments_needing_edit_ratio"], 0.0)
            self.assertGreater(comparison["items"][1]["segments_needing_edit_ratio"], 0.0)
            self.assertEqual(comparison["items"][0]["text_edit_segment_ratio"], 0.0)
            self.assertEqual(comparison["items"][1]["text_edit_segment_ratio"], 1.0)
            self.assertEqual(comparison["items"][1]["channel_edit_segment_ratio"], 0.0)
            self.assertEqual(comparison["items"][1]["timing_edit_segment_ratio"], 0.0)
            self.assertEqual(comparison["items"][1]["missing_reference_segment_ratio"], 0.0)
            self.assertEqual(comparison["items"][1]["extra_candidate_segment_ratio"], 0.0)
            self.assertIsNone(comparison["items"][0]["dominant_review_effort_reason"])
            self.assertEqual(comparison["items"][1]["dominant_review_effort_reason"], "text")
            self.assertEqual(comparison["items"][1]["dominant_review_effort_ratio"], 1.0)
            self.assertEqual(comparison["items"][1]["review_effort_reason_ranking"][0], {"reason": "text", "ratio": 1.0})
            self.assertEqual(comparison["quality_gate"], {"max_practical_cer": 0.1})
            self.assertTrue(comparison["items"][0]["gate_passed"])
            self.assertFalse(comparison["items"][1]["gate_passed"])
            self.assertIn("practical CER", comparison["items"][1]["gate_failures"][0])
            self.assertEqual(
                json.loads(comparison_path.read_text(encoding="utf-8"))["items"][0]["label"],
                "report-good",
            )

    def test_compare_evals_marks_candidate_review_ratio_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.master.json"
            report_path = root / "report.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "candidate.wav", "duration_ms": 3000},
                        "segments": [
                            {
                                "id": "seg_000001",
                                "start_ms": 1000,
                                "end_ms": 2000,
                                "channel": "MIX",
                                "kind": "speech",
                                "text": "ねえ",
                                "needs_review": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["eval-transcript", "-o", str(report_path), str(reference), str(candidate)])

            result, output = run_cli(
                [
                    "compare-evals",
                    "--json",
                    "--max-candidate-review-ratio",
                    "0.0",
                    str(report_path),
                ]
            )

            self.assertEqual(result, 0)
            comparison = json.loads(output)
            self.assertEqual(comparison["quality_gate"], {"max_candidate_review_ratio": 0.0})
            self.assertEqual(comparison["items"][0]["candidate_review_ratio"], 1.0)
            self.assertFalse(comparison["items"][0]["gate_passed"])
            self.assertIn("candidate review ratio", comparison["items"][0]["gate_failures"][0])

    def test_compare_evals_product_gate_marks_reference_type_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            manifest = root / "gold.json"
            report_path = root / "report.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\n[L] ねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,000 --> 00:00:02,000\n[L] ねえ\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "cases": [{"id": "sample", "reference": "reference.srt", "candidate": "candidate.srt"}],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["eval-manifest", "-o", str(report_path), str(manifest)])

            result, output = run_cli(["compare-evals", "--json", "--product-gate", str(report_path)])

            self.assertEqual(result, 0)
            comparison = json.loads(output)
            self.assertEqual(comparison["quality_gate"]["preset"], "local-asmr-v1")
            self.assertEqual(comparison["quality_gate"]["require_reference_type"], "human-reviewed")
            self.assertFalse(comparison["items"][0]["gate_passed"])
            self.assertIn("reference_type", comparison["items"][0]["gate_failures"][0])

    def test_pipeline_readiness_fails_when_pipeline_stages_still_block_asr_only_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_audit = root / "reference-audit.json"
            vad_comparison = root / "vad-comparison.json"
            eval_comparison = root / "eval-comparison.json"
            readiness_path = root / "readiness.json"
            reference_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-audit-suite-v1",
                        "case_count": 1,
                        "summary": {
                            "segment_count": 2,
                            "review_count": 1,
                            "same_channel_overlap_pair_count": 1,
                            "exact_boundary_overlap_pair_count": 0,
                            "long_segment_count": 0,
                            "speech_coverage_ratio": 0.5,
                            "flag_type_counts": {"review_flag_segments": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )
            vad_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-comparison-v1",
                        "quality_gate": {"max_detected_interval_ms": 60000},
                        "items": [
                            {
                                "label": "energy",
                                "gate_passed": False,
                                "gate_failures": ["missed reference duration 25ms > 0ms"],
                                "missed_reference_duration_ms": 25,
                                "extra_detected_duration_ms": 0,
                                "reference_recall": 0.95,
                                "detected_precision": 1.0,
                                "detected_max_interval_ms": 30000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            eval_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-comparison-v1",
                        "items": [
                            {
                                "label": "candidate",
                                "timing_edit_segment_ratio": 0.50,
                                "time_aligned_500ms_ratio": 0.50,
                                "channel_edit_segment_ratio": 0.25,
                                "channel_time_aligned_accuracy": 0.75,
                                "channel_time_aligned_mix_ratio": 0.60,
                                "text_edit_segment_ratio": 1.0,
                                "segments_needing_edit_ratio": 1.0,
                                "practical_cer": 0.20,
                                "dominant_review_effort_reason": "text",
                                "dominant_review_effort_ratio": 1.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "pipeline-readiness",
                    "--json",
                    "--fail-unless-asr-only-ready",
                    "--reference-audit",
                    str(reference_audit),
                    "--vad-comparison",
                    str(vad_comparison),
                    "--eval-comparison",
                    str(eval_comparison),
                    "-o",
                    str(readiness_path),
                ]
            )

            self.assertEqual(result, 1)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-pipeline-readiness-v1")
            self.assertFalse(report["summary"]["asr_only_ready"])
            self.assertEqual(
                report["summary"]["asr_only_blocking_stages"],
                ["reference", "vad_chunking", "alignment", "channel_attribution"],
            )
            self.assertEqual(report["summary"]["next_stage"], "reference")
            self.assertIn("pipeline is not ASR-only ready", error)
            self.assertEqual(
                json.loads(readiness_path.read_text(encoding="utf-8"))["stages"]["vad_chunking"]["status"],
                "fail",
            )

    def test_pipeline_readiness_can_report_that_only_text_asr_remains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_audit = root / "reference-audit.json"
            vad_comparison = root / "vad-comparison.json"
            eval_comparison = root / "eval-comparison.json"
            reference_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-audit-suite-v1",
                        "case_count": 1,
                        "summary": {
                            "segment_count": 1,
                            "review_count": 0,
                            "same_channel_overlap_pair_count": 0,
                            "exact_boundary_overlap_pair_count": 0,
                            "long_segment_count": 0,
                            "speech_coverage_ratio": 0.3,
                            "flag_type_counts": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            vad_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-comparison-v1",
                        "quality_gate": {"max_missed_reference_ms": 50, "min_reference_recall": 0.95},
                        "items": [
                            {
                                "label": "chunked",
                                "gate_passed": True,
                                "gate_failures": [],
                                "missed_reference_duration_ms": 25,
                                "extra_detected_duration_ms": 10,
                                "reference_recall": 0.975,
                                "detected_precision": 0.95,
                                "detected_max_interval_ms": 30000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            eval_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-comparison-v1",
                        "items": [
                            {
                                "label": "candidate",
                                "timing_edit_segment_ratio": 0.0,
                                "time_aligned_500ms_ratio": 1.0,
                                "channel_edit_segment_ratio": 0.0,
                                "channel_time_aligned_accuracy": 1.0,
                                "channel_time_aligned_mix_ratio": 0.0,
                                "text_edit_segment_ratio": 0.5,
                                "segments_needing_edit_ratio": 0.5,
                                "practical_cer": 0.12,
                                "dominant_review_effort_reason": "text",
                                "dominant_review_effort_ratio": 0.5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "pipeline-readiness",
                    "--json",
                    "--fail-unless-asr-only-ready",
                    "--reference-audit",
                    str(reference_audit),
                    "--vad-comparison",
                    str(vad_comparison),
                    "--eval-comparison",
                    str(eval_comparison),
                ]
            )

            self.assertEqual(result, 0, error)
            report = json.loads(output)
            self.assertTrue(report["summary"]["asr_only_ready"])
            self.assertFalse(report["summary"]["production_ready"])
            self.assertEqual(report["summary"]["asr_only_blocking_stages"], [])
            self.assertEqual(report["summary"]["quality_blocking_stages"], ["text_asr"])
            self.assertEqual(report["summary"]["next_stage"], "text_asr")

    def test_pipeline_readiness_can_use_separate_channel_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_audit = root / "reference-audit.json"
            vad_comparison = root / "vad-comparison.json"
            eval_comparison = root / "eval-comparison.json"
            channel_comparison = root / "channel-comparison.json"
            reference_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-audit-suite-v1",
                        "case_count": 1,
                        "summary": {
                            "segment_count": 1,
                            "review_count": 0,
                            "same_channel_overlap_pair_count": 0,
                            "exact_boundary_overlap_pair_count": 0,
                            "long_segment_count": 0,
                            "speech_coverage_ratio": 0.3,
                            "flag_type_counts": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            vad_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-comparison-v1",
                        "quality_gate": {"max_missed_reference_ms": 0},
                        "items": [
                            {
                                "label": "chunked",
                                "gate_passed": True,
                                "gate_failures": [],
                                "missed_reference_duration_ms": 0,
                                "extra_detected_duration_ms": 0,
                                "reference_recall": 1.0,
                                "detected_precision": 1.0,
                                "detected_max_interval_ms": 30000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            eval_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-comparison-v1",
                        "items": [
                            {
                                "label": "asr-candidate",
                                "timing_edit_segment_ratio": 0.0,
                                "time_aligned_500ms_ratio": 1.0,
                                "channel_edit_segment_ratio": 0.0,
                                "channel_time_aligned_accuracy": 1.0,
                                "channel_time_aligned_mix_ratio": 0.0,
                                "text_edit_segment_ratio": 0.0,
                                "segments_needing_edit_ratio": 0.0,
                                "practical_cer": 0.0,
                                "dominant_review_effort_reason": None,
                                "dominant_review_effort_ratio": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            channel_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-comparison-v1",
                        "items": [
                            {
                                "label": "channel-sweep",
                                "channel_edit_segment_ratio": 0.5,
                                "channel_time_aligned_accuracy": 0.5,
                                "channel_time_aligned_mix_ratio": 0.2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "pipeline-readiness",
                    "--json",
                    "--fail-unless-asr-only-ready",
                    "--reference-audit",
                    str(reference_audit),
                    "--vad-comparison",
                    str(vad_comparison),
                    "--eval-comparison",
                    str(eval_comparison),
                    "--channel-comparison",
                    str(channel_comparison),
                ]
            )

            self.assertEqual(result, 1)
            report = json.loads(output)
            self.assertEqual(report["summary"]["asr_only_blocking_stages"], ["channel_attribution"])
            self.assertEqual(report["summary"]["quality_blocking_stages"], ["channel_attribution"])
            self.assertEqual(report["stages"]["alignment"]["status"], "pass")
            self.assertEqual(report["stages"]["text_asr"]["status"], "pass")
            self.assertEqual(report["stages"]["channel_attribution"]["metrics"]["report"], str(channel_comparison))
            self.assertIn("channel_attribution", error)

    def test_pipeline_readiness_product_gate_uses_thresholds_instead_of_zero_edit_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_audit = root / "reference-audit.json"
            vad_comparison = root / "vad-comparison.json"
            eval_comparison = root / "eval-comparison.json"
            reference_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-audit-suite-v1",
                        "case_count": 1,
                        "summary": {
                            "segment_count": 10,
                            "review_count": 0,
                            "same_channel_overlap_pair_count": 0,
                            "exact_boundary_overlap_pair_count": 0,
                            "long_segment_count": 0,
                            "speech_coverage_ratio": 0.4,
                            "flag_type_counts": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            vad_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-comparison-v1",
                        "quality_gate": {"max_missed_reference_ms": 0},
                        "items": [
                            {
                                "label": "chunked",
                                "gate_passed": True,
                                "gate_failures": [],
                                "missed_reference_duration_ms": 0,
                                "extra_detected_duration_ms": 10,
                                "reference_recall": 1.0,
                                "detected_precision": 0.99,
                                "detected_max_interval_ms": 30000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            eval_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-comparison-v1",
                        "items": [
                            {
                                "label": "candidate",
                                "reference_type": "human-reviewed",
                                "timing_edit_segment_ratio": 0.2,
                                "time_aligned_500ms_ratio": 0.95,
                                "channel_edit_segment_ratio": 0.1,
                                "channel_time_aligned_accuracy": 0.9,
                                "channel_time_aligned_mix_ratio": 0.4,
                                "text_edit_segment_ratio": 0.1,
                                "segments_needing_edit_ratio": 0.1,
                                "practical_cer": 0.05,
                                "candidate_review_ratio": 0.0,
                                "dominant_review_effort_reason": "text",
                                "dominant_review_effort_ratio": 0.1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "pipeline-readiness",
                    "--json",
                    "--product-gate",
                    "--fail-unless-asr-only-ready",
                    "--reference-audit",
                    str(reference_audit),
                    "--vad-comparison",
                    str(vad_comparison),
                    "--eval-comparison",
                    str(eval_comparison),
                ]
            )

            self.assertEqual(result, 0, error)
            report = json.loads(output)
            self.assertTrue(report["summary"]["asr_only_ready"])
            self.assertTrue(report["summary"]["production_ready"])
            self.assertEqual(report["summary"]["quality_blocking_stages"], [])
            self.assertEqual(report["quality_gate"]["preset"], "local-asmr-v1")
            self.assertEqual(
                report["stages"]["reference"]["quality_gate"],
                {"required_reference_type": "human-reviewed"},
            )
            self.assertEqual(report["stages"]["reference"]["metrics"]["reference_type"], "human-reviewed")
            self.assertEqual(report["stages"]["alignment"]["quality_gate"], {"min_time_aligned_500ms_ratio": 0.9})
            self.assertEqual(report["stages"]["channel_attribution"]["status"], "pass")
            self.assertEqual(report["stages"]["text_asr"]["status"], "pass")

    def test_pipeline_readiness_product_gate_treats_pseudo_gold_as_reference_blocker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_audit = root / "reference-audit.json"
            vad_comparison = root / "vad-comparison.json"
            eval_comparison = root / "eval-comparison.json"
            reference_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-audit-suite-v1",
                        "case_count": 1,
                        "summary": {
                            "segment_count": 10,
                            "review_count": 0,
                            "same_channel_overlap_pair_count": 0,
                            "exact_boundary_overlap_pair_count": 0,
                            "long_segment_count": 0,
                            "speech_coverage_ratio": 0.4,
                            "flag_type_counts": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            vad_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-comparison-v1",
                        "quality_gate": {"max_missed_reference_ms": 0},
                        "items": [
                            {
                                "label": "chunked",
                                "gate_passed": True,
                                "gate_failures": [],
                                "missed_reference_duration_ms": 0,
                                "extra_detected_duration_ms": 10,
                                "reference_recall": 1.0,
                                "detected_precision": 0.99,
                                "detected_max_interval_ms": 30000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            eval_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-comparison-v1",
                        "items": [
                            {
                                "label": "candidate",
                                "reference_type": "pseudo-gold",
                                "timing_edit_segment_ratio": 0.0,
                                "time_aligned_500ms_ratio": 1.0,
                                "channel_edit_segment_ratio": 0.0,
                                "channel_time_aligned_accuracy": 1.0,
                                "channel_time_aligned_mix_ratio": 0.0,
                                "text_edit_segment_ratio": 0.0,
                                "segments_needing_edit_ratio": 0.0,
                                "practical_cer": 0.0,
                                "candidate_review_ratio": 0.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "pipeline-readiness",
                    "--json",
                    "--product-gate",
                    "--fail-unless-asr-only-ready",
                    "--reference-audit",
                    str(reference_audit),
                    "--vad-comparison",
                    str(vad_comparison),
                    "--eval-comparison",
                    str(eval_comparison),
                ]
            )

            self.assertEqual(result, 1)
            report = json.loads(output)
            self.assertFalse(report["summary"]["asr_only_ready"])
            self.assertFalse(report["summary"]["production_ready"])
            self.assertEqual(report["summary"]["asr_only_blocking_stages"], ["reference"])
            self.assertEqual(report["summary"]["quality_blocking_stages"], ["reference"])
            self.assertEqual(report["stages"]["reference"]["status"], "fail")
            self.assertIn("reference_type", report["stages"]["reference"]["reasons"][0])
            self.assertEqual(report["stages"]["text_asr"]["status"], "pass")
            self.assertIn("reference", error)

    def test_pipeline_readiness_reference_stage_uses_reference_channel_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_audit = root / "reference-audit.json"
            reference_channel_audit = root / "reference-channel-audit.json"
            vad_comparison = root / "vad-comparison.json"
            eval_comparison = root / "eval-comparison.json"
            reference_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-audit-suite-v1",
                        "case_count": 1,
                        "summary": {
                            "segment_count": 10,
                            "review_count": 0,
                            "same_channel_overlap_pair_count": 0,
                            "exact_boundary_overlap_pair_count": 0,
                            "long_segment_count": 0,
                            "speech_coverage_ratio": 0.4,
                            "flag_type_counts": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            reference_channel_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-channel-audit-suite-v1",
                        "case_count": 1,
                        "summary": {
                            "speech_segment_count": 3,
                            "eligible_reference_channel_count": 3,
                            "reference_mix_segment_count": 0,
                            "eligible_count": 3,
                            "energy_labeled_count": 2,
                            "energy_uncertain_count": 1,
                            "match_count": 1,
                            "mismatch_count": 1,
                            "match_ratio": 0.5,
                            "mismatch_ratio": 0.5,
                            "energy_labeled_ratio": 2 / 3,
                            "status_counts": {"match": 1, "mismatch": 1, "uncertain": 1},
                            "reason_counts": {"below_threshold": 1, "left_dominant": 1, "right_dominant": 1},
                            "reference_channel_counts": {"L": 2, "R": 1},
                            "energy_channel_counts": {"L": 1, "MIX": 1, "R": 1},
                        },
                        "cases": [],
                    }
                ),
                encoding="utf-8",
            )
            vad_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-comparison-v1",
                        "quality_gate": {"max_missed_reference_ms": 0},
                        "items": [
                            {
                                "label": "chunked",
                                "gate_passed": True,
                                "gate_failures": [],
                                "missed_reference_duration_ms": 0,
                                "extra_detected_duration_ms": 10,
                                "reference_recall": 1.0,
                                "detected_precision": 0.99,
                                "detected_max_interval_ms": 30000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            eval_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-comparison-v1",
                        "items": [
                            {
                                "label": "candidate",
                                "reference_type": "human-reviewed",
                                "timing_edit_segment_ratio": 0.0,
                                "time_aligned_500ms_ratio": 1.0,
                                "channel_edit_segment_ratio": 0.0,
                                "channel_time_aligned_accuracy": 1.0,
                                "channel_time_aligned_mix_ratio": 0.0,
                                "text_edit_segment_ratio": 0.0,
                                "segments_needing_edit_ratio": 0.0,
                                "practical_cer": 0.0,
                                "candidate_review_ratio": 0.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "pipeline-readiness",
                    "--json",
                    "--product-gate",
                    "--fail-unless-asr-only-ready",
                    "--reference-audit",
                    str(reference_audit),
                    "--reference-channel-audit",
                    str(reference_channel_audit),
                    "--vad-comparison",
                    str(vad_comparison),
                    "--eval-comparison",
                    str(eval_comparison),
                ]
            )

            self.assertEqual(result, 1)
            report = json.loads(output)
            self.assertEqual(report["summary"]["asr_only_blocking_stages"], ["reference"])
            self.assertEqual(report["stages"]["reference"]["status"], "fail")
            self.assertEqual(report["stages"]["reference"]["metrics"]["channel_audit"]["mismatch_count"], 1)
            self.assertEqual(report["stages"]["reference"]["metrics"]["channel_audit"]["energy_uncertain_count"], 1)
            self.assertIn("reference channel labels conflict", report["stages"]["reference"]["reasons"][0])
            self.assertIn("reference", error)

    def test_pipeline_readiness_rejects_gated_vad_without_gate_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vad_comparison = root / "vad-comparison.json"
            vad_comparison.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-vad-coverage-comparison-v1",
                        "quality_gate": {"max_missed_reference_ms": 50},
                        "items": [
                            {
                                "label": "candidate",
                                "missed_reference_duration_ms": 25,
                                "extra_detected_duration_ms": 0,
                                "reference_recall": 0.975,
                                "detected_precision": 0.95,
                                "detected_max_interval_ms": 30000,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                ["pipeline-readiness", "--json", "--vad-comparison", str(vad_comparison)]
            )

            self.assertEqual(result, 1)
            self.assertEqual(output, "")
            self.assertIn("gate_passed must be a boolean", error)

    def test_review_effort_outputs_items_from_eval_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            manifest = root / "gold.json"
            eval_report_path = root / "eval-suite.json"
            review_report_path = root / "review-effort.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\n[L] ねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,900 --> 00:00:03,000\n[R] ね\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "reference_type": "human-reviewed",
                        "cases": [
                            {
                                "id": "front-a",
                                "reference": "reference.srt",
                                "candidate": "candidate.srt",
                                "candidate_id": "qwen-align",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["eval-manifest", "-o", str(eval_report_path), str(manifest)])

            result, output = run_cli(
                [
                    "review-effort",
                    "--json",
                    "-o",
                    str(review_report_path),
                    str(eval_report_path),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-review-effort-v1")
            self.assertEqual(report["item_count"], 1)
            self.assertEqual(report["reason_counts"], {"text": 1, "channel": 1, "timing": 1})
            self.assertEqual(report["items"][0]["case_id"], "front-a")
            self.assertEqual(report["items"][0]["case_candidate_id"], "qwen-align")
            self.assertEqual(
                json.loads(review_report_path.read_text(encoding="utf-8"))["items"][0]["reference_type"],
                "human-reviewed",
            )

    def test_review_pack_creates_audio_clips_from_review_effort_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            audio_path = audio_dir / "front-a.wav"
            review_path = root / "review-effort.json"
            audio_map = root / "audio-map.json"
            case_index = root / "case-index.json"
            pack_dir = root / "review-pack"
            write_mono_wav(audio_path)
            case_index.write_text(
                json.dumps({"format": "custom-asmr-review-case-set-v1", "items": []}),
                encoding="utf-8",
            )
            review_path.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-effort-v1",
                        "source_report": "eval-suite.json",
                        "item_count": 2,
                        "reason_counts": {"text": 2},
                        "items": [
                            {
                                "case_id": "front-a",
                                "reference_id": "seg_000010",
                                "candidate_id": "seg_000010",
                                "start_ms": 0,
                                "end_ms": 1,
                                "reasons": ["text"],
                                "reference_text": "優先",
                                "candidate_text": "優",
                                "priority_score": 9000.0,
                                "priority_rank": 1,
                            },
                            {
                                "case_id": "front-a",
                                "reference_id": "seg_000001",
                                "candidate_id": "seg_000001",
                                "start_ms": 1,
                                "end_ms": 2,
                                "reasons": ["text"],
                                "reference_text": "後",
                                "candidate_text": "",
                                "priority_score": 1000.0,
                                "priority_rank": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audio_map.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-audio-map-v1",
                        "items": [{"case_id": "front-a", "audio": "audio/front-a.wav"}],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "review-pack",
                    "--json",
                    "--audio-map",
                    str(audio_map),
                    "--source-case-index",
                    str(case_index),
                    "-o",
                    str(pack_dir),
                    str(review_path),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-review-pack-v1")
            self.assertEqual(report["clip_count"], 2)
            self.assertEqual(report["source_case_index"], str(case_index))
            self.assertEqual([item["priority_rank"] for item in report["items"]], [1, 2])
            self.assertEqual([item["reference_id"] for item in report["items"]], ["seg_000010", "seg_000001"])
            self.assertEqual(report["items"][0]["source_case_index"], str(case_index))
            self.assertEqual(report["items"][0]["clip_start_ms"], 0)
            self.assertEqual(report["items"][0]["clip_end_ms"], 2)
            clip_path = pack_dir / report["items"][0]["clip_file"]
            self.assertTrue(clip_path.exists())
            self.assertEqual(analyze_wav(clip_path.read_bytes()).duration_ms, 2)
            index = json.loads((pack_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["items"][0]["priority_score"], 9000.0)
            self.assertEqual(index["items"][0]["reference_text"], "優先")
            self.assertEqual(index["source_case_index"], str(case_index))

    def test_review_pack_rejects_missing_source_case_index_before_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "front-a.wav"
            review_path = root / "review-effort.json"
            pack_dir = root / "review-pack"
            write_mono_wav(audio_path)
            review_path.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-effort-v1",
                        "items": [
                            {
                                "case_id": "front-a",
                                "reference_id": "seg_000001",
                                "start_ms": 0,
                                "end_ms": 2,
                                "reasons": ["text"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                [
                    "review-pack",
                    "--audio",
                    str(audio_path),
                    "--source-case-index",
                    str(root / "missing-case-index.json"),
                    "-o",
                    str(pack_dir),
                    str(review_path),
                ]
            )

            self.assertEqual(result, 1)
            self.assertIn("source case index is missing", error)
            self.assertFalse(pack_dir.exists())

    def test_review_pack_rejects_single_audio_for_multiple_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "front-a.wav"
            review_path = root / "review-effort.json"
            write_mono_wav(audio_path)
            review_path.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-effort-v1",
                        "items": [
                            {"case_id": "front-a", "start_ms": 0, "end_ms": 2, "reasons": ["text"]},
                            {"case_id": "front-b", "start_ms": 0, "end_ms": 2, "reasons": ["text"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                [
                    "review-pack",
                    "--audio",
                    str(audio_path),
                    "-o",
                    str(root / "review-pack"),
                    str(review_path),
                ]
            )

            self.assertEqual(result, 1)
            self.assertIn("multiple case_id values requires --audio-map", error)

    def test_review_pack_rejects_non_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "front-a.wav"
            review_path = root / "review-effort.json"
            pack_dir = root / "review-pack"
            pack_dir.mkdir()
            (pack_dir / "old.wav").write_bytes(b"stale")
            write_mono_wav(audio_path)
            review_path.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-effort-v1",
                        "items": [{"start_ms": 0, "end_ms": 2, "reasons": ["text"]}],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                [
                    "review-pack",
                    "--audio",
                    str(audio_path),
                    "-o",
                    str(pack_dir),
                    str(review_path),
                ]
            )

            self.assertEqual(result, 1)
            self.assertIn("output directory must be empty", error)

    def test_review_case_pack_creates_audio_clips_from_reference_review_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "cases"
            audio_dir = case_dir / "audio"
            reference_dir = case_dir / "references"
            audio_dir.mkdir(parents=True)
            reference_dir.mkdir()
            audio_path = audio_dir / "front-a.wav"
            reference_path = reference_dir / "front-a.master.json"
            case_index_path = case_dir / "case-index.json"
            pack_dir = root / "review-case-pack"
            write_stereo_samples(audio_path, [(100, 200)] * 8)
            reference_path.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-master-v1",
                        "source_language": "ja",
                        "audio": {"source_file": "front-a.wav", "duration_ms": 8},
                        "segments": [
                            {
                                "id": "seg_000001",
                                "start_ms": 0,
                                "end_ms": 2,
                                "channel": "MIX",
                                "kind": "speech",
                                "text": "確認済み",
                                "needs_review": False,
                            },
                            {
                                "id": "seg_000002",
                                "start_ms": 2,
                                "end_ms": 4,
                                "channel": "L",
                                "kind": "speech",
                                "text": "前半確認",
                                "needs_review": True,
                            },
                            {
                                "id": "seg_000003",
                                "start_ms": 6,
                                "end_ms": 8,
                                "channel": "R",
                                "kind": "speech",
                                "text": "後半確認",
                                "needs_review": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            case_index_path.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "reference_type": "pseudo-gold",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "audio/front-a.wav",
                                "reference": "references/front-a.master.json",
                                "segments": 3,
                                "review_count": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, output = run_cli(
                [
                    "review-case-pack",
                    "--json",
                    "--context-ms",
                    "1",
                    "-o",
                    str(pack_dir),
                    str(case_index_path),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-review-pack-v1")
            self.assertEqual(report["source_case_index"], str(case_index_path))
            self.assertEqual(report["clip_count"], 2)
            self.assertEqual([item["reference_id"] for item in report["items"]], ["seg_000002", "seg_000003"])
            self.assertEqual([item["priority_rank"] for item in report["items"]], [1, 2])
            self.assertEqual(report["items"][0]["reasons"], ["reference-needs-review"])
            self.assertEqual(report["items"][0]["reference_channel"], "L")
            self.assertEqual(report["items"][0]["reference_text"], "前半確認")
            self.assertEqual(report["items"][0]["candidate_text"], "")
            self.assertEqual(report["items"][0]["clip_start_ms"], 1)
            self.assertEqual(report["items"][0]["clip_end_ms"], 5)
            first_clip = pack_dir / report["items"][0]["clip_file"]
            second_clip = pack_dir / report["items"][1]["clip_file"]
            self.assertTrue(first_clip.exists())
            self.assertTrue(second_clip.exists())
            self.assertEqual(analyze_wav(first_clip.read_bytes()).duration_ms, 4)
            self.assertEqual(analyze_wav(second_clip.read_bytes()).duration_ms, 3)
            index = json.loads((pack_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["items"][1]["clip_start_ms"], 5)
            self.assertEqual(index["items"][1]["clip_end_ms"], 8)

    def test_review_case_pack_rejects_missing_inputs_before_output_side_effects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_index_path = root / "case-index.json"
            pack_dir = root / "review-case-pack"
            case_index_path.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "items": [
                            {
                                "id": "front-a",
                                "audio": "missing.wav",
                                "reference": "missing.master.json",
                                "segments": 1,
                                "review_count": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result, _, error = run_cli_with_stderr(
                [
                    "review-case-pack",
                    "-o",
                    str(pack_dir),
                    str(case_index_path),
                ]
            )

            self.assertEqual(result, 1)
            self.assertIn("audio file is missing", error)
            self.assertFalse(pack_dir.exists())

    def test_eval_manifest_quality_gate_passes_when_thresholds_are_met(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            manifest = root / "gold.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,100 --> 00:00:02,200\nね\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "cases": [{"id": "sample", "reference": "reference.srt", "candidate": "candidate.srt"}],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "eval-manifest",
                    "--json",
                    "--max-practical-cer",
                    "0.60",
                    "--min-time-aligned-500ms-ratio",
                    "1.0",
                    str(manifest),
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output)["summary"]["text_practical"]["edit_distance"], 1)
            self.assertEqual(error, "")

    def test_eval_manifest_reference_type_gate_passes_for_human_reviewed_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            manifest = root / "gold.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "reference_type": "human-reviewed",
                        "cases": [{"id": "sample", "reference": "reference.srt", "candidate": "candidate.srt"}],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                ["eval-manifest", "--json", "--require-reference-type", "human-reviewed", str(manifest)]
            )

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output)["cases"][0]["reference_type"], "human-reviewed")
            self.assertEqual(error, "")

    def test_eval_manifest_reference_type_gate_fails_after_emitting_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            manifest = root / "gold.json"
            report_path = root / "report.json"
            reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            candidate.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "reference_type": "pseudo-gold",
                        "cases": [{"id": "sample", "reference": "reference.srt", "candidate": "candidate.srt"}],
                    }
                ),
                encoding="utf-8",
            )

            result, output, error = run_cli_with_stderr(
                [
                    "eval-manifest",
                    "--json",
                    "-o",
                    str(report_path),
                    "--require-reference-type",
                    "human-reviewed",
                    str(manifest),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output)["cases"][0]["reference_type"], "pseudo-gold")
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["reference_type"], "pseudo-gold")
            self.assertIn("reference type gate failed", error)
            self.assertIn("sample reference_type 'pseudo-gold' != 'human-reviewed'", error)

    def test_transcribe_and_retranscribe_project_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "voice.wav"
            project_root = root / "projects"
            write_mono_wav(audio_path)

            _, created_output = run_cli(
                ["project", "create-audio", "--project-root", str(project_root), "--json", str(audio_path)]
            )
            project_id = json.loads(created_output)["project_id"]
            run_cli(["project", "analyze", "--project-root", str(project_root), project_id])

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=1,
                        channel=channel,
                        kind="speech",
                        text="初回",
                    ),
                )

            with mock.patch("custom_asmr_srt_stack.cli.transcribe_audio", side_effect=fake_transcribe):
                result, output = run_cli(
                    [
                        "project",
                        "transcribe",
                        "--project-root",
                        str(project_root),
                        "--adapter",
                        "openai-compatible",
                        "--endpoint-url",
                        "http://localhost:8000/v1",
                        "--model-id",
                        "gemma-4-e4b",
                        "--json",
                        project_id,
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output)["master"]["segments"][0]["text"], "初回")

            def fake_retranscribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=1,
                        channel=channel,
                        kind="speech",
                        text="再",
                    ),
                )

            with mock.patch("custom_asmr_srt_stack.cli.transcribe_audio", side_effect=fake_retranscribe):
                result, output = run_cli(
                    [
                        "project",
                        "retranscribe",
                        "--project-root",
                        str(project_root),
                        "--adapter",
                        "openai-compatible",
                        "--endpoint-url",
                        "http://localhost:8000/v1",
                        "--model-id",
                        "gemma-4-e4b",
                        "--json",
                        project_id,
                        "seg_000001",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output)["master"]["segments"][0]["text"], "再")

    def test_cli_errors_are_visible_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, stdout, stderr = run_cli_with_stderr(
                ["project", "show", "--project-root", str(Path(tmpdir)), "0" * 32]
            )

            self.assertEqual(result, 1)
            self.assertEqual(stdout, "")
            self.assertIn("error: project not found", stderr)
            self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
