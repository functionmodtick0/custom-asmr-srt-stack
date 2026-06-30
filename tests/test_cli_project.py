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
from custom_asmr_srt_stack.models import Segment


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
            self.assertEqual(json.loads(output)["changed_segments"], 2)
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
            self.assertEqual(json.loads(output)["diagnostics_output"], str(diagnostics_path))
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["format"], "custom-asmr-channel-diagnostics-v1")
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
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-channel-attribution-sweep-v1")
            self.assertEqual(report["setting_count"], 2)
            self.assertEqual([item["changed_segments"] for item in report["items"]], [2, 0])
            comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["items"][0]["label"], "th3_quietm40.eval-report")
            self.assertEqual(comparison["items"][0]["segments_needing_edit"], 0.0)
            self.assertEqual(comparison["items"][1]["segments_needing_edit"], 2.0)
            self.assertTrue((output_dir / "th3_quietm40" / "candidates" / "front-a.master.json").exists())
            self.assertTrue((output_dir / "index.json").exists())

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
            self.assertEqual(report["reference_type_counts"], {"pseudo-gold": 1})
            self.assertEqual(report["reference_review_count"], 1)
            self.assertEqual(report["reference_review_case_count"], 1)
            self.assertEqual(report["reference_review_clear_case_count"], 0)
            self.assertEqual(report["cases_needing_review"], ["front-a"])
            self.assertEqual(report["items"][0]["reference_segments"], 2)
            self.assertEqual(report["items"][0]["reference_review_count"], 1)
            self.assertEqual(json.loads(status_path.read_text(encoding="utf-8"))["case_count"], 1)

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
            pack_dir = root / "review-pack"
            write_mono_wav(audio_path)
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
                    "-o",
                    str(pack_dir),
                    str(review_path),
                ]
            )

            self.assertEqual(result, 0)
            report = json.loads(output)
            self.assertEqual(report["format"], "custom-asmr-review-pack-v1")
            self.assertEqual(report["clip_count"], 2)
            self.assertEqual([item["priority_rank"] for item in report["items"]], [1, 2])
            self.assertEqual([item["reference_id"] for item in report["items"]], ["seg_000010", "seg_000001"])
            self.assertEqual(report["items"][0]["clip_start_ms"], 0)
            self.assertEqual(report["items"][0]["clip_end_ms"], 2)
            clip_path = pack_dir / report["items"][0]["clip_file"]
            self.assertTrue(clip_path.exists())
            self.assertEqual(analyze_wav(clip_path.read_bytes()).duration_ms, 2)
            index = json.loads((pack_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["items"][0]["priority_score"], 9000.0)
            self.assertEqual(index["items"][0]["reference_text"], "優先")

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
