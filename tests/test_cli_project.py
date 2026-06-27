import contextlib
import io
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

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
            self.assertTrue(report_path.exists())

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
