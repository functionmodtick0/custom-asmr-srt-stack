import os
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.vad import aggregate_vad_coverage_reports, parse_vad_intervals, run_vad_command, vad_coverage_report


def make_wav_bytes(duration_ms: int = 100) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(1000)
            wav.writeframes(struct.pack("<h", 0) * duration_ms)
        return Path(tmp.name).read_bytes()


class VadTests(unittest.TestCase):
    def test_parse_vad_intervals_accepts_sorted_non_overlapping_ranges(self):
        intervals = parse_vad_intervals(
            {
                "intervals": [
                    {"start_ms": 100, "end_ms": 500},
                    {"start_ms": 700, "end_ms": 900},
                ]
            },
            duration_ms=1000,
        )

        self.assertEqual(
            intervals,
            (
                {"index": 0, "start_ms": 100, "end_ms": 500},
                {"index": 1, "start_ms": 700, "end_ms": 900},
            ),
        )

    def test_parse_vad_intervals_rejects_overlapping_ranges(self):
        with self.assertRaisesRegex(ValueError, "sorted and non-overlapping"):
            parse_vad_intervals(
                {
                    "intervals": [
                        {"start_ms": 100, "end_ms": 500},
                        {"start_ms": 400, "end_ms": 900},
                    ]
                },
                duration_ms=1000,
            )

    def test_parse_vad_intervals_rejects_ranges_past_duration(self):
        with self.assertRaisesRegex(ValueError, "audio duration"):
            parse_vad_intervals({"intervals": [{"start_ms": 100, "end_ms": 1100}]}, duration_ms=1000)

    def test_vad_coverage_report_uses_reference_union_for_overlapping_speech(self):
        reference = MasterDocument(
            source_language="ja",
            source_file="voice.wav",
            duration_ms=1000,
            segments=(
                Segment("seg_000001", 100, 500, "L", "speech", "左"),
                Segment("seg_000002", 300, 700, "R", "speech", "右"),
            ),
        )

        report = vad_coverage_report(
            reference=reference,
            intervals=(
                {"start_ms": 0, "end_ms": 400},
                {"start_ms": 600, "end_ms": 900},
            ),
            audio_duration_ms=1000,
            source="unit-test",
        )

        self.assertEqual(report["format"], "custom-asmr-vad-coverage-v1")
        self.assertEqual(report["reference_segment_count"], 2)
        self.assertEqual(report["reference_interval_count"], 1)
        self.assertEqual(report["detected_max_interval_ms"], 400)
        self.assertEqual(report["detected_mean_interval_ms"], 350)
        self.assertEqual(report["reference_speech_duration_ms"], 600)
        self.assertEqual(report["detected_speech_duration_ms"], 700)
        self.assertEqual(report["overlap_duration_ms"], 400)
        self.assertEqual(report["missed_reference_duration_ms"], 200)
        self.assertEqual(report["extra_detected_duration_ms"], 300)
        self.assertEqual(
            report["missed_reference_intervals"],
            ({"index": 0, "start_ms": 400, "end_ms": 600, "duration_ms": 200},),
        )
        self.assertEqual(
            report["extra_detected_intervals"],
            (
                {"index": 0, "start_ms": 0, "end_ms": 100, "duration_ms": 100},
                {"index": 1, "start_ms": 700, "end_ms": 900, "duration_ms": 200},
            ),
        )
        self.assertAlmostEqual(report["reference_recall"], 400 / 600)
        self.assertAlmostEqual(report["detected_precision"], 400 / 700)

    def test_aggregate_vad_coverage_reports_uses_duration_weighted_totals(self):
        reports = [
            {
                "format": "custom-asmr-vad-coverage-v1",
                "audio_duration_ms": 100,
                "reference_segment_count": 1,
                "reference_interval_count": 1,
                "detected_interval_count": 1,
                "detected_max_interval_ms": 50,
                "reference_speech_duration_ms": 100,
                "detected_speech_duration_ms": 50,
                "overlap_duration_ms": 50,
                "missed_reference_duration_ms": 50,
                "extra_detected_duration_ms": 0,
            },
            {
                "format": "custom-asmr-vad-coverage-v1",
                "audio_duration_ms": 300,
                "reference_segment_count": 2,
                "reference_interval_count": 2,
                "detected_interval_count": 1,
                "detected_max_interval_ms": 300,
                "reference_speech_duration_ms": 300,
                "detected_speech_duration_ms": 300,
                "overlap_duration_ms": 150,
                "missed_reference_duration_ms": 150,
                "extra_detected_duration_ms": 150,
            },
        ]

        summary = aggregate_vad_coverage_reports(reports)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["reference_speech_duration_ms"], 400)
        self.assertEqual(summary["detected_speech_duration_ms"], 350)
        self.assertEqual(summary["detected_max_interval_ms"], 300)
        self.assertEqual(summary["detected_mean_interval_ms"], 175)
        self.assertEqual(summary["overlap_duration_ms"], 200)
        self.assertAlmostEqual(summary["reference_recall"], 200 / 400)
        self.assertAlmostEqual(summary["detected_precision"], 200 / 350)

    def test_run_vad_command_strips_sensitive_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "vad.py"
            script.write_text(
                "\n".join(
                    [
                        "import json",
                        "import os",
                        "import sys",
                        "request = json.loads(sys.stdin.read())",
                        "assert 'HF_TOKEN' not in os.environ",
                        "assert 'WANDB_API_KEY' not in os.environ",
                        "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''",
                        "assert os.environ['HF_HUB_OFFLINE'] == '1'",
                        "assert os.environ['TMPDIR']",
                        "assert request['audio_info']['duration_ms'] == 100",
                        "print(json.dumps({'intervals': [{'start_ms': 0, 'end_ms': 100}]}))",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"HF_TOKEN": "secret", "WANDB_API_KEY": "secret"}):
                intervals = run_vad_command(make_wav_bytes(), command=[sys.executable, str(script)])

        self.assertEqual(intervals, ({"index": 0, "start_ms": 0, "end_ms": 100},))

    def test_run_vad_command_times_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "slow_vad.py"
            script.write_text("import time\ntime.sleep(1)\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"CASRT_VAD_TIMEOUT_SECONDS": "0.05"}):
                with self.assertRaisesRegex(ValueError, "timed out"):
                    run_vad_command(make_wav_bytes(), command=[sys.executable, str(script)])


if __name__ == "__main__":
    unittest.main()
