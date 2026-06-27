import os
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from custom_asmr_srt_stack.vad import parse_vad_intervals, run_vad_command


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
