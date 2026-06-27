import unittest
import sys
import tempfile
from pathlib import Path

from custom_asmr_srt_stack.alignment import apply_alignment_review_flags, merge_alignment_output, run_alignment_command
from custom_asmr_srt_stack.models import MasterDocument, Segment


def sample_master() -> MasterDocument:
    return MasterDocument(
        source_language="ja",
        source_file="voice.wav",
        duration_ms=60_000,
        segments=(
            Segment(
                id="seg_000001",
                start_ms=0,
                end_ms=1000,
                channel="L",
                kind="speech",
                text="ねえ",
            ),
            Segment(
                id="seg_000002",
                start_ms=2000,
                end_ms=3000,
                channel="R",
                kind="speech",
                text="聞こえる？",
            ),
        ),
    )


class AlignmentTests(unittest.TestCase):
    def test_merge_alignment_output_preserves_text_and_channel(self):
        merged = merge_alignment_output(
            sample_master(),
            {
                "segments": [
                    {"id": "seg_000001", "start_ms": 100, "end_ms": 1200},
                    {"id": "seg_000002", "start_ms": 2200, "end_ms": 3100},
                ]
            },
        )

        self.assertEqual(merged.segments[0].start_ms, 100)
        self.assertEqual(merged.segments[0].channel, "L")
        self.assertEqual(merged.segments[0].text, "ねえ")
        self.assertEqual(merged.segments[1].end_ms, 3100)

    def test_merge_alignment_output_fails_on_missing_ids(self):
        with self.assertRaisesRegex(ValueError, "missing ids: seg_000002"):
            merge_alignment_output(
                sample_master(),
                {"segments": [{"id": "seg_000001", "start_ms": 100, "end_ms": 1200}]},
            )

    def test_review_flags_mark_empty_speech_and_long_segments(self):
        master = MasterDocument(
            source_language="ja",
            source_file="voice.wav",
            duration_ms=60_000,
            segments=(
                Segment(
                    id="seg_000001",
                    start_ms=0,
                    end_ms=1000,
                    channel="MIX",
                    kind="speech",
                    text=" ",
                ),
                Segment(
                    id="seg_000002",
                    start_ms=2000,
                    end_ms=40_000,
                    channel="MIX",
                    kind="speech",
                    text="長い",
                ),
            ),
        )

        reviewed = apply_alignment_review_flags(master)

        self.assertTrue(reviewed.segments[0].needs_review)
        self.assertTrue(reviewed.segments[1].needs_review)

    def test_run_alignment_command_merges_stdout_json(self):
        script = (
            "import json,sys;"
            "json.load(sys.stdin);"
            "print(json.dumps({'segments':["
            "{'id':'seg_000001','start_ms':10,'end_ms':1010},"
            "{'id':'seg_000002','start_ms':2010,'end_ms':3010}"
            "]}))"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(b"audio")

            aligned = run_alignment_command(
                sample_master(),
                audio_file=audio_path,
                command=[sys.executable, "-c", script],
            )

        self.assertEqual(aligned.segments[0].start_ms, 10)
        self.assertEqual(aligned.segments[1].end_ms, 3010)


if __name__ == "__main__":
    unittest.main()
