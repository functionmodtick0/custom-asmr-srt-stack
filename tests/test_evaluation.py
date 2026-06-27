import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.evaluation import (
    evaluate_transcripts,
    levenshtein_distance,
    load_transcript_document,
    normalize_for_cer,
)
from custom_asmr_srt_stack.models import MasterDocument, Segment


def master_with_segments(segments):
    return MasterDocument(source_language="ja", source_file="voice.wav", duration_ms=5000, segments=tuple(segments))


class EvaluationTests(unittest.TestCase):
    def test_levenshtein_distance_counts_character_edits(self):
        self.assertEqual(levenshtein_distance("ねえ", "ね"), 1)
        self.assertEqual(levenshtein_distance("abc", "adc"), 1)

    def test_evaluate_transcripts_reports_text_timing_channel_and_review_metrics(self):
        reference = master_with_segments(
            [
                Segment("seg_000001", 100, 1000, "L", "speech", "ねえ"),
                Segment("seg_000002", 1200, 2000, "R", "speech", "見つかった"),
            ]
        )
        candidate = master_with_segments(
            [
                Segment("seg_000001", 80, 900, "L", "speech", "ねえ"),
                Segment("seg_000002", 1300, 2100, "L", "speech", "見つた", needs_review=True),
            ]
        )

        report = evaluate_transcripts(reference, candidate)

        self.assertEqual(report["format"], "custom-asmr-eval-v1")
        self.assertEqual(report["text"]["edit_distance"], 2)
        self.assertAlmostEqual(report["text"]["cer"], 2 / 7)
        self.assertEqual(report["text_practical"]["mode"], "practical")
        self.assertEqual(report["timing"]["paired_segments"], 2)
        self.assertEqual(report["timing"]["mean_start_error_ms"], 60)
        self.assertEqual(report["timing"]["mean_end_error_ms"], 100)
        self.assertEqual(report["channel"]["comparable_segments"], 2)
        self.assertEqual(report["channel"]["accuracy"], 0.5)
        self.assertEqual(report["review"]["candidate_review_count"], 1)

    def test_practical_cer_normalizes_width_spacing_and_punctuation(self):
        self.assertEqual(normalize_for_cer("ね、 魔女ちゃん！？", mode="practical"), "ね魔女ちゃん")
        self.assertEqual(normalize_for_cer("ABC１２３", mode="practical"), "ABC123")

    def test_load_transcript_document_accepts_srt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ref.srt"
            path.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")

            master = load_transcript_document(path)

        self.assertEqual(master.segments[0].text, "ねえ")
        self.assertEqual(master.segments[0].start_ms, 1000)


if __name__ == "__main__":
    unittest.main()
