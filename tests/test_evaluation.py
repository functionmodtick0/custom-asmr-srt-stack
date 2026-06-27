import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.evaluation import (
    EVAL_SUITE_FORMAT,
    evaluate_manifest,
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

    def test_evaluate_manifest_resolves_relative_paths_and_aggregates_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs = root / "refs"
            candidates = root / "candidates"
            refs.mkdir()
            candidates.mkdir()
            (refs / "a.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nあい\n", encoding="utf-8")
            (candidates / "a.srt").write_text("1\n00:00:01,100 --> 00:00:02,200\nあ\n", encoding="utf-8")
            (refs / "b.srt").write_text("1\n00:00:03,000 --> 00:00:04,000\nかきくけ\n", encoding="utf-8")
            (candidates / "b.srt").write_text("1\n00:00:03,200 --> 00:00:04,200\nかきくけ\n", encoding="utf-8")
            manifest = root / "gold.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "cases": [
                            {"id": "front-a", "reference": "refs/a.srt", "candidate": "candidates/a.srt"},
                            {
                                "id": "front-b",
                                "reference": "refs/b.srt",
                                "candidate": "candidates/b.srt",
                                "candidate_id": "qwen-energy",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_manifest(manifest)

        self.assertEqual(report["format"], EVAL_SUITE_FORMAT)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["cases"][0]["candidate_id"], "a")
        self.assertEqual(report["cases"][1]["candidate_id"], "qwen-energy")
        self.assertEqual(report["summary"]["text"]["edit_distance"], 1)
        self.assertEqual(report["summary"]["text"]["reference_characters"], 6)
        self.assertAlmostEqual(report["summary"]["text"]["cer"], 1 / 6)
        self.assertEqual(report["summary"]["timing"]["paired_segments"], 2)
        self.assertEqual(report["summary"]["timing"]["mean_start_error_ms"], 150)
        self.assertEqual(report["summary"]["timing"]["mean_boundary_error_ms"], 175)

    def test_evaluate_manifest_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "gold.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "cases": [
                            {"id": "dup", "reference": "a.srt", "candidate": "a.srt"},
                            {"id": "dup", "reference": "b.srt", "candidate": "b.srt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicated"):
                evaluate_manifest(manifest)

    def test_load_transcript_document_accepts_srt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ref.srt"
            path.write_text("1\n00:00:01,000 --> 00:00:02,000\nねえ\n", encoding="utf-8")

            master = load_transcript_document(path)

        self.assertEqual(master.segments[0].text, "ねえ")
        self.assertEqual(master.segments[0].start_ms, 1000)


if __name__ == "__main__":
    unittest.main()
