import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.evaluation import (
    EVAL_SUITE_FORMAT,
    REVIEW_EFFORT_FORMAT,
    evaluate_manifest,
    evaluate_transcripts,
    levenshtein_distance,
    load_transcript_document,
    normalize_for_cer,
    review_effort_items_report,
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
        self.assertEqual(report["text_japanese_relaxed"]["mode"], "japanese-relaxed")
        self.assertEqual(report["timing"]["paired_segments"], 2)
        self.assertEqual(report["timing"]["boundary_samples"], 4)
        self.assertEqual(report["timing"]["mean_start_error_ms"], 60)
        self.assertEqual(report["timing"]["mean_end_error_ms"], 100)
        self.assertEqual(report["timing"]["mean_boundary_error_ms"], 80)
        self.assertEqual(report["timing"]["max_boundary_error_ms"], 100)
        self.assertEqual(report["timing"]["within_250ms_count"], 4)
        self.assertEqual(report["timing"]["within_250ms_ratio"], 1.0)
        self.assertEqual(report["timing_time_aligned"]["matched_reference_segments"], 2)
        self.assertEqual(report["timing_time_aligned"]["reference_match_ratio"], 1.0)
        self.assertEqual(report["timing_time_aligned"]["mean_boundary_error_ms"], 80)
        self.assertEqual(report["channel"]["paired_segments"], 2)
        self.assertEqual(report["channel"]["comparable_segments"], 2)
        self.assertEqual(report["channel"]["accuracy"], 0.5)
        self.assertEqual(report["channel"]["confusion"]["L"]["L"], 1)
        self.assertEqual(report["channel"]["confusion"]["R"]["L"], 1)
        self.assertEqual(report["channel"]["candidate_mix_segments"], 0)
        self.assertEqual(report["channel_time_aligned"]["accuracy"], 0.5)
        self.assertEqual(report["review"]["candidate_review_count"], 1)
        self.assertEqual(report["review_effort"]["text_edit_segments"], 1)
        self.assertEqual(report["review_effort"]["channel_edit_segments"], 1)
        self.assertEqual(report["review_effort"]["timing_edit_segments"], 0)
        self.assertEqual(report["review_effort"]["segments_needing_edit"], 1)
        self.assertEqual(report["review_effort"]["segments_needing_edit_ratio"], 0.5)
        self.assertEqual(report["review_effort"]["items"][0]["reference_id"], "seg_000002")
        self.assertEqual(report["review_effort"]["items"][0]["candidate_id"], "seg_000002")
        self.assertEqual(report["review_effort"]["items"][0]["reasons"], ["text", "channel"])
        self.assertEqual(report["review_effort"]["items"][0]["reference_text"], "見つかった")
        self.assertEqual(report["review_effort"]["items"][0]["candidate_text"], "見つた")

    def test_time_aligned_timing_ignores_non_overlapping_extra_candidate_segments(self):
        reference = master_with_segments(
            [
                Segment("seg_000001", 1000, 2000, "L", "speech", "あ"),
                Segment("seg_000002", 3000, 4000, "R", "speech", "い"),
            ]
        )
        candidate = master_with_segments(
            [
                Segment("seg_000001", 0, 500, "MIX", "speech", "noise"),
                Segment("seg_000002", 1100, 1900, "L", "speech", "あ"),
                Segment("seg_000003", 3100, 3900, "R", "speech", "い"),
            ]
        )

        report = evaluate_transcripts(reference, candidate)

        self.assertGreater(report["timing"]["mean_boundary_error_ms"], 1000)
        self.assertEqual(report["timing_time_aligned"]["matched_reference_segments"], 2)
        self.assertEqual(report["timing_time_aligned"]["mean_boundary_error_ms"], 100)
        self.assertEqual(report["channel_time_aligned"]["comparable_segments"], 2)
        self.assertEqual(report["channel_time_aligned"]["accuracy"], 1.0)
        self.assertEqual(report["review_effort"]["extra_candidate_segments"], 1)
        self.assertEqual(report["review_effort"]["segments_needing_edit"], 1)
        self.assertEqual(report["review_effort"]["segments_needing_edit_ratio"], 1 / 3)
        self.assertEqual(report["review_effort"]["items"], [
            {
                "reference_id": None,
                "candidate_id": "seg_000001",
                "start_ms": 0,
                "end_ms": 500,
                "reasons": ["extra_candidate"],
                "reference_text": "",
                "candidate_text": "noise",
                "reference_channel": None,
                "candidate_channel": "MIX",
            }
        ])

    def test_review_effort_items_report_missing_reference_segments(self):
        reference = master_with_segments(
            [
                Segment("seg_000001", 1000, 2000, "L", "speech", "あ"),
                Segment("seg_000002", 3000, 4000, "R", "speech", "い"),
            ]
        )
        candidate = master_with_segments([Segment("seg_000001", 1000, 2000, "L", "speech", "あ")])

        report = evaluate_transcripts(reference, candidate)

        self.assertEqual(report["review_effort"]["missing_reference_segments"], 1)
        self.assertEqual(report["review_effort"]["segments_needing_edit"], 1)
        self.assertEqual(report["review_effort"]["items"][0]["reference_id"], "seg_000002")
        self.assertEqual(report["review_effort"]["items"][0]["candidate_id"], None)
        self.assertEqual(report["review_effort"]["items"][0]["reasons"], ["missing_reference"])

    def test_practical_cer_normalizes_width_spacing_and_punctuation(self):
        self.assertEqual(normalize_for_cer("ね、 魔女ちゃん！？", mode="practical"), "ね魔女ちゃん")
        self.assertEqual(normalize_for_cer("ABC１２３", mode="practical"), "ABC123")

    def test_japanese_relaxed_cer_removes_prolonged_sound_marks(self):
        self.assertEqual(normalize_for_cer("おにいちゃーん〜～", mode="practical"), "おにいちゃーん")
        self.assertEqual(normalize_for_cer("おにいちゃーん〜～", mode="japanese-relaxed"), "おにいちゃん")

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
                        "reference_type": "pseudo-gold",
                        "reference_notes": "stable-ts baseline",
                        "cases": [
                            {"id": "front-a", "reference": "refs/a.srt", "candidate": "candidates/a.srt"},
                            {
                                "id": "front-b",
                                "reference": "refs/b.srt",
                                "candidate": "candidates/b.srt",
                                "candidate_id": "qwen-energy",
                                "reference_type": "human-reviewed",
                                "reference_notes": "spot checked",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_manifest(manifest)

        self.assertEqual(report["format"], EVAL_SUITE_FORMAT)
        self.assertEqual(report["reference_type"], "pseudo-gold")
        self.assertEqual(report["reference_notes"], "stable-ts baseline")
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["summary"]["text_japanese_relaxed"]["mode"], "japanese-relaxed")
        self.assertEqual(report["cases"][0]["candidate_id"], "a")
        self.assertEqual(report["cases"][0]["reference_type"], "pseudo-gold")
        self.assertEqual(report["cases"][0]["reference_notes"], "stable-ts baseline")
        self.assertEqual(report["cases"][1]["candidate_id"], "qwen-energy")
        self.assertEqual(report["cases"][1]["reference_type"], "human-reviewed")
        self.assertEqual(report["cases"][1]["reference_notes"], "spot checked")
        self.assertEqual(report["summary"]["text"]["edit_distance"], 1)
        self.assertEqual(report["summary"]["text"]["reference_characters"], 6)
        self.assertAlmostEqual(report["summary"]["text"]["cer"], 1 / 6)
        self.assertEqual(report["summary"]["timing"]["paired_segments"], 2)
        self.assertEqual(report["summary"]["timing"]["boundary_samples"], 4)
        self.assertEqual(report["summary"]["timing"]["mean_start_error_ms"], 150)
        self.assertEqual(report["summary"]["timing"]["mean_boundary_error_ms"], 175)
        self.assertEqual(report["summary"]["timing"]["max_boundary_error_ms"], 200)
        self.assertEqual(report["summary"]["timing"]["within_250ms_ratio"], 1.0)
        self.assertEqual(report["summary"]["timing_time_aligned"]["matched_reference_segments"], 2)
        self.assertEqual(report["summary"]["timing_time_aligned"]["reference_match_ratio"], 1.0)
        self.assertEqual(report["summary"]["timing_time_aligned"]["within_500ms_ratio"], 1.0)
        self.assertEqual(report["summary"]["review_effort"]["text_edit_segments"], 1)
        self.assertEqual(report["summary"]["review_effort"]["segments_needing_edit"], 1)
        self.assertEqual(report["summary"]["review_effort"]["segments_needing_edit_ratio"], 0.5)
        self.assertNotIn("items", report["summary"]["review_effort"])
        self.assertEqual(report["summary"]["channel"]["paired_segments"], 2)
        self.assertEqual(report["summary"]["channel"]["confusion"]["MIX"]["MIX"], 2)
        self.assertEqual(report["summary"]["channel"]["candidate_mix_ratio"], 1.0)
        self.assertEqual(report["summary"]["channel_time_aligned"]["candidate_mix_ratio"], 1.0)

    def test_review_effort_items_report_extracts_manifest_case_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate = root / "candidate.srt"
            manifest = root / "gold.json"
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
            eval_report = evaluate_manifest(manifest)

        review_report = review_effort_items_report(eval_report, source_report="eval-suite.json")

        self.assertEqual(review_report["format"], REVIEW_EFFORT_FORMAT)
        self.assertEqual(review_report["source_report"], "eval-suite.json")
        self.assertEqual(review_report["item_count"], 1)
        self.assertEqual(review_report["reason_counts"], {"text": 1, "channel": 1, "timing": 1})
        item = review_report["items"][0]
        self.assertEqual(item["case_id"], "front-a")
        self.assertEqual(item["case_candidate_id"], "qwen-align")
        self.assertEqual(item["reference_type"], "human-reviewed")
        self.assertEqual(item["reasons"], ["text", "channel", "timing"])
        self.assertEqual(item["duration_ms"], 2000)
        self.assertEqual(item["start_delta_ms"], 900)
        self.assertEqual(item["end_delta_ms"], 1000)

    def test_review_effort_items_report_extracts_single_eval_report(self):
        reference = master_with_segments([Segment("seg_000001", 1000, 2000, "L", "speech", "ねえ")])
        candidate = master_with_segments([Segment("seg_000001", 1900, 3000, "R", "speech", "ね")])
        eval_report = evaluate_transcripts(reference, candidate)

        review_report = review_effort_items_report(eval_report)

        self.assertEqual(review_report["format"], REVIEW_EFFORT_FORMAT)
        self.assertEqual(review_report["item_count"], 1)
        self.assertEqual(review_report["reason_counts"], {"text": 1, "channel": 1, "timing": 1})
        item = review_report["items"][0]
        self.assertNotIn("case_id", item)
        self.assertEqual(item["duration_ms"], 2000)
        self.assertEqual(item["start_delta_ms"], 900)

    def test_evaluate_manifest_aggregates_channel_reports_when_one_case_has_no_comparable_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs = root / "refs"
            candidates = root / "candidates"
            refs.mkdir()
            candidates.mkdir()
            (refs / "l.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n[L] あ\n", encoding="utf-8")
            (candidates / "l.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n[L] あ\n", encoding="utf-8")
            (refs / "mix.srt").write_text("1\n00:00:03,000 --> 00:00:04,000\n[LR] い\n", encoding="utf-8")
            (candidates / "mix.srt").write_text("1\n00:00:03,000 --> 00:00:04,000\n[LR] い\n", encoding="utf-8")
            manifest = root / "gold.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-eval-manifest-v1",
                        "cases": [
                            {"id": "l", "reference": "refs/l.srt", "candidate": "candidates/l.srt"},
                            {"id": "mix", "reference": "refs/mix.srt", "candidate": "candidates/mix.srt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_manifest(manifest)

        self.assertEqual(report["summary"]["channel_time_aligned"]["comparable_segments"], 1)
        self.assertEqual(report["summary"]["channel_time_aligned"]["accuracy"], 1.0)
        self.assertEqual(report["cases"][1]["report"]["channel_time_aligned"]["accuracy"], None)

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
