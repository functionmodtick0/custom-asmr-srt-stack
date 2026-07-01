import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.evaluation import (
    EVAL_FORMAT,
    EVAL_SUITE_FORMAT,
    REVIEW_EFFORT_FORMAT,
    compare_eval_reports,
    compare_review_effort_reports,
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
        self.assertEqual(report["review_effort"]["text_edit_segment_ratio"], 0.5)
        self.assertEqual(report["review_effort"]["channel_edit_segment_ratio"], 0.5)
        self.assertEqual(report["review_effort"]["timing_edit_segment_ratio"], 0.0)
        self.assertEqual(report["review_effort"]["missing_reference_segment_ratio"], 0.0)
        self.assertEqual(report["review_effort"]["extra_candidate_segment_ratio"], 0.0)
        self.assertEqual(report["review_effort"]["items"][0]["reference_id"], "seg_000002")
        self.assertEqual(report["review_effort"]["items"][0]["candidate_id"], "seg_000002")
        self.assertEqual(report["review_effort"]["items"][0]["reasons"], ["text", "channel"])
        self.assertEqual(report["review_effort"]["items"][0]["reference_text"], "見つかった")
        self.assertEqual(report["review_effort"]["items"][0]["candidate_text"], "見つた")
        self.assertEqual(report["asr_artifacts"]["artifact_segments"], 0)
        self.assertEqual(report["asr_artifacts"]["artifact_segment_ratio"], 0.0)

    def test_evaluate_transcripts_reports_asr_artifact_diagnostics(self):
        reference = master_with_segments(
            [
                Segment("seg_000001", 0, 1000, "MIX", "speech", "ねえ"),
                Segment("seg_000002", 1000, 2000, "MIX", "speech", "そこ"),
                Segment("seg_000003", 2000, 3000, "MIX", "speech", "聞こえる"),
                Segment("seg_000004", 3000, 4000, "MIX", "speech", "はい"),
            ]
        )
        candidate = master_with_segments(
            [
                Segment("seg_000001", 0, 1000, "MIX", "speech", "noise"),
                Segment("seg_000002", 1000, 1400, "MIX", "speech", "あいうえおかきくけこ"),
                Segment("seg_000003", 2000, 4500, "MIX", "speech", "おにいちゃんおにいちゃんおにいちゃん"),
                Segment("seg_000004", 3000, 4000, "MIX", "speech", "はい"),
            ]
        )

        report = evaluate_transcripts(reference, candidate)

        artifacts = report["asr_artifacts"]
        self.assertEqual(artifacts["candidate_segments"], 4)
        self.assertEqual(artifacts["artifact_segments"], 3)
        self.assertEqual(artifacts["non_japanese_text_segments"], 1)
        self.assertEqual(artifacts["high_text_density_segments"], 1)
        self.assertEqual(artifacts["repeated_text_segments"], 1)
        self.assertEqual(artifacts["artifact_segment_ratio"], 0.75)
        self.assertEqual([item["segment_id"] for item in artifacts["items"]], [
            "seg_000001",
            "seg_000002",
            "seg_000003",
        ])
        self.assertEqual(artifacts["items"][0]["reasons"], ["non_japanese_text"])
        self.assertEqual(artifacts["items"][1]["reasons"], ["high_text_density"])
        self.assertEqual(artifacts["items"][2]["reasons"], ["repeated_text"])

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
        self.assertEqual(report["review_effort"]["extra_candidate_segment_ratio"], 1 / 3)
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
        self.assertEqual(report["summary"]["review_effort"]["text_edit_segment_ratio"], 0.5)
        self.assertEqual(report["summary"]["review_effort"]["channel_edit_segment_ratio"], 0.0)
        self.assertEqual(report["summary"]["review_effort"]["timing_edit_segment_ratio"], 0.0)
        self.assertEqual(report["summary"]["review_effort"]["missing_reference_segment_ratio"], 0.0)
        self.assertEqual(report["summary"]["review_effort"]["extra_candidate_segment_ratio"], 0.0)
        self.assertNotIn("items", report["summary"]["review_effort"])
        self.assertEqual(report["summary"]["asr_artifacts"]["candidate_segments"], 2)
        self.assertEqual(report["summary"]["asr_artifacts"]["artifact_segments"], 0)
        self.assertEqual(report["summary"]["asr_artifacts"]["artifact_segment_ratio"], 0.0)
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
        self.assertEqual(review_report["sort"], "priority_score_desc")
        self.assertEqual(review_report["item_count"], 1)
        self.assertEqual(review_report["reason_counts"], {"text": 1, "channel": 1, "timing": 1})
        item = review_report["items"][0]
        self.assertEqual(item["priority_rank"], 1)
        self.assertGreater(item["priority_score"], 0)
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
        self.assertEqual(item["priority_rank"], 1)
        self.assertNotIn("case_id", item)
        self.assertEqual(item["duration_ms"], 2000)
        self.assertEqual(item["start_delta_ms"], 900)

    def test_review_effort_items_report_sorts_by_review_priority(self):
        eval_report = {
            "format": EVAL_FORMAT,
            "review_effort": {
                "items": [
                    {
                        "reference_id": "seg_channel",
                        "candidate_id": "seg_channel",
                        "start_ms": 3000,
                        "end_ms": 4000,
                        "reasons": ["channel"],
                        "reference_text": "あ",
                        "candidate_text": "あ",
                        "reference_channel": "L",
                        "candidate_channel": "R",
                    },
                    {
                        "reference_id": "seg_text",
                        "candidate_id": "seg_text",
                        "start_ms": 1000,
                        "end_ms": 2000,
                        "reasons": ["text"],
                        "reference_text": "見つかった",
                        "candidate_text": "見つた",
                        "reference_channel": "L",
                        "candidate_channel": "L",
                    },
                    {
                        "reference_id": "seg_missing",
                        "candidate_id": None,
                        "start_ms": 2000,
                        "end_ms": 3000,
                        "reasons": ["missing_reference"],
                        "reference_text": "ねえ",
                        "candidate_text": "",
                        "reference_channel": "R",
                        "candidate_channel": None,
                    },
                ]
            },
        }

        review_report = review_effort_items_report(eval_report)

        items = review_report["items"]
        self.assertEqual([item["reference_id"] for item in items], ["seg_missing", "seg_text", "seg_channel"])
        self.assertEqual([item["priority_rank"] for item in items], [1, 2, 3])
        self.assertGreater(items[0]["priority_score"], items[1]["priority_score"])
        self.assertGreater(items[1]["priority_score"], items[2]["priority_score"])

    def test_compare_eval_reports_computes_breakdown_ratios_for_legacy_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "legacy-report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "format": EVAL_FORMAT,
                        "text_practical": {"cer": 0.2},
                        "timing_time_aligned": {"within_500ms_ratio": 0.5},
                        "channel_time_aligned": {"accuracy": 0.25, "candidate_mix_ratio": 0.75},
                        "review_effort": {
                            "reference_segments": 4,
                            "extra_candidate_segments": 1,
                            "text_edit_segments": 2,
                            "channel_edit_segments": 1,
                            "timing_edit_segments": 3,
                            "missing_reference_segments": 1,
                            "segments_needing_edit": 4,
                            "segments_needing_edit_ratio": 0.8,
                        },
                    }
                ),
                encoding="utf-8",
            )

            comparison = compare_eval_reports([report_path])

        item = comparison["items"][0]
        self.assertEqual(item["text_edit_segment_ratio"], 2 / 5)
        self.assertEqual(item["channel_edit_segment_ratio"], 1 / 5)
        self.assertEqual(item["timing_edit_segment_ratio"], 3 / 5)
        self.assertEqual(item["missing_reference_segment_ratio"], 1 / 5)
        self.assertEqual(item["extra_candidate_segment_ratio"], 1 / 5)
        self.assertIsNone(item["asr_artifact_segment_ratio"])

    def test_compare_review_effort_reports_groups_candidate_failures_by_reference_segment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.srt"
            candidate_a = root / "candidate-a.srt"
            candidate_b = root / "candidate-b.srt"
            manifest_a = root / "manifest-a.json"
            manifest_b = root / "manifest-b.json"
            report_a = root / "qwen-report.json"
            report_b = root / "neosophie-report.json"
            reference.write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:01,000 --> 00:00:02,000",
                        "[L] ねえ",
                        "",
                        "2",
                        "00:00:03,000 --> 00:00:04,000",
                        "[R] 見つかった",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            candidate_a.write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:01,000 --> 00:00:02,000",
                        "[L] ねえ",
                        "",
                        "2",
                        "00:00:03,000 --> 00:00:04,000",
                        "[R] 見つた",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            candidate_b.write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:01,900 --> 00:00:03,000",
                        "[R] ねえ",
                        "",
                        "2",
                        "00:00:03,000 --> 00:00:04,000",
                        "[R] 見つかった",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            for manifest, candidate, candidate_id in (
                (manifest_a, candidate_a, "qwen"),
                (manifest_b, candidate_b, "neosophie"),
            ):
                manifest.write_text(
                    json.dumps(
                        {
                            "format": "custom-asmr-eval-manifest-v1",
                            "reference_type": "pseudo-gold",
                            "cases": [
                                {
                                    "id": "front-a",
                                    "reference": reference.name,
                                    "candidate": candidate.name,
                                    "candidate_id": candidate_id,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            report_a.write_text(json.dumps(evaluate_manifest(manifest_a)), encoding="utf-8")
            report_b.write_text(json.dumps(evaluate_manifest(manifest_b)), encoding="utf-8")

            comparison = compare_review_effort_reports([report_a, report_b])

        self.assertEqual(comparison["format"], "custom-asmr-review-effort-comparison-v1")
        self.assertEqual(comparison["report_count"], 2)
        self.assertEqual(comparison["reference_issue_count"], 2)
        self.assertEqual(comparison["summary"]["reference_segments_with_any_pass"], 2)
        self.assertEqual(comparison["summary"]["reference_segments_failed_by_all"], 0)
        by_reference = {item["reference_id"]: item for item in comparison["items"]}
        first = by_reference["seg_000001"]
        self.assertEqual(first["case_id"], "front-a")
        self.assertEqual(first["failed_candidate_count"], 1)
        self.assertEqual(first["passed_candidate_count"], 1)
        self.assertEqual(first["reason_counts"], {"channel": 1, "timing": 1})
        self.assertEqual(
            [(candidate["label"], candidate["passed"], candidate["reasons"]) for candidate in first["candidates"]],
            [
                ("qwen-report", True, []),
                ("neosophie-report", False, ["channel", "timing"]),
            ],
        )
        second = by_reference["seg_000002"]
        self.assertEqual(second["reason_counts"], {"text": 1})
        self.assertEqual(
            [(candidate["label"], candidate["passed"], candidate["reasons"]) for candidate in second["candidates"]],
            [
                ("qwen-report", False, ["text"]),
                ("neosophie-report", True, []),
            ],
        )

    def test_compare_review_effort_reports_keeps_extra_candidate_items_separate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "candidate-report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "format": EVAL_FORMAT,
                        "review_effort": {
                            "items": [
                                {
                                    "reference_id": None,
                                    "candidate_id": "seg_extra",
                                    "start_ms": 0,
                                    "end_ms": 500,
                                    "reasons": ["extra_candidate"],
                                    "reference_text": "",
                                    "candidate_text": "noise",
                                    "reference_channel": None,
                                    "candidate_channel": "MIX",
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            comparison = compare_review_effort_reports([report_path])

        self.assertEqual(comparison["reference_issue_count"], 0)
        self.assertEqual(comparison["extra_candidate_issue_count"], 1)
        self.assertEqual(comparison["extra_candidate_items"][0]["label"], "candidate-report")
        self.assertEqual(comparison["extra_candidate_items"][0]["candidate_segment_id"], "seg_extra")

    def test_compare_review_effort_reports_disambiguates_duplicate_report_stems(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a" / "report.json"
            second = root / "b" / "report.json"
            first.parent.mkdir()
            second.parent.mkdir()
            report = {
                "format": EVAL_FORMAT,
                "review_effort": {
                    "items": [
                        {
                            "reference_id": "seg_000001",
                            "candidate_id": "seg_000001",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "reasons": ["text"],
                            "reference_text": "あ",
                            "candidate_text": "い",
                            "reference_channel": "L",
                            "candidate_channel": "L",
                        }
                    ]
                },
            }
            first.write_text(json.dumps(report), encoding="utf-8")
            second.write_text(json.dumps(report), encoding="utf-8")

            comparison = compare_review_effort_reports([first, second])

        self.assertEqual([candidate["label"] for candidate in comparison["candidates"]], ["report", "report#2"])
        self.assertEqual([candidate["label"] for candidate in comparison["items"][0]["candidates"]], [
            "report",
            "report#2",
        ])

    def test_compare_review_effort_reports_rejects_different_case_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(
                json.dumps(
                    {
                        "format": EVAL_SUITE_FORMAT,
                        "cases": [{"id": "front-a", "candidate_id": "a", "report": {"review_effort": {"items": []}}}],
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "format": EVAL_SUITE_FORMAT,
                        "cases": [{"id": "front-b", "candidate_id": "b", "report": {"review_effort": {"items": []}}}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "same case ids"):
                compare_review_effort_reports([first, second])

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
