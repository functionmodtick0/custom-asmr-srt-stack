import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.pipeline_readiness import reference_stage, vad_stage


def write_vad_coverage(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "format": "custom-asmr-vad-coverage-comparison-v1",
                "quality_gate": {"min_reference_recall": 0.95},
                "items": [
                    {
                        "label": "tuned-vad",
                        "gate_passed": True,
                        "gate_failures": [],
                        "missed_reference_duration_ms": 10,
                        "extra_detected_duration_ms": 20,
                        "reference_recall": 0.99,
                        "detected_precision": 0.98,
                        "detected_max_interval_ms": 30000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def write_eval_report(
    path: Path,
    *,
    practical_cer: float,
    timing_500ms: float,
    channel_aware_cer: float = 0.2,
    channel_aware_timing_500ms: float = 0.8,
    channel_accuracy: float = 0.9,
    mix_ratio: float = 0.1,
    review_ratio: float = 0.1,
) -> None:
    path.write_text(
        json.dumps(
            {
                "format": "custom-asmr-eval-v1",
                "text_practical": {"cer": practical_cer},
                "text_practical_channel_aware": {"cer": channel_aware_cer},
                "timing_time_aligned": {"within_500ms_ratio": timing_500ms},
                "timing_time_aligned_channel_aware": {"within_500ms_ratio": channel_aware_timing_500ms},
                "channel_time_aligned": {"accuracy": channel_accuracy, "candidate_mix_ratio": mix_ratio},
                "review": {"candidate_review_ratio": review_ratio},
                "review_effort": {
                    "segments_needing_edit": review_ratio * 10,
                    "segments_needing_edit_ratio": review_ratio,
                    "text_edit_segment_ratio": review_ratio,
                    "channel_edit_segment_ratio": review_ratio,
                    "timing_edit_segment_ratio": review_ratio,
                    "missing_reference_segment_ratio": 0.0,
                    "extra_candidate_segment_ratio": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )


def write_eval_suite_report(path: Path, *, case_id: str) -> None:
    write_eval_report(path, practical_cer=0.2, timing_500ms=0.8)
    single = json.loads(path.read_text(encoding="utf-8"))
    single.pop("format")
    path.write_text(
        json.dumps(
            {
                "format": "custom-asmr-eval-suite-v1",
                "case_count": 1,
                "reference_type": "pseudo-gold",
                "cases": [{"id": case_id}],
                "summary": single,
            }
        ),
        encoding="utf-8",
    )


class PipelineReadinessTests(unittest.TestCase):
    def test_reference_stage_blocks_explicit_content_unreviewed_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_audit = Path(tmpdir) / "reference-audit.json"
            reference_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-audit-suite-v1",
                        "summary": {
                            "segment_count": 2,
                            "review_count": 0,
                            "content_unreviewed_count": 1,
                            "same_channel_overlap_pair_count": 0,
                            "exact_boundary_overlap_pair_count": 0,
                            "exact_boundary_same_channel_overlap_pair_count": 0,
                            "exact_boundary_cross_channel_overlap_pair_count": 0,
                            "long_segment_count": 0,
                            "speech_coverage_ratio": 0.5,
                            "flag_type_counts": {"content_unreviewed_segments": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )

            stage = reference_stage(reference_audit)

        self.assertEqual(stage["status"], "fail")
        self.assertIn("reference content review evidence missing: 1", stage["reasons"])
        self.assertEqual(stage["metrics"]["content_unreviewed_count"], 1)

    def test_vad_stage_requires_downstream_reports_for_product_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coverage = Path(tmpdir) / "coverage.json"
            write_vad_coverage(coverage)

            stage = vad_stage(coverage, require_downstream=True)

        self.assertEqual(stage["status"], "fail")
        self.assertIn("downstream ASR validation", stage["reasons"][0])
        self.assertEqual(stage["metrics"]["chosen_label"], "tuned-vad")

    def test_vad_stage_rejects_downstream_metric_regressions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coverage = root / "coverage.json"
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            write_vad_coverage(coverage)
            write_eval_report(baseline, practical_cer=0.20, timing_500ms=0.80)
            write_eval_report(candidate, practical_cer=0.21, timing_500ms=0.75)

            stage = vad_stage(
                coverage,
                baseline_eval_file=baseline,
                candidate_eval_file=candidate,
                require_downstream=True,
            )

        self.assertEqual(stage["status"], "fail")
        self.assertIn("practical CER regressed", stage["reasons"][0])
        self.assertIn("time-aligned 500ms ratio regressed", stage["reasons"][1])
        self.assertFalse(stage["metrics"]["downstream_validation"]["passed"])

    def test_vad_stage_passes_coverage_and_no_regression_downstream(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coverage = root / "coverage.json"
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            write_vad_coverage(coverage)
            write_eval_report(baseline, practical_cer=0.20, timing_500ms=0.80)
            write_eval_report(candidate, practical_cer=0.19, timing_500ms=0.82)

            stage = vad_stage(
                coverage,
                baseline_eval_file=baseline,
                candidate_eval_file=candidate,
                require_downstream=True,
            )

        self.assertEqual(stage["status"], "pass")
        self.assertEqual(stage["reasons"], [])
        self.assertTrue(stage["metrics"]["downstream_validation"]["passed"])

    def test_vad_stage_requires_baseline_and_candidate_as_a_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coverage = root / "coverage.json"
            baseline = root / "baseline.json"
            write_vad_coverage(coverage)
            write_eval_report(baseline, practical_cer=0.20, timing_500ms=0.80)

            with self.assertRaisesRegex(ValueError, "must be provided together"):
                vad_stage(coverage, baseline_eval_file=baseline)

    def test_vad_stage_rejects_mismatched_downstream_case_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coverage = root / "coverage.json"
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            write_vad_coverage(coverage)
            write_eval_suite_report(baseline, case_id="front-a")
            write_eval_suite_report(candidate, case_id="front-b")

            with self.assertRaisesRegex(ValueError, "same case ids"):
                vad_stage(
                    coverage,
                    baseline_eval_file=baseline,
                    candidate_eval_file=candidate,
                    require_downstream=True,
                )

    def test_reference_stage_uses_unresolved_channel_audit_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_audit = root / "reference-audit.json"
            channel_audit = root / "channel-audit.json"
            reference_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-audit-suite-v1",
                        "summary": {
                            "segment_count": 1,
                            "review_count": 0,
                            "same_channel_overlap_pair_count": 0,
                            "exact_boundary_overlap_pair_count": 0,
                            "exact_boundary_same_channel_overlap_pair_count": 0,
                            "exact_boundary_cross_channel_overlap_pair_count": 0,
                            "long_segment_count": 0,
                            "speech_coverage_ratio": 1.0,
                            "flag_type_counts": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            channel_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-channel-audit-suite-v1",
                        "summary": {
                            "eligible_reference_channel_count": 1,
                            "energy_labeled_count": 1,
                            "energy_uncertain_count": 0,
                            "match_count": 0,
                            "mismatch_count": 1,
                            "channel_reviewed_count": 1,
                            "reviewed_exception_count": 1,
                            "unresolved_mismatch_count": 0,
                            "unresolved_uncertain_count": 0,
                            "match_ratio": 0.0,
                            "energy_labeled_ratio": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            stage = reference_stage(reference_audit, reference_channel_audit_file=channel_audit)

        self.assertEqual(stage["status"], "pass")
        self.assertEqual(stage["reasons"], [])
        self.assertEqual(stage["metrics"]["channel_audit"]["mismatch_count"], 1)
        self.assertEqual(stage["metrics"]["channel_audit"]["unresolved_mismatch_count"], 0)

    def test_reference_stage_treats_legacy_raw_counts_as_unresolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_audit = root / "reference-audit.json"
            channel_audit = root / "channel-audit.json"
            reference_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-audit-suite-v1",
                        "summary": {
                            "segment_count": 1,
                            "review_count": 0,
                            "same_channel_overlap_pair_count": 0,
                            "exact_boundary_overlap_pair_count": 0,
                            "long_segment_count": 0,
                            "speech_coverage_ratio": 1.0,
                            "flag_type_counts": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            channel_audit.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-reference-channel-audit-suite-v1",
                        "summary": {
                            "eligible_reference_channel_count": 1,
                            "energy_labeled_count": 1,
                            "energy_uncertain_count": 0,
                            "match_count": 0,
                            "mismatch_count": 1,
                            "match_ratio": 0.0,
                            "energy_labeled_ratio": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            stage = reference_stage(reference_audit, reference_channel_audit_file=channel_audit)

        self.assertEqual(stage["status"], "fail")
        self.assertEqual(
            stage["reasons"],
            ["unreviewed reference channel labels conflict with energy: 1"],
        )


if __name__ == "__main__":
    unittest.main()
