import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.pipeline_readiness import reference_stage


class PipelineReadinessTests(unittest.TestCase):
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
