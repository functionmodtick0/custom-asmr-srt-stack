import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.reference_audit import (
    REFERENCE_AUDIT_FORMAT,
    REFERENCE_AUDIT_SUITE_FORMAT,
    audit_master_reference,
    audit_review_case_references,
    reference_audit_review_effort_report,
)


def master_with_segments(segments, *, duration_ms=40000):
    return MasterDocument(source_language="ja", source_file="voice.wav", duration_ms=duration_ms, segments=tuple(segments))


class ReferenceAuditTests(unittest.TestCase):
    def test_audit_master_reference_reports_overlap_and_segmentation_flags_without_text(self):
        master = master_with_segments(
            [
                Segment("seg_000001", 0, 1000, "L", "speech", "ねえ"),
                Segment("seg_000002", 500, 1500, "L", "speech", "そこ"),
                Segment("seg_000003", 0, 1000, "R", "speech", "ねえ"),
                Segment("seg_000004", 2000, 33000, "MIX", "speech", "長い", needs_review=True),
            ]
        )

        report = audit_master_reference(master, case_id="front-a", reference="references/front-a.master.json")

        self.assertEqual(report["format"], REFERENCE_AUDIT_FORMAT)
        self.assertEqual(report["case_id"], "front-a")
        self.assertEqual(report["reference"], "references/front-a.master.json")
        self.assertEqual(report["speech_segment_count"], 4)
        self.assertEqual(report["review_count"], 1)
        self.assertEqual(report["channel_counts"], {"L": 2, "R": 1, "MIX": 1})
        self.assertEqual(report["overlap_pair_count"], 3)
        self.assertEqual(report["same_channel_overlap_pair_count"], 1)
        self.assertEqual(report["cross_channel_overlap_pair_count"], 2)
        self.assertEqual(report["exact_boundary_overlap_pair_count"], 1)
        self.assertEqual(report["pair_overlap_duration_ms"], 2000)
        self.assertEqual(report["long_segment_count"], 1)
        self.assertEqual(report["max_segment_duration_ms"], 31000)
        self.assertAlmostEqual(report["speech_coverage_ratio"], 32500 / 40000)
        self.assertEqual(
            [flag["type"] for flag in report["flags"]],
            ["review_flag_segments", "same_channel_overlap", "exact_boundary_overlap", "long_segment"],
        )
        self.assertNotIn("text", json.dumps(report, ensure_ascii=False))

    def test_default_reference_audit_ignores_tiny_boundary_overlap(self):
        master = master_with_segments(
            [
                Segment("seg_000001", 0, 1000, "L", "speech", "前"),
                Segment("seg_000002", 980, 2000, "L", "speech", "後"),
            ],
            duration_ms=3000,
        )

        product_report = audit_master_reference(master)
        strict_report = audit_master_reference(master, overlap_min_ms=1)

        self.assertEqual(product_report["thresholds"]["overlap_min_ms"], 100)
        self.assertEqual(product_report["overlap_pair_count"], 0)
        self.assertEqual(product_report["same_channel_overlap_pair_count"], 0)
        self.assertEqual(product_report["flags"], [])
        self.assertEqual(strict_report["thresholds"]["overlap_min_ms"], 1)
        self.assertEqual(strict_report["overlap_pair_count"], 1)
        self.assertEqual(strict_report["same_channel_overlap_pair_count"], 1)
        self.assertEqual(strict_report["flags"], [{"type": "same_channel_overlap", "count": 1}])

    def test_audit_review_case_references_aggregates_prepared_case_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            references = root / "references"
            references.mkdir()
            first = master_with_segments(
                [
                    Segment("seg_000001", 0, 1000, "L", "speech", "あ"),
                    Segment("seg_000002", 500, 1200, "L", "speech", "い"),
                ],
                duration_ms=2000,
            )
            second = master_with_segments(
                [
                    Segment("seg_000001", 0, 1000, "R", "speech", "う", needs_review=True),
                    Segment("seg_000002", 1000, 2000, "MIX", "speech", "え"),
                ],
                duration_ms=3000,
            )
            (references / "first.master.json").write_text(
                json.dumps(first.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            (references / "second.master.json").write_text(
                json.dumps(second.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            case_index = root / "case-index.json"
            case_index.write_text(
                json.dumps(
                    {
                        "format": "custom-asmr-review-case-set-v1",
                        "reference_type": "pseudo-gold",
                        "items": [
                            {"id": "first", "reference": "references/first.master.json"},
                            {"id": "second", "reference": "references/second.master.json"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_review_case_references(case_index)

        self.assertEqual(report["format"], REFERENCE_AUDIT_SUITE_FORMAT)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["summary"]["segment_count"], 4)
        self.assertEqual(report["summary"]["speech_segment_count"], 4)
        self.assertEqual(report["summary"]["review_count"], 1)
        self.assertEqual(report["summary"]["overlap_pair_count"], 1)
        self.assertEqual(report["summary"]["same_channel_overlap_pair_count"], 1)
        self.assertEqual(report["summary"]["channel_counts"], {"L": 2, "R": 1, "MIX": 1})
        self.assertEqual(report["summary"]["flag_type_counts"], {"review_flag_segments": 1, "same_channel_overlap": 1})
        self.assertEqual([case["case_id"] for case in report["cases"]], ["first", "second"])

    def test_reference_audit_review_effort_report_exports_packable_queue_without_text(self):
        master = master_with_segments(
            [
                Segment("seg_000001", 0, 1000, "L", "speech", "ねえ"),
                Segment("seg_000002", 500, 1500, "L", "speech", "そこ"),
                Segment("seg_000003", 0, 1000, "R", "speech", "ねえ"),
                Segment("seg_000004", 2000, 33000, "MIX", "speech", "長い", needs_review=True),
            ]
        )
        audit = {
            "format": REFERENCE_AUDIT_SUITE_FORMAT,
            "case_index": "cases/case-index.json",
            "case_count": 1,
            "summary": {},
            "cases": [
                audit_master_reference(master, case_id="front-a", reference="references/front-a.master.json")
            ],
        }

        report = reference_audit_review_effort_report(
            audit,
            source_report="reference-audit.json",
            source_case_index="cases/case-index.json",
        )

        self.assertEqual(report["format"], "custom-asmr-review-effort-v1")
        self.assertEqual(report["source_report"], "reference-audit.json")
        self.assertEqual(report["source_case_index"], "cases/case-index.json")
        self.assertEqual(
            report["reason_counts"],
            {
                "reference-exact-boundary-overlap": 1,
                "reference-long-segment": 1,
                "reference-needs-review": 1,
                "reference-same-channel-overlap": 1,
            },
        )
        self.assertEqual(report["item_count"], 4)
        self.assertEqual([item["priority_rank"] for item in report["items"]], [1, 2, 3, 4])
        self.assertEqual(report["items"][0]["reasons"], ["reference-needs-review"])
        self.assertEqual(report["items"][1]["reasons"], ["reference-exact-boundary-overlap"])
        self.assertEqual(report["items"][2]["reasons"], ["reference-same-channel-overlap"])
        self.assertEqual(report["items"][3]["reasons"], ["reference-long-segment"])
        self.assertNotIn("ねえ", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
