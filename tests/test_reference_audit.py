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


if __name__ == "__main__":
    unittest.main()
