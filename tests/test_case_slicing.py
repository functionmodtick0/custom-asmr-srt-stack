import unittest

from custom_asmr_srt_stack.case_slicing import slice_master_document
from custom_asmr_srt_stack.models import MasterDocument, Segment


class CaseSlicingTests(unittest.TestCase):
    def test_clipping_invalidates_channel_review_but_full_segment_preserves_it(self):
        master = MasterDocument(
            source_language="ja",
            source_file="voice.wav",
            duration_ms=3000,
            segments=(
                Segment("seg_000001", 0, 1000, "L", "speech", "前", channel_reviewed=True),
                Segment("seg_000002", 1000, 2000, "R", "speech", "後", channel_reviewed=True),
            ),
        )

        sliced = slice_master_document(master, start_ms=500, end_ms=2000)

        self.assertFalse(sliced.segments[0].channel_reviewed)
        self.assertTrue(sliced.segments[0].needs_review)
        self.assertTrue(sliced.segments[1].channel_reviewed)
        self.assertFalse(sliced.segments[1].needs_review)


if __name__ == "__main__":
    unittest.main()
