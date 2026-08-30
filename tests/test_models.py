import unittest

from custom_asmr_srt_stack.models import MasterDocument, Segment, make_segment_id


class SegmentModelTests(unittest.TestCase):
    def test_segment_rejects_invalid_timing(self):
        with self.assertRaisesRegex(ValueError, "end_ms must be greater"):
            Segment(
                id="seg_000001",
                start_ms=1000,
                end_ms=1000,
                channel="MIX",
                kind="speech",
                text="こんにちは",
            )

    def test_segment_rejects_channel_labels_as_channel_values(self):
        with self.assertRaisesRegex(ValueError, "unsupported channel"):
            Segment(
                id="seg_000001",
                start_ms=0,
                end_ms=1000,
                channel="[L]",
                kind="speech",
                text="こんにちは",
            )

    def test_master_rejects_duplicate_ids(self):
        segment = Segment(
            id="seg_000001",
            start_ms=0,
            end_ms=1000,
            channel="MIX",
            kind="speech",
            text="こんにちは",
        )
        with self.assertRaisesRegex(ValueError, "duplicate segment id"):
            MasterDocument(
                source_language="ja",
                source_file="audio.wav",
                duration_ms=2000,
                segments=(segment, segment),
            )

    def test_master_round_trips_json_contract(self):
        master = MasterDocument(
            source_language="ja",
            source_file="audio.wav",
            duration_ms=2000,
            segments=(
                Segment(
                    id=make_segment_id(1),
                    start_ms=100,
                    end_ms=900,
                    channel="L",
                    kind="speech",
                    text="ねえ、聞こえてる？",
                    channel_reviewed=True,
                ),
            ),
        )

        parsed = MasterDocument.from_json(master.to_json())

        self.assertEqual(parsed, master)
        self.assertEqual(parsed.to_json()["segments"][0]["channel"], "L")
        self.assertEqual(parsed.to_json()["segments"][0]["text"], "ねえ、聞こえてる？")
        self.assertTrue(parsed.to_json()["segments"][0]["channel_reviewed"])

    def test_segment_rejects_non_boolean_review_fields(self):
        segment = {
            "id": "seg_000001",
            "start_ms": 0,
            "end_ms": 1000,
            "channel": "L",
            "kind": "speech",
            "text": "ねえ",
            "needs_review": "false",
        }

        with self.assertRaisesRegex(ValueError, "segment.needs_review must be a boolean"):
            Segment.from_json(segment)

        segment["needs_review"] = False
        segment["channel_reviewed"] = 1
        with self.assertRaisesRegex(ValueError, "segment.channel_reviewed must be a boolean"):
            Segment.from_json(segment)


if __name__ == "__main__":
    unittest.main()
