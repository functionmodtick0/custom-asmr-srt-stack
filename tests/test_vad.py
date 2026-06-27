import unittest

from custom_asmr_srt_stack.vad import parse_vad_intervals


class VadTests(unittest.TestCase):
    def test_parse_vad_intervals_accepts_sorted_non_overlapping_ranges(self):
        intervals = parse_vad_intervals(
            {
                "intervals": [
                    {"start_ms": 100, "end_ms": 500},
                    {"start_ms": 700, "end_ms": 900},
                ]
            },
            duration_ms=1000,
        )

        self.assertEqual(
            intervals,
            (
                {"index": 0, "start_ms": 100, "end_ms": 500},
                {"index": 1, "start_ms": 700, "end_ms": 900},
            ),
        )

    def test_parse_vad_intervals_rejects_overlapping_ranges(self):
        with self.assertRaisesRegex(ValueError, "sorted and non-overlapping"):
            parse_vad_intervals(
                {
                    "intervals": [
                        {"start_ms": 100, "end_ms": 500},
                        {"start_ms": 400, "end_ms": 900},
                    ]
                },
                duration_ms=1000,
            )

    def test_parse_vad_intervals_rejects_ranges_past_duration(self):
        with self.assertRaisesRegex(ValueError, "audio duration"):
            parse_vad_intervals({"intervals": [{"start_ms": 100, "end_ms": 1100}]}, duration_ms=1000)


if __name__ == "__main__":
    unittest.main()
