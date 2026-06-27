import unittest

from custom_asmr_srt_stack.whisper_vad_onnx import activate_outputs, probabilities_to_intervals


class WhisperVadOnnxTests(unittest.TestCase):
    def test_probabilities_to_intervals_uses_hysteresis_and_padding(self):
        intervals = probabilities_to_intervals(
            [0.1, 0.6, 0.4, 0.34, 0.34, 0.1, 0.1],
            duration_ms=140,
            frame_ms=20,
            threshold=0.5,
            min_speech_ms=20,
            min_silence_ms=40,
            pad_ms=10,
        )

        self.assertEqual(intervals, ({"index": 0, "start_ms": 10, "end_ms": 70},))

    def test_probabilities_to_intervals_filters_short_speech(self):
        intervals = probabilities_to_intervals(
            [0.1, 0.6, 0.1, 0.1],
            duration_ms=80,
            frame_ms=20,
            threshold=0.5,
            min_speech_ms=60,
            min_silence_ms=20,
            pad_ms=0,
        )

        self.assertEqual(intervals, ())

    def test_probabilities_to_intervals_merges_padding_overlap(self):
        intervals = probabilities_to_intervals(
            [0.6, 0.6, 0.1, 0.1, 0.6, 0.6, 0.1],
            duration_ms=140,
            frame_ms=20,
            threshold=0.5,
            min_speech_ms=20,
            min_silence_ms=40,
            pad_ms=30,
        )

        self.assertEqual(intervals, ({"index": 0, "start_ms": 0, "end_ms": 140},))

    def test_probabilities_to_intervals_rejects_invalid_probability(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            probabilities_to_intervals([1.5], duration_ms=20)

    def test_activate_outputs_supports_sigmoid_and_identity(self):
        self.assertEqual(activate_outputs([0.0], "sigmoid"), [0.5])
        self.assertEqual(activate_outputs([0.25], "identity"), [0.25])


if __name__ == "__main__":
    unittest.main()
