import json
import tempfile
import unittest
from pathlib import Path

from custom_asmr_srt_stack.whisper_vad_onnx import (
    EXPECTED_METADATA,
    WhisperVadOnnxSettings,
    activate_outputs,
    load_metadata,
    probabilities_to_intervals,
    validate_model_files,
    validate_model_metadata,
    validate_session_contract,
    validate_settings,
)


class FakeOrtValue:
    def __init__(self, shape):
        self.shape = shape


class FakeOrtSession:
    def __init__(self, *, input_shape, output_shape, providers=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.providers = providers or ["CPUExecutionProvider"]

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return [FakeOrtValue(self.input_shape)]

    def get_outputs(self):
        return [FakeOrtValue(self.output_shape)]


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

    def test_model_files_must_be_dedicated_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "model.onnx"
            metadata = root / "model_metadata.json"
            model.write_bytes(b"")
            metadata.write_text(json.dumps(EXPECTED_METADATA), encoding="utf-8")
            validate_model_files(model, metadata)

            (root / "README.md").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected files"):
                validate_model_files(model, metadata)

    def test_metadata_must_exist_and_match_reviewed_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = Path(tmpdir) / "model_metadata.json"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                load_metadata(metadata)

            invalid = dict(EXPECTED_METADATA)
            invalid["total_duration_ms"] = 10_000
            metadata.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "total_duration_ms"):
                validate_model_metadata(load_metadata(metadata))

    def test_settings_require_cpu(self):
        with self.assertRaisesRegex(ValueError, "force_cpu"):
            validate_settings(WhisperVadOnnxSettings(force_cpu=False))

    def test_session_contract_allows_reviewed_symbolic_batch_only(self):
        validate_session_contract(FakeOrtSession(input_shape=["s6", 80, 3000], output_shape=[1, 1500]))

        with self.assertRaisesRegex(ValueError, "input shape"):
            validate_session_contract(FakeOrtSession(input_shape=["s6", 80, 2999], output_shape=[1, 1500]))

        with self.assertRaisesRegex(ValueError, "CPUExecutionProvider"):
            validate_session_contract(
                FakeOrtSession(
                    input_shape=["s6", 80, 3000],
                    output_shape=[1, 1500],
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
            )


if __name__ == "__main__":
    unittest.main()
