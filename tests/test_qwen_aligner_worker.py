import base64
import hashlib
import io
import json
import socket
import struct
import tempfile
import unittest
import wave
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import custom_asmr_srt_stack.qwen_aligner_worker as qwen_aligner_worker
from custom_asmr_srt_stack.audio import analyze_wav
from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.qwen_aligner_worker import (
    align_master,
    aligned_bounds_ms,
    apply_local_load_kwargs,
    checked_model_path,
    disable_python_network_if_requested,
    qwen_language,
    require_secure_runtime,
    response_for_stdin,
    verify_qwen_asr_package,
)


def mono_wav_bytes(duration_ms: int = 3000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(struct.pack("<h", 100) * duration_ms)
    return output.getvalue()


def sample_master() -> MasterDocument:
    return MasterDocument(
        source_language="ja",
        source_file="voice.wav",
        duration_ms=3000,
        segments=(
            Segment("seg_000001", 0, 1000, "MIX", "speech", "ねえ"),
            Segment("seg_000002", 1200, 2200, "MIX", "speech", "聞こえる"),
        ),
    )


def secure_env() -> dict[str, str]:
    return {
        "CASRT_ALIGNER_ENV_MODE": "offline",
        "CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH": "1",
        "CASRT_QWEN_ALIGNER_LOCAL_FILES_ONLY": "1",
        "CASRT_QWEN_ALIGNER_DISABLE_NETWORK": "1",
    }


@dataclass(frozen=True)
class FakeAlignItem:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class FakeAlignResult:
    items: list[FakeAlignItem]


class FakeAligner:
    def __init__(self) -> None:
        self.calls = []
        self.clip_durations_ms = []

    def align(self, *, audio, text, language):
        self.calls.append({"audio": audio, "text": text, "language": language})
        self.clip_durations_ms.extend(analyze_wav(Path(path).read_bytes()).duration_ms for path in audio)
        return [
            FakeAlignResult([FakeAlignItem("ね", 0.10, 0.20), FakeAlignItem("え", 0.25, 0.80)]),
            FakeAlignResult([FakeAlignItem("聞こえる", 0.05, 0.60)]),
        ]


class FakeRuntime:
    def __init__(self, aligner):
        self.aligner = aligner
        self.model_ids = []

    def load_aligner(self, model_id):
        self.model_ids.append(model_id)
        return self.aligner


class FakeDistribution:
    def __init__(self, version: str, root: Path, record_path: Path) -> None:
        self.version = version
        self.root = root
        self.record_path = record_path

    def locate_file(self, path: str):
        if path.endswith(".dist-info/RECORD"):
            return self.record_path
        return self.root / path


class FakeSpec:
    def __init__(self, origin: Path) -> None:
        self.origin = str(origin)


def write_fake_qwen_asr_distribution(root: Path, *, file_content: str = "package") -> tuple[FakeDistribution, Path]:
    package_init = root / "qwen_asr" / "__init__.py"
    package_init.parent.mkdir()
    package_init.write_text(file_content, encoding="utf-8")
    record_path = root / "qwen_asr-0.0.6.dist-info" / "RECORD"
    record_path.parent.mkdir()
    package_hash = record_hash_entry(package_init)
    record_path.write_text(
        f"qwen_asr/__init__.py,sha256={package_hash},{len(package_init.read_bytes())}\n"
        "qwen_asr-0.0.6.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    return FakeDistribution("0.0.6", root, record_path), package_init


def record_hash_entry(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


class ShortAligner:
    def align(self, *, audio, text, language):
        del audio, text, language
        return []


class QwenAlignerWorkerTests(unittest.TestCase):
    def test_align_master_applies_segment_relative_bounds(self):
        aligner = FakeAligner()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(mono_wav_bytes())

            aligned = align_master(sample_master(), audio_file=audio_path, aligner=aligner)

        self.assertEqual((aligned.segments[0].start_ms, aligned.segments[0].end_ms), (100, 800))
        self.assertEqual((aligned.segments[1].start_ms, aligned.segments[1].end_ms), (1250, 1800))
        self.assertEqual(aligner.calls[0]["text"], ["ねえ", "聞こえる"])
        self.assertEqual(aligner.calls[0]["language"], ["Japanese", "Japanese"])
        self.assertEqual(aligner.clip_durations_ms, [1000, 1000])

    def test_align_master_context_padding_can_move_bounds_outside_original_segment(self):
        aligner = FakeAligner()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(mono_wav_bytes())

            with mock.patch.dict("os.environ", {"CASRT_QWEN_ALIGNER_CONTEXT_MS": "200"}, clear=False):
                aligned = align_master(sample_master(), audio_file=audio_path, aligner=aligner)

        self.assertEqual((aligned.segments[0].start_ms, aligned.segments[0].end_ms), (100, 800))
        self.assertEqual((aligned.segments[1].start_ms, aligned.segments[1].end_ms), (1050, 1600))
        self.assertEqual(aligner.clip_durations_ms, [1200, 1400])

    def test_align_master_fails_when_aligner_result_count_mismatches_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(mono_wav_bytes())

            with self.assertRaisesRegex(ValueError, "returned 0 results for 2 speech segments"):
                align_master(sample_master(), audio_file=audio_path, aligner=ShortAligner())

    def test_aligned_bounds_rejects_implausibly_short_spans(self):
        result = FakeAlignResult([FakeAlignItem("う", 0.100, 0.101)])

        self.assertIsNone(aligned_bounds_ms(result, 1000))

    def test_aligned_bounds_rejects_low_coverage_spans(self):
        result = FakeAlignResult([FakeAlignItem("ね", 0.100, 0.400)])

        with mock.patch.dict("os.environ", {"CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO": "0.75"}, clear=False):
            self.assertIsNone(aligned_bounds_ms(result, 1000))

    def test_aligned_bounds_uses_original_duration_for_context_coverage(self):
        result = FakeAlignResult([FakeAlignItem("ね", 1.100, 1.800)])

        self.assertEqual(aligned_bounds_ms(result, 3000, coverage_duration_ms=1000), (1100, 1800))

    def test_aligned_bounds_rejects_invalid_coverage_ratio(self):
        result = FakeAlignResult([FakeAlignItem("ね", 0.100, 0.900)])

        with mock.patch.dict("os.environ", {"CASRT_QWEN_ALIGNER_MIN_COVERAGE_RATIO": "1.1"}, clear=False):
            with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
                aligned_bounds_ms(result, 1000)

    def test_response_for_stdin_outputs_alignment_contract(self):
        aligner = FakeAligner()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(mono_wav_bytes())
            request = {"audio_file": str(audio_path), "master": sample_master().to_json()}

            with mock.patch.dict("os.environ", secure_env(), clear=False):
                response = response_for_stdin(FakeRuntime(aligner), json.dumps(request), model_id=str(tmpdir))

        self.assertEqual(response["segments"][0], {"id": "seg_000001", "start_ms": 100, "end_ms": 800})

    def test_response_for_stdin_fails_when_secure_env_is_missing(self):
        response = response_for_stdin(FakeRuntime(FakeAligner()), "{}", model_id="aligner-model")

        self.assertFalse(response["ok"])
        self.assertIn("CASRT_ALIGNER_ENV_MODE=offline", response["error"])
        self.assertNotIn("traceback", response)

    def test_response_for_stdin_reports_missing_model_id(self):
        with mock.patch.dict("os.environ", secure_env(), clear=False):
            response = response_for_stdin(FakeRuntime(FakeAligner()), "{}", model_id="")

        self.assertFalse(response["ok"])
        self.assertIn("model-id", response["error"])

    def test_qwen_language_defaults_to_japanese(self):
        self.assertEqual(qwen_language("ja"), "Japanese")
        self.assertEqual(qwen_language("unknown"), "Japanese")

    def test_local_loading_flags_are_added_when_local_path_is_required(self):
        with mock.patch.dict("os.environ", {"CASRT_QWEN_ALIGNER_REQUIRE_LOCAL_MODEL_PATH": "1"}, clear=False):
            kwargs = apply_local_load_kwargs({})

        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])

    def test_local_model_path_requirement_rejects_repo_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            self.assertEqual(checked_model_path(str(model_dir), "model_id"), str(model_dir.resolve()))
            with self.assertRaisesRegex(ValueError, "existing local model directory"):
                checked_model_path("Qwen/Qwen3-ForcedAligner-0.6B", "model_id")

    def test_secure_runtime_requires_all_fail_closed_flags(self):
        for missing_name in secure_env():
            env = secure_env()
            env.pop(missing_name)
            with self.subTest(missing_name=missing_name), mock.patch.dict("os.environ", env, clear=True):
                with self.assertRaisesRegex(ValueError, missing_name):
                    require_secure_runtime()

    def test_qwen_asr_package_fingerprint_accepts_expected_record_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            distribution, package_init = write_fake_qwen_asr_distribution(Path(tmpdir))
            digest = hashlib.sha256(distribution.record_path.read_bytes()).hexdigest()

            with mock.patch.object(qwen_aligner_worker.importlib_metadata, "distribution", return_value=distribution):
                with mock.patch.object(qwen_aligner_worker, "QWEN_ASR_EXPECTED_RECORD_SHA256", digest):
                    with mock.patch.object(
                        qwen_aligner_worker.importlib.util,
                        "find_spec",
                        return_value=FakeSpec(package_init),
                    ):
                        verify_qwen_asr_package()

    def test_qwen_asr_package_fingerprint_rejects_version_or_record_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            distribution, _ = write_fake_qwen_asr_distribution(Path(tmpdir))

            with mock.patch.object(
                qwen_aligner_worker.importlib_metadata,
                "distribution",
                return_value=FakeDistribution("0.0.7", Path(tmpdir), distribution.record_path),
            ):
                with self.assertRaisesRegex(ValueError, "not the pinned"):
                    verify_qwen_asr_package()

            with mock.patch.object(
                qwen_aligner_worker.importlib_metadata,
                "distribution",
                return_value=distribution,
            ):
                with self.assertRaisesRegex(ValueError, "RECORD hash"):
                    verify_qwen_asr_package()

    def test_qwen_asr_package_fingerprint_rejects_installed_file_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            distribution, package_init = write_fake_qwen_asr_distribution(Path(tmpdir))
            digest = hashlib.sha256(distribution.record_path.read_bytes()).hexdigest()
            package_init.write_text("tampered", encoding="utf-8")

            with mock.patch.object(qwen_aligner_worker.importlib_metadata, "distribution", return_value=distribution):
                with mock.patch.object(qwen_aligner_worker, "QWEN_ASR_EXPECTED_RECORD_SHA256", digest):
                    with self.assertRaisesRegex(ValueError, "installed file hash mismatch"):
                        verify_qwen_asr_package()

    def test_qwen_asr_package_fingerprint_rejects_import_origin_hijack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            distribution, _ = write_fake_qwen_asr_distribution(root)
            digest = hashlib.sha256(distribution.record_path.read_bytes()).hexdigest()
            hijack = root / "other" / "__init__.py"
            hijack.parent.mkdir()
            hijack.write_text("hijack", encoding="utf-8")

            with mock.patch.object(qwen_aligner_worker.importlib_metadata, "distribution", return_value=distribution):
                with mock.patch.object(qwen_aligner_worker, "QWEN_ASR_EXPECTED_RECORD_SHA256", digest):
                    with mock.patch.object(
                        qwen_aligner_worker.importlib.util,
                        "find_spec",
                        return_value=FakeSpec(hijack),
                    ):
                        with self.assertRaisesRegex(ValueError, "import origin"):
                            verify_qwen_asr_package()

    def test_network_guard_blocks_python_socket_creation(self):
        original_socket = socket.socket
        original_create_connection = socket.create_connection
        try:
            with mock.patch.dict("os.environ", {"CASRT_QWEN_ALIGNER_DISABLE_NETWORK": "1"}, clear=False):
                disable_python_network_if_requested()

            with self.assertRaisesRegex(OSError, "network access is disabled"):
                socket.socket()
            with self.assertRaisesRegex(OSError, "network access is disabled"):
                socket.create_connection(("127.0.0.1", 9), timeout=0.1)

            import ssl

            self.assertTrue(hasattr(ssl, "SSLSocket"))
        finally:
            socket.socket = original_socket
            socket.create_connection = original_create_connection
            qwen_aligner_worker._NETWORK_DISABLED = False


if __name__ == "__main__":
    unittest.main()
