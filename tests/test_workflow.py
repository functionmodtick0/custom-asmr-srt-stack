import base64
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from custom_asmr_srt_stack.audio import analyze_wav
from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.transcription import ModelEndpoint
from custom_asmr_srt_stack.workflow import analyze_project, transcribe_project
from custom_asmr_srt_stack.workflow import retranscribe_segment as retranscribe_workflow_segment


def write_stereo_wav(path: Path, duration_ms: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(struct.pack("<hh", 100, 200) * duration_ms)


def write_stereo_samples(path: Path, samples: list[tuple[int, int]]) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        frames = bytearray()
        for left, right in samples:
            frames.extend(struct.pack("<hh", left, right))
        wav.writeframes(bytes(frames))


def write_mono_wav(path: Path, duration_ms: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(struct.pack("<h", 100) * duration_ms)


class WorkflowTests(unittest.TestCase):
    def test_production_transcription_uses_fixed_stereo_chunks_and_ignores_experiment_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "fixed.wav"
            write_stereo_wav(audio_path, duration_ms=45_000)
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "fixed.wav",
                "audio/wav",
                base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            )
            analyzed = analyze_project(store, created["project_id"])
            calls = []

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                del endpoint, mime_type, source_language
                duration_ms = analyze_wav(audio_bytes).duration_ms
                calls.append((channel, duration_ms))
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=duration_ms,
                        channel=channel,
                        kind="speech",
                        text=f"{channel}:{duration_ms}",
                    ),
                )

            with mock.patch.dict(
                os.environ,
                {
                    "CASRT_LOCAL_ASR_CHANNEL_MODE": "mix",
                    "CASRT_VAD_COMMAND": "missing-vad-command",
                    "CASRT_ALIGNER_COMMAND": "missing-aligner-command",
                    "CASRT_QWEN_ENERGY_MAX_CHUNK_MS": "1",
                },
            ):
                legacy_metadata = {
                    **analyzed["metadata"],
                    "chunks": [{"start_ms": 0, "end_ms": 45_000}],
                }
                master = transcribe_project(
                    store,
                    created["project_id"],
                    ModelEndpoint(
                        adapter="local-qwen-hf-asr",
                        endpoint_url=None,
                        model_id="/models/qwen3-asr-1.7b-hf",
                    ),
                    legacy_metadata,
                    source_language="ja",
                    transcribe_audio_func=fake_transcribe,
                )

            self.assertEqual(
                calls,
                [
                    ("L", 20_000),
                    ("L", 20_000),
                    ("L", 5_000),
                    ("R", 20_000),
                    ("R", 20_000),
                    ("R", 5_000),
                ],
            )
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.channel) for segment in master.segments],
                [
                    (0, 20_000, "L"),
                    (0, 20_000, "R"),
                    (20_000, 40_000, "L"),
                    (20_000, 40_000, "R"),
                    (40_000, 45_000, "L"),
                    (40_000, 45_000, "R"),
                ],
            )

    def test_transcribe_project_offsets_segments_at_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "long.wav"
            write_stereo_wav(audio_path, duration_ms=20_001)
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "long.wav",
                "audio/wav",
                base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            )
            analyzed = analyze_project(store, created["project_id"])
            calls = []

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                del endpoint, mime_type, source_language
                duration_ms = analyze_wav(audio_bytes).duration_ms
                calls.append((channel, duration_ms))
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=duration_ms,
                        channel=channel,
                        kind="speech",
                        text=f"{channel}:{duration_ms}",
                    ),
                )

            master = transcribe_project(
                store,
                created["project_id"],
                ModelEndpoint(
                    adapter="openai-compatible",
                    endpoint_url="http://localhost:8000/v1",
                    model_id="gemma-4-e4b",
                ),
                analyzed["metadata"],
                source_language="ja",
                transcribe_audio_func=fake_transcribe,
            )

            self.assertEqual(calls, [("L", 20_000), ("L", 1), ("R", 20_000), ("R", 1)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.channel, segment.text) for segment in master.segments],
                [
                    (0, 20_000, "L", "L:20000"),
                    (0, 20_000, "R", "R:20000"),
                    (20_000, 20_001, "L", "L:1"),
                    (20_000, 20_001, "R", "R:1"),
                ],
            )

    def test_local_transformers_uses_production_chunk_limit_and_stereo_channels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "local.wav"
            write_stereo_samples(audio_path, [(200, 200)] * 65_000)
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "local.wav",
                "audio/wav",
                base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            )
            analyzed = analyze_project(store, created["project_id"])
            calls = []

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                del endpoint, mime_type, source_language
                duration_ms = analyze_wav(audio_bytes).duration_ms
                calls.append((channel, duration_ms))
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=duration_ms,
                        channel=channel,
                        kind="speech",
                        text=f"{channel}:{duration_ms}",
                    ),
                )

            master = transcribe_project(
                store,
                created["project_id"],
                ModelEndpoint(
                    adapter="local-transformers",
                    endpoint_url=None,
                    model_id="google/gemma-4-E4B-it",
                ),
                analyzed["metadata"],
                source_language="ja",
                transcribe_audio_func=fake_transcribe,
            )

            self.assertEqual(
                calls,
                [
                    ("L", 20_000),
                    ("L", 20_000),
                    ("L", 20_000),
                    ("L", 5_000),
                    ("R", 20_000),
                    ("R", 20_000),
                    ("R", 20_000),
                    ("R", 5_000),
                ],
            )
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.channel, segment.text) for segment in master.segments],
                [
                    (0, 20_000, "L", "L:20000"),
                    (0, 20_000, "R", "R:20000"),
                    (20_000, 40_000, "L", "L:20000"),
                    (20_000, 40_000, "R", "R:20000"),
                    (40_000, 60_000, "L", "L:20000"),
                    (40_000, 60_000, "R", "R:20000"),
                    (60_000, 65_000, "L", "L:5000"),
                    (60_000, 65_000, "R", "R:5000"),
                ],
            )

    def test_local_asr_adapters_use_the_same_fixed_stereo_path(self):
        for adapter, model_id in (
            ("local-qwen-asr", "Qwen/Qwen3-ASR-1.7B"),
            ("local-qwen-hf-asr", "/models/qwen3-asr-1.7b-hf"),
            ("local-granite-asr", "/models/granite-speech-4.1-2b"),
        ):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                audio_path = root / "qwen.wav"
                write_stereo_samples(audio_path, [(200, 200)] * 10_000)
                store = ProjectStore(root / "projects")
                created = store.create_from_audio(
                    "qwen.wav",
                    "audio/wav",
                    base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                )
                analyzed = analyze_project(store, created["project_id"])
                calls = []

                def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                    del endpoint, mime_type, source_language
                    calls.append((channel, analyze_wav(audio_bytes).duration_ms))
                    return (
                        Segment(
                            id="ignored",
                            start_ms=0,
                            end_ms=10_000,
                            channel=channel,
                            kind="speech",
                            text=f"{channel}-speech",
                        ),
                    )

                master = transcribe_project(
                    store,
                    created["project_id"],
                    ModelEndpoint(
                        adapter=adapter,
                        endpoint_url=None,
                        model_id=model_id,
                    ),
                    analyzed["metadata"],
                    source_language="ja",
                    transcribe_audio_func=fake_transcribe,
                )

                self.assertEqual(calls, [("L", 10_000), ("R", 10_000)])
                self.assertEqual(
                    [(segment.channel, segment.text) for segment in master.segments],
                    [("L", "L-speech"), ("R", "R-speech")],
                )

    def test_mono_project_transcribes_mix_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "mono.wav"
            write_mono_wav(audio_path, duration_ms=10_000)
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "mono.wav",
                "audio/wav",
                base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            )
            analyzed = analyze_project(store, created["project_id"])
            calls = []

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                del endpoint, mime_type, source_language
                duration_ms = analyze_wav(audio_bytes).duration_ms
                calls.append((channel, duration_ms))
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=duration_ms,
                        channel=channel,
                        kind="speech",
                        text=f"{channel}-speech",
                    ),
                )

            master = transcribe_project(
                store,
                created["project_id"],
                ModelEndpoint(
                    adapter="local-qwen-hf-asr",
                    endpoint_url=None,
                    model_id="/models/qwen3-asr-1.7b-hf",
                ),
                analyzed["metadata"],
                source_language="ja",
                transcribe_audio_func=fake_transcribe,
            )

            self.assertEqual(calls, [("MIX", 10_000)])
            self.assertEqual(
                [(segment.channel, segment.text) for segment in master.segments],
                [("MIX", "MIX-speech")],
            )

    def test_production_transcription_does_not_drop_silence_intervals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "qwen-silence.wav"
            write_stereo_samples(
                audio_path,
                ([(2000, 2000)] * 1000) + ([(0, 0)] * 1000) + ([(2000, 2000)] * 1000),
            )
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "qwen-silence.wav",
                "audio/wav",
                base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            )
            analyzed = analyze_project(store, created["project_id"])
            calls = []

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                del endpoint, mime_type, source_language
                duration_ms = analyze_wav(audio_bytes).duration_ms
                calls.append((channel, duration_ms))
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=duration_ms,
                        channel=channel,
                        kind="speech",
                        text=str(duration_ms),
                    ),
                )

            master = transcribe_project(
                store,
                created["project_id"],
                ModelEndpoint(
                    adapter="local-qwen-asr",
                    endpoint_url=None,
                    model_id="Qwen/Qwen3-ASR-1.7B",
                ),
                analyzed["metadata"],
                source_language="ja",
                transcribe_audio_func=fake_transcribe,
            )

            self.assertEqual(calls, [("L", 3000), ("R", 3000)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.channel, segment.text) for segment in master.segments],
                [(0, 3000, "L", "3000"), (0, 3000, "R", "3000")],
            )

    def test_transcribe_project_requires_analysis_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProjectStore(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "analyzed before transcription"):
                transcribe_project(
                    store,
                    "0" * 32,
                    ModelEndpoint(
                        adapter="openai-compatible",
                        endpoint_url="http://localhost:8000/v1",
                        model_id="gemma-4-e4b",
                    ),
                    {"channels": {"MIX": "audio/mix.wav"}},
                    source_language="ja",
                    transcribe_audio_func=lambda *args, **kwargs: (),
                )

    def test_local_transformers_retranscribe_segment_splits_target_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "local.wav"
            write_stereo_wav(audio_path, duration_ms=70_000)
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "local.wav",
                "audio/wav",
                base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            )
            analyzed = analyze_project(store, created["project_id"])
            master = MasterDocument(
                source_language="ja",
                source_file="local.wav",
                duration_ms=70_000,
                segments=(
                    Segment(
                        id="seg_000001",
                        start_ms=5_000,
                        end_ms=70_000,
                        channel="L",
                        kind="speech",
                        text="古い",
                    ),
                ),
            )
            calls = []

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                del endpoint, mime_type, source_language
                duration_ms = analyze_wav(audio_bytes).duration_ms
                calls.append((channel, duration_ms))
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=duration_ms,
                        channel=channel,
                        kind="speech",
                        text=f"{duration_ms}",
                    ),
                )

            updated = retranscribe_workflow_segment(
                store,
                created["project_id"],
                master,
                analyzed["metadata"],
                segment_id="seg_000001",
                model_endpoint=ModelEndpoint(
                    adapter="local-transformers",
                    endpoint_url=None,
                    model_id="google/gemma-4-E4B-it",
                ),
                source_language="ja",
                transcribe_audio_func=fake_transcribe,
            )

            self.assertEqual(calls, [("L", 20_000), ("L", 20_000), ("L", 20_000), ("L", 5_000)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.text) for segment in updated.segments],
                [
                    (5_000, 25_000, "20000"),
                    (25_000, 45_000, "20000"),
                    (45_000, 65_000, "20000"),
                    (65_000, 70_000, "5000"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
