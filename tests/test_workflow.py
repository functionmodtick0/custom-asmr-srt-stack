import base64
import os
import struct
import sys
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


class WorkflowTests(unittest.TestCase):
    def test_transcribe_project_sends_analyzed_channel_chunks_and_offsets_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "long.wav"
            write_stereo_wav(audio_path, duration_ms=180_001)
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

            self.assertEqual(calls, [("L", 180_000), ("L", 1), ("R", 180_000), ("R", 1)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.channel, segment.text) for segment in master.segments],
                [
                    (0, 180_000, "L", "L:180000"),
                    (0, 180_000, "R", "R:180000"),
                    (180_000, 180_001, "L", "L:1"),
                    (180_000, 180_001, "R", "R:1"),
                ],
            )

    def test_local_transformers_transcribe_project_splits_chunks_to_audio_limit(self):
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

            self.assertEqual(calls, [("MIX", 30_000), ("MIX", 30_000), ("MIX", 5_000)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.channel, segment.text) for segment in master.segments],
                [
                    (0, 30_000, "MIX", "MIX:30000"),
                    (30_000, 60_000, "MIX", "MIX:30000"),
                    (60_000, 65_000, "MIX", "MIX:5000"),
                ],
            )

    def test_local_qwen_asr_transcribe_project_uses_mix_first(self):
        for adapter, model_id in (
            ("local-qwen-asr", "Qwen/Qwen3-ASR-1.7B"),
            ("local-qwen-hf-asr", "/models/qwen3-asr-1.7b-hf"),
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
                            text="MIX-first",
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

                self.assertEqual(calls, [("MIX", 10_000)])
                self.assertEqual([(segment.channel, segment.text) for segment in master.segments], [("MIX", "MIX-first")])

    def test_local_qwen_asr_transcribe_project_uses_energy_chunks(self):
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

            self.assertEqual(calls, [("MIX", 1200), ("MIX", 1200)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.text) for segment in master.segments],
                [(0, 1200, "1200"), (1800, 3000, "1200")],
            )

    def test_local_qwen_asr_energy_chunks_can_be_tuned_by_env(self):
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

            with mock.patch.dict(
                os.environ,
                {
                    "CASRT_QWEN_ENERGY_MIN_SILENCE_MS": "800",
                    "CASRT_QWEN_ENERGY_PAD_MS": "400",
                },
            ):
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

            self.assertEqual(calls, [("MIX", 1400), ("MIX", 1400)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.text) for segment in master.segments],
                [(0, 1400, "1400"), (1600, 3000, "1400")],
            )

    def test_local_qwen_asr_energy_chunks_can_be_bounded_by_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "qwen-long-scene.wav"
            write_stereo_samples(audio_path, [(2000, 2000)] * 3000)
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "qwen-long-scene.wav",
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

            with mock.patch.dict(os.environ, {"CASRT_QWEN_ENERGY_MAX_CHUNK_MS": "1000"}):
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

            self.assertEqual(calls, [("MIX", 1000), ("MIX", 1000), ("MIX", 1000)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.text) for segment in master.segments],
                [(0, 1000, "1000"), (1000, 2000, "1000"), (2000, 3000, "1000")],
            )

    def test_local_qwen_asr_transcribe_project_uses_vad_command_when_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "qwen-vad.wav"
            vad_script = root / "vad.py"
            write_stereo_samples(audio_path, [(2000, 2000)] * 3000)
            vad_script.write_text(
                "\n".join(
                    [
                        "import json",
                        "import pathlib",
                        "import sys",
                        "request = json.loads(sys.stdin.read())",
                        "assert pathlib.Path(request['audio_file']).exists()",
                        "assert request['audio_info']['duration_ms'] == 3000",
                        "print(json.dumps({'intervals': [{'start_ms': 500, 'end_ms': 1500}]}))",
                    ]
                ),
                encoding="utf-8",
            )
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "qwen-vad.wav",
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

            with mock.patch.dict(os.environ, {"CASRT_VAD_COMMAND": f"{sys.executable} {vad_script}"}):
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

            self.assertEqual(calls, [("MIX", 1000)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.text) for segment in master.segments],
                [(500, 1500, "1000")],
            )

    def test_local_qwen_asr_attributes_mix_segment_to_louder_channel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "qwen-left.wav"
            write_stereo_samples(audio_path, [(6000, 100)] * 2000)
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "qwen-left.wav",
                "audio/wav",
                base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            )
            analyzed = analyze_project(store, created["project_id"])

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                del endpoint, audio_bytes, mime_type, channel, source_language
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=2000,
                        channel="MIX",
                        kind="speech",
                        text="左",
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

            self.assertEqual([(segment.channel, segment.text) for segment in master.segments], [("L", "左")])

    def test_local_transformers_attributes_mix_segment_to_louder_channel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "gemma-right.wav"
            write_stereo_samples(audio_path, [(100, 6000)] * 2000)
            store = ProjectStore(root / "projects")
            created = store.create_from_audio(
                "gemma-right.wav",
                "audio/wav",
                base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            )
            analyzed = analyze_project(store, created["project_id"])
            calls = []

            def fake_transcribe(endpoint, audio_bytes, *, mime_type, channel, source_language):
                del endpoint, audio_bytes, mime_type, source_language
                calls.append(channel)
                return (
                    Segment(
                        id="ignored",
                        start_ms=0,
                        end_ms=2000,
                        channel="MIX",
                        kind="speech",
                        text="右",
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

            self.assertEqual(calls, ["MIX"])
            self.assertEqual([(segment.channel, segment.text) for segment in master.segments], [("R", "右")])

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

            self.assertEqual(calls, [("L", 30_000), ("L", 30_000), ("L", 5_000)])
            self.assertEqual(
                [(segment.start_ms, segment.end_ms, segment.text) for segment in updated.segments],
                [(5_000, 35_000, "30000"), (35_000, 65_000, "30000"), (65_000, 70_000, "5000")],
            )


if __name__ == "__main__":
    unittest.main()
