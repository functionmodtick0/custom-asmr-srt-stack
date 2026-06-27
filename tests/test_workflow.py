import base64
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from custom_asmr_srt_stack.audio import analyze_wav
from custom_asmr_srt_stack.models import Segment
from custom_asmr_srt_stack.projects import ProjectStore
from custom_asmr_srt_stack.transcription import ModelEndpoint
from custom_asmr_srt_stack.workflow import analyze_project, transcribe_project


def write_stereo_wav(path: Path, duration_ms: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(struct.pack("<hh", 100, 200) * duration_ms)


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


if __name__ == "__main__":
    unittest.main()
