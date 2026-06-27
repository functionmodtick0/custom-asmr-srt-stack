import io
import struct
import unittest
import wave

from custom_asmr_srt_stack.audio import (
    analyze_wav,
    chunk_intervals,
    normalize_audio_to_wav,
    slice_wav,
    speech_intervals_by_energy,
    split_wav_channels,
)


def make_stereo_wav(samples):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        frames = bytearray()
        for left, right in samples:
            frames.extend(struct.pack("<h", left))
            frames.extend(struct.pack("<h", right))
        wav.writeframes(bytes(frames))
    return output.getvalue()


def make_mono_wav(samples):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    return output.getvalue()


def read_mono_samples(audio_bytes):
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return [struct.unpack("<h", frames[index : index + 2])[0] for index in range(0, len(frames), 2)]


class AudioPipelineTests(unittest.TestCase):
    def test_analyze_wav_reports_duration_and_channels(self):
        info = analyze_wav(make_stereo_wav([(100, 300), (200, 400)]))

        self.assertEqual(info.duration_ms, 2)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.sample_rate, 1000)
        self.assertEqual(info.sample_width, 2)
        self.assertEqual(info.frame_count, 2)

    def test_split_wav_channels_creates_left_right_and_mix(self):
        _, channels = split_wav_channels(make_stereo_wav([(100, 300), (200, 400)]))

        self.assertEqual(set(channels), {"L", "R", "MIX"})
        self.assertEqual(read_mono_samples(channels["L"]), [100, 200])
        self.assertEqual(read_mono_samples(channels["R"]), [300, 400])
        self.assertEqual(read_mono_samples(channels["MIX"]), [200, 300])

    def test_chunk_intervals_cover_duration_without_overflow(self):
        self.assertEqual(
            chunk_intervals(450, max_chunk_ms=200),
            [
                {"index": 0, "start_ms": 0, "end_ms": 200},
                {"index": 1, "start_ms": 200, "end_ms": 400},
                {"index": 2, "start_ms": 400, "end_ms": 450},
            ],
        )

    def test_speech_intervals_by_energy_splits_on_long_silence(self):
        audio = make_mono_wav(([2000] * 1000) + ([0] * 1000) + ([2000] * 1000))

        intervals = speech_intervals_by_energy(
            audio,
            threshold_dbfs=-40,
            window_ms=100,
            min_silence_ms=500,
            pad_ms=100,
        )

        self.assertEqual(intervals, [{"index": 0, "start_ms": 0, "end_ms": 1100}, {"index": 1, "start_ms": 1900, "end_ms": 3000}])

    def test_normalize_audio_keeps_valid_wav(self):
        audio = make_stereo_wav([(100, 300)])

        self.assertEqual(normalize_audio_to_wav(audio, file_name="voice.wav"), audio)

    def test_normalize_audio_fails_visibly_for_invalid_audio(self):
        with self.assertRaisesRegex(ValueError, "ffmpeg could not decode audio|audio is not WAV"):
            normalize_audio_to_wav(b"not audio", file_name="voice.mp3", mime_type="audio/mpeg")

    def test_slice_wav_returns_selected_time_range(self):
        audio = make_stereo_wav([(100, 300), (200, 400), (500, 700)])

        sliced = slice_wav(audio, start_ms=1, end_ms=3)

        info = analyze_wav(sliced)
        self.assertEqual(info.duration_ms, 2)
        _, channels = split_wav_channels(sliced)
        self.assertEqual(read_mono_samples(channels["L"]), [200, 500])


if __name__ == "__main__":
    unittest.main()
