from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioInfo:
    duration_ms: int
    channels: int
    sample_rate: int
    sample_width: int
    frame_count: int

    def to_json(self) -> dict[str, int]:
        return {
            "duration_ms": self.duration_ms,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "sample_width": self.sample_width,
            "frame_count": self.frame_count,
        }


def analyze_wav(audio_bytes: bytes) -> AudioInfo:
    with open_wave(audio_bytes) as wav:
        frame_count = wav.getnframes()
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

    if sample_rate <= 0:
        raise ValueError("WAV sample_rate must be positive")
    return AudioInfo(
        duration_ms=round((frame_count / sample_rate) * 1000),
        channels=channels,
        sample_rate=sample_rate,
        sample_width=sample_width,
        frame_count=frame_count,
    )


def split_wav_channels(audio_bytes: bytes) -> tuple[AudioInfo, dict[str, bytes]]:
    with open_wave(audio_bytes) as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        frames = wav.readframes(frame_count)

    info = analyze_wav(audio_bytes)
    if channels == 1:
        return info, {"MIX": audio_bytes}
    if channels != 2:
        raise ValueError("only mono and stereo WAV files are supported")
    if sample_width != 2:
        raise ValueError("stereo channel split currently requires 16-bit PCM WAV")

    left = bytearray()
    right = bytearray()
    mix = bytearray()
    for offset in range(0, len(frames), channels * sample_width):
        left_sample = frames[offset : offset + sample_width]
        right_sample = frames[offset + sample_width : offset + (2 * sample_width)]
        left.extend(left_sample)
        right.extend(right_sample)
        mixed_sample = round((pcm16_to_int(left_sample) + pcm16_to_int(right_sample)) / 2)
        mix.extend(int_to_pcm16(mixed_sample))

    return info, {
        "L": write_wav(bytes(left), channels=1, sample_width=sample_width, sample_rate=sample_rate),
        "R": write_wav(bytes(right), channels=1, sample_width=sample_width, sample_rate=sample_rate),
        "MIX": write_wav(bytes(mix), channels=1, sample_width=sample_width, sample_rate=sample_rate),
    }


def chunk_intervals(duration_ms: int, max_chunk_ms: int = 180_000) -> list[dict[str, int]]:
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")
    if max_chunk_ms <= 0:
        raise ValueError("max_chunk_ms must be positive")
    if duration_ms == 0:
        return []

    intervals = []
    start_ms = 0
    while start_ms < duration_ms:
        end_ms = min(duration_ms, start_ms + max_chunk_ms)
        intervals.append({"index": len(intervals), "start_ms": start_ms, "end_ms": end_ms})
        start_ms = end_ms
    return intervals


def open_wave(audio_bytes: bytes) -> wave.Wave_read:
    try:
        return wave.open(io.BytesIO(audio_bytes), "rb")
    except wave.Error as error:
        raise ValueError(f"unsupported WAV audio: {error}") from error


def write_wav(frames: bytes, *, channels: int, sample_width: int, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    return output.getvalue()


def pcm16_to_int(sample: bytes) -> int:
    return struct.unpack("<h", sample)[0]


def int_to_pcm16(value: int) -> bytes:
    return struct.pack("<h", max(-32768, min(32767, value)))
