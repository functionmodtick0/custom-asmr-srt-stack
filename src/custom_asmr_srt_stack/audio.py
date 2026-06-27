from __future__ import annotations

import io
import shutil
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


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


def slice_wav(audio_bytes: bytes, *, start_ms: int, end_ms: int) -> bytes:
    if start_ms < 0:
        raise ValueError("start_ms must be non-negative")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")

    with open_wave(audio_bytes) as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        start_frame = min(frame_count, round((start_ms / 1000) * sample_rate))
        end_frame = min(frame_count, round((end_ms / 1000) * sample_rate))
        if end_frame <= start_frame:
            raise ValueError("selected audio range is empty")
        wav.setpos(start_frame)
        frames = wav.readframes(end_frame - start_frame)
    return write_wav(frames, channels=channels, sample_width=sample_width, sample_rate=sample_rate)


def normalize_audio_to_wav(audio_bytes: bytes, *, file_name: str | None = None, mime_type: str | None = None) -> bytes:
    if not audio_bytes:
        raise ValueError("audio_bytes must not be empty")
    try:
        analyze_wav(audio_bytes)
        return audio_bytes
    except ValueError:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("audio is not WAV and ffmpeg is not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        input_path = root / f"input{input_extension(file_name=file_name, mime_type=mime_type)}"
        output_path = root / "normalized.wav"
        input_path.write_bytes(audio_bytes)
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "unknown ffmpeg error"
            raise ValueError(f"ffmpeg could not decode audio: {detail}")
        normalized = output_path.read_bytes()
    analyze_wav(normalized)
    return normalized


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


def input_extension(*, file_name: str | None, mime_type: str | None) -> str:
    if file_name:
        suffix = Path(file_name).suffix.lower()
        if suffix:
            return suffix
    if mime_type:
        subtype = mime_type.split("/", 1)[-1].split(";", 1)[0].lower()
        if subtype == "mpeg":
            return ".mp3"
        if subtype:
            return f".{subtype}"
    return ".bin"
