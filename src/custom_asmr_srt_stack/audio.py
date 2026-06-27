from __future__ import annotations

import io
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from array import array
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


def speech_intervals_by_energy(
    audio_bytes: bytes,
    *,
    threshold_dbfs: float = -48.0,
    window_ms: int = 100,
    min_silence_ms: int = 800,
    min_speech_ms: int = 200,
    pad_ms: int = 400,
) -> list[dict[str, int]]:
    info = analyze_wav(audio_bytes)
    if info.duration_ms == 0:
        return []
    if info.sample_width != 2:
        return [{"index": 0, "start_ms": 0, "end_ms": info.duration_ms}]
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if min_silence_ms < 0:
        raise ValueError("min_silence_ms must be non-negative")
    if min_speech_ms < 0:
        raise ValueError("min_speech_ms must be non-negative")
    if pad_ms < 0:
        raise ValueError("pad_ms must be non-negative")

    with open_wave(audio_bytes) as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        frames = wav.readframes(frame_count)

    window_frames = max(1, round(sample_rate * (window_ms / 1000)))
    bytes_per_frame = channels * info.sample_width
    threshold = dbfs_to_pcm16_amplitude(threshold_dbfs)
    speech_ranges: list[tuple[int, int]] = []
    for start_frame in range(0, frame_count, window_frames):
        end_frame = min(frame_count, start_frame + window_frames)
        chunk = frames[start_frame * bytes_per_frame : end_frame * bytes_per_frame]
        if pcm16_rms(chunk) >= threshold:
            speech_ranges.append((round(start_frame / sample_rate * 1000), round(end_frame / sample_rate * 1000)))

    if not speech_ranges:
        return [{"index": 0, "start_ms": 0, "end_ms": info.duration_ms}]

    merged: list[tuple[int, int]] = []
    current_start, current_end = speech_ranges[0]
    for start_ms, end_ms in speech_ranges[1:]:
        if start_ms - current_end <= min_silence_ms:
            current_end = end_ms
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start_ms, end_ms
    merged.append((current_start, current_end))

    padded: list[dict[str, int]] = []
    for start_ms, end_ms in merged:
        if end_ms - start_ms < min_speech_ms:
            continue
        padded.append(
            {
                "index": len(padded),
                "start_ms": max(0, start_ms - pad_ms),
                "end_ms": min(info.duration_ms, end_ms + pad_ms),
            }
        )
    return padded or [{"index": 0, "start_ms": 0, "end_ms": info.duration_ms}]


def pcm16_rms(frames: bytes) -> float:
    if not frames:
        return 0.0
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) == 0:
        return 0.0
    return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5


def dbfs_to_pcm16_amplitude(dbfs: float) -> float:
    return (2**15 - 1) * (10 ** (dbfs / 20))


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
