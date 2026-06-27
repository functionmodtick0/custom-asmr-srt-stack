from __future__ import annotations

import io
import math
import re
import sys
import wave
from array import array

PCM16_MAX = 32767
ASR_TARGET_RMS_DBFS = -24.0
ASR_MAX_PEAK_DBFS = -3.0
ASR_MAX_GAIN = 4.0
JAPANESE_CHAR_CLASS = r"\u3040-\u30ff\u3400-\u9fff々〆〤ヶー"
JAPANESE_OR_PUNCTUATION_CLASS = JAPANESE_CHAR_CLASS + r"。、，,.！？!?…「」『』（）【】《》〈〉"


def clean_transcription_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    prefixes = ("Transcription:", "Transcript:", "文字起こし:", "書き起こし:")
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = strip_non_japanese_noise_edges(text)
    text = re.sub(fr"([{JAPANESE_CHAR_CLASS}])\s+([{JAPANESE_CHAR_CLASS}])", r"\1\2", text)
    text = re.sub(fr"\s+([{JAPANESE_OR_PUNCTUATION_CLASS}])", r"\1", text)
    text = re.sub(fr"([{JAPANESE_OR_PUNCTUATION_CLASS}])\s+", r"\1", text)
    return text.strip()


def strip_non_japanese_noise_edges(text: str) -> str:
    text = re.sub(fr"^[^{JAPANESE_OR_PUNCTUATION_CLASS}]+", "", text)
    return re.sub(fr"[^{JAPANESE_OR_PUNCTUATION_CLASS}]+$", "", text)


def prepare_audio_for_asr(audio_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        params = wav.getparams()
        frames = wav.readframes(wav.getnframes())

    if params.sampwidth != 2 or not frames:
        return audio_bytes

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return audio_bytes

    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak = max(abs(sample) for sample in samples)
    if rms <= 0 or peak <= 0:
        return audio_bytes

    target_rms = dbfs_to_pcm_amplitude(ASR_TARGET_RMS_DBFS)
    max_peak = dbfs_to_pcm_amplitude(ASR_MAX_PEAK_DBFS)
    gain = min(ASR_MAX_GAIN, target_rms / rms, max_peak / peak)
    if gain <= 1.01:
        return audio_bytes

    boosted = array("h", (clip_pcm16(round(sample * gain)) for sample in samples))
    if sys.byteorder != "little":
        boosted.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams(params)
        wav.writeframes(boosted.tobytes())
    return output.getvalue()


def dbfs_to_pcm_amplitude(dbfs: float) -> float:
    return PCM16_MAX * (10 ** (dbfs / 20))


def clip_pcm16(value: int) -> int:
    return max(-32768, min(32767, value))
