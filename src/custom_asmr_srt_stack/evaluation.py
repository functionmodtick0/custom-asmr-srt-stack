from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from custom_asmr_srt_stack.models import MasterDocument, Segment
from custom_asmr_srt_stack.srt import parse_srt

EVAL_FORMAT = "custom-asmr-eval-v1"


def load_transcript_document(path: Path, *, source_language: str = "ja") -> MasterDocument:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".srt":
        return parse_srt(content, source_language=source_language, source_file=path.name)
    return MasterDocument.from_json(json.loads(content))


def evaluate_transcripts(reference: MasterDocument, candidate: MasterDocument) -> dict[str, Any]:
    reference_speech = speech_segments(reference)
    candidate_speech = speech_segments(candidate)
    reference_text = normalize_for_cer("".join(segment.text for segment in reference_speech))
    candidate_text = normalize_for_cer("".join(segment.text for segment in candidate_speech))
    distance = levenshtein_distance(reference_text, candidate_text)
    reference_chars = len(reference_text)
    paired = list(zip(reference_speech, candidate_speech))
    timing_errors = timing_error_summary(paired)
    channel_summary = channel_accuracy_summary(paired)
    review_count = sum(1 for segment in candidate.segments if segment.needs_review)

    return {
        "format": EVAL_FORMAT,
        "reference_segments": len(reference_speech),
        "candidate_segments": len(candidate_speech),
        "text": {
            "cer": 0.0 if reference_chars == 0 and len(candidate_text) == 0 else distance / max(1, reference_chars),
            "edit_distance": distance,
            "reference_characters": reference_chars,
            "candidate_characters": len(candidate_text),
        },
        "timing": timing_errors,
        "channel": channel_summary,
        "review": {
            "candidate_review_count": review_count,
            "candidate_review_ratio": review_count / max(1, len(candidate.segments)),
        },
    }


def speech_segments(master: MasterDocument) -> tuple[Segment, ...]:
    return tuple(segment for segment in master.segments if segment.kind == "speech" and segment.text)


def normalize_for_cer(text: str) -> str:
    return re.sub(r"\s+", "", text)


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            deletion = previous[right_index] + 1
            insertion = current[right_index - 1] + 1
            substitution = previous[right_index - 1] + (0 if left_char == right_char else 1)
            current.append(min(deletion, insertion, substitution))
        previous = current
    return previous[-1]


def timing_error_summary(paired_segments: list[tuple[Segment, Segment]]) -> dict[str, Any]:
    if not paired_segments:
        return {
            "paired_segments": 0,
            "mean_start_error_ms": None,
            "mean_end_error_ms": None,
            "mean_boundary_error_ms": None,
        }

    start_errors = [abs(reference.start_ms - candidate.start_ms) for reference, candidate in paired_segments]
    end_errors = [abs(reference.end_ms - candidate.end_ms) for reference, candidate in paired_segments]
    return {
        "paired_segments": len(paired_segments),
        "mean_start_error_ms": mean(start_errors),
        "mean_end_error_ms": mean(end_errors),
        "mean_boundary_error_ms": mean(start_errors + end_errors),
    }


def channel_accuracy_summary(paired_segments: list[tuple[Segment, Segment]]) -> dict[str, Any]:
    comparable = [
        (reference, candidate)
        for reference, candidate in paired_segments
        if reference.channel in {"L", "R"} and candidate.channel in {"L", "R"}
    ]
    if not comparable:
        return {
            "comparable_segments": 0,
            "accuracy": None,
        }
    correct = sum(1 for reference, candidate in comparable if reference.channel == candidate.channel)
    return {
        "comparable_segments": len(comparable),
        "accuracy": correct / len(comparable),
    }


def mean(values: list[int]) -> float:
    return sum(values) / len(values)
