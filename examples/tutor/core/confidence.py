from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from examples.common.openai_utils import TokenLogprob


@dataclass(frozen=True, slots=True)
class AnswerTokenConfidence:
    confidence: float
    mean_logprob: float | None
    token_count: int
    available: bool
    reason: str = ""


def _last_boxed_answer_span(text: str) -> tuple[int, int] | None:
    boxed_idx = text.rfind("\\boxed")
    fbox_idx = text.rfind("\\fbox")
    marker_idx = max(boxed_idx, fbox_idx)
    if marker_idx < 0:
        return None

    if marker_idx == boxed_idx and text.startswith("\\boxed ", marker_idx):
        start = marker_idx + len("\\boxed ")
        end = text.find("$", start)
        if end < 0:
            end = len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return (start, end) if start < end else None

    left_brace = text.find("{", marker_idx)
    if left_brace < 0:
        return None
    depth = 0
    for idx in range(left_brace, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                start = left_brace + 1
                return (start, idx) if start < idx else None
    return None


def _token_bytes(token_logprob: TokenLogprob) -> bytes:
    if token_logprob.bytes is not None:
        return bytes(token_logprob.bytes)
    return token_logprob.token.encode("utf-8")


def compute_answer_token_confidence(
    token_logprobs: Sequence[TokenLogprob],
) -> AnswerTokenConfidence:
    if not token_logprobs:
        return AnswerTokenConfidence(0.0, None, 0, False, "empty_output")

    chunks = [_token_bytes(item) for item in token_logprobs]
    try:
        decoded = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Student confidence received invalid UTF-8 token bytes."
        ) from exc
    answer_span = _last_boxed_answer_span(decoded)
    if answer_span is None:
        return AnswerTokenConfidence(0.0, None, 0, False, "missing_boxed_answer")

    answer_start, answer_end = answer_span
    answer_start_byte = len(decoded[:answer_start].encode("utf-8"))
    answer_end_byte = len(decoded[:answer_end].encode("utf-8"))
    selected_logprobs: list[float] = []
    token_start_byte = 0
    for item, chunk in zip(token_logprobs, chunks, strict=True):
        token_end_byte = token_start_byte + len(chunk)
        if token_end_byte > answer_start_byte and token_start_byte < answer_end_byte:
            selected_logprobs.append(float(item.logprob))
        token_start_byte = token_end_byte
    if not selected_logprobs:
        return AnswerTokenConfidence(0.0, None, 0, False, "missing_answer_tokens")
    if not all(math.isfinite(logprob) for logprob in selected_logprobs):
        raise ValueError("Student confidence received a non-finite output logprob.")

    mean_logprob = sum(selected_logprobs) / len(selected_logprobs)
    confidence = math.exp(min(0.0, mean_logprob))
    return AnswerTokenConfidence(
        confidence=max(0.0, min(1.0, confidence)),
        mean_logprob=mean_logprob,
        token_count=len(selected_logprobs),
        available=True,
    )
