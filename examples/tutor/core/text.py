from __future__ import annotations

import re


def strip_reasoning_for_context(text: str) -> str:
    text = text or ""
    text = re.sub(
        r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<think(?:ing)?\b[^>]*>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"</?think(?:ing)?\b[^>]*>", "", text, flags=re.IGNORECASE).strip()


def strip_think_tags(text: str) -> str:
    return strip_reasoning_for_context(text)


def compact_text(text: str, max_chars: int = 240) -> str:
    compact = " ".join(strip_reasoning_for_context(text).split())
    if not compact:
        return "(empty)"
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."
