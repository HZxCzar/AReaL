from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.tutor.core.text import strip_reasoning_for_context
from examples.tutor.prompts import (
    PERSONALITY_GATE_V3_NO_LAST_STUDENT_MESSAGE,
    PERSONALITY_GATE_V3_SYSTEM_PROMPT,
    PERSONALITY_GATE_V3_USER_TEMPLATE,
)

NO_PREFERENCE = "none"


@dataclass(slots=True)
class PreferenceGateDecision:
    preference: str
    passed: bool
    reasoning: str
    raw_output: str
    error: str | None
    attempts: int
    complaint: str = ""


def _read_json(path: str, *, label: str) -> dict[str, Any]:
    file_path = Path(path)
    if not path or not file_path.is_file():
        raise ValueError(f"{label} file not found: {file_path}")
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {file_path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {file_path}")
    return value


def load_preferences(path: str) -> dict[str, str]:
    payload = _read_json(path, label="preference prompt")
    entries = payload.get("personalities")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("preference prompt JSON needs a non-empty personalities object")
    result: dict[str, str] = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict) or not str(entry.get("preference") or "").strip():
            raise ValueError(f"preference {name!r} has no preference text")
        result[str(name)] = str(entry["preference"]).strip()
    return result


def load_complaints(path: str) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    payload = _read_json(path, label="preference complaint")

    def clean(value: Any, *, label: str) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ValueError(f"{label} must be an array")
        lines = tuple(str(item).strip() for item in value if str(item).strip())
        if not lines:
            raise ValueError(f"{label} must not be empty")
        return lines

    bare = clean(payload.get("bare"), label="complaints.bare")
    explained = payload.get("explain")
    if not isinstance(explained, dict):
        raise ValueError("complaints.explain must be an object")
    return bare, {
        str(name): clean(lines, label=f"complaints.explain.{name}")
        for name, lines in explained.items()
    }


def parse_gate_reply(text: str) -> tuple[bool, str, str | None]:
    """Parse the same V3 binary XML contract as the tutor preference gate."""

    body = strip_reasoning_for_context(str(text or "")).strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
        body = re.sub(r"\n?```$", "", body).strip()
    match = re.fullmatch(
        r"<reasoning>(.*?)</reasoning>\s*<verdict>\s*([^<]*?)\s*</verdict>",
        body,
        flags=re.DOTALL,
    )
    if match is None:
        return False, "", "preference gate reply was not valid tagged XML"
    reasoning = match.group(1).strip()
    verdict = match.group(2).strip().upper()
    if verdict == "PASS":
        return True, reasoning, None
    if verdict == "FAIL":
        return False, reasoning, None
    return False, reasoning, f"preference gate verdict was {verdict!r}, not PASS/FAIL"


class PreferenceGate:
    """Evaluation-only V3 gate plus the exact shared scripted complaints."""

    def __init__(
        self,
        *,
        prompts_path: str,
        complaints_path: str,
        retries: int,
        explain_ratio: float,
        seed: int,
    ) -> None:
        self.preferences = load_preferences(prompts_path)
        self.bare_complaints, self.explained_complaints = load_complaints(
            complaints_path
        )
        self.retries = int(retries)
        self.explain_ratio = float(explain_ratio)
        self.seed = int(seed)

    def validate_names(self, names: list[str]) -> None:
        missing = sorted(
            name
            for name in names
            if name != NO_PREFERENCE and name not in self.preferences
        )
        missing_complaints = sorted(
            name
            for name in names
            if name != NO_PREFERENCE and name not in self.explained_complaints
        )
        if missing:
            raise ValueError(f"unknown evaluation preferences: {missing}")
        if self.explain_ratio > 0.0 and missing_complaints:
            raise ValueError(
                f"missing explained complaints for preferences: {missing_complaints}"
            )

    def complaint(
        self,
        *,
        preference: str,
        problem: str,
        conversation_type: str,
        turn_idx: int,
    ) -> str:
        rng = random.Random(
            f"{self.seed}:{problem}:{conversation_type}:{preference}:{turn_idx}"
        )
        explained = self.explained_complaints.get(preference, ())
        if explained and rng.random() < self.explain_ratio:
            return rng.choice(explained)
        return rng.choice(self.bare_complaints)

    async def judge(
        self,
        *,
        client: Any,
        preference: str,
        problem: str,
        last_student_message: str | None,
        teacher_message: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
    ) -> PreferenceGateDecision:
        if preference == NO_PREFERENCE:
            raise ValueError("the open student must not call the preference gate")
        definition = self.preferences[preference]
        user_prompt = PERSONALITY_GATE_V3_USER_TEMPLATE.format(
            preference=definition,
            task=problem.strip(),
            last_student_message=(
                str(last_student_message or "").strip()
                or PERSONALITY_GATE_V3_NO_LAST_STUDENT_MESSAGE
            ),
            teacher_message=teacher_message.strip(),
        )
        raw_output = ""
        last_error = "preference gate produced no verdict"
        for attempt in range(1, self.retries + 1):
            try:
                outputs = await client.generate(
                    [
                        {
                            "role": "system",
                            "content": PERSONALITY_GATE_V3_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    n=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
                raw_output = outputs[0] if outputs else ""
            except Exception as exc:  # fail closed, exactly as the tutor gate
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            passed, reasoning, error = parse_gate_reply(raw_output)
            if error is None:
                return PreferenceGateDecision(
                    preference=preference,
                    passed=passed,
                    reasoning=reasoning,
                    raw_output=raw_output,
                    error=None,
                    attempts=attempt,
                )
            last_error = error
        return PreferenceGateDecision(
            preference=preference,
            passed=False,
            reasoning="",
            raw_output=raw_output,
            error=last_error,
            attempts=self.retries,
        )
