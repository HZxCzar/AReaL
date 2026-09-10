"""Offline-evaluation-only leak JSON recovery and request diagnostics."""

from __future__ import annotations

import json
from typing import Any

from examples.tutor.core.callers import ApiAuxiliaryCaller
from examples.tutor.core.parsers import parse_leak_check_result
from examples.tutor.core.text import strip_reasoning_for_context
from examples.tutor.core.types import LeakCheckResult
from examples.tutor.prompts import (
    RAWBASE_LEAK_CHECK_SYSTEM_PROMPT,
    RAWBASE_LEAK_CHECK_USER_TEMPLATE,
    render_prompt,
)

LEAK_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "leak_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "feedback": {"type": "string"},
                "leaked": {"type": "boolean"},
            },
            "required": ["feedback", "leaked"],
            "additionalProperties": False,
        },
    },
}


def parse_eval_leak_result(text: str) -> LeakCheckResult:
    """Accept one complete verdict; never guess escapes or a boolean value."""
    result = parse_leak_check_result(text)
    if result.parse_error is None:
        return result
    decoder = json.JSONDecoder()
    candidates = []
    offset = 0
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            offset = start + 1
            continue
        offset = end
        if isinstance(value, dict) and "leaked" in value:
            candidates.append(value)
    if len(candidates) == 1:
        recovered = parse_leak_check_result(json.dumps(candidates[0]))
        if recovered.parse_error is None:
            recovered.raw_output = text
            recovered.raw_result["eval_json_object_recovered"] = True
            return recovered
    return result


class EvalLeakCheckMixin:
    """Used only by RecordingTutorWorkflow, never the training workflow."""

    async def _run_rawbase_leak_check(
        self,
        task: str,
        ground_truth: str,
        teacher_action: str,
        *,
        aux_caller: Any = None,
    ) -> LeakCheckResult:
        prompt = render_prompt(
            RAWBASE_LEAK_CHECK_USER_TEMPLATE,
            ground_truth=ground_truth,
            teacher_action=strip_reasoning_for_context(teacher_action),
        )
        caller = aux_caller or self._make_auxiliary_caller(engine=None)
        diagnostic = {
            "task": task,
            "ground_truth": ground_truth,
            "teacher_action": teacher_action,
            "messages": [
                {"role": "system", "content": RAWBASE_LEAK_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "attempts": [],
        }
        # First request is unchanged. Only failures get at most two local,
        # schema-constrained retries, without regenerating teacher/student turns.
        for attempt in range(3):
            active_caller = caller
            if attempt:
                if not isinstance(caller, ApiAuxiliaryCaller):
                    break
                active_caller = ApiAuxiliaryCaller(
                    caller.caller,
                    request_overrides={
                        **caller.request_overrides,
                        "response_format": LEAK_RESPONSE_FORMAT,
                    },
                )
            reply = await self._call_auxiliary_prompt(
                system_prompt=RAWBASE_LEAK_CHECK_SYSTEM_PROMPT,
                user_prompt=prompt,
                aux_caller=active_caller,
                rid_prefix="rawbase-leak-check",
            )
            result = parse_eval_leak_result(reply.text)
            if reply.error:
                result = LeakCheckResult(
                    raw_output=reply.raw_text,
                    leaked=True,
                    feedback=f"Leak-check call failed: {reply.error}",
                    parse_error=reply.error,
                    raw_result={},
                )
            diagnostic["attempts"].append(
                {
                    "attempt": attempt + 1,
                    "schema_constrained": bool(attempt),
                    "raw_output": reply.raw_text,
                    "parsed_input": reply.text,
                    "call_error": reply.error,
                    "parse_error": result.parse_error,
                    "leaked": result.leaked,
                    "feedback": result.feedback,
                }
            )
            if result.parse_error is None:
                break
        result.raw_result["method"] = "rawbase_llm"
        result.raw_result["eval_leak_attempts"] = len(diagnostic["attempts"])
        self.leak_check_diagnostics.append(diagnostic)
        return result
