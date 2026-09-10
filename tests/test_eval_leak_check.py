import json

import pytest

from examples.tutor.core.callers import ApiAuxiliaryCaller, TextCallResult
from examples.tutor.scripts.eval_leak_check import (
    EvalLeakCheckMixin,
    parse_eval_leak_result,
)


@pytest.mark.parametrize("prefix", ["", "```json\n", r"Compare \frac{20}{3}. "])
def test_valid_verdict_ignores_math_braces(prefix):
    raw = prefix + json.dumps({"feedback": r"Compare \$4", "leaked": False})
    result = parse_eval_leak_result(raw)
    assert result.parse_error is None
    assert result.leaked is False
    assert result.raw_output == raw


@pytest.mark.parametrize(
    "raw",
    [
        r'{"feedback":"The answer is \$4", "leaked":false}',
        '{"feedback":"cut off',
        '{"feedback":"no", "leaked":"false"}',
        '{"feedback":"yes", "leaked":true} {"feedback":"no", "leaked":false}',
    ],
)
def test_invalid_or_ambiguous_verdict_stays_failed(raw):
    result = parse_eval_leak_result(raw)
    assert result.parse_error
    assert result.leaked is True


class FakeEval(EvalLeakCheckMixin):
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.leak_check_diagnostics = []

    async def _call_auxiliary_prompt(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.replies)


@pytest.mark.asyncio
async def test_failure_retries_only_judge_and_records_raw_outputs():
    bad = r'{"feedback":"\$4", "leaked":false}'
    good = '{"feedback":"No match", "leaked":false}'
    workflow = FakeEval([TextCallResult(bad, bad), TextCallResult(good, good)])
    caller = ApiAuxiliaryCaller(None, request_overrides={"temperature": 0.0})
    result = await workflow._run_rawbase_leak_check(
        "task", "4", "hint", aux_caller=caller
    )
    assert result.parse_error is None and not result.leaked
    assert len(workflow.calls) == 2
    assert "response_format" not in caller.request_overrides
    retry = workflow.calls[1]["aux_caller"].request_overrides
    assert retry["response_format"]["type"] == "json_schema"
    assert retry["temperature"] == 0.0
    assert workflow.calls[0]["user_prompt"] == workflow.calls[1]["user_prompt"]
    diagnostic = workflow.leak_check_diagnostics[0]
    assert diagnostic["teacher_action"] == "hint"
    assert diagnostic["attempts"][0]["raw_output"] == bad
    assert diagnostic["attempts"][0]["parse_error"]


@pytest.mark.asyncio
async def test_exhaustion_does_not_turn_failure_into_pass():
    workflow = FakeEval([TextCallResult("", "", "timeout")] * 3)
    result = await workflow._run_rawbase_leak_check(
        "task", "4", "hint", aux_caller=ApiAuxiliaryCaller(None)
    )
    assert result.leaked and result.parse_error == "timeout"
    assert len(workflow.calls) == 3
    assert len(workflow.leak_check_diagnostics[0]["attempts"]) == 3


@pytest.mark.asyncio
async def test_valid_initial_verdict_does_not_retry():
    good = '{"feedback":"Contains answer", "leaked":true}'
    workflow = FakeEval([TextCallResult(good, good)])
    result = await workflow._run_rawbase_leak_check(
        "task", "4", "hint", aux_caller=ApiAuxiliaryCaller(None)
    )
    assert result.leaked and result.parse_error is None
    assert len(workflow.calls) == 1
