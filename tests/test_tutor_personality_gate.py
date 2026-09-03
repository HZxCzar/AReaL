#!/usr/bin/env python3
"""The personality gate: parsing, sampling, complaints, couplings, and the arm.

A personality is one prompt. On a sampled teacher turn an auxiliary model either
answers its binary prompt or chooses among all six personalities plus NONE; a failed
match means the student does not answer and a complaint takes its slot. This checks
the parts of that which can be checked without a GPU or an endpoint.

What it deliberately does check, because each is a silent failure otherwise:

  - an unclean verdict ends as FAIL, not as PASS. Fail-open would disable the
    mechanism while every metric kept reporting compliance.
  - an unsampled turn is neither compliant nor gated. Counting it either way turns
    the sample rate itself into a compliance number.
  - the open gate contributes no name segment, so existing arms' cell names and the
    per-student metric series they feed are unchanged.
  - every personality cites a source, because the categories are learner types from
    prior work and the citation is what makes that claim checkable.

Run from the repo root with the venv and .env active.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import random
import sys
import tempfile
from types import SimpleNamespace

from examples.tutor.configs import (
    NO_PERSONALITY,
    TutorPersonalityConfig,
    TutorStudentAxesConfig,
    TutorStudentMaskConfig,
    TutorStudentModelConfig,
)
from examples.tutor.core.types import (
    PersonalityGateResult,
    PublicHistoryState,
    StudentTurnState,
    TurnTrace,
)
from examples.tutor.prompts import PERSONALITY_GATE_USER_TEMPLATE
from examples.tutor.workflow import (
    TutorAgentWorkflow,
    _parse_personality_classifier_reply,
    _parse_personality_gate_reply,
    load_personality_complaints,
    load_personality_prompts,
)

FAILURES: list[str] = []
REPO = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = REPO / "examples/tutor/prompt_pools/personality_prompts_v1.json"
COMPLAINTS = REPO / "examples/tutor/prompt_pools/personality_complaints_v1.json"


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}  {detail}")


def raises(fn, fragment: str) -> tuple[bool, str]:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the point is to inspect the message
        return fragment in str(exc), f"{type(exc).__name__}: {exc}"
    return False, "no exception"


def _student(**kwargs) -> dict:
    base = dict(
        name="s",
        base_url="http://localhost/v1",
        model="m",
        mode="text",
        mask=TutorStudentMaskConfig(mode="full"),
    )
    base.update(kwargs)
    return base


def _stub(**overrides) -> SimpleNamespace:
    """A stub carrying only what the gate methods read off self."""
    stub = SimpleNamespace(
        personality_prompts={
            "instrumental": {"source": "Nelson-Le Gall (1981)", "preference": "PREF"}
        },
        personality_complaints_bare=("bare one", "bare two"),
        personality_complaints_explain={"instrumental": ("explain one", "explain two")},
        personality_gate_sample_rate=1.0,
        personality_explain_ratio=1.0,
        personality_gate_retries=3,
        personality_active=True,
        prompt_pool_seed=0,
        _personality_fallback_rng=random.Random(0),
        student_model_runtimes={},
    )
    for key, value in overrides.items():
        setattr(stub, key, value)
    for method in (
        "_run_personality_gate",
        "_personality_complaint",
        "_personality_rng",
        "_personality_metrics",
    ):
        setattr(
            stub,
            method,
            getattr(TutorAgentWorkflow, method).__get__(stub, TutorAgentWorkflow),
        )
    # A staticmethod, so it needs no binding -- but it does have to be present, since
    # the metrics helper reads it off self.
    stub._student_metric_name = TutorAgentWorkflow._student_metric_name
    return stub


def _reply(text: str = "", error: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(text=text, raw_text=text, error=error)


def main() -> int:
    print("\n[1] the verdict parser is strict, and unclean means FAIL")
    passed, reason, err = _parse_personality_gate_reply(
        "<reasoning>compares two concrete choices</reasoning>"
        "<verdict>PASS</verdict>"
    )
    check("a clean XML PASS parses", passed and err is None, f"{passed} {err}")
    check("the XML reasoning is kept", reason == "compares two concrete choices")
    passed, reason, err = _parse_personality_gate_reply(
        r"<reasoning>uses \\sin and \\pi but not the requested method</reasoning>"
        "<verdict>FAIL</verdict>"
    )
    check(
        "LaTeX backslashes cannot break an XML FAIL",
        not passed and err is None,
        f"{passed} {err}",
    )
    passed, reason, err = _parse_personality_gate_reply(
        '{"reasoning": "asks a question", "verdict": "PASS"}'
    )
    check("a legacy JSON PASS parses", passed and err is None, f"{passed} {err}")
    check("the reasoning is kept", reason == "asks a question", reason)
    passed, _, err = _parse_personality_gate_reply(
        '{"reasoning": "states the step", "verdict": "FAIL"}'
    )
    check("a clean FAIL parses", not passed and err is None, f"{passed} {err}")
    passed, _, err = _parse_personality_gate_reply(
        '```json\n{"reasoning": "r", "verdict": "pass"}\n```'
    )
    check(
        "a fenced, lowercase verdict parses", passed and err is None, f"{passed} {err}"
    )
    passed, _, err = _parse_personality_gate_reply(
        'Sure! {"reasoning": "r", "verdict": "FAIL"} hope that helps'
    )
    check("an object inside prose parses", not passed and err is None, f"{err}")
    for label, text in (
        ("empty", ""),
        ("no object", "PASS"),
        ("invalid json", '{"verdict": PASS}'),
        ("missing verdict", '{"reasoning": "r"}'),
        ("other verdict", '{"reasoning": "r", "verdict": "MAYBE"}'),
    ):
        passed, _, err = _parse_personality_gate_reply(text)
        check(
            f"{label} is a parse error and not a pass",
            err is not None and not passed,
            f"{err}",
        )

    classifier_labels = (
        "feedback",
        "hinting",
        "instructing",
        "explaining",
        "modeling",
        "questioning",
        "NONE",
    )
    label, reason, err = _parse_personality_classifier_reply(
        '{"reasoning": "It mainly clarifies the concept.", "decision": "EXPLAINING"}',
        classifier_labels,
    )
    check(
        "a clean named classifier decision parses",
        label == "explaining" and reason and err is None,
        f"{label} {reason} {err}",
    )
    label, _, err = _parse_personality_classifier_reply(
        '{"reasoning": "", "decision": "EXPLAINING"}',
        classifier_labels,
    )
    check(
        "classifier JSON requires its reasoning",
        label is None and err is not None,
        str(err),
    )

    classifier_prompts = {
        name: {"source": "source", "preference": f"definition for {name}"}
        for name in classifier_labels
        if name != "NONE"
    }
    captured_json: dict[str, str] = {}

    async def classify_with_reasoning(**kwargs):
        captured_json.update(kwargs)
        return _reply(
            '{"reasoning": "It mainly clarifies the concept.", '
            '"decision": "EXPLAINING"}'
        )

    json_stub = _stub(
        personality_prompts=classifier_prompts,
        personality_gate_decision_mode="classifier",
        personality_gate_prompt_version="v2",
        _call_auxiliary_prompt=classify_with_reasoning,
    )
    json_result = asyncio.run(
        json_stub._run_personality_gate(
            "explaining",
            "TEACHER MESSAGE",
            task="TASK",
            turn_idx=2,
            previous_student_message="LAST STUDENT MESSAGE",
        )
    )
    check(
        "classifier routes by its named decision",
        json_result.passed
        and json_result.classification_label == "explaining"
        and bool(json_result.reason),
        str(json_result),
    )
    check(
        "classifier sees all choices and the last student message",
        all(
            name.upper() in captured_json.get("system_prompt", "")
            for name in classifier_labels
        )
        and "LAST STUDENT MESSAGE" in captured_json.get("user_prompt", ""),
        captured_json.get("system_prompt", "")[-200:],
    )

    print("\n[2] retries, then FAIL -- never fail-open")
    calls = {"n": 0}

    async def always_broken(**_kwargs):
        calls["n"] += 1
        return _reply(error="connection reset")

    stub = _stub(_call_auxiliary_prompt=always_broken)
    result = asyncio.run(
        stub._run_personality_gate("instrumental", "anything", task="TASK", turn_idx=1)
    )
    check("a broken check closes the gate", not result.passed, str(result.passed))
    check("the error is recorded", bool(result.error), str(result.error))
    check("it retried gate_retries times", calls["n"] == 3, str(calls["n"]))

    calls["n"] = 0

    async def garbage_then_verdict(**_kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            return _reply("I think it is fine")
        return _reply('{"reasoning": "r", "verdict": "PASS"}')

    stub = _stub(_call_auxiliary_prompt=garbage_then_verdict)
    result = asyncio.run(
        stub._run_personality_gate("instrumental", "anything", task="TASK", turn_idx=1)
    )
    check("a retry can recover", result.passed and result.error is None, str(result))
    check("attempts are counted", result.attempts == 2, str(result.attempts))

    print("\n[3] the open gate and the sampling rate")
    stub = _stub(_call_auxiliary_prompt=always_broken)
    check(
        "no personality means no gate at all",
        asyncio.run(stub._run_personality_gate("", "x", task="TASK", turn_idx=1))
        is None,
    )
    check(
        f"{NO_PERSONALITY!r} means no gate at all",
        asyncio.run(
            stub._run_personality_gate(NO_PERSONALITY, "x", task="TASK", turn_idx=1)
        )
        is None,
    )
    calls["n"] = 0
    stub = _stub(personality_gate_sample_rate=0.0, _call_auxiliary_prompt=always_broken)
    unsampled = [
        asyncio.run(
            stub._run_personality_gate("instrumental", "x", task="TASK", turn_idx=turn)
        )
        for turn in range(1, 6)
    ]
    check("rate 0 never calls the model", calls["n"] == 0, str(calls["n"]))
    check(
        "an unsampled turn is marked unsampled and does not gate",
        all(not r.sampled and r.passed for r in unsampled),
        str(unsampled[0]),
    )

    async def always_pass(**_kwargs):
        calls["n"] += 1
        return _reply('{"reasoning": "r", "verdict": "PASS"}')

    calls["n"] = 0
    stub = _stub(personality_gate_sample_rate=1.0, _call_auxiliary_prompt=always_pass)
    sampled = asyncio.run(
        stub._run_personality_gate("instrumental", "x", task="TASK", turn_idx=1)
    )
    check("rate 1 always calls", calls["n"] == 1, str(calls["n"]))
    check("a sampled turn is marked sampled", sampled.sampled, str(sampled))

    print("\n[4] the prompt carries the preference, the message, and the problem")
    captured: dict[str, str] = {}

    async def capture(**kwargs):
        captured.update(kwargs)
        return _reply('{"reasoning": "r", "verdict": "PASS"}')

    stub = _stub(_call_auxiliary_prompt=capture)
    asyncio.run(
        stub._run_personality_gate(
            "instrumental", "  TEACHER SAID THIS  ", task="THE PROBLEM TEXT", turn_idx=2
        )
    )
    user_prompt = captured.get("user_prompt", "")
    check("the preference is in the prompt", "PREF" in user_prompt)
    check("the teacher message is in the prompt", "TEACHER SAID THIS" in user_prompt)
    check(
        "the template shape is used",
        "<student_preference>" in user_prompt and "<teacher_message>" in user_prompt,
    )
    system_prompt = captured.get("system_prompt", "")
    check(
        "the problem reaches the system prompt",
        "TEACHER SAID THIS" not in system_prompt
        and "THE PROBLEM TEXT" in system_prompt,
        system_prompt[-120:],
    )
    check(
        "the template was formatted, not passed through",
        "{task}" not in system_prompt,
        system_prompt[-80:],
    )
    # Inside the object, not in the surrounding prose -- the prose says "give your
    # verdict" first, and what matters is the key order the model is asked to emit.
    skeleton = user_prompt[user_prompt.index('{"') :]
    check(
        "reasoning is requested before the verdict in the object",
        skeleton.index("reasoning") < skeleton.index("verdict"),
        skeleton.splitlines()[0] if skeleton else "",
    )

    print("\n[5] the complaint comes from the file, by ratio")
    stub = _stub(personality_explain_ratio=1.0)
    drawn = {stub._personality_complaint("instrumental", turn_idx=t) for t in range(40)}
    check(
        "ratio 1 always names the remedy",
        drawn <= {"explain one", "explain two"},
        str(drawn),
    )
    stub = _stub(personality_explain_ratio=0.0)
    drawn = {stub._personality_complaint("instrumental", turn_idx=t) for t in range(40)}
    check("ratio 0 always draws bare", drawn <= {"bare one", "bare two"}, str(drawn))
    stub = _stub(personality_explain_ratio=1.0)
    drawn = {stub._personality_complaint("unknown", turn_idx=t) for t in range(20)}
    check(
        "a personality with no explain list falls back to bare",
        drawn <= {"bare one", "bare two"},
        str(drawn),
    )

    print("\n[6] compliance is a rate over SAMPLED turns")

    def trace(turn_idx: int, result: PersonalityGateResult | None) -> TurnTrace:
        return TurnTrace(
            turn_idx=turn_idx,
            tutor_state=None,
            tutor_raw_output="",
            tutor_visible_output="",
            leaked=False,
            student_output="",
            judge_correct=False,
            judge_feedback="",
            reward=0.0,
            reward_components={},
            public_history_before=[],
            public_history_after=[],
            personality_gate_result=result,
        )

    def gate(passed: bool, sampled: bool = True, error: str | None = None):
        return PersonalityGateResult(
            raw_output="", passed=passed, reason="", error=error, sampled=sampled
        )

    runtime = SimpleNamespace(
        name="s-text-original-instrumental", personality="instrumental"
    )
    stub = _stub(student_model_runtimes={"s": runtime})
    traces = [
        trace(1, gate(True)),
        trace(2, gate(False, sampled=True)),
        trace(3, gate(True, sampled=False)),
        trace(4, gate(True, sampled=False)),
    ]
    metrics = stub._personality_metrics(traces, student_name=runtime.name)
    prefix = "personality/instrumental"
    check(
        "compliance divides by sampled turns only",
        abs(metrics[f"{prefix}/compliance"] - 0.5) < 1e-9,
        str(metrics.get(f"{prefix}/compliance")),
    )
    check(
        "gate_calls is the sampled count",
        metrics[f"{prefix}/gate_calls"] == 2.0,
        str(metrics.get(f"{prefix}/gate_calls")),
    )
    check(
        "gated turns are counted",
        metrics[f"{prefix}/gated_turns"] == 1.0,
        str(metrics.get(f"{prefix}/gated_turns")),
    )
    check(
        "turn 1 compliance is reported separately",
        metrics[f"{prefix}/compliance_turn1"] == 1.0,
        str(metrics.get(f"{prefix}/compliance_turn1")),
    )
    check(
        "an episode with a gated turn is not clean",
        metrics[f"{prefix}/clean_episode"] == 0.0,
        str(metrics.get(f"{prefix}/clean_episode")),
    )
    errored = stub._personality_metrics(
        [trace(1, gate(False, error="timeout"))], student_name=runtime.name
    )
    check(
        "gate errors are reported so an outage is not read as the teacher",
        errored[f"{prefix}/gate_error"] == 1.0,
        str(errored.get(f"{prefix}/gate_error")),
    )
    check(
        "no personality means no personality metrics",
        stub._personality_metrics([trace(1, None)], student_name=runtime.name) == {},
    )

    print("\n[7] a personality implies a text student with nothing masked")
    ok, detail = raises(
        lambda: TutorStudentModelConfig(
            **_student(mode="code", personality="instrumental")
        ),
        "requires mode 'text'",
    )
    check("code plus personality is refused", ok, detail)
    ok, detail = raises(
        lambda: TutorStudentModelConfig(
            **_student(
                mask=TutorStudentMaskConfig(mode="student_fade"),
                personality="instrumental",
            )
        ),
        "requires mask.mode 'full'",
    )
    check("a mask plus personality is refused", ok, detail)
    entry = TutorStudentModelConfig(**_student(personality="instrumental"))
    check("text plus unmasked is accepted", entry.personality == "instrumental")
    entry = TutorStudentModelConfig(**_student(mode="code", personality=NO_PERSONALITY))
    check("the open gate is allowed on any student", entry.mode == "code")

    print(
        "\n[8] expansion is a triple product, and the open gate is invisible in names"
    )
    template = TutorStudentModelConfig(**_student(name="qwen3-1.7b", weight=1.0))
    axes = TutorStudentAxesConfig(
        behaviors=["text"],
        informations={"original": TutorStudentMaskConfig(mode="full")},
        personalities=[NO_PERSONALITY, "instrumental", "executive"],
        template=template,
    )
    expanded = axes.expand()
    names = [entry.name for entry in expanded]
    check("one cell per personality", len(expanded) == 3, str(names))
    check(
        "the open gate adds no name segment",
        "qwen3-1.7b-text-original" in names,
        str(names),
    )
    check(
        "the others are named for their personality",
        {"qwen3-1.7b-text-original-instrumental", "qwen3-1.7b-text-original-executive"}
        <= set(names),
        str(names),
    )
    check(
        "the open-gate cell carries no personality",
        next(e for e in expanded if e.name == "qwen3-1.7b-text-original").personality
        == "",
    )
    check(
        "weight is split across the product",
        all(abs(entry.weight - 1.0 / 3.0) < 1e-9 for entry in expanded),
        str([entry.weight for entry in expanded]),
    )
    unchanged = TutorStudentAxesConfig(
        behaviors=["text", "code"],
        informations={"original": TutorStudentMaskConfig(mode="full")},
        template=template,
    ).expand()
    check(
        "a block that says nothing about personalities expands as before",
        [entry.name for entry in unchanged]
        == ["qwen3-1.7b-text-original", "qwen3-1.7b-code-original"],
        str([entry.name for entry in unchanged]),
    )
    ok, detail = raises(
        lambda: TutorStudentAxesConfig(
            behaviors=["text"],
            informations={"original": TutorStudentMaskConfig(mode="full")},
            personalities=["deep", "deep"],
            template=template,
        ).expand(),
        "duplicates",
    )
    check("duplicate personalities are refused", ok, detail)

    print("\n[9] the loaders demand a source, and the shipped files satisfy them")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "p.json"
        path.write_text(
            json.dumps({"personalities": {"x": {"preference": "p"}}}), encoding="utf-8"
        )
        ok, detail = raises(lambda: load_personality_prompts(str(path)), "no 'source'")
        check("a personality without a source is refused", ok, detail)
        path.write_text(
            json.dumps({"personalities": {"x": {"source": "s"}}}), encoding="utf-8"
        )
        ok, detail = raises(
            lambda: load_personality_prompts(str(path)), "empty preference"
        )
        check("a personality without a preference is refused", ok, detail)
        path.write_text(
            json.dumps(
                {"personalities": {NO_PERSONALITY: {"source": "s", "preference": "p"}}}
            ),
            encoding="utf-8",
        )
        ok, detail = raises(lambda: load_personality_prompts(str(path)), "open gate")
        check("the open gate may not have a prompt", ok, detail)

    check("the shipped prompt file exists", PROMPTS.is_file(), str(PROMPTS))
    check("the shipped complaint file exists", COMPLAINTS.is_file(), str(COMPLAINTS))
    prompts = load_personality_prompts(str(PROMPTS))
    bare, explain = load_personality_complaints(str(COMPLAINTS))
    # Not a fixed count: personalities are added as the literature turns them up, and
    # a test that hard-codes the number fails on every addition for no reason. What
    # matters is that each shipped one is complete, which the loop below checks.
    check("personalities ship", len(prompts) >= 6, str(sorted(prompts)))
    for name, entry in sorted(prompts.items()):
        check(f"{name}: cites a source", bool(entry["source"]), entry["source"][:40])
        check(
            f"{name}: states both a PASS and a FAIL clause",
            "PASS" in entry["preference"] and "FAIL" in entry["preference"],
        )
        check(f"{name}: has explain complaints", bool(explain.get(name)))
    check("bare complaints are shared and non-empty", len(bare) >= 5, str(len(bare)))
    check(
        "the preference fits the template without KeyError",
        "PREF"
        not in PERSONALITY_GATE_USER_TEMPLATE.format(
            preference="X", teacher_message="Y"
        ),
    )

    print("\n[10] the sampling and ratio bounds are enforced")
    for field, value in (
        ("gate_sample_rate", 1.5),
        ("gate_sample_rate", -0.1),
        ("explain_ratio", 2.0),
        ("gate_retries", 0),
        ("gated_turn_visibility", "student_only"),
    ):
        ok, detail = raises(
            lambda field=field, value=value: TutorPersonalityConfig(**{field: value}),
            field,
        )
        check(f"{field}={value} is refused", ok, detail)
    default = TutorPersonalityConfig()
    check("sampling defaults to 0.5", default.gate_sample_rate == 0.5)
    check(
        "failed turns are shared by default for backward compatibility",
        default.gated_turn_visibility == "shared",
    )
    teacher_only = TutorPersonalityConfig(gated_turn_visibility=" Teacher_Only ")
    check(
        "teacher_only visibility is accepted and normalized",
        teacher_only.gated_turn_visibility == "teacher_only",
    )
    classifier_mode = TutorPersonalityConfig(gate_decision_mode=" Classifier ")
    check(
        "classifier is accepted and normalized",
        classifier_mode.gate_decision_mode == "classifier",
    )
    check("the remedy is named by default", default.explain_ratio == 1.0)
    check("three retries by default", default.gate_retries == 3)

    print("\n[11] teacher_only removes failed exchanges from every student view")
    workflow = object.__new__(TutorAgentWorkflow)
    workflow.free_chat_enabled = True
    visible_before = PublicHistoryState(
        summary="visible round 1",
        turn_count=1,
        turns=[
            {"role": "teacher", "content": "visible teacher 1"},
            {"role": "student", "content": "visible student 1"},
        ],
    )
    failed = SimpleNamespace(
        turn_idx=2,
        student_state=SimpleNamespace(
            public_history=visible_before,
            previous_student_output="visible student 1",
        ),
        tutor_visible_output="hidden failed teacher",
        student_output="hidden rule complaint",
        personality_gated=True,
        public_history_after=[
            *visible_before.turns,
            {"role": "teacher", "content": "hidden failed teacher"},
            {"role": "student", "content": "hidden rule complaint"},
        ],
    )
    workflow.personality_gated_turn_visibility = "shared"
    shared = workflow._student_visible_history_after(failed)
    check(
        "shared mode preserves the old failed-turn transcript",
        [turn["content"] for turn in shared.turns][-2:]
        == ["hidden failed teacher", "hidden rule complaint"],
        str(shared.turns),
    )

    workflow.personality_gated_turn_visibility = "teacher_only"
    hidden = workflow._student_visible_history_after(failed)
    check(
        "teacher_only retains only the prefix before a failed gate",
        hidden.turns == visible_before.turns and hidden.turn_count == 1,
        str(hidden.turns),
    )
    (
        live_after_failure,
        live_output_after_failure,
        live_history_filtered,
    ) = workflow._advance_student_visible_history(
        complete_history_after=PublicHistoryState(
            summary="complete including failure",
            turn_count=2,
            turns=list(failed.public_history_after),
        ),
        student_visible_history=visible_before,
        previous_student_output="visible student 1",
        tutor_visible_output=failed.tutor_visible_output,
        current_student_output=failed.student_output,
        personality_gated=True,
        history_already_filtered=False,
    )
    check(
        "the live dialogue branch does not advance on a failed gate",
        live_after_failure is visible_before
        and live_output_after_failure == "visible student 1"
        and live_history_filtered,
    )
    workflow.student_mask_active = False
    next_real_student_messages = workflow._build_student_messages(
        StudentTurnState(
            task="TASK",
            public_history=live_after_failure,
            previous_student_output=live_output_after_failure,
            latest_tutor_visible_output="visible teacher 3",
        )
    )
    next_real_student_context = "\n".join(
        message["content"] for message in next_real_student_messages
    )
    check(
        "the next real student request contains neither failed-side message",
        "hidden failed teacher" not in next_real_student_context
        and "hidden rule complaint" not in next_real_student_context
        and "visible teacher 3" in next_real_student_context,
        next_real_student_context,
    )
    failed_anchor = workflow._turn_generalization_anchor(failed)
    check(
        "the final re-test anchor also excludes the failed exchange",
        failed_anchor.public_history.turns == visible_before.turns
        and failed_anchor.previous_student_output == "visible student 1",
        str(failed_anchor.public_history.turns),
    )

    passed_after_failure = SimpleNamespace(
        turn_idx=3,
        student_state=SimpleNamespace(
            public_history=live_after_failure,
            previous_student_output=live_output_after_failure,
        ),
        tutor_visible_output="visible teacher 3",
        student_output="visible student 3",
        personality_gated=False,
        public_history_after=[
            *failed.public_history_after,
            {"role": "teacher", "content": "visible teacher 3"},
            {"role": "student", "content": "visible student 3"},
        ],
    )
    resumed = workflow._student_visible_history_after(passed_after_failure)
    resumed_text = [turn["content"] for turn in resumed.turns]
    check(
        "a later passing turn resumes from the filtered prefix",
        resumed_text
        == [
            "visible teacher 1",
            "visible student 1",
            "visible teacher 3",
            "visible student 3",
        ],
        str(resumed_text),
    )
    check(
        "the student-visible turn count ignores the failed exchange",
        resumed.turn_count == 2,
        str(resumed.turn_count),
    )
    live_after_pass, live_output_after_pass, still_filtered = (
        workflow._advance_student_visible_history(
            complete_history_after=PublicHistoryState(
                summary="complete through turn 3",
                turn_count=3,
                turns=list(passed_after_failure.public_history_after),
            ),
            student_visible_history=live_after_failure,
            previous_student_output=live_output_after_failure,
            tutor_visible_output=passed_after_failure.tutor_visible_output,
            current_student_output=passed_after_failure.student_output,
            personality_gated=False,
            history_already_filtered=live_history_filtered,
        )
    )
    check(
        "the next real student call resumes on the filtered branch",
        live_after_pass.turns == resumed.turns
        and live_output_after_pass == "visible student 3"
        and still_filtered,
        str(live_after_pass.turns),
    )
    first_pass = SimpleNamespace(
        personality_gated=False,
        tutor_visible_output="visible teacher 1",
        student_output="visible student 1",
    )
    filtered_cross_eval = workflow._cross_eval_transcript(
        [first_pass, failed, passed_after_failure],
        hide_personality_gated_turns=True,
    )
    check(
        "cross-eval re-tests use the same filtered transcript",
        [turn["content"] for turn in filtered_cross_eval]
        == [
            "visible teacher 1",
            "visible student 1",
            "visible teacher 3",
            "visible student 3",
        ],
        str(filtered_cross_eval),
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES[:6])}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
