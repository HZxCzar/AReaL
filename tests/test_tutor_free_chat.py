#!/usr/bin/env python3
"""Check the free-chat rollout: prompts, message layout, and the guard rails.

The prompt-assembly checks build the workflow with ``object.__new__`` and set only
the attributes the methods under test read. That keeps them hermetic and fast
while still exercising the real methods -- these are pure string assembly, so
there is nothing a full construction would add.

The guard-rail checks go through the real ``__init__``, because a
misconfiguration that only surfaces after 40 minutes of training is the thing
they exist to prevent.

Run from the repo root with the venv and .env active, and PYTHONPATH set to the
repo root (run_official.sh does this; a bare `python tests/...` picks up whichever
worktree the editable install points at).
"""
from __future__ import annotations

import asyncio

import sys
from dataclasses import asdict

from areal.api.cli_args import load_expr_config
from examples.tutor.configs import TutorConfig
from examples.tutor.core.types import (
    PublicHistoryState,
    StudentTurnState,
    TeacherPreSolveResult,
    TutorPrivateFeedback,
    TutorTurnState,
)
from examples.tutor.prompts import FREE_CHAT_STUDENT_SYSTEM_PROMPT
from examples.tutor.workflow import TutorAgentWorkflow

FAILURES: list[str] = []
BASE = "examples/tutor/configs/math/0810"
ALLOCATIONS = ("2gpu", "4gpu")
ARMS = ("base", "leak-reward", "leak-terminate")
# Everything a 4gpu arm is allowed to differ from its 2gpu twin by. `ref` is in
# here because its backend follows ${actor.backend}; it is checked separately.
# Anything else differing means the allocation file changed the experiment rather
# than the hardware. Every remaining field is compared with the trial name
# normalised out, since half of them interpolate it into a path.
ALLOCATION_ONLY = {"trial_name", "cluster", "rollout", "actor", "ref",
                   "stats_logger"}
TASK = "Let f(x) = x^2 - 4x + 7. What is the minimum value of f?"
GROUND_TRUTH = "3"


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def load(allocation: str, name: str) -> TutorConfig:
    config, _ = load_expr_config(
        ["--config", f"{BASE}/{allocation}/{name}.yaml"], TutorConfig
    )
    return config


def _normalize(value, trial_name: str):
    """Replace the trial name wherever it was interpolated into a value.

    saver, recover, evaluator and debug_trace_dir all carry ${trial_name}, so two
    otherwise identical arms differ everywhere the name appears. Normalising it
    away leaves a comparison that still fails on a real setting change.
    """
    if isinstance(value, dict):
        return {key: _normalize(item, trial_name) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item, trial_name) for item in value]
    if isinstance(value, str):
        return value.replace(trial_name, "<TRIAL>")
    return value


def make_workflow(**attrs: object) -> TutorAgentWorkflow:
    workflow = object.__new__(TutorAgentWorkflow)
    defaults = {
        "free_chat_enabled": True,
        "free_chat_budget": 5,
        "max_turns": 5,
        "enable_thinking": False,
        "teacher_show_ground_truth": False,
        "teacher_anti_leak_instruction_enabled": True,
        "teacher_adaptive_instruction_enabled": False,
        "teacher_pre_enabled": False,
        "teacher_history_tags": "stripped",
        "dataset_type": "math",
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(workflow, key, value)
    return workflow


def tutor_state(
    *,
    turns: list[dict[str, str]] | None = None,
    turn_idx: int = 1,
    pre_solve: TeacherPreSolveResult | None = None,
) -> TutorTurnState:
    return TutorTurnState(
        task=TASK,
        ground_truth=GROUND_TRUTH,
        public_history=PublicHistoryState(turns=list(turns or [])),
        previous_tutor_visible_output="",
        previous_feedback=TutorPrivateFeedback(),
        turn_idx=turn_idx,
        max_turns=5,
        teacher_pre_solve_result=pre_solve,
    )


def main() -> int:
    print("\n[1] every arm loads and agrees with itself")
    loaded = {
        (allocation, name): load(allocation, name)
        for allocation in ALLOCATIONS
        for name in ARMS
    }
    for (allocation, name), config in loaded.items():
        prefix = f"{allocation}/{name}:"
        check(f"{prefix} free_chat.enabled", config.free_chat.enabled)
        check(f"{prefix} budget == max_turns == 5",
              config.free_chat.budget == 5 and config.max_turns == 5,
              f"budget={config.free_chat.budget} max_turns={config.max_turns}")
        check(f"{prefix} max_turn_penalty is 0",
              config.reward.max_turn_penalty == 0.0,
              f"got {config.reward.max_turn_penalty}")
        check(f"{prefix} retest is the reward",
              config.student_generalize.retest_original
              and config.student_generalize.retest_reward > 0.0
              and config.student_generalize.enabled)
        check(f"{prefix} k4 g8",
              config.student_generalize.replays == 4
              and config.gconfig.n_samples == 8)
        check(f"{prefix} no transfer levels",
              not config.student_generalize.level1_enabled
              and not config.student_generalize.level2_enabled)
        # Not a style preference. Under 'stripped' the teacher is handed its own
        # past turns with the tags removed, imitates them, and the malformed rate
        # climbs with depth -- 6.3% to 41.7% over depths 1-5 measured on this
        # rollout. A fixed budget puts every episode in that tail.
        check(f"{prefix} teacher history keeps its tag skeleton",
              config.teacher_history_tags == "masked",
              f"got {config.teacher_history_tags!r}")

    print("\n[2] the leak arms differ in exactly one key, at both allocations")
    for allocation in ALLOCATIONS:
        reward_arm = loaded[(allocation, "leak-reward")]
        terminate_arm = loaded[(allocation, "leak-terminate")]
        check(f"{allocation}: leak-reward is reward_only",
              reward_arm.leak_handling_mode == "reward_only")
        check(f"{allocation}: leak-terminate is terminate",
              terminate_arm.leak_handling_mode == "terminate")
        differing = [
            field
            for field in ("max_turns", "format_handling_mode", "seed",
                          "teacher_anti_leak_instruction_enabled")
            if getattr(reward_arm, field) != getattr(terminate_arm, field)
        ]
        check(f"{allocation}: nothing else differs", differing == [], str(differing))
        check(f"{allocation}: same reward settings",
              asdict(reward_arm.reward) == asdict(terminate_arm.reward))
        check(f"{allocation}: same student_generalize settings",
              asdict(reward_arm.student_generalize)
              == asdict(terminate_arm.student_generalize))
        check(f"{allocation}: distinct trial names",
              reward_arm.trial_name != terminate_arm.trial_name)

    print("\n[3] 4gpu changes the allocation and nothing else")
    for name in ARMS:
        two, four = loaded[("2gpu", name)], loaded[("4gpu", name)]
        check(f"{name}: 4gpu asks for 4 GPUs",
              four.cluster.n_gpus_per_node == 4
              and two.cluster.n_gpus_per_node == 2,
              f"2gpu={two.cluster.n_gpus_per_node} 4gpu={four.cluster.n_gpus_per_node}")
        check(f"{name}: 4gpu shards generation and the actor",
              four.rollout.backend == "sglang:d2p1t1"
              and four.actor.backend == "fsdp:d2p1t1",
              f"{four.rollout.backend} / {four.actor.backend}")
        check(f"{name}: the ref model follows the actor's sharding",
              four.ref.backend == four.actor.backend
              and two.ref.backend == two.actor.backend,
              f"{four.ref.backend} vs {four.actor.backend}")
        two_fields = _normalize(asdict(two), two.trial_name)
        four_fields = _normalize(asdict(four), four.trial_name)
        drifted = sorted(
            key
            for key in two_fields
            if key not in ALLOCATION_ONLY and two_fields[key] != four_fields[key]
        )
        check(f"{name}: nothing outside the allocation drifted",
              drifted == [], str(drifted))
        check(f"{name}: trial names cannot collide across allocations",
              two.trial_name != four.trial_name,
              f"both {two.trial_name}")

    print("\n[4] the teacher opens cold, with the budget and the task")
    workflow = make_workflow()
    messages = workflow._build_tutor_messages(tutor_state())
    check("turn 1 is a single system message",
          len(messages) == 1 and messages[0]["role"] == "system",
          f"got {[m['role'] for m in messages]}")
    system = messages[0]["content"]
    check("states the budget", "You have 5 turn budgets" in system)
    check("carries the task", TASK in system)
    check("says the student is tested alone afterwards",
          "solve the problem from scratch" in system)
    check("carries the output format contract",
          "<reasoning>" in system and "<output>" in system)
    check("carries the anti-leak clause",
          "Do not reveal the problem's answer" in system)
    check("withholds the ground truth", GROUND_TRUTH not in system.replace(TASK, ""))
    check("training prompt == rollout prompt",
          workflow._teacher_system_for_state(tutor_state(), clean=True) == system)

    print("\n[5] masked history hands the teacher its own tags back")
    masked = make_workflow(teacher_history_tags="masked")
    stripped = make_workflow(teacher_history_tags="stripped")
    history = [
        {"role": "teacher", "content": "What does Vieta give you?"},
        {"role": "student", "content": "a + b = m and ab = 2."},
    ]
    own_turn_masked = masked._build_tutor_messages(
        tutor_state(turns=history, turn_idx=2)
    )[1]
    own_turn_stripped = stripped._build_tutor_messages(
        tutor_state(turns=history, turn_idx=2)
    )[1]
    check("the teacher's own turn comes back as assistant",
          own_turn_masked["role"] == "assistant"
          and own_turn_stripped["role"] == "assistant")
    check("masked restores the tag skeleton",
          "<reasoning>" in own_turn_masked["content"]
          and "<output>" in own_turn_masked["content"],
          own_turn_masked["content"][:80])
    check("masked hides the earlier private reasoning",
          "omitted from this transcript" in own_turn_masked["content"])
    check("masked keeps the visible text intact",
          "What does Vieta give you?" in own_turn_masked["content"])
    check("stripped is the bare text it used to be",
          own_turn_stripped["content"].strip() == "What does Vieta give you?",
          own_turn_stripped["content"][:80])
    check("the student never sees the tag protocol either way",
          all(
              "<reasoning>" not in message["content"]
              for message in masked._build_student_messages(
                  StudentTurnState(
                      task=TASK,
                      public_history=PublicHistoryState(turns=history),
                      previous_student_output="a + b = m and ab = 2.",
                      latest_tutor_visible_output="Now substitute.",
                  )
              )
          ))

    print("\n[6] the switchable blocks are switchable, in order")
    plain = make_workflow(teacher_anti_leak_instruction_enabled=False)
    check("anti-leak off removes it",
          "Do not reveal the problem's answer"
          not in plain._free_chat_teacher_system(tutor_state()))
    thinking = make_workflow(enable_thinking=True)
    check("enable_thinking removes the tag contract",
          "<reasoning>" not in thinking._free_chat_teacher_system(tutor_state()))
    with_draft = make_workflow(teacher_pre_enabled=True)
    drafted = with_draft._free_chat_teacher_system(
        tutor_state(
            pre_solve=TeacherPreSolveResult(
                enabled=True, mode="filter_solver", accepted=True,
                raw_output="Complete the square: f(x) = (x-2)^2 + 3.",
            )
        )
    )
    check("accepted pre-solve draft is appended",
          "You've solved this problem" in drafted
          and "Complete the square" in drafted)
    check("the draft comes last",
          drafted.index("You've solved this problem")
          > drafted.index("Do not reveal the problem's answer"))
    rejected = with_draft._free_chat_teacher_system(
        tutor_state(
            pre_solve=TeacherPreSolveResult(
                enabled=True, mode="filter_solver", accepted=False,
                raw_output="nonsense",
            )
        )
    )
    check("a rejected pre-solve is not appended",
          "You've solved this problem" not in rejected)

    print("\n[7] the student is told nothing")
    student_messages = workflow._build_student_messages(
        StudentTurnState(
            task=TASK,
            public_history=PublicHistoryState(turns=[]),
            previous_student_output="",
            latest_tutor_visible_output="Let's start. What does the graph look like?",
        )
    )
    student_system = student_messages[0]["content"]
    check("system prompt is the one-liner, verbatim",
          student_system == FREE_CHAT_STUDENT_SYSTEM_PROMPT,
          repr(student_system))
    check("the task is not in the student's prompt", TASK not in student_system)
    check("no mention of math", "math" not in student_system.lower())
    check("the teacher's opening is the first user turn",
          [m["role"] for m in student_messages] == ["system", "user"]
          and "What does the graph look like?" in student_messages[1]["content"])

    print("\n[8] the re-test replays the conversation and then shows the task")
    conversation = [
        {"role": "teacher", "content": "What happens at the vertex?"},
        {"role": "student", "content": "I am not sure how to find it."},
    ]
    probe = workflow._build_student_probe_messages(
        episode_artifact=_episode(conversation),
        anchor=_anchor(conversation),
        level="original",
        transfer_task="",
    )
    check("same one-line system prompt as the conversation",
          probe[0]["content"] == FREE_CHAT_STUDENT_SYSTEM_PROMPT)
    check("the conversation is replayed with roles flipped for the student",
          [m["role"] for m in probe] == ["system", "user", "assistant", "user"],
          f"got {[m['role'] for m in probe]}")
    final = probe[-1]["content"]
    check("final turn asks for a solution from scratch",
          "After these conversations, try to solve the problem from scratch:" in final)
    check("final turn is where the task first appears", TASK in final)
    check("final turn asks for a boxed answer", "\\boxed{}" in final)

    print("\n[9] the guard rails fire at startup, not at step 40")
    base_kwargs = dict(
        dataset_type="math",
        answer_scorer="math",
        max_turns=5,
        free_chat={"enabled": True, "budget": 5},
        student_generalize_enabled=True,
        student_generalize_retest_original=True,
        student_generalize_retest_reward=1.0,
        max_turn_penalty=0.0,
        leak_handling_mode="reward_only",
    )
    for label, override in (
        ("student_generalize disabled", {"student_generalize_enabled": False}),
        ("retest_original off", {"student_generalize_retest_original": False}),
        ("retest_reward 0", {"student_generalize_retest_reward": 0.0}),
        ("max_turn_penalty non-zero", {"max_turn_penalty": -1.0}),
        ("budget 0 and max_turns 0", {"max_turns": 0, "free_chat": {
            "enabled": True, "budget": 0}}),
    ):
        kwargs = dict(base_kwargs)
        kwargs.update(override)
        try:
            TutorAgentWorkflow(**kwargs)
        except ValueError as exc:
            check(f"rejected: {label}", True)
            del exc
        except Exception as exc:  # noqa: BLE001
            check(f"rejected: {label}", False,
                  f"raised {type(exc).__name__} instead of ValueError: {exc}")
        else:
            check(f"rejected: {label}", False, "constructed without complaint")

    print("\n[10] no phantom student turn is synthesised into the history")
    # _run_public_summary_update rebuilds the student's opening attempt whenever
    # the history is empty. Free chat has no opening attempt, and an empty history
    # is its normal turn-1 state, so the fallback used to write
    # "Here is my attempt at this problem:" with nothing after it into the
    # teacher's own context from turn 2 on, and into the re-test transcript.
    updated = asyncio.run(
        workflow._run_public_summary_update(
            old_public_history=PublicHistoryState(),
            previous_student_answer="",
            tutor_visible_output="What does Vieta give you?",
            current_student_answer="a + b = m and ab = 2.",
            env_feedback="",
        )
    )
    check("the first round produces exactly two turns",
          len(updated.turns) == 2,
          f"got {updated.turns}")
    check("the teacher speaks first in the stored history",
          updated.turns and updated.turns[0]["role"] == "teacher",
          f"got {[t['role'] for t in updated.turns]}")
    check("no synthesised initial attempt in the turns",
          all("Here is my attempt" not in turn["content"] for turn in updated.turns))
    check("no synthesised initial attempt in the summary",
          "Student round 0" not in updated.summary, updated.summary[:120])
    legacy_updated = asyncio.run(
        make_workflow(free_chat_enabled=False)._run_public_summary_update(
            old_public_history=PublicHistoryState(),
            previous_student_answer="my first try",
            tutor_visible_output="t",
            current_student_answer="s",
            env_feedback="",
        )
    )
    check("free_chat off still reconstructs the opening attempt, as before",
          len(legacy_updated.turns) == 3
          and "Here is my attempt" in legacy_updated.turns[0]["content"],
          f"got {[t['role'] for t in legacy_updated.turns]}")

    print("\n[11] the success series reports the re-test, not the in-chat judge")
    # `solved` used to come from success_round > 0, which reads the per-turn answer
    # judge. Free chat has no per-turn judge, so every success series sat at a flat
    # zero while the thing that decides the episode was only visible under
    # generalize/*. All of them now read this score.
    score = TutorAgentWorkflow._free_chat_outcome_score
    check("3 of 4 replays correct scores 0.75",
          score([_retest(replay_correct=3, replay_count=4)]) == 0.75,
          f"got {score([_retest(replay_correct=3, replay_count=4)])}")
    check("4 of 4 scores 1.0",
          score([_retest(replay_correct=4, replay_count=4)]) == 1.0)
    check("0 of 4 scores 0.0",
          score([_retest(replay_correct=0, replay_count=4)]) == 0.0)
    check("a skipped re-test scores 0, not a missing value",
          score([_retest(skipped=True)]) == 0.0)
    check("an unattempted re-test scores 0",
          score([_retest(attempted=False)]) == 0.0)
    check("no re-test at all scores 0", score([]) == 0.0 and score(None) == 0.0)
    check("a transfer level is never mistaken for the re-test",
          score([_retest(level="level1", replay_correct=4, replay_count=4)]) == 0.0)
    check("replays=1 falls back to the single judge result",
          score([_retest(replay_count=0, judge_correct=True)]) == 1.0
          and score([_retest(replay_count=0, judge_correct=False)]) == 0.0)

    print("\n[12] leak-terminate re-tests the transcript WITHOUT the leaking turn")
    # The terminate branch stores `public_history_after = public_before` for the
    # leaking turn and never gives it a student_state, so the last turn that
    # counts as an anchor is the last COMPLETED round. That is what makes the
    # leaked answer worth nothing to the student.
    round_one = _turn(1, [
        {"role": "teacher", "content": "What is the vertex form?"},
        {"role": "student", "content": "I do not know."},
    ], completed=True)
    round_two = _turn(2, [
        {"role": "teacher", "content": "Complete the square."},
        {"role": "student", "content": "(x-2)^2 + something?"},
    ], completed=True)
    leaked_round = _turn(3, [], completed=False, leaked=True)
    anchor = workflow._student_generalization_anchor(
        _episode_with([round_one, round_two, leaked_round]), allow_unsuccessful=True
    )
    check("the anchor is a completed round, not the leaking one",
          anchor is not None and anchor.reward_turn_idx == 2,
          f"got {None if anchor is None else anchor.reward_turn_idx}")
    replayed = "" if anchor is None else " ".join(
        turn["content"] for turn in anchor.public_history.turns
    )
    check("the leaking turn is absent from the replayed transcript",
          "LEAKED" not in replayed, replayed[:120])
    check("the completed rounds are present",
          "Complete the square." in replayed)
    leak_on_turn_one = workflow._student_generalization_anchor(
        _episode_with([_turn(1, [], completed=False, leaked=True)]),
        allow_unsuccessful=True,
    )
    check("a leak on turn 1 skips the re-test instead of running it on nothing",
          leak_on_turn_one is None,
          f"got {leak_on_turn_one}")

    print("\n[13] the configured retest_reward actually reaches the case")
    # retest_reward is read from config into an attribute AND used where the
    # re-test case is built. Only the second one decides what an episode is worth,
    # and a missing wire there is invisible to every prompt check above -- which
    # is exactly how it was missed the first time.
    rewarded = make_workflow(
        student_generalize_retest_original=True,
        student_generalize_retest_reward=1.0,
        student_generalize_level1_enabled=False,
        student_generalize_level2_enabled=False,
        student_generalize_bank={},
    )
    row = {"id": "x", "task": TASK, "ground_truth": GROUND_TRUTH}
    cases = rewarded._student_generalization_cases(row)
    check("the original re-test case exists", "original" in cases)
    check("its reward is the configured retest_reward",
          cases["original"].reward == 1.0,
          f"got {cases['original'].reward}")
    legacy_cases = make_workflow(
        free_chat_enabled=False,
        student_generalize_retest_original=True,
        student_generalize_retest_reward=0.0,
        student_generalize_level1_enabled=False,
        student_generalize_level2_enabled=False,
        student_generalize_bank={},
    )._student_generalization_cases(row)
    check("default 0.0 keeps the re-test unrewarded, as before",
          legacy_cases["original"].reward == 0.0,
          f"got {legacy_cases['original'].reward}")

    print("\n[14] free_chat off leaves the old rollout byte-identical")
    legacy = make_workflow(free_chat_enabled=False, free_chat_budget=0)
    legacy.teacher_system_prompt = "You are a careful math tutor."
    legacy.teacher_prompt_pool = ()
    legacy_system = legacy._teacher_system_for_state(tutor_state(), clean=True)
    check("legacy teacher prompt still starts from teacher_system_prompt",
          legacy_system.startswith("You are a careful math tutor."))
    check("legacy teacher prompt still appends the task block",
          "Here is the math problem:" in legacy_system)
    legacy.student_system_prompt = "You are a real student solving the task."
    legacy.student_prompt_pool = ()
    legacy_student = legacy._build_student_messages(
        StudentTurnState(
            task=TASK,
            public_history=PublicHistoryState(turns=[]),
            previous_student_output="",
            latest_tutor_visible_output="hello",
        )
    )[0]["content"]
    check("legacy student prompt still carries the task",
          TASK in legacy_student
          and legacy_student.startswith("You are a real student solving the task."))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all free-chat checks passed")
    return 0


def _retest(
    *,
    level: str = "original",
    skipped: bool = False,
    attempted: bool = True,
    replay_correct: int = 0,
    replay_count: int = 4,
    judge_correct: bool = False,
):
    from examples.tutor.core.types import JudgeResult
    from examples.tutor.workflow import StudentGeneralizationResult

    return StudentGeneralizationResult(
        level=level,
        skipped=skipped,
        attempted=attempted,
        replay_correct=replay_correct,
        replay_count=replay_count,
        judge_result=JudgeResult(
            raw_output="", correct=judge_correct, feedback="", parse_error=None,
            raw_result={},
        ),
    )


def _turn(
    turn_idx: int,
    conversation: list[dict[str, str]],
    *,
    completed: bool,
    leaked: bool = False,
):
    """One TurnArtifact.

    ``completed`` false is the leak-terminate shape: the turn carries no
    student_state and its `public_history_after` equals `public_history_before`,
    so the turn it represents never entered the transcript.
    """
    from examples.tutor.core.types import LeakCheckResult, TurnArtifact

    before = [] if turn_idx == 1 else [{"role": "teacher", "content": "earlier"}]
    return TurnArtifact(
        turn_idx=turn_idx,
        tutor_state=tutor_state(turn_idx=turn_idx),
        tutor_messages=[],
        tutor_response=None,
        tutor_raw_output="",
        tutor_visible_output="LEAKED: the answer is 3" if leaked else "fine",
        leak_result=LeakCheckResult(
            raw_output="", leaked=leaked, feedback="", parse_error=None,
            raw_result={},
        ),
        public_history_before=before,
        public_history_after=(before if not completed else list(conversation)),
        student_state=(
            None
            if not completed
            else StudentTurnState(
                task=TASK,
                public_history=PublicHistoryState(turns=list(conversation)),
                previous_student_output="",
                latest_tutor_visible_output="fine",
            )
        ),
        student_output=conversation[-1]["content"] if completed else "",
    )


def _episode_with(turns: list):
    episode = _episode([{"role": "teacher", "content": "x"}])
    episode.turns = turns
    return episode


def _episode(conversation: list[dict[str, str]]):
    from examples.tutor.core.types import EpisodeArtifact, JudgeResult

    return EpisodeArtifact(
        task=TASK,
        ground_truth=GROUND_TRUTH,
        initial_student_answer="",
        initial_student_error=None,
        initial_judge_result=JudgeResult(
            raw_output="", correct=False, feedback="", parse_error=None,
            raw_result={},
        ),
        turns=[],
        termination_reason="max_turns",
        pre_success=False,
        leak_count=0,
        latest_student_answer=conversation[-1]["content"],
    )


def _anchor(conversation: list[dict[str, str]]):
    from examples.tutor.workflow import StudentGeneralizationAnchor

    return StudentGeneralizationAnchor(
        public_history=PublicHistoryState(
            summary="", turn_count=1, turns=list(conversation)
        ),
        previous_student_output=conversation[-1]["content"],
        teacher_feedback=conversation[0]["content"],
        reward_turn_idx=1,
    )


if __name__ == "__main__":
    raise SystemExit(main())
