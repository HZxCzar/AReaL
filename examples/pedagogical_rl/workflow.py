from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from examples.pedagogical_rl import cross_eval
from examples.pedagogical_rl.api import (
    PedagogicalAPIClient,
    PedagogicalEngineClient,
)
from examples.pedagogical_rl.config import (
    PedagogicalAPIModelConfig,
    PedagogicalEvaluationConfig,
    PedagogicalGenerationConfig,
    PedagogicalTeacherPreConfig,
)
from examples.pedagogical_rl.cross_eval_config import CrossEvalConfig
from examples.pedagogical_rl.judges import run_whole_dialogue_judge
from examples.pedagogical_rl.preference import NO_PREFERENCE, PreferenceGate
from examples.pedagogical_rl.prompts import WHOLE_DIALOGUE_JUDGE_PROMPTS
from examples.pedagogical_rl.scoring import native_answer_correct, score_unified_answer
from examples.pedagogical_rl.state import (
    ClassroomEpisode,
    ConversationType,
    NativeJudgeDecision,
    student_visible_text,
)
from examples.tutor.core.math import score_math_answer
from examples.tutor.core.parsers import parse_leak_check_result
from examples.tutor.prompts import (
    FILTER_SOLVER_SYSTEM_PROMPT,
    FILTER_SOLVER_USER_TEMPLATE,
    RAWBASE_LEAK_CHECK_SYSTEM_PROMPT,
    RAWBASE_LEAK_CHECK_USER_TEMPLATE,
    render_prompt,
)

from areal import workflow_context
from areal.api import RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging, stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("PedagogicalRLWorkflow")


def _coerce_config(config_cls: type, value: Any) -> Any:
    if isinstance(value, config_cls):
        return value
    if is_dataclass(value):
        value = asdict(value)
    elif not isinstance(value, dict):
        try:
            from omegaconf import OmegaConf

            value = OmegaConf.to_container(value, resolve=True)
        except Exception:
            pass
    if not isinstance(value, dict):
        raise TypeError(
            f"cannot convert {type(value).__name__} to {config_cls.__name__}"
        )
    return config_cls(**value)


class PedagogicalRLWorkflow(RolloutWorkflow):
    """PedagogicalRL's classroom method running on an AReaL teacher actor.

    Training selects either PedagogicalRL's whole-dialogue leak gate or AReaL's
    turn-level leak gate. Evaluation completes one shared dialogue while
    recording both leak judges without terminating on either.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: Any,
        student_model: PedagogicalAPIModelConfig | dict[str, Any],
        judge_model: PedagogicalAPIModelConfig | dict[str, Any],
        generation: PedagogicalGenerationConfig | dict[str, Any],
        teacher_pre: PedagogicalTeacherPreConfig | dict[str, Any] | None = None,
        evaluation: PedagogicalEvaluationConfig | dict[str, Any] | None = None,
        cross_eval: CrossEvalConfig | dict[str, Any] | None = None,
        debug_trace_dir: str = "",
        debug_trace_every_n_rollouts: int = 10,
        eval_repeat_count: int = 1,
        require_complete_group: bool | None = None,
        student_client: PedagogicalAPIClient | None = None,
        judge_client: PedagogicalAPIClient | None = None,
        actor_client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.gconfig = gconfig
        self.tokenizer = (
            load_hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.student_model = _coerce_config(PedagogicalAPIModelConfig, student_model)
        self.judge_model = _coerce_config(PedagogicalAPIModelConfig, judge_model)
        self.generation = _coerce_config(PedagogicalGenerationConfig, generation)
        self.teacher_pre = _coerce_config(
            PedagogicalTeacherPreConfig, teacher_pre or {}
        )
        self.evaluation = _coerce_config(
            PedagogicalEvaluationConfig, evaluation or {}
        )
        self.cross_eval_config = _coerce_config(CrossEvalConfig, cross_eval or {})
        if student_client is None and self.student_model.mode != "api":
            raise ValueError(
                "the student must use mode='api'; offline launchers provide a "
                "local OpenAI-compatible server"
            )
        self.student_client = student_client or PedagogicalAPIClient(self.student_model)
        self.judge_client = judge_client
        if self.judge_client is None and self.judge_model.mode == "api":
            self.judge_client = PedagogicalAPIClient(self.judge_model)
        self._engine_judge_clients: dict[int, PedagogicalEngineClient] = {}
        self.preference_gate: PreferenceGate | None = None
        gated_preferences = [
            name
            for name in self.evaluation.preference_names
            if name != NO_PREFERENCE
        ]
        if self.evaluation.matrix_enabled and gated_preferences:
            self.preference_gate = PreferenceGate(
                prompts_path=self.evaluation.preference_prompts_path,
                complaints_path=self.evaluation.preference_complaints_path,
                retries=self.evaluation.preference_gate_retries,
                explain_ratio=self.evaluation.preference_explain_ratio,
                seed=self.student_model.seed,
            )
            self.preference_gate.validate_names(self.evaluation.preference_names)
        self.debug_trace_dir = debug_trace_dir
        self.debug_trace_every_n_rollouts = int(debug_trace_every_n_rollouts)
        self.eval_repeat_count = int(eval_repeat_count)
        if self.eval_repeat_count < 1:
            raise ValueError("eval_repeat_count must be positive")
        self._eval_repeat_outcomes: dict[int, list[float]] = {}
        self.actor_client_factory = actor_client_factory or ArealOpenAI
        # Verified presolve can reject an individual sub-rollout. The remote
        # grouped wrapper must drop the whole problem group rather than feed a
        # partial group into fixed-size GRPO normalization.
        self.require_complete_group = (
            bool(self.teacher_pre.enabled)
            if require_complete_group is None
            else bool(require_complete_group)
        )

    def _new_actor_client(self, engine: Any) -> Any:
        return self.actor_client_factory(
            engine=engine,
            tokenizer=self.tokenizer,
            chat_template_type="concat",
            engine_max_tokens=self.gconfig.max_tokens,
        )

    def _new_judge_client(self, engine: Any) -> Any:
        if self.judge_client is not None:
            return self.judge_client
        if self.judge_model.mode != "self":
            raise RuntimeError(
                f"unsupported judge model mode: {self.judge_model.mode!r}"
            )
        key = id(engine)
        if key not in self._engine_judge_clients:
            self._engine_judge_clients[key] = PedagogicalEngineClient(
                self.judge_model,
                engine=engine,
                tokenizer=self.tokenizer,
                base_gconfig=self.gconfig,
            )
        return self._engine_judge_clients[key]

    async def _teacher_turn(self, client: Any, episode: ClassroomEpisode) -> str:
        response = await client.chat.completions.create(
            messages=episode.teacher_messages(),
            n=1,
            max_completion_tokens=self.generation.max_tokens_per_teacher_turn,
            max_total_tokens=self.gconfig.max_tokens,
            temperature=self.gconfig.temperature,
            top_p=self.gconfig.top_p,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content or ""

    async def _student_turn(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        max_tokens: int,
    ) -> list[str]:
        return await self.student_client.generate(
            messages,
            n=n,
            max_tokens=max_tokens,
            temperature=self.generation.student_temperature,
            top_p=self.generation.student_top_p,
        )

    async def _run_native_judges(
        self,
        episode: ClassroomEpisode,
        *,
        judge_client: Any,
        stop_on_reject: bool,
        rules: tuple[str, ...] | None = None,
    ) -> list[NativeJudgeDecision]:
        hidden = episode.hidden_conversation()
        all_decisions: list[NativeJudgeDecision] = []
        selected_rules = rules or tuple(WHOLE_DIALOGUE_JUDGE_PROMPTS)

        async def call(prompt: str) -> list[str]:
            return await judge_client.generate(
                [{"role": "user", "content": prompt}],
                n=1,
                max_tokens=self.generation.max_tokens_per_judge_attempt,
                temperature=self.generation.judge_temperature,
                top_p=self.generation.judge_top_p,
            )

        for rule in selected_rules:
            decisions = await run_whole_dialogue_judge(
                rule=rule,
                conversation=hidden,
                call=call,
                attempts=self.generation.number_judge_attempts,
            )
            all_decisions.extend(decisions)
            if stop_on_reject and any(d.rejected for d in decisions):
                break
        episode.native_judges.extend(all_decisions)
        return all_decisions

    async def _rawbase_leak_check(
        self,
        episode: ClassroomEpisode,
        teacher_output: str,
        *,
        judge_client: Any,
    ) -> tuple[bool, str | None, str, str]:
        visible = student_visible_text(
            teacher_output,
            output_format=episode.teacher_output_format,
        )
        prompt = render_prompt(
            RAWBASE_LEAK_CHECK_USER_TEMPLATE,
            ground_truth=episode.answer,
            teacher_action=visible,
        )
        try:
            outputs = await judge_client.generate(
                [
                    {"role": "system", "content": RAWBASE_LEAK_CHECK_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                n=1,
                max_tokens=self.generation.max_tokens_per_judge_attempt,
                temperature=0.0,
                top_p=1.0,
            )
            raw_output = outputs[0]
            result = parse_leak_check_result(raw_output)
            return result.leaked, result.parse_error, result.feedback, raw_output
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return True, error, f"Rawbase leak check failed: {error}", ""

    async def _run_teacher_pre_solve(
        self,
        actor_client: Any,
        episode: ClassroomEpisode,
        *,
        judge_client: Any,
    ) -> str | None:
        messages = [
            {"role": "system", "content": FILTER_SOLVER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FILTER_SOLVER_USER_TEMPLATE.format(task=episode.problem),
            },
        ]
        attempt_count = self.teacher_pre.attempts if self.teacher_pre.verify else 1
        max_tokens = (
            self.teacher_pre.max_tokens
            if self.teacher_pre.max_tokens > 0
            else self.generation.max_tokens_per_teacher_turn
        )
        for attempt_idx in range(attempt_count):
            try:
                response = await actor_client.chat.completions.create(
                    messages=messages,
                    n=1,
                    store=False,
                    max_completion_tokens=max_tokens,
                    max_total_tokens=self.gconfig.max_tokens,
                    temperature=self.gconfig.temperature,
                    top_p=self.gconfig.top_p,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw_output = response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning(
                    "Teacher presolve attempt %s/%s failed: %s",
                    attempt_idx + 1,
                    attempt_count,
                    exc,
                )
                continue

            if not self.teacher_pre.verify:
                return raw_output

            verification = await score_unified_answer(
                task=episode.problem,
                ground_truth=episode.answer,
                student_answer=raw_output,
                judge=judge_client,
                judge_max_tokens=self.generation.max_tokens_per_judge_attempt,
            )
            if verification.correct:
                return raw_output

        logger.warning(
            "Teacher presolve rejected after %s attempt(s) (verify=%s)",
            attempt_count,
            self.teacher_pre.verify,
        )
        return None

    @staticmethod
    def _native_rejected(episode: ClassroomEpisode, rule: str) -> bool:
        return any(
            decision.rule == rule and decision.rejected
            for decision in episode.native_judges
        )

    def _native_reward(self, episode: ClassroomEpisode) -> dict[str, float]:
        end_reward = (
            sum(
                native_answer_correct(solution, episode.answer)
                for solution in episode.final_solutions
            )
            / len(episode.final_solutions)
            if episode.final_solutions
            else -self.generation.extra_penalty_for_rejected_judges
        )
        if episode.failed_native_judges and episode.final_solutions:
            end_reward -= self.generation.extra_penalty_for_rejected_judges

        teacher_outputs: list[str] = []
        for message in episode.conversation:
            if message["role"] != "teacher":
                continue
            teacher_outputs.append(message["content"])

        format_reward = 0.0
        if episode.teacher_output_format == "unified_xml":
            thinking_reward = 0.0
            if episode.format_failed:
                format_reward = self.generation.format_error_penalty
        else:
            missing_think_penalty = 0.0
            used_thinking = 0
            for content in teacher_outputs:
                if content.count("<think>") != content.count("</think>"):
                    missing_think_penalty -= 0.5
                elif content.count("<think>") > 0:
                    used_thinking += 1
            thinking_reward = missing_think_penalty + (
                0.5 * used_thinking / len(teacher_outputs) if teacher_outputs else 0.0
            )
        if episode.final_solutions:
            eoc_reward = 0.1 if episode.teacher_ended else 0.0
        else:
            # PedagogicalRL's Conversation reward helpers return zero when the
            # native judge gate rejects the dialogue and no final solutions are
            # sampled. The rejection penalty and length penalty still apply.
            if episode.teacher_output_format == "native":
                thinking_reward = 0.0
            eoc_reward = 0.0
        length_reward = (
            -0.5
            if any(
                len(self.tokenizer.encode(output))
                >= self.generation.max_tokens_per_teacher_turn - 1
                for output in teacher_outputs
            )
            else 0.0
        )
        return {
            "end_rm_reward": float(end_reward),
            "thinking_reward": float(thinking_reward),
            "format_reward": float(format_reward),
            "end_of_conversation_reward": float(eoc_reward),
            "length_reward": float(length_reward),
            "total_reward": float(
                end_reward
                + thinking_reward
                + format_reward
                + eoc_reward
                + length_reward
            ),
        }

    async def _train_episode(
        self,
        actor_client: Any,
        episode: ClassroomEpisode,
        *,
        judge_client: Any,
    ) -> tuple[float, dict[str, float]]:
        if episode.conversation_type is ConversationType.ATTEMPTED:
            initial = await self._student_turn(
                episode.initial_student_messages(),
                n=1,
                max_tokens=self.generation.max_tokens_per_student_attempt,
            )
            episode.add_initial_attempt(initial[0])

        while not episode.should_stop_dialogue(
            tokenizer=self.tokenizer,
            max_teacher_turns=self.generation.max_teacher_turns,
            max_tokens_in_conversation=self.generation.max_tokens_in_conversation,
        ):
            teacher_output = await self._teacher_turn(actor_client, episode)
            episode.add_teacher(teacher_output)
            if episode.format_failed:
                break
            if self.generation.leak_judge_mode == "turn":
                leaked, error, feedback, raw_output = await self._rawbase_leak_check(
                    episode,
                    teacher_output,
                    judge_client=judge_client,
                )
                episode.leak_checks.append(
                    {
                        "teacher_turn": episode.teacher_turns,
                        "student_visible": student_visible_text(
                            teacher_output,
                            output_format=episode.teacher_output_format,
                        ),
                        "leaked": leaked,
                        "parse_error": error,
                        "feedback": feedback,
                        "raw_output": raw_output,
                    }
                )
                if leaked:
                    episode.leak_failed = True
                    episode.termination_reason = (
                        "turn_leak_judge_error" if error else "turn_leak"
                    )
                    break
            if episode.should_stop_dialogue(
                tokenizer=self.tokenizer,
                max_teacher_turns=self.generation.max_teacher_turns,
                max_tokens_in_conversation=self.generation.max_tokens_in_conversation,
            ):
                break
            student = await self._student_turn(
                episode.student_messages(),
                n=1,
                max_tokens=self.generation.max_tokens_per_student_turn,
            )
            episode.add_student(student[0])

        if not episode.format_failed:
            native_rules = (
                tuple(WHOLE_DIALOGUE_JUDGE_PROMPTS)
                if self.generation.leak_judge_mode == "pedagogical_rl"
                else ("follows_pedagogical_values",)
            )
            await self._run_native_judges(
                episode,
                judge_client=judge_client,
                stop_on_reject=True,
                rules=native_rules,
            )
            if not episode.leak_failed and not episode.failed_native_judges:
                episode.final_solutions = await self._student_turn(
                    episode.student_messages(final=True),
                    n=self.generation.number_student_attempts,
                    max_tokens=self.generation.max_tokens_per_student_attempt,
                )
        components = self._native_reward(episode)
        return components["total_reward"], components

    # ------------------------------------------------------------------
    # The tutor / PedagogicalRL cross. Evaluation only; training is untouched.
    #
    # The instrument is examples/pedagogical_rl/cross_eval.py, imported by both
    # arms. Everything below is the adapter from this workflow's clients to the
    # plain async callables it takes. The mirror of this block lives in
    # examples/tutor/workflow.py under the same heading.
    # ------------------------------------------------------------------

    def _cross_eval_selected(self, problem: str) -> bool:
        """Whether this problem is in the crossed subset.

        A stable hash of the problem text, not a counter or a seed: the two arms
        run different code over the eval split in a different order, and this is
        what makes them cross exactly the same problems anyway.
        """

        if not self.cross_eval_config.enabled:
            return False
        rate = float(self.cross_eval_config.sample_rate)
        if rate >= 1.0:
            return True
        digest = hashlib.sha256(problem.encode("utf-8")).hexdigest()
        return (int(digest[:8], 16) % 10_000) < rate * 10_000

    async def _run_cross_eval(
        self,
        *,
        actor_client: Any,
        episode: ClassroomEpisode,
        judge_client: Any,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        config = self.cross_eval_config
        draft = str(episode.teacher_draft or "")

        async def teacher_call(messages, *, rid_prefix: str = "xeval") -> str:
            del rid_prefix
            response = await actor_client.chat.completions.create(
                messages=messages,
                n=1,
                max_completion_tokens=self.generation.max_tokens_per_teacher_turn,
                max_total_tokens=self.gconfig.max_tokens,
                temperature=self.gconfig.temperature,
                top_p=self.gconfig.top_p,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content or ""

        async def student_call(
            messages, *, n: int = 1, max_tokens=None, rid_prefix="xeval", timeout=None
        ) -> list[str]:
            del rid_prefix, timeout
            return await self.student_client.generate(
                messages,
                n=n,
                max_tokens=int(
                    max_tokens or self.generation.max_tokens_per_student_attempt
                ),
                temperature=self.generation.student_temperature,
                top_p=self.generation.student_top_p,
            )

        async def judge_call(messages, *, rid_prefix: str = "xeval-judge") -> str:
            del rid_prefix
            outputs = await judge_client.generate(
                messages,
                n=1,
                max_tokens=self.generation.max_tokens_per_judge_attempt,
                temperature=self.generation.judge_temperature,
                top_p=self.generation.judge_top_p,
            )
            return outputs[0]

        async def answer_judge(*, task: str, ground_truth: str, answer: str) -> bool:
            # The tutor arm's judge: AReaL's exact math scorer, then its answer
            # judge prompt on the same auxiliary model. score_unified_answer is
            # that path, so "our scoring" means the same thing on both arms.
            result = await score_unified_answer(
                task=task,
                ground_truth=ground_truth,
                student_answer=answer,
                judge=judge_client,
            )
            return bool(result.correct)

        return await cross_eval.run_cross_eval(
            task=episode.problem,
            ground_truth=episode.answer,
            own_protocol="classroom",
            own_transcript=episode.hidden_conversation(),
            own_initial_attempt=episode.initial_attempt or "",
            teacher_call=teacher_call,
            student_call=student_call,
            judge_call=judge_call,
            answer_judge=answer_judge,
            tokenizer=self.tokenizer,
            free_chat=cross_eval.FreeChatSpec(
                budget=int(config.free_chat.budget),
                enable_thinking=bool(self.generation.use_thinking),
                # The tutor arm's teacher is shown the ground truth only when
                # its own teacher_show_ground_truth is on, and that arm leaves
                # it off. Handing it to this teacher would make the free_chat
                # column measure two different setups.
                show_ground_truth=False,
                student_has_not_seen_problem=False,
                max_student_tokens=int(config.free_chat.max_student_tokens),
                # This arm has no teacher_history_tags of its own -- the setting
                # belongs to our protocol -- so it comes from the shared block,
                # where the tutor arm checks it against what it rolls out with.
                teacher_history_tags=str(config.free_chat.teacher_history_tags),
                teacher_draft=draft,
            ),
            classroom=cross_eval.ClassroomSpec(
                max_teacher_turns=int(config.classroom.max_teacher_turns),
                max_tokens_in_conversation=int(
                    config.classroom.max_tokens_in_conversation
                ),
                max_tokens_per_student_turn=int(
                    config.classroom.max_tokens_per_student_turn
                ),
                max_tokens_per_student_attempt=int(
                    config.classroom.max_tokens_per_student_attempt
                ),
                include_thinking=bool(config.classroom.include_thinking),
                teacher_draft=draft,
            ),
            retest=cross_eval.RetestSpec(
                replays=int(config.retest.replays),
                max_tokens=int(config.retest.max_tokens),
            ),
            interview=cross_eval.InterviewSpec(
                attempts=int(config.interview.attempts),
                max_tokens=int(config.interview.max_tokens),
                student_name=str(config.interview.student_name or ""),
                timeout=config.interview.timeout,
            ),
            leak_judges=(
                cross_eval.LeakJudgeSpec(
                    turn_enabled=bool(config.leak_judges.turn_enabled),
                    native_enabled=bool(config.leak_judges.native_enabled),
                    native_attempts=int(config.leak_judges.native_attempts),
                    native_max_retries=int(config.leak_judges.native_max_retries),
                )
                if config.leak_judges.enabled
                else None
            ),
            run_other_protocol=bool(config.run_other_protocol),
        )

    async def _eval_episode(
        self,
        actor_client: Any,
        episode: ClassroomEpisode,
        *,
        judge_client: Any,
    ) -> tuple[float | None, dict[str, float], dict[str, Any]]:
        """Run native PedagogicalRL evaluation plus the agreed preference gate.

        The preference layer is evaluation-only. A failed teacher turn and its
        scripted complaint stay in teacher history but are absent from the real
        student's context, the native whole-dialogue judges, and the final test.
        """

        eval_details: dict[str, Any] = {}
        if self.evaluation.compute_initial_attempts:
            episode.evaluation_initial_solutions = await self._student_turn(
                episode.no_tutor_attempt_messages(),
                n=self.generation.number_student_attempts,
                max_tokens=self.generation.max_tokens_per_student_attempt,
            )
        initial_results = [
            native_answer_correct(solution, episode.answer)
            for solution in episode.evaluation_initial_solutions
        ]
        initial_correct = (
            sum(initial_results) / len(initial_results) if initial_results else 0.0
        )
        eval_details["initial"] = [
            {"solution": solution, "correct": correct}
            for solution, correct in zip(
                episode.evaluation_initial_solutions,
                initial_results,
                strict=True,
            )
        ]

        if episode.conversation_type is ConversationType.ATTEMPTED:
            initial = await self._student_turn(
                episode.initial_student_messages(),
                n=1,
                max_tokens=self.generation.max_tokens_per_student_attempt,
            )
            episode.add_initial_attempt(initial[0])

        while not episode.should_stop_dialogue(
            tokenizer=self.tokenizer,
            max_teacher_turns=self.generation.max_teacher_turns,
            max_tokens_in_conversation=self.generation.max_tokens_in_conversation,
        ):
            teacher_output = await self._teacher_turn(actor_client, episode)
            episode.add_teacher(teacher_output)
            if episode.format_failed:
                break
            visible_output = student_visible_text(
                teacher_output,
                output_format=episode.teacher_output_format,
            )

            if self.evaluation.record_turn_leak_diagnostic:
                leaked, error, feedback, raw_output = await self._rawbase_leak_check(
                    episode,
                    teacher_output,
                    judge_client=judge_client,
                )
                episode.leak_checks.append(
                    {
                        "teacher_turn": episode.teacher_turns,
                        "student_visible": visible_output,
                        "leaked": leaked,
                        "parse_error": error,
                        "feedback": feedback,
                        "raw_output": raw_output,
                    }
                )

            if episode.teacher_ended:
                break

            if episode.preference != NO_PREFERENCE:
                if self.preference_gate is None:
                    raise RuntimeError(
                        "a preference evaluation row was scheduled without a gate"
                    )
                decision = await self.preference_gate.judge(
                    client=judge_client,
                    preference=episode.preference,
                    problem=episode.problem,
                    last_student_message=episode.latest_real_student_message(),
                    teacher_message=visible_output,
                    max_tokens=self.evaluation.preference_gate_max_tokens,
                    temperature=self.evaluation.preference_gate_temperature,
                    top_p=self.evaluation.preference_gate_top_p,
                    top_k=self.evaluation.preference_gate_top_k,
                    min_p=self.evaluation.preference_gate_min_p,
                )
                if not decision.passed:
                    episode.hide_latest_teacher_from_student()
                    decision.complaint = self.preference_gate.complaint(
                        preference=episode.preference,
                        problem=episode.problem,
                        conversation_type=episode.conversation_type.value,
                        turn_idx=episode.teacher_turns,
                    )
                    # Teacher sees this synthetic student turn. The real student,
                    # native judges, and final interview do not.
                    episode.add_student(
                        decision.complaint,
                        student_visible=False,
                    )
                episode.preference_gate_checks.append(asdict(decision))
                if not decision.passed:
                    continue

            if episode.should_stop_dialogue(
                tokenizer=self.tokenizer,
                max_teacher_turns=self.generation.max_teacher_turns,
                max_tokens_in_conversation=self.generation.max_tokens_in_conversation,
            ):
                break
            student = await self._student_turn(
                episode.student_messages(),
                n=1,
                max_tokens=self.generation.max_tokens_per_student_turn,
            )
            episode.add_student(student[0])

        if not episode.format_failed:
            # Official eval sets ignore_rejected_judge=true: both native judges
            # are diagnostics, and the final attempts are always sampled.
            await self._run_native_judges(
                episode,
                judge_client=judge_client,
                stop_on_reject=False,
            )
            episode.final_solutions = await self._student_turn(
                episode.student_messages(final=True),
                n=self.generation.number_student_attempts,
                max_tokens=self.generation.max_tokens_per_student_attempt,
            )

        final_results = [
            native_answer_correct(solution, episode.answer)
            for solution in episode.final_solutions
        ]
        final_correct = (
            sum(final_results) / len(final_results) if final_results else 0.0
        )
        final_any_correct = float(any(final_results))
        eval_details["final"] = [
            {"solution": solution, "correct": correct}
            for solution, correct in zip(
                episode.final_solutions, final_results, strict=True
            )
        ]
        reward = float(final_correct)
        raw_improvement = reward - float(initial_correct)
        final_correct_math = (
            sum(
                float(
                    bool(score_math_answer(episode.problem, episode.answer, s).correct)
                )
                for s in episode.final_solutions
            )
            / len(episode.final_solutions)
            if episode.final_solutions
            else 0.0
        )
        turn_leak = float(episode.turn_leak_observed)
        native_leak = float(self._native_rejected(episode, "does_not_leak_answer"))
        leak_aware_improvement = 0.0 if native_leak else raw_improvement
        leak_aware_final = 0.0 if native_leak else reward
        turns = float(episode.teacher_turns)
        sampled_gates = len(episode.preference_gate_checks)
        passed_gates = sum(
            int(bool(check["passed"])) for check in episode.preference_gate_checks
        )
        gate_compliance = (
            passed_gates / sampled_gates if sampled_gates else 1.0
        )
        gate_errors = sum(
            int(bool(check.get("error")))
            for check in episode.preference_gate_checks
        )
        student_metric_prefix = f"student/{self.student_model.model}"
        matrix_prefix = (
            f"ped_eval/{episode.conversation_type.value.lower()}/"
            f"{episode.preference}"
        )
        metrics = {
            "reward": reward,
            "final_correct": reward,
            "final_any_correct": final_any_correct,
            "initial_correct": float(initial_correct),
            "improvement/raw": raw_improvement,
            "improvement/leak_aware": leak_aware_improvement,
            "accuracy/raw": reward,
            "accuracy/turn_leak_gate": 0.0 if turn_leak else reward,
            "accuracy/pedagogical_rl_leak_gate": leak_aware_final,
            "accuracy/pedagogical_rl_leak_aware": leak_aware_final,
            "pre_solved": float(initial_correct),
            "solved": reward,
            "leaks": native_leak,
            "turns": turns,
            "stop/leak": 0.0,
            "invalid_success_due_to_leak": float(bool(reward) and bool(native_leak)),
            f"{student_metric_prefix}/selected": 1.0,
            f"{student_metric_prefix}/pre_solved": float(initial_correct),
            f"{student_metric_prefix}/solved": reward,
            f"{student_metric_prefix}/turns": turns,
            f"{student_metric_prefix}/reward": reward,
            "leak_judge_error": float(
                any(bool(check.get("parse_error")) for check in episode.leak_checks)
            ),
            "native_judge_rejected": float(episode.failed_native_judges),
            "format_errors": float(episode.format_failed),
            "stop/format_error": float(episode.format_failed),
            "turn_leak/any": turn_leak,
            "native_leak/rejected": native_leak,
            "native_pedagogy/rejected": float(
                self._native_rejected(episode, "follows_pedagogical_values")
            ),
            "preference_gate/compliance": gate_compliance,
            "preference_gate/passed": float(passed_gates),
            "preference_gate/sampled": float(sampled_gates),
            "preference_gate/errors": float(gate_errors),
            "ped_eval/final_correct": reward,
            "ped_eval/final_any_correct": final_any_correct,
            "ped_eval/final_correct_math": final_correct_math,
            "ped_eval/initial_correct": float(initial_correct),
            "ped_eval/delta": raw_improvement,
            "ped_eval/delta_leak_aware": leak_aware_improvement,
            "ped_eval/leaked": native_leak,
            "ped_eval/turns": turns,
            "ped_eval/attempt_errors": 0.0,
            f"{matrix_prefix}/selected": 1.0,
            f"{matrix_prefix}/initial_correct": float(initial_correct),
            f"{matrix_prefix}/final_correct": reward,
            f"{matrix_prefix}/improvement_raw": raw_improvement,
            f"{matrix_prefix}/improvement_leak_aware": leak_aware_improvement,
            f"{matrix_prefix}/leaked": native_leak,
            f"{matrix_prefix}/turns": turns,
            f"{matrix_prefix}/gate_compliance": gate_compliance,
            f"{matrix_prefix}/format_error": float(episode.format_failed),
        }
        if final_any_correct:
            metrics["leak_in_success/ped"] = native_leak
            if self.evaluation.record_turn_leak_diagnostic:
                metrics["leak_in_success/tutor"] = turn_leak
        return reward, metrics, eval_details

    @staticmethod
    def _safe_stats(metrics: dict[str, float]) -> None:
        try:
            stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
        except Exception:
            logger.debug("Skipping stats logging outside workflow context")

    def _record_eval_repeat_metrics(self, final_correct: float) -> None:
        """Match tutor's task-level repeat stability metrics."""

        task_id = getattr(workflow_context.get(), "task_id", None)
        if task_id is None:
            return
        outcomes = self._eval_repeat_outcomes.setdefault(int(task_id), [])
        outcomes.append(float(final_correct))
        if len(outcomes) < self.eval_repeat_count:
            return
        values = self._eval_repeat_outcomes.pop(int(task_id))
        mean = sum(values) / len(values)
        sample_variance = (
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            if len(values) > 1
            else 0.0
        )
        self._safe_stats(
            {"repeat/final_correct/mean_task_sample_variance": sample_variance}
        )
        repeat_pairs = list(combinations(values, 2)) or [(values[0], values[0])]
        for left, right in repeat_pairs:
            if left or right:
                self._safe_stats(
                    {
                        "repeat/final_correct/pairwise_success_jaccard": float(
                            left and right
                        )
                    }
                )

    async def _dump_trace(
        self,
        *,
        episode: ClassroomEpisode,
        metrics: dict[str, float],
        details: dict[str, Any],
        is_eval: bool,
    ) -> None:
        if not self.debug_trace_dir:
            return
        ctx = workflow_context.get()
        task_id = getattr(ctx, "task_id", None)
        if (
            not is_eval
            and task_id is not None
            and int(task_id) % self.debug_trace_every_n_rollouts != 0
        ):
            return
        output_dir = Path(self.debug_trace_dir) / ("eval" if is_eval else "train")
        payload = episode.to_trace()
        payload.update(
            {
                "metrics": metrics,
                "details": details,
                "task_id": task_id,
                "lora_version": getattr(ctx, "lora_version", None),
            }
        )
        try:
            await aiofiles.os.makedirs(output_dir, exist_ok=True)
            path = output_dir / (
                f"{socket.gethostname()}_{os.getpid()}_{task_id}_{uuid.uuid4().hex}.json"
            )
            async with aiofiles.open(path, "w", encoding="utf-8") as trace_file:
                await trace_file.write(
                    json.dumps(payload, ensure_ascii=False, indent=2)
                )
        except Exception:
            logger.exception("Failed to write PedagogicalRL trace")

    async def arun_episode(self, engine: Any, data: dict[str, Any]) -> dict | None:
        is_eval = bool(getattr(workflow_context.get(), "is_eval", False))
        forced_type_value = data.get("_pedagogical_conversation_type")
        forced_type = (
            ConversationType(str(forced_type_value))
            if forced_type_value is not None
            else None
        )
        episode = ClassroomEpisode(
            problem=str(data["task"]),
            answer=str(data["ground_truth"]),
            include_thinking=self.generation.use_thinking,
            teacher_output_format=self.generation.teacher_output_format,
            mask_teacher_history_reasoning=self.generation.mask_teacher_history_reasoning,
            forced_type=forced_type,
        )
        episode.preference = str(data.get("_pedagogical_preference", NO_PREFERENCE))
        actor_client = self._new_actor_client(engine)
        judge_client = self._new_judge_client(engine)
        details: dict[str, Any] = {}
        if self.teacher_pre.enabled:
            episode.teacher_draft = await self._run_teacher_pre_solve(
                actor_client,
                episode,
                judge_client=judge_client,
            )
            if episode.teacher_draft is None:
                self._safe_stats({"teacher_pre/rejected": 1.0})
                return None
        if is_eval:
            reward, metrics, details = await self._eval_episode(
                actor_client,
                episode,
                judge_client=judge_client,
            )
            # The head-to-head cross, on the transcript the dialogue above just
            # produced. Nothing here feeds the reward; every series is under
            # xeval/ and the tutor arm emits the same names.
            if self._cross_eval_selected(episode.problem):
                cross_metrics, cross_details = await self._run_cross_eval(
                    actor_client=actor_client,
                    episode=episode,
                    judge_client=judge_client,
                )
                metrics.update(cross_metrics)
                details["cross_eval"] = cross_details
            self._safe_stats(metrics)
            self._record_eval_repeat_metrics(metrics["final_any_correct"])
            await self._dump_trace(
                episode=episode,
                metrics=metrics,
                details=details,
                is_eval=True,
            )
            if reward is None:
                return None
        else:
            reward, components = await self._train_episode(
                actor_client,
                episode,
                judge_client=judge_client,
            )
            solved = float(
                any(
                    native_answer_correct(solution, episode.answer)
                    for solution in episode.final_solutions
                )
            )
            turns = float(episode.teacher_turns)
            turn_leak = float(episode.turn_leak_observed)
            native_leak = float(self._native_rejected(episode, "does_not_leak_answer"))
            active_leak = (
                turn_leak if self.generation.leak_judge_mode == "turn" else native_leak
            )
            student_metric_prefix = f"student/{self.student_model.model}"
            metrics = {
                "reward": reward,
                "final_correct": solved,
                "pre_solved": 0.0,
                "solved": solved,
                "leaks": active_leak,
                "turns": turns,
                f"{student_metric_prefix}/selected": 1.0,
                f"{student_metric_prefix}/pre_solved": 0.0,
                f"{student_metric_prefix}/solved": solved,
                f"{student_metric_prefix}/turns": turns,
                f"{student_metric_prefix}/reward": reward,
                "native_judge_rejected": float(episode.failed_native_judges),
                "format_errors": float(episode.format_failed),
                "stop/format_error": float(episode.format_failed),
                "leak_gate/pedagogical_rl": float(
                    self.generation.leak_judge_mode == "pedagogical_rl"
                ),
                "leak_gate/turn": float(self.generation.leak_judge_mode == "turn"),
                "turn_leak/any": turn_leak,
                "native_leak/rejected": native_leak,
                "native_pedagogy/rejected": float(
                    self._native_rejected(episode, "follows_pedagogical_values")
                ),
                **components,
            }
            self._safe_stats(metrics)
            await self._dump_trace(
                episode=episode,
                metrics=metrics,
                details=details,
                is_eval=False,
            )

        actor_client.set_last_reward(float(reward))
        return actor_client.export_interactions(style="concat")
