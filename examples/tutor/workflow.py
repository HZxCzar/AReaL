from __future__ import annotations

import json
import logging as py_logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

try:
    from areal import workflow_context
    from areal.api import RolloutWorkflow
    from areal.experimental.openai import ArealOpenAI
    from areal.utils import logging, stats_tracker
    from areal.utils.hf_utils import load_hf_tokenizer
except Exception:  # pragma: no cover - lightweight local test environments
    class _DummyWorkflowContext:
        @staticmethod
        def stat_scope():
            return "examples"

    class _DummyTracker:
        @staticmethod
        def get(_scope):
            return _DummyTracker()

        def scalar(self, **_metrics):
            return None

    class _DummyLogging:
        @staticmethod
        def getLogger(name: str):
            return py_logging.getLogger(name)

    workflow_context = _DummyWorkflowContext()
    RolloutWorkflow = object
    ArealOpenAI = Any
    stats_tracker = _DummyTracker()
    logging = _DummyLogging()
    def load_hf_tokenizer(path):  # type: ignore[no-redef]
        return path

from examples.common.chat_budget import ChatContextBudget
from examples.common.episode_budget import EpisodeTokenBudget
from examples.common.openai_utils import AsyncLLMCaller, AuxModelConfig, make_teacher_client
from examples.common.parsing import join_errors, parse_json_dict

logger = logging.getLogger("TutorWorkflow")


@dataclass(slots=True)
class JudgeResult:
    raw_output: str
    correct: bool
    feedback: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class LeakCheckResult:
    raw_output: str
    leaked: bool
    feedback: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class GeneratedProblemResult:
    raw_output: str
    task: str
    ground_truth: str
    similarity_notes: str
    parse_error: str | None
    raw_result: dict[str, Any]


def _safe_scalar(**metrics: Any) -> None:
    try:
        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
    except Exception:
        logger.debug("Skipping stats logging outside workflow context.")


class TutorAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: Any | None = None,
        tokenizer: str | Any | None = None,
        max_turns: int = 6,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 512,
        tool_call_parser: str = "qwen25",
        reasoning_parser: str = "qwen3",
        aux_base_url: str = "http://127.0.0.1:30000/v1",
        aux_model: str = "qwen-aux",
        aux_api_key: str = "EMPTY",
        aux_timeout: int = 120,
        aux_max_tokens: int = 1024,
        aux_temperature: float = 0.7,
        aux_top_p: float | None = None,
        max_concurrent_aux_calls: int = 8,
        api_params_config_path: str | None = None,
        api_params_key: str | None = None,
        primary_success_reward: float = 1.0,
        transfer_bonus_reward: float = 0.5,
        transfer_success_reward: float = 1.2,
        transfer_fail_reward: float = 0.6,
        teacher_system_prompt: str = "",
        student_system_prompt: str = "",
        judge_system_prompt: str = "",
        leak_check_system_prompt: str = "",
        generator_system_prompt: str = "",
        debug_trace_dir: str | None = None,
        debug_trace_every_n_rollouts: int = 1,
        max_episode_total_tokens: int | None = None,
        token_budget_penalty: float = -0.2,
        tokenizer_path: str | None = None,
        model_context_length: int | None = None,
        context_window_margin: int = 256,
    ):
        self.max_turns = max_turns
        self.gconfig = gconfig
        self.temperature = (
            gconfig.temperature if gconfig is not None else temperature
        )
        self.top_p = gconfig.top_p if gconfig is not None else top_p
        self.max_completion_tokens = (
            gconfig.max_new_tokens if gconfig is not None else max_completion_tokens
        )
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.primary_success_reward = primary_success_reward
        self.transfer_bonus_reward = transfer_bonus_reward
        self.transfer_success_reward = transfer_success_reward
        self.transfer_fail_reward = transfer_fail_reward
        self.teacher_system_prompt = teacher_system_prompt.strip()
        self.student_system_prompt = student_system_prompt.strip()
        self.judge_system_prompt = judge_system_prompt.strip()
        self.leak_check_system_prompt = leak_check_system_prompt.strip()
        self.generator_system_prompt = generator_system_prompt.strip()
        self.debug_trace_dir = debug_trace_dir.strip() if debug_trace_dir else ""
        self.debug_trace_every_n_rollouts = max(1, int(debug_trace_every_n_rollouts))
        self.max_episode_total_tokens = max_episode_total_tokens
        self.token_budget_penalty = token_budget_penalty
        self.last_history: list[dict[str, Any]] = []
        self.tokenizer = (
            load_hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.teacher_context_budget = ChatContextBudget(
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            safety_margin=context_window_margin,
        )
        aux_config = AuxModelConfig(
            base_url=aux_base_url,
            model=aux_model,
            api_key=aux_api_key,
            timeout=aux_timeout,
            max_tokens=aux_max_tokens,
            temperature=aux_temperature,
            top_p=aux_top_p,
            max_concurrency=max_concurrent_aux_calls,
            api_params_config_path=api_params_config_path,
            api_params_key=api_params_key,
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            context_window_margin=context_window_margin,
        )
        self.aux_caller = AsyncLLMCaller(aux_config)

    async def arun_episode(self, engine, data: dict[str, Any]):
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            tool_call_parser=self.tool_call_parser,
            reasoning_parser=self.reasoning_parser,
            chat_template_type="concat",
        )
        result = await self._run_episode(data, direct_client=client)
        if result is None:
            return None
        total_reward, last_completion_id = result
        if last_completion_id is None:
            return None
        client.set_reward(last_completion_id, total_reward)
        client.apply_reward_discount(turn_discount=1.0)
        interactions = client.export_interactions(style="concat")
        if len(interactions) != 1:
            raise RuntimeError(
                f"Tutor rollout should export exactly 1 interaction, got {len(interactions)}"
            )
        return interactions

    async def run(self, data: dict[str, Any], **extra_kwargs):
        teacher_client = make_teacher_client(extra_kwargs)
        result = await self._run_episode(data, external_client=teacher_client)
        if result is None:
            return {}
        total_reward, _ = result
        return total_reward

    async def _run_episode(
        self,
        data: dict[str, Any],
        direct_client: Any | None = None,
        external_client: Any | None = None,
    ) -> tuple[float, str | None] | None:
        if (direct_client is None) == (external_client is None):
            raise ValueError("Exactly one teacher client must be provided.")
        history: list[dict[str, Any]] = []
        total_reward = 0.0
        last_completion_id: str | None = None
        task = str(data["task"])
        ground_truth = str(data["ground_truth"])
        student_answer, student_error = await self._run_student(
            task, teacher_action=None, history=[]
        )
        initial_student_answer = student_answer
        judge_result = self._score_aime_answer(task, ground_truth, student_answer)
        latest_student_answer = student_answer
        latest_judge_result = judge_result
        pre_solved = judge_result.correct
        leak_count = 0
        termination_reason = "pre_solved" if pre_solved else "max_turns"
        transfer_success = False
        token_budget = EpisodeTokenBudget(
            max_episode_total_tokens=self.max_episode_total_tokens,
        )
        teacher_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.teacher_system_prompt},
            {
                "role": "user",
                "content": self._build_teacher_initial_prompt(
                    task=task,
                    ground_truth=ground_truth,
                    initial_student_answer=initial_student_answer,
                    latest_judge_result=latest_judge_result,
                    pre_solved=pre_solved,
                    round_idx=1,
                ),
            },
        ]

        for round_idx in range(1, self.max_turns + 1):
            if direct_client is not None:
                prompt_tokens = self.teacher_context_budget.count_message_tokens(
                    teacher_messages
                )
                safe_max_completion_tokens = self.max_completion_tokens
            else:
                safe_max_completion_tokens, prompt_tokens = self.teacher_context_budget.clamp_max_completion_tokens(
                    teacher_messages,
                    self.max_completion_tokens,
                )
                if self.max_episode_total_tokens is not None:
                    safe_max_completion_tokens = min(
                        safe_max_completion_tokens,
                        max(0, self.max_episode_total_tokens - prompt_tokens),
                    )
            if safe_max_completion_tokens <= 0:
                total_reward += self.token_budget_penalty
                history.append(
                    {
                        "round_idx": round_idx,
                        "teacher_action": "",
                        "reward": self.token_budget_penalty,
                        "turn_prompt_tokens": prompt_tokens,
                        "turn_completion_tokens": 0,
                        "turn_total_tokens": prompt_tokens,
                        "termination_feedback": (
                            "Episode terminated before teacher generation because prompt length "
                            "exhausted the available episode or context budget."
                        ),
                    }
                )
                break
            if direct_client is not None:
                create_kwargs = {
                    "messages": teacher_messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_completion_tokens": safe_max_completion_tokens,
                }
                if self.max_episode_total_tokens is not None:
                    create_kwargs["max_total_tokens"] = self.max_episode_total_tokens
                try:
                    response = await direct_client.chat.completions.create(
                        **create_kwargs
                    )
                except ValueError as exc:
                    if _is_max_total_tokens_error(exc):
                        total_reward += self.token_budget_penalty
                        history.append(
                            {
                                "round_idx": round_idx,
                                "teacher_action": "",
                                "reward": self.token_budget_penalty,
                                "turn_prompt_tokens": prompt_tokens,
                                "turn_completion_tokens": 0,
                                "turn_total_tokens": prompt_tokens,
                                "termination_feedback": (
                                    "Episode terminated before teacher generation because "
                                    "the real concat prompt exhausted the token budget: "
                                    f"{exc}"
                                ),
                                "length_penalty": self.token_budget_penalty,
                            }
                        )
                        termination_reason = "episode_total_token_budget"
                        break
                    raise
            else:
                response = await external_client.chat.completions.create(
                    model="default",
                    messages=teacher_messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_completion_tokens=safe_max_completion_tokens,
                )
            last_completion_id = response.id
            teacher_message = response.choices[0].message
            teacher_action = _strip_think_tags(teacher_message.content or "").strip()
            budget_snapshot = token_budget.observe_turn(
                response=response,
                prompt_text=teacher_messages[-1]["content"],
                completion_text=teacher_action,
            )

            if pre_solved:
                termination_reason = "pre_solved"
                history.append(
                    {
                        "round_idx": round_idx,
                        "teacher_action": teacher_action,
                        "reward": 0.0,
                        "pre_solved_terminal": True,
                        "turn_prompt_tokens": budget_snapshot.turn_prompt_tokens,
                        "turn_completion_tokens": budget_snapshot.turn_completion_tokens,
                        "turn_total_tokens": budget_snapshot.turn_total_tokens,
                        "termination_feedback": budget_snapshot.stop_feedback,
                    }
                )
                break

            leak_result = await self._run_leak_check(task, ground_truth, teacher_action)
            if leak_result.leaked:
                leak_count += 1
                reward = -0.6
                if budget_snapshot.stop_reason is not None:
                    reward += self.token_budget_penalty
                total_reward += reward
                history.append(
                    {
                        "round_idx": round_idx,
                        "teacher_action": teacher_action,
                        "reward": reward,
                        "leak_detected": True,
                        "leak_feedback": leak_result.feedback,
                        "turn_prompt_tokens": budget_snapshot.turn_prompt_tokens,
                        "turn_completion_tokens": budget_snapshot.turn_completion_tokens,
                        "turn_total_tokens": budget_snapshot.turn_total_tokens,
                        "termination_feedback": budget_snapshot.stop_feedback,
                    }
                )
                if budget_snapshot.stop_reason is not None:
                    termination_reason = budget_snapshot.stop_reason
                    break
                termination_reason = "max_turns" if round_idx >= self.max_turns else "continue"
                if round_idx < self.max_turns:
                    teacher_messages.extend(
                        [
                            teacher_message.model_dump(exclude_none=True),
                            {
                                "role": "user",
                                "content": self._build_teacher_followup_prompt(
                                    latest_record=history[-1],
                                    latest_judge_result=latest_judge_result,
                                    round_idx=round_idx + 1,
                                    pre_solved=pre_solved,
                                ),
                            },
                        ]
                    )
                continue

            student_answer, student_error = await self._run_student(
                task, teacher_action, history=history
            )
            judge_result = self._score_aime_answer(task, ground_truth, student_answer)
            latest_student_answer = student_answer
            latest_judge_result = judge_result

            reward = 0.0
            record: dict[str, Any] = {
                "round_idx": round_idx,
                "teacher_action": teacher_action,
                "student_answer": student_answer,
                "student_error": student_error,
                "judge_feedback": judge_result.feedback,
                "judge_correct": judge_result.correct,
                "leak_detected": False,
                "turn_prompt_tokens": budget_snapshot.turn_prompt_tokens,
                "turn_completion_tokens": budget_snapshot.turn_completion_tokens,
                "turn_total_tokens": budget_snapshot.turn_total_tokens,
                "termination_feedback": budget_snapshot.stop_feedback,
            }
            record["student_visible_summary"] = self._build_student_visible_summary(record)
            if judge_result.correct:
                transfer_result = await self._run_transfer_round(task, ground_truth)
                transfer_success = transfer_result["transfer_success"]
                step_factor = (self.max_turns - round_idx + 1) / max(self.max_turns, 1)
                reward = self.primary_success_reward * step_factor
                if transfer_success:
                    reward += self.transfer_bonus_reward * step_factor
                record["step_factor"] = step_factor
                record["primary_reward"] = self.primary_success_reward * step_factor
                record["transfer_bonus_reward"] = (
                    self.transfer_bonus_reward * step_factor if transfer_success else 0.0
                )
                record.update(transfer_result)
                termination_reason = (
                    "success_transfer_pass"
                    if transfer_success
                    else "success_transfer_fail"
                )
            elif round_idx >= self.max_turns:
                termination_reason = "max_turns"
            else:
                termination_reason = "continue"

            final_reward = reward
            if budget_snapshot.stop_reason is not None:
                final_reward += self.token_budget_penalty
                record["length_penalty"] = self.token_budget_penalty
            total_reward += final_reward
            record["reward"] = final_reward
            history.append(record)
            if budget_snapshot.stop_reason is not None and termination_reason == "continue":
                termination_reason = budget_snapshot.stop_reason
                break
            if judge_result.correct:
                break
            if round_idx < self.max_turns:
                teacher_messages.extend(
                    [
                        teacher_message.model_dump(exclude_none=True),
                        {
                            "role": "user",
                            "content": self._build_teacher_followup_prompt(
                                latest_record=record,
                                latest_judge_result=latest_judge_result,
                                round_idx=round_idx + 1,
                                pre_solved=pre_solved,
                            ),
                        },
                    ]
                )

        success_rounds = [
            int(record["round_idx"])
            for record in history
            if bool(record.get("judge_correct", False))
        ]
        leak_rounds = [
            int(record["round_idx"])
            for record in history
            if bool(record.get("leak_detected", False))
        ]
        primary_reward_sum = sum(
            float(record.get("primary_reward", 0.0) or 0.0) for record in history
        )
        transfer_bonus_sum = sum(
            float(record.get("transfer_bonus_reward", 0.0) or 0.0)
            for record in history
        )
        length_penalty_count = sum(1 for record in history if "length_penalty" in record)
        length_penalty_sum = sum(
            float(record.get("length_penalty", 0.0) or 0.0) for record in history
        )
        max_completion_tokens = max(
            [int(record.get("turn_completion_tokens", 0) or 0) for record in history]
            or [0]
        )
        max_total_tokens = max(
            [int(record.get("turn_total_tokens", 0) or 0) for record in history]
            or [0]
        )
        avg_completion_tokens = sum(
            int(record.get("turn_completion_tokens", 0) or 0) for record in history
        ) / max(len(history), 1)
        avg_total_tokens = sum(
            int(record.get("turn_total_tokens", 0) or 0) for record in history
        ) / max(len(history), 1)
        _safe_scalar(
            reward=total_reward,
            num_turns=len(history),
            leak_count=leak_count,
            primary_success=bool(latest_judge_result.correct),
            transfer_success=transfer_success,
            term_budget=float(termination_reason == "episode_total_token_budget"),
            term_max_turns=float(termination_reason == "max_turns"),
            term_success=float(
                termination_reason
                in {"success_transfer_pass", "success_transfer_fail"}
            ),
            term_pre_solved=float(termination_reason == "pre_solved"),
            term_continue=float(termination_reason == "continue"),
            length_penalty_rate=float(length_penalty_count > 0),
            length_penalty_count=length_penalty_count,
            length_penalty_sum=length_penalty_sum,
            avg_completion_tokens=avg_completion_tokens,
            avg_total_tokens=avg_total_tokens,
            max_completion_tokens=max_completion_tokens,
            max_total_tokens=max_total_tokens,
            primary_reward_sum=primary_reward_sum,
            transfer_bonus_sum=transfer_bonus_sum,
            success_round=success_rounds[0] if success_rounds else 0,
            leak_first_round=leak_rounds[0] if leak_rounds else 0,
        )
        self.last_history = [dict(record) for record in history]
        self._maybe_dump_debug_trace(
            task=task,
            ground_truth=ground_truth,
            initial_student_answer=initial_student_answer,
            latest_student_answer=latest_student_answer,
            total_reward=total_reward,
            history=history,
            termination_reason=termination_reason,
            primary_success=bool(latest_judge_result.correct),
            transfer_success=bool(transfer_success),
            leak_count=leak_count,
        )
        if last_completion_id is None:
            return None
        return total_reward, last_completion_id

    async def _run_student(
        self,
        task: str,
        teacher_action: str | None,
        history: list[dict[str, Any]],
    ) -> tuple[str, str | None]:
        teacher_feedback = teacher_action or "(none, produce the first answer attempt)"
        visible_history = self._student_visible_history_summaries(history)
        prompt = dedent(
            f"""\
            Task:
            {task}

            Visible student history:
            {"No previous visible turns." if not visible_history else "\n".join(visible_history)}

            Current teacher feedback:
            {teacher_feedback}

            Reply with only the student's next answer attempt.
            """
        ).strip()
        try:
            answer = await self.aux_caller.call_text(
                [
                    {"role": "system", "content": self.student_system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            return answer, None
        except Exception as exc:
            return "", f"Student call failed: {exc}"

    async def _run_leak_check(
        self, task: str, ground_truth: str, teacher_action: str
    ) -> LeakCheckResult:
        prompt = dedent(
            f"""\
            Task:
            {task}

            Ground Truth:
            {ground_truth}

            Teacher Message:
            {teacher_action or "(empty)"}

            Return JSON only with this schema:
            {{
              "leaked": false,
              "feedback": "short explanation"
            }}
            """
        ).strip()
        try:
            raw_output = await self.aux_caller.call_text(
                [
                    {"role": "system", "content": self.leak_check_system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            return LeakCheckResult("", False, f"Leak check failed: {exc}", str(exc), {})
        return self._parse_leak_check_result(raw_output)

    async def _run_transfer_round(
        self, task: str, ground_truth: str
    ) -> dict[str, Any]:
        generation = await self._run_transfer_generation(task, ground_truth)
        payload: dict[str, Any] = {
            "transfer_triggered": True,
            "transfer_task": generation.task,
            "transfer_ground_truth": generation.ground_truth,
            "transfer_generation_error": generation.parse_error,
            "transfer_success": False,
        }
        if not generation.task or not generation.ground_truth:
            return payload
        answer, answer_error = await self._run_student(
            generation.task, teacher_action=None, history=[]
        )
        judge_result = self._score_aime_answer(
            generation.task, generation.ground_truth, answer
        )
        payload.update(
            {
                "transfer_student_answer": answer,
                "transfer_student_error": answer_error,
                "transfer_judge_feedback": judge_result.feedback,
                "transfer_success": judge_result.correct,
            }
        )
        return payload

    async def _run_transfer_generation(
        self, task: str, ground_truth: str
    ) -> GeneratedProblemResult:
        prompt = dedent(
            f"""\
            Original Task:
            {task}

            Original Ground Truth:
            {ground_truth}

            Create one new, self-contained problem that is clearly similar in structure and solution method, but not a restatement of the original problem.
            Return JSON only with this schema:
            {{
              "task": "new problem statement",
              "ground_truth": "final answer only",
              "similarity_notes": "optional short note"
            }}
            """
        ).strip()
        try:
            raw_output = await self.aux_caller.call_text(
                [
                    {"role": "system", "content": self.generator_system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            error = f"Generator call failed: {exc}"
            return GeneratedProblemResult("", "", "", "", error, {})
        parsed, parse_error = parse_json_dict(raw_output)
        if parsed is None:
            return GeneratedProblemResult(raw_output, "", "", "", parse_error, {})
        task_value = parsed.get("task", "")
        gt_value = parsed.get("ground_truth", "")
        notes = parsed.get("similarity_notes", "")
        if not isinstance(task_value, str):
            parse_error = join_errors(parse_error, '"task" must be a string')
            task_value = ""
        if not isinstance(gt_value, str):
            parse_error = join_errors(parse_error, '"ground_truth" must be a string')
            gt_value = ""
        if not isinstance(notes, str):
            notes = str(notes)
        return GeneratedProblemResult(
            raw_output=raw_output,
            task=task_value.strip(),
            ground_truth=gt_value.strip(),
            similarity_notes=notes.strip(),
            parse_error=parse_error,
            raw_result=parsed,
        )

    def _build_teacher_initial_prompt(
        self,
        task: str,
        ground_truth: str,
        initial_student_answer: str,
        round_idx: int,
        latest_judge_result: JudgeResult,
        pre_solved: bool,
    ) -> str:
        pre_solved_note = (
            "The student already solved the task during reset. Your next action will terminate the episode with reward 0."
            if pre_solved
            else "The student still needs guidance."
        )
        return dedent(
            f"""\
            Task:
            {task}

            Ground Truth:
            {ground_truth}

            Turn 0:
            - Student initial answer: {initial_student_answer or '(empty)'}
            - Initial judge result: {'correct' if latest_judge_result.correct else 'incorrect'}
            - Initial judge feedback: {latest_judge_result.feedback}
            - Current round: {round_idx - 1}/{self.max_turns}
            - Remaining rounds: {max(self.max_turns - round_idx + 1, 0)}
            - Pre-solved: {pre_solved}
            - Note: {pre_solved_note}

            Reply with concise tutoring guidance only. Do not reveal the final answer directly.
            """
        ).strip()

    def _build_teacher_followup_prompt(
        self,
        latest_record: dict[str, Any],
        latest_judge_result: JudgeResult,
        round_idx: int,
        pre_solved: bool,
    ) -> str:
        lines = [
            f"Turn {latest_record['round_idx']} update:",
            f"- Current round: {round_idx - 1}/{self.max_turns}",
            f"- Remaining rounds: {max(self.max_turns - round_idx + 1, 0)}",
            f"- Pre-solved: {pre_solved}",
        ]
        if latest_record.get("leak_detected"):
            lines.append(
                "- Env feedback: "
                + (
                    latest_record.get("leak_feedback")
                    or "The previous teacher turn leaked the answer and the student did not see it."
                )
            )
            lines.append(
                f"- Latest judge result: {'correct' if latest_judge_result.correct else 'incorrect'}"
            )
            lines.append(f"- Latest judge feedback: {latest_judge_result.feedback}")
        else:
            lines.append(
                f"- Student reply: {latest_record.get('student_answer', '(empty)') or '(empty)'}"
            )
            lines.append(
                f"- Judge result: {'correct' if latest_record.get('judge_correct') else 'incorrect'}"
            )
            lines.append(
                f"- Judge feedback: {latest_record.get('judge_feedback') or '(empty)'}"
            )
            if latest_record.get("transfer_triggered"):
                lines.append(
                    f"- Transfer success: {bool(latest_record.get('transfer_success'))}"
                )
                lines.append(
                    f"- Transfer judge feedback: {latest_record.get('transfer_judge_feedback') or '(empty)'}"
                )
            if latest_record.get("termination_feedback"):
                lines.append(
                    f"- Budget feedback: {latest_record.get('termination_feedback')}"
                )
        lines.append(
            "Reply with concise tutoring guidance only. Do not reveal the final answer directly."
        )
        return "\n".join(lines)

    def _score_aime_answer(
        self, task: str, ground_truth: str, student_answer: str
    ) -> JudgeResult:
        extracted_answer = _official_extract_aime_answer(student_answer)
        normalized_prediction = _official_strip_string(extracted_answer) if extracted_answer else ""
        normalized_target = _official_strip_string(ground_truth)
        correct = _official_is_equiv(extracted_answer, ground_truth)
        raw_result = {
            "method": "lm_eval_aime_exact_match",
            "task": task,
            "student_answer": student_answer,
            "extracted_answer": extracted_answer,
            "normalized_prediction": normalized_prediction,
            "normalized_target": normalized_target,
        }
        return JudgeResult(
            raw_output=json.dumps(
                {
                    "correct": correct,
                    "feedback": "Correct." if correct else "Incorrect.",
                    "scoring": raw_result,
                },
                ensure_ascii=True,
                indent=2,
            ),
            correct=correct,
            feedback="Correct." if correct else "Incorrect.",
            parse_error=None,
            raw_result=raw_result,
        )

    def _parse_leak_check_result(self, raw_output: str) -> LeakCheckResult:
        parsed, parse_error = parse_json_dict(raw_output)
        if parsed is None:
            lowered = raw_output.lower()
            leaked = '"leaked": true' in lowered or re.search(r"\byes\b", lowered) is not None
            return LeakCheckResult(
                raw_output=raw_output,
                leaked=leaked,
                feedback="Failed to parse leak-check output.",
                parse_error=parse_error,
                raw_result={},
            )
        leaked = parsed.get("leaked")
        feedback = parsed.get("feedback", "")
        if not isinstance(leaked, bool):
            parse_error = join_errors(parse_error, '"leaked" must be a boolean')
            leaked = False
        if not isinstance(feedback, str):
            parse_error = join_errors(parse_error, '"feedback" must be a string')
            feedback = str(feedback)
        return LeakCheckResult(
            raw_output=raw_output,
            leaked=leaked,
            feedback=feedback or (
                "The teacher revealed the answer directly. The student did not see this turn."
                if leaked
                else "No answer leakage detected."
            ),
            parse_error=parse_error,
            raw_result=parsed,
        )

    def _student_visible_history_summaries(
        self, history: list[dict[str, Any]]
    ) -> list[str]:
        summaries: list[str] = []
        for record in history:
            if record.get("leak_detected"):
                continue
            summary = record.get("student_visible_summary")
            if not isinstance(summary, str) or not summary.strip():
                summary = self._build_student_visible_summary(record)
            summaries.append(summary)
        return summaries

    def _build_student_visible_summary(self, record: dict[str, Any]) -> str:
        teacher_text = _compact_text(record.get("teacher_action", "(empty)"))
        student_text = _compact_text(record.get("student_answer", "(empty)"))
        judge_text = _compact_text(record.get("judge_feedback", "(empty)"))
        return (
            f"Turn {record['round_idx']}: Teacher guidance: {teacher_text}. "
            f"Student reply: {student_text}. Judge feedback: {judge_text}."
        )

    def _maybe_dump_debug_trace(
        self,
        *,
        task: str,
        ground_truth: str,
        initial_student_answer: str,
        latest_student_answer: str,
        total_reward: float,
        history: list[dict[str, Any]],
        termination_reason: str,
        primary_success: bool,
        transfer_success: bool,
        leak_count: int,
    ) -> None:
        if not self.debug_trace_dir:
            return
        try:
            ctx = workflow_context.get()
            task_id = ctx.task_id
            if task_id is None:
                return
            if task_id % self.debug_trace_every_n_rollouts != 0:
                return

            out_dir = Path(self.debug_trace_dir) / ("eval" if ctx.is_eval else "train")
            out_dir.mkdir(parents=True, exist_ok=True)
            file_path = out_dir / f"task_{task_id:08d}_{int(time.time() * 1000)}.json"

            rounds: list[dict[str, Any]] = []
            for record in history:
                rounds.append(
                    {
                        "round_idx": int(record.get("round_idx", 0)),
                        "teacher_action": record.get("teacher_action", ""),
                        "student_answer": record.get("student_answer", ""),
                        "student_error": record.get("student_error"),
                        "judge_feedback": record.get("judge_feedback"),
                        "judge_correct": bool(record.get("judge_correct", False)),
                        "reward": float(record.get("reward", 0.0) or 0.0),
                        "step_factor": record.get("step_factor"),
                        "primary_reward": record.get("primary_reward"),
                        "transfer_bonus_reward": record.get("transfer_bonus_reward"),
                        "leak_detected": bool(record.get("leak_detected", False)),
                        "leak_feedback": record.get("leak_feedback"),
                        "turn_prompt_tokens": int(record.get("turn_prompt_tokens", 0) or 0),
                        "turn_completion_tokens": int(record.get("turn_completion_tokens", 0) or 0),
                        "turn_total_tokens": int(record.get("turn_total_tokens", 0) or 0),
                        "termination_feedback": record.get("termination_feedback"),
                        "transfer_success": bool(record.get("transfer_success", False)),
                        "transfer_task": record.get("transfer_task"),
                        "transfer_student_answer": record.get("transfer_student_answer"),
                    }
                )

            payload = {
                "task_id": task_id,
                "is_eval": bool(ctx.is_eval),
                "termination_reason": termination_reason,
                "total_reward": float(total_reward),
                "num_turns": len(history),
                "primary_success": bool(primary_success),
                "transfer_success": bool(transfer_success),
                "leak_count": int(leak_count),
                "primary_reward_sum": float(
                    sum(float(record.get("primary_reward", 0.0) or 0.0) for record in history)
                ),
                "transfer_bonus_sum": float(
                    sum(
                        float(record.get("transfer_bonus_reward", 0.0) or 0.0)
                        for record in history
                    )
                ),
                "length_penalty_sum": float(
                    sum(float(record.get("length_penalty", 0.0) or 0.0) for record in history)
                ),
                "task": task,
                "ground_truth": ground_truth,
                "initial_student_answer": initial_student_answer,
                "latest_student_answer": latest_student_answer,
                "rounds": rounds,
            }
            file_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Tutor debug trace dumped to %s", os.fspath(file_path))
        except Exception:
            logger.exception("Failed to dump tutor debug trace.")


def _strip_think_tags(text: str) -> str:
    text = text or ""
    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<think\b[^>]*>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE).strip()


def _is_max_total_tokens_error(exc: Exception) -> bool:
    return "exceeds max_total_tokens" in str(exc)


def _compact_text(text: str, max_chars: int = 240) -> str:
    compact = " ".join((text or "").split())
    if not compact:
        return "(empty)"
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_string += "\\sqrt{" + a + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _official_strip_string(string: str) -> str:
    return _strip_string(string)


def _official_is_equiv(prediction: str, reference: str) -> bool:
    return _official_strip_string(prediction) == _official_strip_string(reference)


def _official_extract_aime_answer(response: str) -> str:
    matches = list(
        re.finditer(r"(?:^|[^0-9])([0-9]{1,4})(?:[^0-9]|$)", response or "")
    )
    if not matches:
        return (response or "").strip()
    return matches[-1].group(1)
