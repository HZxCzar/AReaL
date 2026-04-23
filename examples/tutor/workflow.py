from __future__ import annotations

import json
import logging as py_logging
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Any

try:
    from areal import workflow_context
    from areal.utils import logging, stats_tracker
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
    stats_tracker = _DummyTracker()
    logging = _DummyLogging()

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


class TutorAgentWorkflow:
    def __init__(
        self,
        max_turns: int = 6,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 512,
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
        transfer_success_reward: float = 1.2,
        transfer_fail_reward: float = 0.6,
        teacher_system_prompt: str = "",
        student_system_prompt: str = "",
        judge_system_prompt: str = "",
        leak_check_system_prompt: str = "",
        generator_system_prompt: str = "",
        max_episode_total_tokens: int | None = None,
        token_budget_penalty: float = -0.2,
    ):
        self.max_turns = max_turns
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.transfer_success_reward = transfer_success_reward
        self.transfer_fail_reward = transfer_fail_reward
        self.teacher_system_prompt = teacher_system_prompt.strip()
        self.student_system_prompt = student_system_prompt.strip()
        self.judge_system_prompt = judge_system_prompt.strip()
        self.leak_check_system_prompt = leak_check_system_prompt.strip()
        self.generator_system_prompt = generator_system_prompt.strip()
        self.max_episode_total_tokens = max_episode_total_tokens
        self.token_budget_penalty = token_budget_penalty
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
        )
        self.aux_caller = AsyncLLMCaller(aux_config)

    async def run(self, data: dict[str, Any], **extra_kwargs):
        teacher_client = make_teacher_client(extra_kwargs)
        history: list[dict[str, Any]] = []
        rewards: dict[str, float] = {}
        task = str(data["task"])
        ground_truth = str(data["ground_truth"])
        student_answer, student_error = await self._run_student(task, teacher_action=None)
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

        for round_idx in range(1, self.max_turns + 1):
            prompt = self._build_teacher_prompt(
                task=task,
                ground_truth=ground_truth,
                latest_student_answer=latest_student_answer,
                latest_judge_result=latest_judge_result,
                history=history,
                round_idx=round_idx,
                pre_solved=pre_solved,
            )
            response = await teacher_client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": self.teacher_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=self.max_completion_tokens,
            )
            teacher_action = _strip_think_tags(
                response.choices[0].message.content or ""
            ).strip()
            budget_snapshot = token_budget.observe_turn(
                response=response,
                prompt_text=prompt,
                completion_text=teacher_action,
            )

            if pre_solved:
                rewards[response.id] = 0.0
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
                rewards[response.id] = reward
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
                continue

            student_answer, student_error = await self._run_student(task, teacher_action)
            judge_result = self._score_aime_answer(task, ground_truth, student_answer)
            latest_student_answer = student_answer
            latest_judge_result = judge_result

            reward = -0.1
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
            if judge_result.correct:
                transfer_result = await self._run_transfer_round(task, ground_truth)
                transfer_success = transfer_result["transfer_success"]
                reward = (
                    self.transfer_success_reward
                    if transfer_success
                    else self.transfer_fail_reward
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
            rewards[response.id] = final_reward
            record["reward"] = final_reward
            history.append(record)
            if budget_snapshot.stop_reason is not None and termination_reason == "continue":
                termination_reason = budget_snapshot.stop_reason
                break
            if judge_result.correct:
                break

        _safe_scalar(
            reward=sum(rewards.values()),
            num_turns=len(rewards),
            leak_count=leak_count,
            primary_success=bool(latest_judge_result.correct),
            transfer_success=transfer_success,
        )
        return rewards

    async def _run_student(
        self, task: str, teacher_action: str | None
    ) -> tuple[str, str | None]:
        teacher_feedback = teacher_action or "(none, produce the first answer attempt)"
        prompt = dedent(
            f"""\
            Task:
            {task}

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
        answer, answer_error = await self._run_student(generation.task, teacher_action=None)
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

    def _build_teacher_prompt(
        self,
        task: str,
        ground_truth: str,
        latest_student_answer: str,
        latest_judge_result: JudgeResult,
        history: list[dict[str, Any]],
        round_idx: int,
        pre_solved: bool,
    ) -> str:
        lines = [f"1. Student initial answer: {latest_student_answer or '(empty)'}"]
        idx = 2
        for record in history:
            lines.append(
                f"{idx}. Teacher turn {record['round_idx']}: {record.get('teacher_action', '(empty)') or '(empty)'}"
            )
            idx += 1
            if record.get("leak_detected"):
                lines.append(
                    f"{idx}. Leak feedback after teacher turn {record['round_idx']}: {record.get('leak_feedback') or 'The teacher leaked the answer.'}"
                )
            else:
                lines.append(
                    f"{idx}. Student reply after teacher turn {record['round_idx']}: {record.get('student_answer', '(empty)') or '(empty)'}"
                )
            idx += 1
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

            Teacher-Side History:
            {"\n".join(lines)}

            Current Status:
            - Current round: {round_idx - 1}/{self.max_turns}
            - Remaining rounds: {max(self.max_turns - round_idx + 1, 0)}
            - Latest student answer: {latest_student_answer or "(empty)"}
            - Latest judge result: {"correct" if latest_judge_result.correct else "incorrect"}
            - Latest judge feedback: {latest_judge_result.feedback}
            - Pre-solved: {pre_solved}
            - Note: {pre_solved_note}

            Reply with concise tutoring guidance only. Do not reveal the final answer directly.
            """
        ).strip()

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


def _strip_think_tags(text: str) -> str:
    return re.sub(r"</?think>", "", text or "", flags=re.IGNORECASE).strip()


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
