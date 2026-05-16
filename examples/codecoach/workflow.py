from __future__ import annotations

import asyncio
import json
import logging as py_logging
import re
import subprocess
import sys
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
from examples.common.files import make_run_dir
from examples.common.episode_budget import EpisodeTokenBudget
from examples.common.openai_utils import AsyncLLMCaller, AuxModelConfig, make_teacher_client
from examples.common.parsing import join_errors, parse_json_dict

logger = logging.getLogger("CodeCoachWorkflow")


@dataclass(slots=True)
class StudentReply:
    raw_output: str
    message: str
    replace_code: bool
    code: str | None
    parse_error: str | None


@dataclass(slots=True)
class EvalResult:
    score: float
    target_ratio: float
    validity: float
    success: bool
    raw_result: dict[str, Any]
    error: str | None = None


def _safe_scalar(**metrics: Any) -> None:
    try:
        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
    except Exception:
        logger.debug("Skipping stats logging outside workflow context.")


class CodeCoachAgentWorkflow(RolloutWorkflow):
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
        student_base_url: str = "http://127.0.0.1:30000/v1",
        student_model: str = "qwen-student",
        student_api_key: str = "EMPTY",
        student_timeout: int = 120,
        student_max_tokens: int = 4096,
        student_temperature: float = 0.7,
        student_top_p: float | None = None,
        max_concurrent_students: int = 8,
        student_request_params: dict[str, Any] | None = None,
        work_dir_root: str = "examples/codecoach/artifacts",
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
        self.work_dir_root = work_dir_root
        self.max_episode_total_tokens = max_episode_total_tokens
        self.token_budget_penalty = token_budget_penalty
        self.tokenizer = (
            load_hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.teacher_context_budget = ChatContextBudget(
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            safety_margin=context_window_margin,
        )
        self.student_caller = AsyncLLMCaller(
            AuxModelConfig(
                base_url=student_base_url,
                model=student_model,
                api_key=student_api_key,
                timeout=student_timeout,
                max_tokens=student_max_tokens,
                temperature=student_temperature,
                top_p=student_top_p,
                max_concurrency=max_concurrent_students,
                request_params=student_request_params or {},
                tokenizer_path=tokenizer_path,
                context_length=model_context_length,
                context_window_margin=context_window_margin,
            )
        )

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
                f"CodeCoach rollout should export exactly 1 interaction, got {len(interactions)}"
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
        run_dir = make_run_dir(self.work_dir_root, "codecoach")
        task_markdown = str(data["task_markdown"])
        current_code = str(data["initial_code"])
        evaluator_path = str(data["evaluator_path"])
        target_score = float(data["target_score"])
        entry_function = str(data["entry_function"])
        eval_timeout_sec = int(data.get("eval_timeout_sec", 120))

        initial_eval = self._evaluate_code(
            evaluator_path=evaluator_path,
            target_score=target_score,
            code=current_code,
            eval_root=run_dir / "initial_eval",
            eval_timeout_sec=eval_timeout_sec,
        )
        best_eval = initial_eval
        best_ratio_history: list[float] = []
        total_reward = 0.0
        last_completion_id: str | None = None
        history: list[dict[str, Any]] = []
        token_budget = EpisodeTokenBudget(
            max_episode_total_tokens=self.max_episode_total_tokens,
        )
        teacher_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are the teacher in a code-coaching environment. Reply with concise natural-language guidance only. Do not output code.",
            },
            {
                "role": "user",
                "content": self._build_teacher_initial_prompt(
                    task_markdown=task_markdown,
                    current_code=current_code,
                    current_eval=initial_eval,
                    best_eval=best_eval,
                    entry_function=entry_function,
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
                        "code_updated": False,
                        "eval_result": history[-1]["eval_result"] if history else initial_eval,
                        "best_eval": best_eval,
                        "reward": self.token_budget_penalty,
                        "error": None,
                        "turn_prompt_tokens": prompt_tokens,
                        "turn_completion_tokens": 0,
                        "turn_total_tokens": prompt_tokens,
                        "termination_feedback": (
                            "Episode terminated before teacher generation because prompt length "
                            "exhausted the available episode or context budget."
                        ),
                        "length_penalty": self.token_budget_penalty,
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
                                "code_updated": False,
                                "eval_result": (
                                    history[-1]["eval_result"]
                                    if history
                                    else initial_eval
                                ),
                                "best_eval": best_eval,
                                "reward": self.token_budget_penalty,
                                "error": None,
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
            teacher_action = _strip_think_tags(teacher_message.content or "")
            budget_snapshot = token_budget.observe_turn(
                response=response,
                prompt_text=teacher_messages[-1]["content"],
                completion_text=teacher_action,
            )
            raw_student_output = await self._call_student_with_retry(
                self._build_student_messages(
                    task_markdown=task_markdown,
                    current_code=current_code,
                    teacher_action=teacher_action,
                    entry_function=entry_function,
                    history=history,
                ),
                round_idx=round_idx,
            )
            student_reply = self._parse_student_reply(raw_student_output)
            update_error = student_reply.parse_error
            code_updated = False
            candidate_code = current_code
            if student_reply.replace_code:
                if student_reply.code:
                    candidate_code = student_reply.code
                    code_updated = True
                else:
                    update_error = join_errors(
                        update_error, "replace_code=true but code is empty"
                    )

            eval_result = self._evaluate_code(
                evaluator_path=evaluator_path,
                target_score=target_score,
                code=candidate_code,
                eval_root=run_dir / f"round_{round_idx:02d}_eval",
                eval_timeout_sec=eval_timeout_sec,
            )
            current_code = candidate_code
            if eval_result.target_ratio > best_eval.target_ratio:
                best_eval = eval_result
            best_ratio_history.append(best_eval.target_ratio)
            reward = self._current_auc_gain(best_ratio_history, initial_eval.target_ratio)
            final_reward = reward
            if budget_snapshot.stop_reason is not None:
                final_reward += self.token_budget_penalty
            total_reward += final_reward
            record = {
                "round_idx": round_idx,
                "teacher_action": teacher_action,
                "student_reply": student_reply,
                "code_updated": code_updated,
                "eval_result": eval_result,
                "best_eval": best_eval,
                "reward": final_reward,
                "error": join_errors(update_error, eval_result.error),
                "turn_prompt_tokens": budget_snapshot.turn_prompt_tokens,
                "turn_completion_tokens": budget_snapshot.turn_completion_tokens,
                "turn_total_tokens": budget_snapshot.turn_total_tokens,
                "termination_feedback": budget_snapshot.stop_feedback,
            }
            if budget_snapshot.stop_reason is not None:
                record["length_penalty"] = self.token_budget_penalty
            history.append(record)
            if (
                round_idx >= self.max_turns
                or eval_result.target_ratio >= 1.0
                or budget_snapshot.stop_reason is not None
            ):
                break
            teacher_messages.extend(
                [
                    teacher_message.model_dump(exclude_none=True),
                    {
                        "role": "user",
                        "content": self._build_teacher_followup_prompt(
                            current_code=current_code,
                            latest_record=record,
                            best_eval=best_eval,
                            entry_function=entry_function,
                            round_idx=round_idx + 1,
                        ),
                    },
                ]
            )

        final_eval = history[-1]["eval_result"] if history else initial_eval
        _safe_scalar(
            reward=total_reward,
            num_turns=len(history),
            score=final_eval.score,
            best_score=best_eval.score,
            auc_gain=self._current_auc_gain(best_ratio_history, initial_eval.target_ratio),
        )
        if last_completion_id is None:
            return None
        return total_reward, last_completion_id

    def _build_teacher_initial_prompt(
        self,
        task_markdown: str,
        current_code: str,
        current_eval: EvalResult,
        best_eval: EvalResult,
        entry_function: str,
        round_idx: int,
    ) -> str:
        return dedent(
            f"""\
            Task:
            {task_markdown}

            Budget:
            - Current round: {round_idx - 1}/{self.max_turns}
            - Remaining rounds: {max(self.max_turns - round_idx + 1, 0)}

            Current evaluation:
            - Score: {current_eval.score:.6f}
            - Target ratio: {current_eval.target_ratio:.6f}
            - Validity: {current_eval.validity:.6f}
            - Best score: {best_eval.score:.6f}
            - Best target ratio: {best_eval.target_ratio:.6f}
            - Required entry function: {entry_function}

            Current code:
            ```python
            {current_code}
            ```
            """
        ).strip()

    def _build_teacher_followup_prompt(
        self,
        current_code: str,
        latest_record: dict[str, Any],
        best_eval: EvalResult,
        entry_function: str,
        round_idx: int,
    ) -> str:
        student_reply: StudentReply = latest_record["student_reply"]
        eval_result: EvalResult = latest_record["eval_result"]
        error_text = latest_record["error"] or "None"
        return dedent(
            f"""\
            Round {latest_record['round_idx']} update:
            - Current round: {round_idx - 1}/{self.max_turns}
            - Remaining rounds: {max(self.max_turns - round_idx + 1, 0)}
            - Student message: {json.dumps(student_reply.message)}
            - Code updated: {latest_record['code_updated']}
            - Error: {json.dumps(error_text)}
            - Score: {eval_result.score:.6f}
            - Target ratio: {eval_result.target_ratio:.6f}
            - Validity: {eval_result.validity:.6f}
            - Best score: {best_eval.score:.6f}
            - Best target ratio: {best_eval.target_ratio:.6f}
            - Required entry function: {entry_function}

            Current code:
            ```python
            {current_code}
            ```
            """
        ).strip()

    def _build_student_messages(
        self,
        task_markdown: str,
        current_code: str,
        teacher_action: str,
        entry_function: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        lines = []
        for record in history[-5:]:
            eval_result: EvalResult = record["eval_result"]
            student_reply: StudentReply = record["student_reply"]
            error_text = record["error"] or "None"
            lines.append(
                f'Round {record["round_idx"]}: score={eval_result.score:.6f}, '
                f'best={record["best_eval"].score:.6f}, updated={record["code_updated"]}, '
                f'student_message={json.dumps(student_reply.message)}, error={json.dumps(error_text)}'
            )
        prompt = dedent(
            f"""\
            You are the student programmer.
            Output a single JSON object with this schema:
            {{
              "message": "string",
              "replace_code": true,
              "code": "full python file or null"
            }}

            Rules:
            - Output valid JSON only.
            - If you do not want to change the code, set "replace_code" to false and "code" to null.
            - If you change the code, "code" must be a complete executable Python file.
            - Keep the required entry function `{entry_function}`.

            Task:
            {task_markdown}

            Teacher guidance:
            {teacher_action}

            Recent history:
            {"No previous rounds." if not lines else "\n".join(lines)}

            Current code:
            ```python
            {current_code}
            ```
            """
        ).strip()
        return [
            {
                "role": "system",
                "content": "You are a careful coding student. Output strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ]

    async def _call_student_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        round_idx: int,
        max_attempts: int = 3,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await self.student_caller.call_text(messages)
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Student call failed after {max_attempts} attempts "
                        f"at round {round_idx}."
                    ) from exc
                logger.warning(
                    "Student call failed on attempt %s/%s at round %s; retrying: %s",
                    attempt,
                    max_attempts,
                    round_idx,
                    exc,
                )
                await asyncio.sleep(float(attempt))
        raise RuntimeError(
            f"Student call failed after {max_attempts} attempts at round {round_idx}."
        ) from last_exc

    def _parse_student_reply(self, raw_output: str) -> StudentReply:
        parsed, parse_error = parse_json_dict(raw_output)
        if parsed is None:
            return StudentReply(raw_output, "", False, None, parse_error)
        message = parsed.get("message", "")
        replace_code = parsed.get("replace_code", False)
        code = parsed.get("code")
        if not isinstance(message, str):
            parse_error = join_errors(parse_error, '"message" must be a string')
            message = str(message)
        if not isinstance(replace_code, bool):
            parse_error = join_errors(parse_error, '"replace_code" must be a boolean')
            replace_code = False
        if code is not None and not isinstance(code, str):
            parse_error = join_errors(parse_error, '"code" must be a string or null')
            code = None
        return StudentReply(raw_output, message.strip(), replace_code, code, parse_error)

    def _current_auc_gain(
        self, best_ratio_history: list[float], initial_ratio: float
    ) -> float:
        area = sum(best_ratio - initial_ratio for best_ratio in best_ratio_history)
        return area / float(self.max_turns)

    def _evaluate_code(
        self,
        evaluator_path: str,
        target_score: float,
        code: str,
        eval_root: Path,
        eval_timeout_sec: int,
    ) -> EvalResult:
        eval_root.mkdir(parents=True, exist_ok=True)
        code_path = eval_root / "candidate.py"
        result_path = eval_root / "result.json"
        stdout_path = eval_root / "stdout.log"
        stderr_path = eval_root / "stderr.log"
        code_path.write_text(code, encoding="utf-8")
        result: dict[str, Any] | None = None
        error: str | None = None
        try:
            process = subprocess.run(
                [sys.executable, evaluator_path, str(code_path), str(result_path)],
                capture_output=True,
                text=True,
                timeout=eval_timeout_sec,
                check=False,
            )
            stdout_path.write_text(process.stdout or "", encoding="utf-8")
            stderr_path.write_text(process.stderr or "", encoding="utf-8")
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            if process.returncode != 0 and result is None:
                error = f"Evaluator exited with code {process.returncode}"
        except subprocess.TimeoutExpired:
            error = f"Evaluator timed out after {eval_timeout_sec}s"
        except Exception as exc:
            error = str(exc)

        if result is None:
            result = {}
        raw_score = result.get(
            "sum_radii",
            result.get("score", result.get("eval_score", result.get("combined_score", 0.0))),
        )
        score = float(raw_score or 0.0)
        target_ratio = result.get("target_ratio")
        if target_ratio is None:
            target_ratio = score / target_score if target_score > 0 else 0.0
        validity = float(result.get("validity", 1.0 if score > 0 else 0.0))
        success = bool(result.get("success", validity > 0 and score > 0))
        error = error or result.get("error")
        return EvalResult(
            score=score,
            target_ratio=float(target_ratio or 0.0),
            validity=validity,
            success=success,
            raw_result=result,
            error=error,
        )


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
