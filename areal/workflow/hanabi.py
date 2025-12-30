import asyncio
import json
import os
import math
import random
import time
import uuid
from collections import defaultdict, deque
from typing import Optional

import aiofiles
import aiofiles.os
import aiohttp
import colorama
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from realhf.base import logging
from realhf.impl.environment import hanabi_env  # noqa: F401 - ensure registration
from realhf.impl.environment.hanabi_env import HanabiEnv


logger = logging.getLogger("Hanabi workflow")
DEFAULT_HANABI_COLORS = list(hanabi_env.COLORS)
DEFAULT_HANABI_RANK_COUNTS = dict(hanabi_env.RANK_COUNTS)
DEFAULT_TEACHER_OBSERVATION_KWARGS = dict(
    use_individual_thoughts = True,
    use_global_obs = True,
)


def _parse_rank_counts(raw_counts: dict | None) -> dict[int, int] | None:
    if not raw_counts:
        return None
    parsed: dict[int, int] = {}
    for key, value in raw_counts.items():
        if value is None:
            continue
        try:
            rank = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Rank keys must be integers, got {key!r}.") from exc
        parsed[rank] = int(value)
    return parsed or None


def _resolve_colors(raw_colors: list[str] | None, num_colors: int | None) -> list[str]:
    colors: list[str] = (
        [str(c).lower() for c in raw_colors if str(c).strip()] if raw_colors else DEFAULT_HANABI_COLORS
    )
    if num_colors is not None:
        if num_colors <= 0:
            raise ValueError("Number of colors must be positive.")
        colors = colors[:num_colors]
    if not colors:
        raise ValueError("At least one color is required for Hanabi.")
    return colors


def _build_color_letter_map(colors: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used_letters: set[str] = set()
    for idx, color in enumerate(colors):
        base_letter = (color[:1] or str(idx)).upper()
        letter = base_letter
        suffix = 1
        while letter in used_letters:
            suffix += 1
            letter = f"{base_letter}{suffix}"
        mapping[color] = letter
        used_letters.add(letter)
    return mapping


def _sanitize_for_json(value):
    """Recursively convert objects to JSON-serializable structures."""

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return 0.0 if math.isinf(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(k): _sanitize_for_json(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in value]
    return str(value)


def _extract_three_questions(text: str) -> list[str]:
    """Parse up to three questions from the model output."""
    import re

    questions: list[str] = []
    if ("Q1:" in text) and ("Q2:" in text) and ("Q3:" in text):
        q1 = text[text.find("Q1:") + 3 : text.find("Q2:")].strip()
        q2 = text[text.find("Q2:") + 3 : text.find("Q3:")].strip()
        q3 = text[text.find("Q3:") + 3 :].strip()
        questions = [q1, q2, q3]

    if len(questions) < 3:
        for m in re.findall(
            r"Q\s*([123])\s*[:：]\s*(.+?)(?=(?:\nQ\s*[123]\s*[:：])|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            _, q = m
            q = q.strip()
            questions.append(q)
            if len(questions) >= 3:
                break

    if len(questions) < 3:
        cand = re.findall(r"([^?？]+[?？])", text, flags=re.DOTALL)
        cand = [c.strip() for c in cand if c.strip()]
        for c in cand:
            questions.append(c)
            if len(questions) >= 3:
                break

    return questions[:3]


def _detect_api_provider(api_key: str) -> str:
    key = api_key.lower()
    if key.startswith("sk-ant") or key.startswith("anthropic"):
        return "claude"
    return "openai"


class RateLimiter:
    """Asynchronous rate limiter supporting per-second and per-minute limits."""

    def __init__(self, per_second: int, per_minute: int):
        self.per_second = per_second
        self.per_minute = per_minute
        self._lock = asyncio.Lock()
        self._req_timestamps: deque[float] = deque()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            while self._req_timestamps and now - self._req_timestamps[0] > 60:
                self._req_timestamps.popleft()

            while True:
                now = time.monotonic()
                while self._req_timestamps and now - self._req_timestamps[0] > 60:
                    self._req_timestamps.popleft()

                last_1s = [t for t in self._req_timestamps if now - t <= 1.0]
                last_60s = len(self._req_timestamps)

                if len(last_1s) < self.per_second and last_60s < self.per_minute:
                    self._req_timestamps.append(now)
                    return

                next_allowed = min(
                    (1.0 - (now - last_1s[0])) if last_1s else 0.0,
                    (60.0 - (now - self._req_timestamps[0])) if self._req_timestamps else 0.0,
                )
                await asyncio.sleep(max(next_allowed, 0.01))


class HanabiWorkflow(RolloutWorkflow):
    """Workflow for cooperative Hanabi rollouts with teacher-student supervision."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int = 150,
        turn_discount: float = 1.0,
        dump_dir: str | None = None,
        env_kwargs: dict | None = None,
        teacher_rollout: InferenceEngine | None = None,
        teacher_tokenizer: PreTrainedTokenizerFast | None = None,
        student_api_key: str | None = None,
        student_api_model: str | None = None,
        teacher_api_key: str | None = None,
        teacher_api_model: str | None = None,
        sft_reg: float = 0.0,
        misplay_penalty_factor: float = 0.1,
        use_question_tokens: bool = False,
        teacher_process_reward: bool = False,
        teacher_obs_kwargs: dict | None = None,
        process_reward_coef: float = 0.2,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.dump_dir = dump_dir
        raw_env_kwargs = env_kwargs or {}
        self._hanabi_rank_counts = _parse_rank_counts(raw_env_kwargs.get("rank_counts"))
        self._hanabi_colors = _resolve_colors(
            raw_env_kwargs.get("colors"), raw_env_kwargs.get("num_colors")
        )
        self.env_kwargs = raw_env_kwargs
        self.teacher_obs_kwargs = teacher_obs_kwargs or dict()
        for k, v in DEFAULT_TEACHER_OBSERVATION_KWARGS.items():
            if k not in self.teacher_obs_kwargs:
                self.teacher_obs_kwargs[k] = v
        self.misplay_penalty_factor = misplay_penalty_factor
        self.sft_reg = sft_reg
        self.process_reward_coef = process_reward_coef
        self.teacher_process_reward = teacher_process_reward
        self.student_api_key = (student_api_key or "").strip()
        self.teacher_api_key = (teacher_api_key or "").strip()
        self.student_api_provider = (
            _detect_api_provider(self.student_api_key)
            if self.student_api_key
            else None
        )
        self.teacher_api_provider = (
            _detect_api_provider(self.teacher_api_key)
            if self.teacher_api_key
            else None
        )
        self.student_api_model = (
            (student_api_model or "").strip() if self.student_api_key else ""
        )
        self.teacher_api_model = (
            (teacher_api_model or "").strip() if self.teacher_api_key else ""
        )
        if self.student_api_key and not self.student_api_model:
            default_student_model = (
                "gpt-4o-2024-11-20"
                if self.student_api_provider == "openai"
                else "claude-3-7-sonnet-20250219"
            )
            self.student_api_model = (
                os.getenv("AREAL_STUDENT_API_MODEL", "").strip()
                or default_student_model
            )
        if self.teacher_api_key and not self.teacher_api_model:
            default_teacher_model = (
                "gpt-4o-2024-11-20"
                if self.teacher_api_provider == "openai"
                else "claude-3-7-sonnet-20250219"
            )
            self.teacher_api_model = (
                os.getenv("AREAL_TEACHER_API_MODEL", "").strip()
                or default_teacher_model
            )
        self.teacher_rollout = teacher_rollout if not self.teacher_api_key else None
        self.teacher_tokenizer = teacher_tokenizer
        self.use_summary = True
        self.use_question_tokens = use_question_tokens
        self._student_api_session: aiohttp.ClientSession | None = None
        self._teacher_api_session: aiohttp.ClientSession | None = None
        self.rate_limiter = RateLimiter(per_second=2, per_minute=60)
        self._max_retries = 6
        self._base_backoff = 1.0
        self._max_backoff = 10.0

        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _ensure_api_session(self, attr: str) -> aiohttp.ClientSession:
        session = getattr(self, attr)
        if session is None or session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            connector = aiohttp.TCPConnector(limit=64)
            session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            setattr(self, attr, session)
        return session

    def _compute_backoff(self, attempt: int) -> float:
        backoff = min(self._base_backoff * (2 ** attempt), self._max_backoff)
        jitter = random.uniform(0.0, 0.25 * backoff)
        return backoff + jitter

    def _retry_after_seconds(self, retry_after: str) -> Optional[float]:
        try:
            secs = float(retry_after.strip())
            if secs >= 0:
                return secs
        except Exception:
            pass
        return None

    async def _request_json_with_retries(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        *,
        headers: dict,
        json: dict,
    ) -> dict:
        last_err_text = ""
        for attempt in range(self._max_retries + 1):
            try:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    json=json,
                ) as resp:
                    if resp.status < 400:
                        return await resp.json()

                    last_err_text = await resp.text()
                    retryable = resp.status in {408, 409, 425, 429, 500, 502, 503, 504}
                    if not retryable or attempt == self._max_retries:
                        raise RuntimeError(
                            f"API request failed with status {resp.status}: {last_err_text}"
                        )

                    retry_after_hdr = resp.headers.get("Retry-After")
                    sleep_for = (
                        self._retry_after_seconds(retry_after_hdr)
                        if retry_after_hdr
                        else None
                    )
                    if sleep_for is None:
                        sleep_for = self._compute_backoff(attempt)

                    await asyncio.sleep(sleep_for)
                    continue

            except (
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientOSError,
                aiohttp.ClientConnectionError,
                aiohttp.ClientPayloadError,
                asyncio.TimeoutError,
            ) as e:
                if attempt == self._max_retries:
                    raise RuntimeError(
                        f"Network or timeout error after retries: {e}"
                    ) from e
                await asyncio.sleep(self._compute_backoff(attempt))
                continue

        raise RuntimeError(f"Request failed after retries. Last error: {last_err_text}")

    async def _api_chat_completion(
        self,
        prompt: str,
        cfg: GenerationHyperparameters,
        api_key: str,
        api_model: str,
        provider: str,
        session_attr: str,
        rid: str,
    ) -> str:
        await self.rate_limiter.acquire()

        session = await self._ensure_api_session(session_attr)
        max_tokens = max(1, cfg.max_new_tokens)
        temperature = max(0.0, cfg.temperature)
        top_p = min(max(cfg.top_p, 0.0), 1.0)
        stop_words = cfg.stop or []

        if provider == "claude":
            url = os.getenv(
                "ANTHROPIC_API_URL",
                os.getenv("AREAL_CLAUDE_API_URL", "https://api.anthropic.com/v1/messages"),
            )
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": api_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                "metadata": {"rid": rid},
            }
            if stop_words:
                payload["stop_sequences"] = stop_words
        else:
            base_url = os.getenv(
                "OPENAI_API_BASE",
                os.getenv("AREAL_OPENAI_API_BASE", "https://matrixllm.alipay.com/v1"),
            )
            url = base_url.rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": api_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": 1,
            }
            if stop_words:
                payload["stop"] = stop_words

        data = await self._request_json_with_retries(
            session, "POST", url, headers=headers, json=payload
        )

        if provider == "claude":
            content = data.get("content", [])
            texts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(texts)

        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "")

    def _configure_hanabi_env(self) -> None:
        hanabi_env.COLORS = self._hanabi_colors
        hanabi_env.COLOR_TO_LETTER = _build_color_letter_map(self._hanabi_colors)
        hanabi_env.LETTER_TO_COLOR = {
            v: k for k, v in hanabi_env.COLOR_TO_LETTER.items()
        }
        hanabi_env.RANK_COUNTS = self._hanabi_rank_counts or DEFAULT_HANABI_RANK_COUNTS

    def _build_env(self) -> HanabiEnv:
        self._configure_hanabi_env()
        return HanabiEnv(**self.env_kwargs)

    def _build_api_response(
        self,
        input_ids: list[int],
        text: str,
        tokenizer: PreTrainedTokenizerFast,
    ) -> ModelResponse:
        output_tokens = tokenizer.encode(text, add_special_tokens=False)
        return ModelResponse(
            input_tokens=list(input_ids),
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            tokenizer=tokenizer,
        )

    @staticmethod
    def _response_to_tensordict(
        resp: ModelResponse, *, sft_ppo_mask: int = 0, agent_idx: int = -1
    ) -> dict[str, torch.Tensor]:
        full_ids = resp.input_tokens + resp.output_tokens
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
            "logprobs": torch.tensor(
                [0.0] * resp.input_len + resp.output_logprobs,
                dtype=torch.float32,
            ).unsqueeze(0),
            "loss_mask": torch.tensor(
                [0] * resp.input_len + [1] * resp.output_len,
                dtype=torch.long,
            ).unsqueeze(0),
            "versions": torch.tensor(
                [-1] * resp.input_len + resp.output_versions,
                dtype=torch.long,
            ).unsqueeze(0),
            "rewards": torch.zeros(1, dtype=torch.float32),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.bool).unsqueeze(0),
            "sft_ppo_mask": torch.tensor([sft_ppo_mask], dtype=torch.long),
            "agent_idx": torch.tensor([agent_idx], dtype=torch.long),
            "step_indicator": torch.tensor([0], dtype=torch.int32),
        }

    @staticmethod
    def _parse_action_from_completion(text: str) -> str:
        """Extract the agent action from the completion text."""
        import re

        match = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match[-1].strip().lower()
        return ""

    async def _run_one_episode(self, engine: InferenceEngine, data, rid):
        env = self._build_env()
        obs, guide, info = await env.sreset()
        teacher_obs = info.get("teacher_observation", obs)
        state = info.get("state", obs)

        results = []
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []
        total_reward = 0.0
        traj_len = [0, 0]
        summaries = defaultdict(list)
        qa_logs = []
        teacher_logs = []
        episode_steps: list[dict] = []
        episode_uuid = uuid.uuid4().hex

        prev_thought: str = ""

        agent_result_indices: list[int] = []
        step_rewards: list[float] = []
        process_rewards: list[float] = []
        success_count = 0
        misplay_occurred = False

        t_qgen_total = 0.0
        t_agent_answer_total = 0.0
        t_teacher_answer_total = 0.0
        t_action_build_total = 0.0
        t_action_gen_total = 0.0
        t_action_decode_total = 0.0
        t_env_step_total = 0.0
        t_summary_total = 0.0
        t_pack_tensors_total = 0.0
        t_tokenize_total = 0.0

        turns_done = 0

        for turn in range(self.max_turns):
            turns_done += 1
            current_player = env.agent_player
            player_idx = env.players.index(env.agent_player)
            prev_summary = summaries[current_player][-1] if summaries[current_player] else "None yet."

            # ========== 1) Agent self-generates 3 questions ==========
            qgen_prompt = (
                f"{obs}\n"
                "Generate exactly three concise, high-value questions about this Hanabi turn.\n"
                "Questions may target different angles: safety of plays, information-token economy, and partner intent.\n"
                "Questions must be grounded in the visible state and recent events.\n"
                "Output strictly in this format:\nQ1: ...\nQ2: ...\nQ3: ...\n"
            )

            t0 = time.perf_counter()
            qgen_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": qgen_prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            t_tokenize_total += time.perf_counter() - t0

            qgen_cfg = self.gconfig.new(n_samples=1, max_new_tokens=2048)
            qgen_resp: ModelResponse | None = None
            if self.student_api_key:
                t0 = time.perf_counter()
                qgen_text = await self._api_chat_completion(
                    qgen_prompt,
                    qgen_cfg,
                    self.student_api_key,
                    self.student_api_model,
                    self.student_api_provider or "openai",
                    "_student_api_session",
                    f"{rid}-qgen-{turn}",
                )
                t_qgen_total += time.perf_counter() - t0
                qgen_resp = self._build_api_response(qgen_ids, qgen_text, self.tokenizer)
            else:
                req = ModelRequest(
                    rid=f"{rid}-qgen-{turn}",
                    input_ids=qgen_ids,
                    gconfig=qgen_cfg,
                    tokenizer=self.tokenizer,
                )
                t0 = time.perf_counter()
                qgen_resp = await engine.agenerate(req)
                t_qgen_total += time.perf_counter() - t0
                qgen_text = self.tokenizer.decode(
                    qgen_resp.output_tokens,
                    skip_special_tokens=True
                )

            if self.use_question_tokens and qgen_resp is not None:
                t0 = time.perf_counter()
                results.append(self._response_to_tensordict(qgen_resp, sft_ppo_mask=0, agent_idx=player_idx))
                t_pack_tensors_total += time.perf_counter() - t0

            self_questions = _extract_three_questions(qgen_text)
            if not self_questions:
                self_questions = [
                    "What card is most likely safe to play right now?",
                    "Should I give a hint, play, or discard this turn?",
                    "Which teammate card is urgent to protect with a hint?",
                ]

            # ========== 2) Agent and teacher answer 3 questions ==========
            agent_answer_tasks = []
            agent_answer_inputs: list[list[int]] = []
            agent_question_prompts: list[str] = []
            agent_answer_cfg = self.gconfig.new(n_samples=1, max_new_tokens=2048)
            for qi, q in enumerate(self_questions):
                aprompt = (
                    f"{obs}\n"
                    f"Previous summary: {prev_summary}\n"
                    f"You asked: {q}\n"
                    "Answer concisely using public clues, discard history and visible hands; avoid speculation beyond standard Hanabi reasoning."
                )
                agent_question_prompts.append(aprompt)
                t0 = time.perf_counter()
                a_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": aprompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                t_tokenize_total += time.perf_counter() - t0
                agent_answer_inputs.append(a_ids)

                if self.student_api_key:
                    agent_answer_tasks.append(
                        self._api_chat_completion(
                            aprompt,
                            agent_answer_cfg,
                            self.student_api_key,
                            self.student_api_model,
                            self.student_api_provider or "openai",
                            "_student_api_session",
                            f"{rid}-ans-{turn}-{qi}",
                        )
                    )
                else:
                    a_req = ModelRequest(
                        rid=f"{rid}-ans-{turn}-{qi}",
                        input_ids=a_ids,
                        gconfig=agent_answer_cfg,
                        tokenizer=self.tokenizer,
                    )
                    agent_answer_tasks.append(engine.agenerate(a_req))

            t0 = time.perf_counter()
            agent_answers: list[str]
            agent_answer_resps: list[ModelResponse]
            if self.student_api_key:
                agent_answers = await asyncio.gather(*agent_answer_tasks)
                agent_answer_resps = [
                    self._build_api_response(ids, ans, self.tokenizer)
                    for ids, ans in zip(agent_answer_inputs, agent_answers)
                ]
            else:
                agent_answer_resps = await asyncio.gather(*agent_answer_tasks)
                agent_answers = [
                    self.tokenizer.decode(
                        r.output_tokens, 
                        skip_special_tokens=True
                    ) for r in agent_answer_resps
                ]
            t_agent_answer_total += time.perf_counter() - t0

            if self.use_question_tokens:
                t0 = time.perf_counter()
                for a_resp in agent_answer_resps:
                    results.append(
                        self._response_to_tensordict(a_resp, sft_ppo_mask=0, agent_idx=player_idx)
                    )
                t_pack_tensors_total += time.perf_counter() - t0

            teacher_answer_tasks = []
            teacher_answer_inputs: list[list[int]] = []
            teacher_prompts: list[str] = []
            teacher_answer_cfg = self.gconfig.new(n_samples=1, max_new_tokens=8192)
            if self.teacher_rollout or self.teacher_api_key:
                teacher_tokenizer = self.teacher_tokenizer or self.tokenizer
                for qi, q in enumerate(self_questions):
                    _teacher_obs = teacher_obs
                    _players_thoughts = "=== Players' Inner Thoughts ===\n\n" + "\n\n\n".join([f"Inner thought of {p}:\n```\n{summaries[p][-1]}\n```" for p in summaries.keys() if len(summaries[p]) > 0])
                    if not self.teacher_obs_kwargs["use_global_obs"]:
                        _teacher_obs = obs
                        _players_thoughts = ""
                    elif not self.teacher_obs_kwargs["use_individual_thoughts"]:
                        _teacher_obs = teacher_obs
                        _players_thoughts = ""

                    taprompt = (
                        f"{_teacher_obs}\n\n"
                        f"{_players_thoughts}\n\n"
                        f"# Question to be Answered\nQuestion: {q}\n"
                        "Answer concisely using all privileged information, including hidden hands and players' inner thoughts, from the perspective of the current active player."
                    )

                    if self.teacher_process_reward:
                            taprompt = (
                            f"{_teacher_obs}\n\n"
                            f"{_players_thoughts}\n\n"
                            f"# Judge Question Answer Pair\nQuestion: {q}\nAnswer: {agent_answers[qi]}\n\n"
                            "Judge whether the answer is correct to the question using all privileged information, including hidden hands and players' inner thoughts, from the perspective of the current active player.\n"
                            "Reply with [CORRECT] if the answer is correct and [WRONG] is the answer is wrong.\n"
                            "Format your output as:\n"
                            "<analysis> your reasoning process </analysis>\n<result> correctness of the answer </result> "
                        )
                    
                    teacher_prompts.append(taprompt)
                    if self.teacher_api_key:
                        t0 = time.perf_counter()
                        ta_ids = teacher_tokenizer.apply_chat_template(
                            [{"role": "user", "content": taprompt}],
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        t_tokenize_total += time.perf_counter() - t0
                        teacher_answer_tasks.append(
                            self._api_chat_completion(
                                taprompt,
                                teacher_answer_cfg,
                                self.teacher_api_key,
                                self.teacher_api_model,
                                self.teacher_api_provider or "openai",
                                "_teacher_api_session",
                                f"{rid}-tans-{turn}-{qi}",
                            )
                        )
                        teacher_answer_inputs.append(ta_ids)
                    else:
                        t0 = time.perf_counter()
                        ta_ids = teacher_tokenizer.apply_chat_template(
                            [{"role": "user", "content": taprompt}],
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        t_tokenize_total += time.perf_counter() - t0
                        teacher_answer_inputs.append(ta_ids)
                        ta_req = ModelRequest(
                            rid=f"{rid}-tans-{turn}-{qi}",
                            input_ids=ta_ids,
                            gconfig=teacher_answer_cfg,
                            tokenizer=teacher_tokenizer,
                        )
                        teacher_answer_tasks.append(self.teacher_rollout.agenerate(ta_req))

            teacher_answers: list[str] = []
            teacher_resps: list[ModelResponse] = []
            if teacher_answer_tasks:
                t0 = time.perf_counter()
                raw_teacher_resps = await asyncio.gather(*teacher_answer_tasks)
                teacher_tokenizer = self.teacher_tokenizer or self.tokenizer
                if self.teacher_api_key:
                    teacher_answers = list(raw_teacher_resps)
                    teacher_resps = [
                        self._build_api_response(ids, ans, teacher_tokenizer)
                        for ids, ans in zip(teacher_answer_inputs, teacher_answers)
                    ]
                else:
                    teacher_resps = list(raw_teacher_resps)
                    teacher_answers = [
                        teacher_tokenizer.decode(
                            r.output_tokens,
                            skip_special_tokens=True
                        ).split("</think>")[-1]
                        for r in teacher_resps
                    ]
                t_teacher_answer_total += time.perf_counter() - t0

            if self.use_question_tokens and teacher_resps and not self.teacher_process_reward:
                t0 = time.perf_counter()
                for qi, t_ans in enumerate(teacher_answers):
                    student_prompt = (
                        agent_question_prompts[qi]
                        if qi < len(agent_question_prompts)
                        else "" # teacher_prompts[qi]
                        if qi < len(teacher_prompts)
                        else ""
                    )
                    assert len(student_prompt) > 0
                    prompt_ids = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": student_prompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    resp = self._build_api_response(prompt_ids, t_ans, self.tokenizer)
                    results.append(
                        self._response_to_tensordict(resp, sft_ppo_mask=1, agent_idx=player_idx)
                    )
                t_pack_tensors_total += time.perf_counter() - t0
            
            _process_reward = 0.0
            if self.use_question_tokens and teacher_resps and self.teacher_process_reward:
                for qi, t_ans in enumerate(teacher_answers):
                    _process_reward += float("CORRECT" in t_ans)
            process_rewards.append(_process_reward)

            summary_prompt: str | None = None
            agent_summary: str | None = None
            thought: str = ""

            qa_pairs = []
            for qi, q in enumerate(self_questions):
                qa_pairs.append(
                    {
                        "question": q,
                        "agent_answer": agent_answers[qi]
                        if qi < len(agent_answers)
                        else "",
                        "teacher_answer": teacher_answers[qi]
                        if qi < len(teacher_answers)
                        else "",
                        "teacher_prompt": teacher_prompts[qi]
                        if qi < len(teacher_prompts)
                        else "",
                    }
                )

            qa_block = "\n".join(
                [
                    f"{i + 1}) {self_questions[i]}\nAnswer: {agent_answers[i].strip()}"
                    for i in range(len(agent_answers))
                ]
            )

            # ========== 3) Agent generates action ==========
            t0 = time.perf_counter()
            action_prompt = (
                f"{obs}\n"
                f"Previous turn summary: {prev_summary}\n"
                f"Recent thought: {prev_thought or 'None yet.'}\n\n"
                f"Your self-questions and answers:\n{qa_block}\n"
                f"Use the information above to prose a single action.\n {guide}"
            )
            action_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": action_prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            t_action_build_total += time.perf_counter() - t0

            action_cfg = self.gconfig.new(n_samples=1, max_new_tokens=4096)
            req = ModelRequest(
                rid=f"{rid}-act-{turn}",
                input_ids=action_ids,
                gconfig=action_cfg,
                tokenizer=self.tokenizer,
            )
            if self.student_api_key:
                t0 = time.perf_counter()
                completion_str = await self._api_chat_completion(
                    action_prompt,
                    action_cfg,
                    self.student_api_key,
                    self.student_api_model,
                    self.student_api_provider or "openai",
                    "_student_api_session",
                    f"{rid}-act-{turn}",
                )
                t_action_gen_total += time.perf_counter() - t0
                resp = self._build_api_response(action_ids, completion_str, self.tokenizer)
            else:
                t0 = time.perf_counter()
                resp = await engine.agenerate(req)
                t_action_gen_total += time.perf_counter() - t0
                t0 = time.perf_counter()
                completion_str = self.tokenizer.decode(
                    resp.output_tokens,
                    # skip_special_tokens=True
                ).replace("<|im_end|>", "")
                t_action_decode_total += time.perf_counter() - t0

            if self.student_api_key:
                t_action_decode_total += 0.0

            # ========== 4) Prepares training data ==========
            full_ids = resp.input_tokens + resp.output_tokens
            seq = torch.tensor(full_ids, dtype=torch.long)
            logprobs = torch.tensor(
                [0.0] * resp.input_len + resp.output_logprobs,
                dtype=torch.float32,
            )
            loss_mask = torch.tensor(
                [0] * resp.input_len + [1] * resp.output_len,
                dtype=torch.long,
            )
            versions = torch.tensor(
                [-1] * resp.input_len + resp.output_versions,
                dtype=torch.long,
            )

            t0 = time.perf_counter()
            next_obs, next_guide, _env_reward, done, _, info = await env.step(
                (data.get("query_id", ""), [completion_str])
            )
            t_env_step_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            res = {
                "input_ids": seq.unsqueeze(0),
                "logprobs": logprobs.unsqueeze(0),
                "loss_mask": loss_mask.unsqueeze(0),
                "versions": versions.unsqueeze(0),
                "rewards": torch.tensor([0.0], dtype=torch.float32),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.bool).unsqueeze(0),
                "sft_ppo_mask": torch.tensor([0], dtype=torch.long),
                "agent_idx": torch.tensor([player_idx], dtype=torch.long),
                "step_indicator": torch.tensor([0], dtype=torch.int32)
            }
            t_pack_tensors_total += time.perf_counter() - t0

            results.append(res)

            parsed_action = self._parse_action_from_completion(completion_str)
            event_msg = info.get("event", "").lower()
            env_step_reward = _env_reward
            step_reward = env_step_reward
            if parsed_action.startswith("play ") and "successfully played" in event_msg:
                success_count += 1
            if parsed_action.startswith("play ") and "misplayed" in event_msg:
                misplay_occurred = True

            step_rewards.append(step_reward)
            agent_seq_len = len(full_ids)


            prompt_strs.append(action_prompt)
            completions_strs.append(completion_str)
            rewards.append(step_reward)
            seqlens.append(agent_seq_len)
            traj_len[0] += resp.input_len
            traj_len[1] += resp.output_len

            import re

            t = re.findall(r"<think>(.*?)</think>", completion_str, re.DOTALL)
            thought = t[-1].strip() if t else ""
            m = re.findall(r"<answer>(.*?)</answer>", completion_str, re.DOTALL)
            action_txt = m[-1].strip().lower() if m else ""

            # ========== 5) Agent produces summary ==========
            if self.use_summary:
                summary_prompt = (
                    f"{obs}\n"
                    # Structured summary update required

                    f"Previous summary: {prev_summary}"
                    f"Latest thought: {thought or 'None yet.'}\n"
                    f"Last action: {action_txt or 'Invalid action'}.\n" 
                    f"Provide a summary of the Hanabi game state for guiding the future turns from the perspective of the current active player {current_player}. You may include:\n"
                    "1. Public state recap (score, tokens, decks, fireworks, discards);\n"
                    "2. Inferred info about your own hand (safe/risky/unknown);\n"
                    "3. Teammate intentions and hint priorities;\n"
                    "4. Next-step focus for upcoming turns."
                )
                t0 = time.perf_counter()
                summary_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": summary_prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                t_tokenize_total += time.perf_counter() - t0
                summary_req = ModelRequest(
                    rid=f"{rid}-sum-{turn}",
                    input_ids=summary_ids,
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=2048),
                    tokenizer=self.tokenizer,
                )
                summary_cfg = summary_req.gconfig
                if self.student_api_key:
                    t0 = time.perf_counter()
                    agent_summary = await self._api_chat_completion(
                        summary_prompt,
                        summary_cfg,
                        self.student_api_key,
                        self.student_api_model,
                        self.student_api_provider or "openai",
                        "_student_api_session",
                        f"{rid}-sum-{turn}",
                    )
                    t_summary_total += time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    summary_resp = await engine.agenerate(summary_req)
                    t_summary_total += time.perf_counter() - t0
                    agent_summary = self.tokenizer.decode(
                        summary_resp.output_tokens, 
                        skip_special_tokens=True
                    )
                    results.append(
                        self._response_to_tensordict(summary_resp, sft_ppo_mask=0, agent_idx=player_idx)
                    )
                summaries[current_player].append(agent_summary)

                qa_logs.append(
                    {
                        "agent": current_player,
                        "role": "player",
                        "QAs": [
                            {"question": self_questions[i], "answer": agent_answers[i]}
                            for i in range(len(agent_answers))
                        ],
                        "summary": agent_summary,
                        "thought": thought,
                    }
                )
            else:
                qa_logs.append(
                    {
                        "agent": current_player,
                        "role": "player",
                        "QAs": [
                            {"question": self_questions[i], "answer": agent_answers[i]}
                            for i in range(len(agent_answers))
                        ],
                        "thought": thought
                    }
                )

            prev_thought = thought

            if teacher_answers:
                teacher_logs.append(
                    {
                        "agent": current_player,
                        "role": "teacher",
                        "QAs": [
                            {
                                "question": self_questions[i],
                                "answer": teacher_answers[i] if i < len(teacher_answers) else "",
                                "privileged": teacher_obs,
                            }
                            for i in range(len(self_questions))
                        ],
                    }
                )

            agent_result_indices.append(len(results) - 1)
            env_info = { # Build SFT data
                k: _sanitize_for_json(v)
                for k, v in info.items()
                if k not in {"teacher_observation"}
            }
            next_teacher_obs = info.get("teacher_observation", next_obs)
            next_state = info.get("state", next_obs)
            step_log = {
                "turn": turn,
                "player": current_player,
                "observation": obs,
                "guide": guide,
                "qgen_prompt": qgen_prompt,
                "qgen_response": qgen_text,
                "previous_summary": prev_summary,
                "qa_pairs": qa_pairs,
                "agent_question_prompts": agent_question_prompts,
                "teacher_prompts": teacher_prompts,
                "action_prompt": action_prompt,
                "action_completion": completion_str,
                "parsed_action": parsed_action,
                "env_event": info.get("event", ""),
                "env_info": env_info,
                "env_reward": env_step_reward,
                "adjusted_reward": step_reward,
                "agent_answers": agent_answers,
                "teacher_answers": teacher_answers,
                "process_reward": process_reward,
                "agent_summary": agent_summary,
                "summary_prompt": summary_prompt,
                "thought": thought,
                "next_observation": next_obs,
                "next_guide": next_guide,
                "next_teacher_observation": next_teacher_obs,
                "done": bool(done),
                "timestamp": time.time(),
            }
            episode_steps.append(step_log)

            obs = next_obs # prepare obs for next turn
            guide = next_guide
            state = next_state
            teacher_obs = info.get("teacher_observation", obs)

            if done or turn == self.max_turns - 1:
                break
        
        assert len(process_rewards) == len(agent_result_indices)

        running_return = 0.0 # cpalculate return
        returns = []
        for r in reversed(step_rewards):
            running_return += r
            returns.append(running_return)
        returns.reverse()
        final_total_reward = running_return 

        prev_idx = 0
        for idx, ret in zip(agent_result_indices, returns):
            ret = ret + self.process_reward_coef * process_rewards[idx]
            for i in range(prev_idx, idx + 1):
                results[i]["rewards"] = torch.tensor([ret], dtype=torch.float32)
            prev_idx = idx + 1

        for step_log, ret in zip(episode_steps, returns):
            step_log["discounted_return"] = ret

        rewards = returns
        total_reward = final_total_reward

        # ========== 6) Gather statistics and logging ==========
        stats = {}
        if hasattr(env, "get_stats"):
            try:
                stats = env.get_stats()
            except Exception:
                logger.error("Failed to collect stats from Hanabi environment.")

        logger.info(
            "Hanabi trajectory finished after %s turns with score %s and total reward %.2f. Abs reward: %.2f",
            turns_done,
            stats.get("score", 0),
            total_reward,
            sum(abs(r) for r in step_rewards),
        )

        avg_div = max(1, turns_done)
        logging_vals = [
            turns_done,
            traj_len[0] + traj_len[1],
            traj_len[0],
            traj_len[1],
            total_reward,
            sum(process_rewards) / len(process_rewards),
            stats.get("score", 0),
            stats.get("info_tokens", 0),
            stats.get("fuse_tokens", 0),
            stats.get("deck_remaining", 0),
            env.answer_format_record[0] / max(env.answer_format_record[1], 1e-6),
            t_qgen_total / avg_div,
            t_agent_answer_total / avg_div,
            (t_teacher_answer_total / avg_div) if teacher_logs else 0.0,
            t_action_build_total / avg_div,
            t_action_gen_total / avg_div,
            t_action_decode_total / avg_div,
            t_env_step_total / avg_div,
            t_summary_total / avg_div if self.use_summary else 0.0,
            t_pack_tensors_total / avg_div,
            t_tokenize_total / avg_div,
        ]

        log_tensor = torch.tensor(logging_vals, dtype=torch.float32).unsqueeze(0)
        if results:
            results[0]["logging"] = log_tensor
            zero_log = torch.zeros_like(log_tensor)
            for i in range(1, len(results)):
                results[i]["logging"] = zero_log

        trajectory = []
        if hasattr(env, "get_trajectory"):
            try:
                trajectory = env.get_trajectory()
            except Exception:
                logger.error("Failed to fetch trajectory history from Hanabi env.")

        episode_summary = { # Summary for SFT data construction
            "episode_id": episode_uuid,
            "turns": turns_done,
            "total_reward": total_reward,
            "final_score": stats.get("score", 0),
            "stats": _sanitize_for_json(stats),
            "returns": returns,
            "step_rewards": step_rewards,
            "success_count": success_count,
            "misplay_occurred": misplay_occurred,
            "qa_events": len(qa_logs),
            "process_rewards": process_rewards,
        }

        return (
            results,
            prompt_strs,
            completions_strs,
            rewards,
            seqlens,
            trajectory,
            qa_logs,
            teacher_logs,
            episode_steps,
            episode_summary,
        )

    async def arun_episode(self, engine: InferenceEngine, data):
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        episodes = await asyncio.gather(*tasks)

        results = []
        for res_list, *_ in episodes:
            results.extend(res_list)

        if self.dump_dir is not None and results:
            version = engine.get_version()
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            file_path = os.path.join(dump_path, f"{qid}.jsonl")
            async with aiofiles.open(file_path, "a") as f:
                for episode_idx, (
                    _,
                    p_list,
                    c_list,
                    r_list,
                    sl_list,
                    traj,
                    qa_logs,
                    t_logs,
                    step_logs,
                    episode_info,
                ) in enumerate(episodes):
                    record = {
                        "query_id": qid,
                        "episode_index": episode_idx,
                        "prompts": p_list,
                        "completions": c_list,
                        "sequence_lengths": sl_list,
                        "returns": r_list,
                        "trajectory": traj,
                        "qa_logs": qa_logs,
                        "teacher_logs": t_logs,
                        "steps": step_logs,
                        "summary": {
                            **episode_info,
                            "model_version": version,
                        },
                        "step_rewards": episode_info.get("step_rewards", []),
                        "timestamp": time.time(),
                    }
                    await f.write(json.dumps(_sanitize_for_json(record), indent = 2) + "\n")

        return concat_padded_tensors(results)