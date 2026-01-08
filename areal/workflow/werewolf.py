import asyncio
import os
import uuid
import time

import aiofiles
import aiofiles.os
import aiohttp
import colorama
import random
import torch
import json
import re
from typing import Dict, Deque, Optional
from collections import defaultdict, deque
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from realhf.base import logging
from realhf.impl.environment.werewolf_env import WerewolfEnv

logger = logging.getLogger("Werewolf workflow")

DEFAULT_TEACHER_OBSERVATION_KWARGS = dict(
    use_individual_thoughts=True,
    use_global_obs=True,
)


def _sanitize_for_json(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    return str(obj)


def _extract_three_questions(text: str) -> list[str]:
    """
    Try to parse three questions from the model output. Accepts formats like:
      Q1: ...
      Q2: ...
      Q3: ...
    Falls back to splitting into sentences ending with '?'. Returns up to 3.
    """
    qs = []
    # Pattern 1: Q1...Q2...Q3
    if ("Q1:" in text) and ("Q2:" in text) and ("Q3:" in text):
        q1 = text[text.find("Q1:") + 3 : text.find("Q2:")].strip()
        q2 = text[text.find("Q2:") + 3 : text.find("Q3:")].strip()
        q3 = text[text.find("Q3:") + 3 :].strip()
        qs = [q1, q2, q3]
    # Pattern 2: Q1:/Q2:/Q3:
    if len(qs) < 3:
        for m in re.findall(
            r"Q\s*([123])\s*[:：]\s*(.+?)(?=(?:\nQ\s*[123]\s*[:：])|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            _, q = m
            q = q.strip()
            if q.endswith(("</s>", "</answer>", "</assistant>")):
                q = re.sub(r"</s>|</answer>|</assistant>", "", q).strip()
            qs.append(q)
    # If not found, try to split by question marks.
    if len(qs) < 3:
        cand = re.findall(r"([^?？]+[?？])", text, flags=re.DOTALL)
        cand = [c.strip() for c in cand if c.strip()]
        for c in cand:
            if len(qs) >= 3:
                break
            qs.append(c)
    return qs[:3]


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
        self._req_timestamps: Deque[float] = deque()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            # Drop timestamps older than 60s
            while self._req_timestamps and now - self._req_timestamps[0] > 60:
                self._req_timestamps.popleft()

            # If we’re exceeding limits, wait
            while True:
                now = time.monotonic()
                # Clean up old timestamps again inside loop
                while self._req_timestamps and now - self._req_timestamps[0] > 60:
                    self._req_timestamps.popleft()

                # Count requests within 1s window and 60s window
                last_1s = [t for t in self._req_timestamps if now - t <= 1.0]
                last_60s = len(self._req_timestamps)

                if len(last_1s) < self.per_second and last_60s < self.per_minute:
                    self._req_timestamps.append(now)
                    return  # allowed to proceed

                # Sleep until earliest timestamp expires
                next_allowed = min(
                    (1.0 - (now - last_1s[0])) if last_1s else 0,
                    (60.0 - (now - self._req_timestamps[0])) if self._req_timestamps else 0,
                )
                await asyncio.sleep(max(next_allowed, 0.01))

class WerewolfWorkflow(RolloutWorkflow):
    """
    Workflow for running the werewolf game.
    Agent now self-generates 3 questions each turn, answers them,
    and uses the answers to guide action-making. Teacher also answers
    these questions to provide data.
    """
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int = 70,
        turn_discount: float = 1.0,
        dump_dir: str | None = None,
        env_kwargs: dict | None = None,
        role: str = "villager",
        opp_rollout: InferenceEngine | None = None,
        opp_tokenizer: PreTrainedTokenizerFast | None = None,
        teacher_rollout: InferenceEngine | None = None,
        teacher_tokenizer: PreTrainedTokenizerFast | None = None,
        opp_api_key: str | None=None,
        opp_api_model: str | None = None,
        teacher_api_key: str | None = None,
        teacher_api_model: str | None = None,
        questions: list[str] | None = None,  # kept for backward-compat but unused now
        teacher_obs_kwargs: dict | None = None,
        teacher_process_reward: bool = False,
        process_reward_coef: float = 0.2,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.dump_dir = dump_dir
        self.env_kwargs = env_kwargs or {}
        self.teacher_obs_kwargs = teacher_obs_kwargs or dict()
        for k, v in DEFAULT_TEACHER_OBSERVATION_KWARGS.items():
            if k not in self.teacher_obs_kwargs:
                self.teacher_obs_kwargs[k] = v
        self.role = role
        self.opp_rollout = opp_rollout
        self.opp_tokenizer = opp_tokenizer
        self.use_teacher = True
        self.process_reward_coef = process_reward_coef
        self.teacher_process_reward = teacher_process_reward
        self.teacher_rollout = teacher_rollout
        self.teacher_tokenizer = teacher_tokenizer
        self.opp_api_key = (opp_api_key or "").strip()
        self.teacher_api_key = (teacher_api_key or "").strip()
        self.opp_api_provider = (
            _detect_api_provider(self.opp_api_key) if self.opp_api_key else None
        )
        self.teacher_api_provider = (
            _detect_api_provider(self.teacher_api_key)
            if self.teacher_api_key
            else None
        )
        self.opp_api_model = (opp_api_model or "").strip() if self.opp_api_key else ""
        self.teacher_api_model = (
            (teacher_api_model or "").strip() if self.teacher_api_key else ""
        )
        if self.opp_api_key:
            default_model = (
                "gpt-4o-2024-11-20"
                if self.opp_api_provider == "openai"
                else "claude-3-7-sonnet-20250219"
            )
            if not self.opp_api_model:
                self.opp_api_model = os.getenv(
                    "AREAL_OPP_API_MODEL", default_model
                )
            self.opp_api_model = (
                (self.opp_api_model or "").strip() or default_model
            )
        if self.teacher_api_key:
            default_teacher_model = (
                "gpt-4o-2024-11-20"
                if self.teacher_api_provider == "openai"
                else "claude-3-7-sonnet-20250219"
            )
            if not self.teacher_api_model:
                self.teacher_api_model = os.getenv(
                    "AREAL_TEACHER_API_MODEL", default_teacher_model
                )
            self.teacher_api_model = (
                (self.teacher_api_model or "").strip() or default_teacher_model
            )
        self._opp_api_session: aiohttp.ClientSession | None = None
        self._teacher_api_session: aiohttp.ClientSession | None = None
        self.rate_limiter = RateLimiter(per_second=2, per_minute=60)
        self.use_summary = True

        # API retry configs
        self._max_retries = 6
        self._base_backoff = 1.0
        self._max_backoff = 10.0

        # Deprecated path: we no longer use predefined questions
        self.answer_questions = True
        self.questions = []

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
        # Exponential backoff with jitter, capped
        backoff = min(self._base_backoff * (2 ** attempt), self._max_backoff)
        jitter = random.uniform(0.0, 0.25 * backoff)
        return backoff + jitter

    def _retry_after_seconds(self, retry_after: str) -> Optional[float]:
        # Retry-After can be seconds or an HTTP date; we honor only seconds here safely.
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
        last_err_text = None

        for attempt in range(self._max_retries + 1):
            # Rate-limit every attempt (initial + retries)
            await self.rate_limiter.acquire()

            logger.info(f"Starting API generation attempt {attempt}.")

            try:
                async with session.request(method, url, headers=headers, json=json) as resp:
                    # Successful JSON
                    if 200 <= resp.status < 300:
                        return await resp.json()

                    # Read text for diagnostics
                    text = await resp.text()
                    last_err_text = text

                    # Decide if retryable
                    retryable = (resp.status == 429) or (resp.status == 408) or (500 <= resp.status < 600)

                    if not retryable or attempt == self._max_retries:
                        raise RuntimeError(f"API request failed with status {resp.status}: {text}")

                    # Honor Retry-After if present (esp. on 429)
                    retry_after_hdr = resp.headers.get("Retry-After")
                    sleep_for = self._retry_after_seconds(retry_after_hdr) if retry_after_hdr else None
                    if sleep_for is None:
                        sleep_for = self._compute_backoff(attempt)

                    await asyncio.sleep(sleep_for)
                    continue

            except (aiohttp.ServerDisconnectedError,
                    aiohttp.ClientOSError,
                    aiohttp.ClientConnectionError,
                    aiohttp.ClientPayloadError,
                    asyncio.TimeoutError) as e:
                if attempt == self._max_retries:
                    raise RuntimeError(f"Network or timeout error after retries: {e}") from e
                await asyncio.sleep(self._compute_backoff(attempt))
                continue

        # Shouldn’t reach here
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
        # Await rate limiter
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
                "max_completion_tokens": max_tokens,
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
        else:
            choices = data.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            resp = message.get("content", "")
            if resp:
                logger.info(f"API call successful with resp: {resp}")
            return resp

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

    def _build_question_generation_prompt(
        self,
        obs: str,
        prev_summary: str,
        current_role: str,
        phase: str,
    ) -> str:
        """Build phase-aware and role-specific question generation prompt.

        Args:
            obs: Current observation
            prev_summary: Previous summary for context
            current_role: Agent's current role (werewolf, villager, witch, etc.)
            phase: Current game phase (night, discussion, day, hunter)

        Returns:
            Formatted prompt string for question generation
        """
        qa_target_role = "a werewolf" if current_role != "werewolf" else "biggest living threat"

        # Build phase-aware and role-specific examples
        example_questions = []

        # Role-specific strategic questions
        if current_role == "witch":
            if phase == "night":
                example_questions.append("Should I use my heal or poison ability this turn, and on which player?")
            else:
                example_questions.append("Based on recent events, who is most likely a werewolf?")
        elif current_role == "foreseer":
            if phase == "night":
                example_questions.append("Which player should I check tonight to maximize information gain?")
            else:
                example_questions.append("How should I strategically reveal or use my investigation results?")
        elif current_role == "hunter":
            example_questions.append("If I die, which player should I shoot and why?")
        elif current_role == "werewolf":
            if phase == "night":
                example_questions.append("Which non-werewolf player should we target to maximize our advantage?")
            else:
                example_questions.append("What narrative should I push to deflect suspicion from werewolves?")
        else:  # villager
            example_questions.append(f"Which player is {qa_target_role}, and what evidence supports this?")

        # Phase-specific questions
        if phase == "discussion":
            example_questions.append("Whose statements during discussion seem inconsistent or suspicious?")
            example_questions.append("What is each player's likely role based on their behavior and claims?")
        elif phase == "day":
            example_questions.append("Who should be voted out to maximize our team's winning chances?")
            example_questions.append("Which players are most likely aligned with werewolves based on voting patterns?")
        elif phase == "night":
            if current_role == "werewolf":
                example_questions.append("Which player poses the biggest threat to werewolves and should be eliminated?")
            else:
                example_questions.append("What actions should I take tonight to best help my team?")

        # Ensure we have at least 3 examples, add generic ones if needed
        if len(example_questions) < 3:
            example_questions.append("What are the key threats or suspicious behaviors I should focus on?")
        if len(example_questions) < 3:
            example_questions.append("Which players are confirmed dead and what roles did they have?")

        # Take first 3 examples
        example_questions = example_questions[:3]
        examples_text = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(example_questions)])

        prompt = (
            "You are playing Werewolf. Generate investigative questions for yourself.\n\n"
            "Context:\n"
            f"{obs}\n"
            f"Previous summary (keep in mind, do not rewrite): {prev_summary}\n\n"
            "Task:\n"
            "- Produce exactly three investigative questions that help you make better decisions.\n"
            "- Questions should probe player alignment, deception, threats, and strategic options.\n"
            "- Questions can involve inference and theory-of-mind reasoning about what others know or believe.\n"
            "- Focus on situationally relevant questions given your role and the current game phase.\n"
            "- Do NOT make decisions or statements—only ask questions.\n\n"
            "Good examples for your situation:\n"
            f"{examples_text}\n\n"
            "Output format (exactly):\n"
            "Q1: ...\n"
            "Q2: ...\n"
            "Q3: ...\n"
        )
        return prompt

    
    def _build_agent_answer_prompt(
        self,
        obs: str,
        prev_summary: str,
        question: str,
    ) -> str:
        """Build prompt for agent to answer their own investigative question.

        Args:
            obs: Current observation
            prev_summary: Previous summary for context
            question: The question to answer

        Returns:
            Formatted prompt string for agent answer generation
        """
        prompt = (
            "You are answering your own investigative question to guide your decision-making in Werewolf. You are now playing as the current active player.\n\n"
            "Context:\n"
            f"{obs}\n"
            f"Summary at last decision-making time of the active player:\n```\n{prev_summary}\n```\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Instructions:\n"
            "- Provide a thoughtful, evidence-based answer grounded in the current game state\n"
            "- Use theory-of-mind reasoning: consider what other players know, believe, and might be trying to accomplish\n"
            "- Make inferences from observable behavior, voting patterns, statements, and claims\n"
            "- Be concrete and specific, citing evidence when possible\n"
            "- Keep your answer concise and actionable (within 5 sentences)\n\n"
            "Provide your answer:"
        )
        return prompt


    @staticmethod
    def _response_to_tensordict(
        resp: ModelResponse, *, sft_ppo_mask: int = 0, agent_idx: int = -1, reward=0
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
            "rewards": torch.tensor([reward], dtype=torch.float32),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.bool).unsqueeze(0),
            "sft_ppo_mask": torch.tensor([sft_ppo_mask], dtype=torch.long),
            "agent_idx": torch.tensor([agent_idx], dtype=torch.long),
        }

    async def _run_one_episode(self, engine: InferenceEngine, data, rid):
        env = WerewolfEnv(**self.env_kwargs)
        obs, guide, info = await env.sreset()
        teacher_obs = info.get("teacher_observation", obs)

        results = []  # Final return tensor
        prompt_strs = []  # All prompts
        completions_strs = []  # All completions
        rewards = []  # All rewards
        seqlens = []
        vill_total = 0.0 # Villager side total reward
        were_total = 0.0 # Werewolf side total reward
        traj_len = [0, 0]  # input, output
        summaries = defaultdict(list)
        agent_thoughts: Dict[str, str] = {}
        qa_logs = []
        teacher_logs = []
        step_logs: list[dict] = []
        agent_result_indices: list[int] = []  # Track which results correspond to agent actions
        step_rewards: list[list[float]] = []  # Track rewards for each step: [villager_reward, werewolf_reward]
        process_rewards: list[float] = []
        agent_roles: list[str] = []  # Track role of agent for each turn
        episode_info: dict = {
            "episode_id": rid,
            "format_reward_scale": None,
            "vill_reward_total": 0.0,
            "were_reward_total": 0.0,
            "step_rewards": [],
        }

        # ---- timing accumulators (seconds) ----
        t_qgen_total = 0.0
        t_qgen_decode_total = 0.0
        t_agent_answer_total = 0.0
        t_teacher_answer_total = 0.0
        t_action_build_total = 0.0
        t_action_gen_total = 0.0
        t_action_decode_total = 0.0
        t_env_step_total = 0.0
        t_summary_agent_total = 0.0
        t_summary_teacher_total = 0.0
        t_pack_tensors_total = 0.0
        t_tokenize_total = 0.0

        turns_done = 0

        for turn in range(self.max_turns):
            turns_done += 1
            turn_obs = obs
            turn_guide = guide
            # Store the current agent and get its summary
            current_agent = env.agent_player
            current_role = env.agent_role
            use_opp_generation = (
                ((current_role != "werewolf" and self.role == "werewolf") or (current_role == "werewolf" and self.role == "villager"))
                and (self.opp_rollout or self.opp_api_key)
            )
            prev_summary = summaries[current_agent][-1] if summaries[current_agent] else "The player has not made actions or speaked yet."

            # ========== 1) Agent self-generates 3 questions ==========
            qgen_prompt = self._build_question_generation_prompt(
                obs=obs,
                prev_summary=prev_summary,
                current_role=current_role,
                phase=env.phase,
            )

            t0 = time.perf_counter()
            qgen_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": qgen_prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            t_tokenize_total += time.perf_counter() - t0
            
            qgen_cfg = self.gconfig.new(n_samples=1, max_new_tokens=4096)
            qgen_resp: ModelResponse | None = None
            if use_opp_generation and self.opp_api_key:
                t0 = time.perf_counter()
                qgen_text = await self._api_chat_completion(
                    qgen_prompt,
                    qgen_cfg,
                    self.opp_api_key,
                    self.opp_api_model,
                    self.opp_api_provider or "openai",
                    "_opp_api_session",
                    f"{rid}-qgen-{turn}",
                )
                t_qgen_total += time.perf_counter() - t0
                t_qgen_decode_total += 0.0
            else:
                if use_opp_generation:
                    qgen_req = ModelRequest(
                        rid=f"{rid}-qgen-{turn}",
                        input_ids=qgen_ids,
                        gconfig=qgen_cfg,
                        tokenizer=self.opp_tokenizer or self.tokenizer,
                    )
                    t0 = time.perf_counter()
                    qgen_resp = await self.opp_rollout.agenerate(qgen_req)
                else:
                    qgen_req = ModelRequest(
                        rid=f"{rid}-qgen-{turn}",
                        input_ids=qgen_ids,
                        gconfig=qgen_cfg,
                        tokenizer=self.tokenizer,
                    )
                    t0 = time.perf_counter()
                    qgen_resp = await engine.agenerate(qgen_req)
                t_qgen_total += time.perf_counter() - t0

                decode_tok = self.opp_tokenizer if (use_opp_generation and self.opp_tokenizer) else self.tokenizer
                t0 = time.perf_counter()
                qgen_text = decode_tok.decode(qgen_resp.output_tokens, skip_special_tokens=True)
                t_qgen_decode_total += time.perf_counter() - t0

            # Add question generation to training data (only for student agent, not opponent)
            if not use_opp_generation and qgen_resp is not None:
                t0 = time.perf_counter()
                player_idx = env.roles.index(current_agent) if current_agent in env.roles else -1
                results.append(self._response_to_tensordict(qgen_resp, sft_ppo_mask=0, agent_idx=player_idx))
                t_pack_tensors_total += time.perf_counter() - t0

            self_questions = _extract_three_questions(qgen_text)
            if not self_questions:
                # Safety fallback - use role-aware questions
                qa_target_role = "a werewolf" if current_role != "werewolf" else "biggest living threat"
                if current_role == "werewolf":
                    self_questions = [
                        "Which non-werewolf player poses the biggest threat to us?",
                        "What narrative should I push to deflect suspicion from werewolves?",
                        "Which players are most likely to suspect werewolves?",
                    ]
                elif current_role == "witch":
                    self_questions = [
                        "Should I use my heal or poison ability now, and on whom?",
                        "Based on recent events, who is most likely a werewolf?",
                        "Which players have exhibited suspicious behavior?",
                    ]
                elif current_role == "foreseer":
                    self_questions = [
                        "Which player should I investigate to gain the most information?",
                        "How should I use my investigation results strategically?",
                        f"Which player is {qa_target_role}, and what evidence supports this?",
                    ]
                else:  # villager or hunter
                    self_questions = [
                        f"Which player do you think is {qa_target_role} and why?",
                        "What action should I take now to maximize team success?",
                        "Which players are confirmed dead and what roles did they have?",
                    ]

            # ========== 2) Agent answers the 3 questions ==========
            agent_answer_tasks = []
            agent_answer_inputs: list[list[int]] = []  # Store input_ids for later
            agent_answer_prompts: list[str] = []  # Store prompts for SFT data generation
            agent_answer_cfg = self.gconfig.new(n_samples=1, max_new_tokens=2048)
            for qi, q in enumerate(self_questions):
                aprompt = self._build_agent_answer_prompt(
                    obs=obs,
                    prev_summary=prev_summary,
                    question=q,
                )
                agent_answer_prompts.append(aprompt)  # Store for later use in SFT data
                t0 = time.perf_counter()
                a_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": aprompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                t_tokenize_total += time.perf_counter() - t0
                agent_answer_inputs.append(a_ids)

                if use_opp_generation and self.opp_api_key:
                    agent_answer_tasks.append(
                        self._api_chat_completion(
                            aprompt,
                            agent_answer_cfg,
                            self.opp_api_key,
                            self.opp_api_model,
                            self.opp_api_provider or "openai",
                            "_opp_api_session",
                            f"{rid}-ans-{turn}-{qi}",
                        )
                    )
                elif use_opp_generation:
                    a_req = ModelRequest(
                        rid=f"{rid}-ans-{turn}-{qi}",
                        input_ids=a_ids,
                        gconfig=agent_answer_cfg,
                        tokenizer=self.opp_tokenizer or self.tokenizer,
                    )
                    agent_answer_tasks.append(self.opp_rollout.agenerate(a_req))
                else:
                    a_req = ModelRequest(
                        rid=f"{rid}-ans-{turn}-{qi}",
                        input_ids=a_ids,
                        gconfig=agent_answer_cfg,
                        tokenizer=self.tokenizer,
                    )
                    agent_answer_tasks.append(engine.agenerate(a_req))

            # Only gather the agent's answer now
            agent_answers: list[str]
            agent_answer_resps: list[ModelResponse] = []
            if use_opp_generation and self.opp_api_key:
                t0 = time.perf_counter()
                agent_answers = await asyncio.gather(*agent_answer_tasks)
                t_agent_answer_total += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                agent_ans_resps = await asyncio.gather(*agent_answer_tasks)
                if use_opp_generation and self.opp_tokenizer:
                    agent_answers = [
                        self.opp_tokenizer.decode(r.output_tokens, skip_special_tokens=True)
                        for r in agent_ans_resps
                    ]
                else:
                    agent_answers = [
                        self.tokenizer.decode(r.output_tokens, skip_special_tokens=True)
                        for r in agent_ans_resps
                    ]
                agent_answer_resps = agent_ans_resps
                t_agent_answer_total += time.perf_counter() - t0

            # Teacher answers using privileged observation from environment
            teacher_answers: list[str] = []
            _teacher_obs_used = obs  # Default to student observation
            if (not use_opp_generation) and (self.teacher_rollout or self.teacher_api_key):
                teacher_answer_tasks = []
                teacher_answer_cfg = self.gconfig.new(n_samples=1, max_new_tokens=2048)
                teacher_tok = self.teacher_tokenizer or self.tokenizer

                # Build agent thoughts section for privileged information
                agent_thoughts_info = ""
                if agent_thoughts and self.teacher_obs_kwargs["use_individual_thoughts"]:
                    agent_thoughts_info = "\n\n=== Players' Inner Thoughts (Privileged) ===\n"
                    for agent_key, thought in agent_thoughts.items():
                        if thought:  # Only include non-empty thoughts
                            agent_thoughts_info += f"{agent_key}: {thought}\n"

                # Determine which observation to use based on configuration
                _teacher_obs = teacher_obs if self.teacher_obs_kwargs["use_global_obs"] else obs
                _teacher_obs_used = _teacher_obs  # Track what was actually used for logging

                for qi, q in enumerate(self_questions):
                    # Use teacher observation from environment which includes all privileged information
                    taprompt = (
                        f"{_teacher_obs}"
                        f"{agent_thoughts_info}\n\n"
                        f"Question to answer: {q}\n\n"
                        "Instructions:\n"
                        "- Use the privileged information above (including player roles, memories, and inner thoughts) to provide a helpful, concrete answer, from the perspective of the current active player.\n"
                        "- Be concise and specific, grounded in the game state.\n"
                        "- Keep to within 5 sentences."
                    )
                    if self.teacher_process_reward:
                        taprompt = (
                            f"{_teacher_obs}"
                            f"{agent_thoughts_info}\n\n"
                            f"# Judge Question Answer Pair\n"
                            f"Question: {q}\nAnswer: {agent_answers[qi]}\n"
                            "Instructions:\n"
                            "Judge whether the answer is correct to the question using all privileged information, (including player roles, memories, and inner thoughts).\n"
                            "Reply with [CORRECT] if the answer is correct and [WRONG] is the answer is wrong.\n"
                            "Format your output as:\n"
                            "<analysis> your reasoning process </analysis>\n<result> correctness of the answer </result> "
                        )
                    t0 = time.perf_counter()
                    ta_ids = teacher_tok.apply_chat_template(
                        [{"role": "user", "content": taprompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    t_tokenize_total += time.perf_counter() - t0

                    if self.teacher_api_key:
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
                    else:
                        ta_req = ModelRequest(
                            rid=f"{rid}-tans-{turn}-{qi}",
                            input_ids=ta_ids,
                            gconfig=teacher_answer_cfg,
                            tokenizer=teacher_tok,
                        )
                        teacher_answer_tasks.append(self.teacher_rollout.agenerate(ta_req))

                if teacher_answer_tasks:
                    if self.teacher_api_key:
                        t0 = time.perf_counter()
                        teacher_answers = await asyncio.gather(*teacher_answer_tasks)
                        t_teacher_answer_total += time.perf_counter() - t0
                    else:
                        t0 = time.perf_counter()
                        teacher_ans_resps = await asyncio.gather(*teacher_answer_tasks)
                        if self.teacher_tokenizer:
                            teacher_answers = [
                                self.teacher_tokenizer.decode(
                                    r.output_tokens, skip_special_tokens=True
                                )
                                for r in teacher_ans_resps
                            ]
                        else:
                            teacher_answers = [
                                self.tokenizer.decode(
                                    r.output_tokens, skip_special_tokens=True
                                )
                                for r in teacher_ans_resps
                            ]
                        t_teacher_answer_total += time.perf_counter() - t0

            _process_reward = 0.0
            _process_rewards = []
            if teacher_answers and self.teacher_process_reward:
                for qi, t_ans in enumerate(teacher_answers):
                    _process_rewards.append(float("CORRECT" in t_ans))
                _process_reward = sum(_process_rewards)
            if not use_opp_generation:
                process_rewards.append(_process_reward)

            # Add agent answers to training data (only for student agent, not opponent)
            if not use_opp_generation and agent_answer_resps:
                t0 = time.perf_counter()
                player_idx = env.roles.index(current_agent) if current_agent in env.roles else -1
                for qi, a_resp in enumerate(agent_answer_resps):
                    _reward = 0.0
                    if len(_process_rewards) > 0:
                        _reward = _process_rewards[qi] - 1
                    results.append(
                        self._response_to_tensordict(a_resp, sft_ppo_mask=0, agent_idx=player_idx, reward=_reward)
                    )
                t_pack_tensors_total += time.perf_counter() - t0

            # ========== 3) Use agent's Q&A to guide action generation ==========
            qa_block = "\n".join([
                f"{i+1}) {self_questions[i]}\nAnswer: {agent_answers[i].strip()}"
                for i in range(len(agent_answers))
            ])
            t0 = time.perf_counter()

            if self.use_summary:
                action_prompt = (
                    f"{obs}\n"
                    "You are playing as the active player. You must propose one concrete action for this Werewolf turn.\n\n"
                    f"Inner thought of the player at last decision-making step (from the perspective of the active player): \n```\n{prev_summary}\n```\n\n"
                    "Your self-questions and answers:\n"
                    f"{qa_block}\n\n"
                    # "Required output format:\n"
                    # "<think>step-by-step reasoning tied to the observation and answers</think> "
                    # "<answer>the single action you will take now</answer>\n\n"
                    f"{guide}"
                )
            else:
                action_prompt = (
                    f"{obs}\n\n"
                    "You are playing as the active player. You must propose one concrete action for this Werewolf turn.\n\n"
                    "Your self-questions and answers:\n"
                    f"{qa_block}\n\n"
                    # "Required output format:\n"
                    # "<think>step-by-step reasoning tied to the observation and answers</think> "
                    # "<answer>the single action you will take now</answer>\n\n"
                    f"{guide}"
                )

            action_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": action_prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True
            )
            t_action_build_total += time.perf_counter() - t0
            t_tokenize_total += 0.0  # (already included in build section)

            if use_opp_generation and self.opp_api_key:
                action_cfg = self.gconfig.new(n_samples=1, max_new_tokens=8192)
                t0 = time.perf_counter()
                completion_str = await self._api_chat_completion(
                    action_prompt,
                    action_cfg,
                    self.opp_api_key,
                    self.opp_api_model,
                    self.opp_api_provider or "openai",
                    "_opp_api_session",
                    rid,
                )
                t_action_gen_total += time.perf_counter() - t0
                t0 = time.perf_counter()
                resp = self._build_api_response(action_ids, completion_str, self.tokenizer)
                t_action_decode_total += time.perf_counter() - t0
            elif use_opp_generation:
                req = ModelRequest(
                    rid=rid,
                    input_ids=action_ids,
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=8192),
                    tokenizer=self.opp_tokenizer or self.tokenizer,
                )
                t0 = time.perf_counter()
                resp = await self.opp_rollout.agenerate(req)
                t_action_gen_total += time.perf_counter() - t0
                t0 = time.perf_counter()
                completion_str = (
                    self.opp_tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
                    if self.opp_tokenizer
                    else self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
                )
                t_action_decode_total += time.perf_counter() - t0
            else:
                req = ModelRequest(
                    rid=rid,
                    input_ids=action_ids,
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=8192),
                    tokenizer=self.tokenizer,
                )
                t0 = time.perf_counter()
                resp = await engine.agenerate(req)
                t_action_gen_total += time.perf_counter() - t0
                t0 = time.perf_counter()
                completion_str = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
                t_action_decode_total += time.perf_counter() - t0
            
            action_response = completion_str
            if "<answer>" in action_response and "</answer>" not in action_response:
                action_response += "</answer>"

            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            prompt_str = self.tokenizer.decode(action_ids, skip_special_tokens=True)

            # Get next env state
            t0 = time.perf_counter()
            next_obs, next_guide, reward_list, done, _, info = await env.step(
                (data.get("query_id", ""), [action_response])
            )
            t_env_step_total += time.perf_counter() - t0
            next_teacher_obs = info.get("teacher_observation", next_obs)

            format_reward_scale = (env.answer_format_record[0] / env.answer_format_record[1])
            reward_list = [rew * format_reward_scale for rew in reward_list]
            vill_total += float(reward_list[0])
            were_total += float(reward_list[1])

            if self.role == "both":
                reward = reward_list[0] + reward_list[1]
            elif self.role == "werewolf":
                reward = reward_list[1]
            else:
                reward = reward_list[0]

            # ========== 4) Build PPO training data ==========
            t0 = time.perf_counter()
            if not use_opp_generation:
                player_idx = env.roles.index(current_agent) if current_agent in env.roles else -1
                res = {
                    "input_ids": torch.tensor(seq).unsqueeze(0),
                    "loss_mask": torch.tensor(loss_mask).unsqueeze(0),
                    "logprobs": torch.tensor(logprobs).unsqueeze(0),
                    "versions": torch.tensor(versions).unsqueeze(0),
                    "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                    "rewards": torch.tensor([0.0], dtype=torch.float32),
                    "sft_ppo_mask": torch.tensor([0], dtype=torch.long),
                    "agent_idx": torch.tensor([player_idx], dtype=torch.long),
                }
                results.append(res)
            step_rewards.append(reward_list)  # Track both villager and werewolf rewards
            agent_roles.append(current_role)  # Track agent role for this turn
            t_pack_tensors_total += time.perf_counter() - t0

            prompt_strs.append(prompt_str)
            completions_strs.append(completion_str)
            rewards.append(reward)
            seqlens.append(len(seq))
            traj_len[0] += resp.input_len
            traj_len[1] += resp.output_len
            episode_info["step_rewards"].append(reward)
            episode_info["format_reward_scale"] = format_reward_scale
            episode_info["vill_reward_total"] = vill_total
            episode_info["were_reward_total"] = were_total

            # ========== 5) Build SFT training data ==========
            # Use student prompts (without privileged info) as input, teacher answers as target
            if (self.teacher_rollout or self.teacher_api_key) and len(teacher_answers) > 0 and not use_opp_generation and not self.teacher_process_reward:
                t0 = time.perf_counter()
                player_idx = env.roles.index(current_agent) if current_agent in env.roles else -1
                for qi, (aprompt, t_ans) in enumerate(zip(agent_answer_prompts, teacher_answers)):
                    # Reuse the student prompt from step 2
                    prompt_ids = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": aprompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thining=False,
                    )
                    # Build synthetic response from student prompt + teacher answer
                    resp = self._build_api_response(prompt_ids, t_ans, self.tokenizer)
                    results.append(
                        self._response_to_tensordict(resp, sft_ppo_mask=1, agent_idx=player_idx)
                    )
                t_pack_tensors_total += time.perf_counter() - t0

            # ========== 6) Agent summarization, log agent thinking and Q&As ==========
            t = re.findall(r"<think>(.*?)</think>", action_response, re.DOTALL)
            think_str = t[-1].strip().lower() if t else ""
            agent_thoughts[f"{current_agent} ({current_role})"] = think_str
            agent_summary = None
            if self.use_summary:
                m = re.findall(r"<answer>(.*?)</answer>", action_response, re.DOTALL)
                action_txt = m[-1].strip().lower() if m else ""
                summary_prompt = (
                    f"{obs}\n\nYou are playing as the active player. You selected action: {action_txt}. "
                    f"Your latest memory at last decision-making step: \n```\n{prev_summary}\n```\n\nProvide an updated memory covering the game states and notable information, "
                    "to guide your future planning and next moves. Be concise and brief."
                )
                summary_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": summary_prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                summary_req = ModelRequest(
                    rid=f"{rid}-s-{turn}",
                    input_ids=summary_ids,
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=2048),
                    tokenizer=self.tokenizer,
                )
                if use_opp_generation and self.opp_api_key:
                    summary_tasks = [self._api_chat_completion(
                        summary_prompt,
                        self.gconfig.new(n_samples=1, max_new_tokens=2048),
                        self.opp_api_key,
                        self.opp_api_model,
                        self.opp_api_provider or "openai",
                        "_opp_api_session",
                        f"{rid}-s-{turn}",
                    )]
                elif use_opp_generation:
                    summary_tasks = [self.opp_rollout.agenerate(summary_req)]
                else:
                    summary_tasks = [engine.agenerate(summary_req)]
                t0 = time.perf_counter()
                summary_resps = await asyncio.gather(*summary_tasks)
                t_sum = time.perf_counter() - t0
                t_summary_agent_total += t_sum

                if use_opp_generation and self.opp_api_key:
                    agent_summary = summary_resps[0]
                elif use_opp_generation and self.opp_tokenizer:
                    agent_summary = self.opp_tokenizer.decode(summary_resps[0].output_tokens, skip_special_tokens=True)
                else:
                    agent_summary = self.tokenizer.decode(summary_resps[0].output_tokens, skip_special_tokens=True)
                summaries[current_agent].append(agent_summary)

                # Add summary to training data (only for student agent, not opponent)
                if not use_opp_generation:
                    t0 = time.perf_counter()
                    player_idx = env.roles.index(current_agent) if current_agent in env.roles else -1
                    summary_resp = summary_resps[0]
                    results.append(
                        self._response_to_tensordict(summary_resp, sft_ppo_mask=0, agent_idx=player_idx)
                    )
                    t_pack_tensors_total += time.perf_counter() - t0

                qa_logs.append(
                    {
                        "agent": current_agent,
                        "role": current_role,
                        "QAs": [{"question": self_questions[i], "answer": agent_answers[i]} for i in range(len(self_questions))],
                        "summary_prompt": summary_prompt,
                        "summary": agent_summary,
                    }
                )
            else:  # Do not use summary
                qa_logs.append(
                    {
                        "agent": current_agent,
                        "role": current_role,
                        "QAs": [{"question": self_questions[i], "answer": agent_answers[i]} for i in range(len(self_questions))],
                    }
                )
            teacher_summary = None
            if self.teacher_rollout or self.teacher_api_key:
                teacher_entry = {
                    "agent": current_agent,
                    "role": current_role,
                    "QAs": [
                        {
                            "question": self_questions[i],
                            "privileged_info": _teacher_obs_used,  # Use the observation that was actually used
                            "answer": teacher_answers[i] if i < len(teacher_answers) else "",
                        }
                        for i in range(len(self_questions))
                    ],
                }
                teacher_logs.append(teacher_entry)
                teacher_summary = teacher_entry

            # Track the final result index after all training data for this turn is added
            agent_result_indices.append(len(results) - 1)

            step_logs.append(
                {
                    "turn": turn + 1,
                    "agent": current_agent,
                    "role": current_role,
                    "observation": turn_obs,
                    "guide": turn_guide,
                    "previous_summary": prev_summary,
                    "self_questions": self_questions,
                    "next_observation": next_obs,
                    "next_guide": next_guide,
                    "reward": reward,
                    "reward_breakdown": {
                        "villager": reward_list[0],
                        "werewolf": reward_list[1],
                        "format_scale": format_reward_scale,
                    },
                    "done": bool(done),
                    "teacher_supervision": teacher_summary,
                    "agent_thought": think_str,
                    "qgen_prompt": qgen_prompt,
                    "qgen_response": qgen_text,
                    "summary_prompt": summary_prompt,
                    "agent_summary": agent_summary if self.use_summary else None,
                    "agent_answer_prompts": agent_answer_prompts,
                    "agent_answers": agent_answers,
                    "teacher_answers": teacher_answers,
                    "action_prompt": action_prompt,
                    "action_completion": completion_str,
                    "process_rewards": _process_rewards,
                }
            )

            if done or (turn == self.max_turns - 1):
                logger.info(f"Trajectory ended with {turn + 1} turns, total reward: {sum(rewards)}.")
                break
            obs = next_obs
            guide = next_guide
            teacher_obs = next_teacher_obs

        # Calculate discounted returns separately for villagers and werewolves
        running_return_villager = 0.0
        running_return_werewolf = 0.0
        returns_villager = []
        returns_werewolf = []

        # Calculate returns backward through time
        for reward_pair in reversed(step_rewards):
            running_return_villager += reward_pair[0]  # Villager reward
            running_return_werewolf += reward_pair[1]  # Werewolf reward
            returns_villager.append(running_return_villager)
            returns_werewolf.append(running_return_werewolf)

        returns_villager.reverse()
        returns_werewolf.reverse()

        assert len(agent_result_indices) == len(agent_roles)

        # Assign returns to training examples based on agent role
        prev_idx = 0
        for turn_idx, (idx, role) in enumerate(zip(agent_result_indices, agent_roles)):
            # Choose return based on agent's role
            if role == "werewolf":
                ret = returns_werewolf[turn_idx]
            else:
                ret = returns_villager[turn_idx]

            # Assign return to all training examples from this turn
            for i in range(prev_idx, idx + 1):
                results[i]["rewards"] = torch.tensor([ret], dtype=torch.float32) + results[i]["rewards"] * self.process_reward_coef
            prev_idx = idx + 1

        # Add discounted return to step logs (both returns for analysis)
        for step_log, ret_vill, ret_were, role in zip(step_logs, returns_villager, returns_werewolf, agent_roles):
            step_log["discounted_return_villager"] = ret_vill
            step_log["discounted_return_werewolf"] = ret_were
            step_log["discounted_return"] = ret_were if role == "werewolf" else ret_vill

        # Update rewards and total_reward for logging
        final_total_reward = running_return_villager + running_return_werewolf
        rewards = [ret_were if role == "werewolf" else ret_vill
                   for ret_vill, ret_were, role in zip(returns_villager, returns_werewolf, agent_roles)]
        total_reward = final_total_reward

        # Stats logging
        stats = {}
        if hasattr(env, "get_stats"):
            try:
                stats = env.get_stats()
            except Exception:
                logger.error("No stats are available for this trajectory.")

        # ---- finalize timing aggregates ----
        avg_div = max(1, turns_done)
        timing_vals = [
            # t_qgen_total,                 # 19: total time generating questions
            # t_qgen_decode_total,          # 20: total time decoding questions
            # t_agent_answer_total,         # 21: total time answering (agent) incl. decodes
            # t_teacher_answer_total,       # 22: total time answering (teacher)
            # t_action_build_total,         # 23: total time building action prompt (+tokenize build)
            # t_action_gen_total,           # 24: total time generating action
            # t_action_decode_total,        # 25: total time decoding action
            # t_env_step_total,             # 26: total env.step time
            # t_summary_agent_total,        # 27: total time generating agent summary
            # t_summary_teacher_total,      # 28: total time generating teacher summary
            # t_pack_tensors_total,         # 29: total time packing tensors
            # t_tokenize_total,             # 30: total time spent in tokenizer.apply_chat_template
            # Averages per turn (useful to monitor)
            t_qgen_total / avg_div,               # 31: avg qgen
            t_agent_answer_total / avg_div,       # 32: avg agent answers
            t_teacher_answer_total / avg_div,     # 33: avg teacher answers
            t_action_gen_total / avg_div,         # 34: avg action gen
            # t_env_step_total / avg_div,           # 35: avg env step
            (t_summary_agent_total / avg_div),    # 36: avg agent summary
        ]

        logging_vals = [
            avg_div,                                # 0
            traj_len[0] + traj_len[1],              # 1
            traj_len[0],                            # 2
            traj_len[1],                            # 3
            vill_total,                             # 4
            were_total,                             # 5
            sum(process_rewards) / len(process_rewards),
            stats.get("vill_wins", 0),              # 6
            stats.get("were_wins", 0),              # 7
            stats.get("werewolf_kills", 0),         # 8
            stats.get("werewolf_correct_kills", 0), # 9
            stats.get("villager_correct_votes", 0), # 10
            stats.get("villager_wrong_votes", 0),   # 11
            stats.get("witch_heals", 0),            # 12
            stats.get("witch_correct_heals", 0),    # 13
            stats.get("witch_poisons", 0),          # 14
            stats.get("witch_correct_poisons", 0),  # 15
            stats.get("hunter_shots", 0),           # 16
            stats.get("hunter_correct_shots", 0),   # 17
            env.answer_format_record[0] / env.answer_format_record[1], # 18
        ] + timing_vals                              # 19+ timing slots as documented above

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
                logger.error("Failed to get trajectory from env.")

        return (
            results,
            prompt_strs,
            completions_strs,
            rewards,
            seqlens,
            trajectory,
            qa_logs,
            teacher_logs,
            step_logs,
            episode_info,
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
                            "turns": len(step_logs),
                        },
                        "timestamp": time.time(),
                    }
                    await f.write(json.dumps(_sanitize_for_json(record), indent=2) + "\n")

        return concat_padded_tensors(results)
