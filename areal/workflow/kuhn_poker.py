import asyncio
import json
import os
import random
import re
import time
import uuid
from collections import deque
from typing import Optional

import aiofiles
import aiofiles.os
import aiohttp
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.workflow_api import RolloutWorkflow
from realhf.impl.environment import kuhn_poker_env
from realhf.impl.environment.kuhn_poker_env import KuhnPokerConfig, KuhnPokerEnv
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

logger = logging.getLogger("Kuhn Poker workflow")


def _detect_api_provider(api_key: str) -> str:
    key = api_key.lower()
    if key.startswith("sk-ant") or key.startswith("anthropic"):
        return "claude"
    return "openai"


def _sanitize_for_json(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    return str(value)


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
                    (60.0 - (now - self._req_timestamps[0]))
                    if self._req_timestamps
                    else 0.0,
                )
                await asyncio.sleep(max(next_allowed, 0.01))


class KuhnPokerWorkflow(RolloutWorkflow):
    """Workflow for running Kuhn Poker rollouts with optional opponent/teacher models."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int = 6,
        turn_discount: float = 1.0,
        dump_dir: str | None = None,
        env_kwargs: dict | None = None,
        player_id: int = 0,
        opp_rollout: InferenceEngine | None = None,
        opp_tokenizer: PreTrainedTokenizerFast | None = None,
        teacher_rollout: InferenceEngine | None = None,
        teacher_tokenizer: PreTrainedTokenizerFast | None = None,
        opp_api_key: str | None = None,
        opp_api_model: str | None = None,
        teacher_api_key: str | None = None,
        teacher_api_model: str | None = None,
        teacher_process_reward: bool = False,
        process_reward_coef: float = 0.2,
    ):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if not (0.0 < turn_discount <= 1.0):
            raise ValueError("turn_discount must be in (0, 1].")

        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.dump_dir = dump_dir
        self.env_kwargs = env_kwargs or {}
        self.player_id = player_id
        self.opp_rollout = opp_rollout
        self.opp_tokenizer = opp_tokenizer
        self.teacher_rollout = teacher_rollout
        self.teacher_tokenizer = teacher_tokenizer
        # self.teacher_process_reward = teacher_process_reward
        self.process_reward_coef = process_reward_coef

        self.opp_api_key = (opp_api_key or "").strip()
        self.teacher_api_key = (teacher_api_key or "").strip()
        self.opp_api_provider = (
            _detect_api_provider(self.opp_api_key) if self.opp_api_key else None
        )
        self.teacher_api_provider = (
            _detect_api_provider(self.teacher_api_key) if self.teacher_api_key else None
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
                self.opp_api_model = os.getenv("AREAL_OPP_API_MODEL", default_model)
            self.opp_api_model = (self.opp_api_model or "").strip() or default_model

        if self.teacher_api_key:
            default_model = (
                "gpt-4o-2024-11-20"
                if self.teacher_api_provider == "openai"
                else "claude-3-7-sonnet-20250219"
            )
            if not self.teacher_api_model:
                self.teacher_api_model = os.getenv(
                    "AREAL_TEACHER_API_MODEL", default_model
                )
            self.teacher_api_model = (
                (self.teacher_api_model or "").strip() or default_model
            )

        self._opp_api_session: aiohttp.ClientSession | None = None
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
        backoff = min(self._base_backoff * (2**attempt), self._max_backoff)
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
        headers: dict[str, str],
        json: dict,
    ):
        last_err_text = ""
        for attempt in range(self._max_retries + 1):
            try:
                async with session.request(
                    method, url, headers=headers, json=json
                ) as resp:
                    if 200 <= resp.status < 300:
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
            ) as exc:
                if attempt == self._max_retries:
                    raise RuntimeError(
                        f"Network or timeout error after retries: {exc}"
                    ) from exc
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
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
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

        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "")

    @staticmethod
    def _build_api_response(
        input_ids: list[int], text: str, tokenizer: PreTrainedTokenizerFast
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
        resp: ModelResponse, reward: float = 0.0
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
        }

    def _build_env(self) -> KuhnPokerEnv:
        env_kwargs = dict(self.env_kwargs)
        if "built_in_opponent" not in env_kwargs:
            env_kwargs["built_in_opponent"] = "none"
        if self.opp_rollout or self.opp_api_key:
            env_kwargs["built_in_opponent"] = "none"
        if "opponent_player" not in env_kwargs:
            env_kwargs["opponent_player"] = 1 - self.player_id
        config = KuhnPokerConfig(**env_kwargs)
        return KuhnPokerEnv(config)

    def _build_turn_prompt(
        self,
        env: KuhnPokerEnv,
        observation: str,
        legal_actions: dict[int, str],
        player_id: int,
    ) -> list[dict[str, str]]:
        prefix = env.get_prompt(mode="prefix", player_id=player_id)
        actions_str = ", ".join(legal_actions.values())
        user_prompt = (
            f"{prefix['user']}"
            "CURRENT GAME STATE:\n"
            f"{observation}\n\n"
            "LEGAL ACTIONS:\n"
            f"{actions_str}\n\n"
            "Please keep your answers concise."
        )
        return [
            {"role": "system", "content": prefix["system"]},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_action(self, response: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    async def _generate_action_text(
        self,
        engine: InferenceEngine,
        tokenizer: PreTrainedTokenizerFast,
        prompt_messages: list[dict[str, str]],
        rid: str,
    ) -> tuple[ModelResponse, str]:
        input_ids = list(
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                # enable_thinking=False,
            )
        )
        req = ModelRequest(
            rid=rid,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1, max_new_tokens=2048),
            tokenizer=tokenizer,
        )
        resp = await engine.agenerate(req)
        text = tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
        return resp, text

    async def _generate_action_text_api(
        self,
        prompt_messages: list[dict[str, str]],
        tokenizer: PreTrainedTokenizerFast,
        rid: str,
        api_key: str,
        api_model: str,
        provider: str,
        session_attr: str,
    ) -> tuple[ModelResponse, str]:
        prompt = "\n\n".join([m["content"] for m in prompt_messages])
        text = await self._api_chat_completion(
            prompt,
            self.gconfig,
            api_key,
            api_model,
            provider,
            session_attr,
            rid,
        )
        input_ids = list(
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                # enable_thinking=False,
            )
        )
        resp = self._build_api_response(input_ids, text, tokenizer)
        return resp, text

    async def _run_one_episode(self, engine: InferenceEngine, data: dict, rid: str):
        env = self._build_env()
        seed = data.get("seed")
        if seed is None:
            seed = random.randint(0, 1_000_000)
        initial_observation, execute_results = env.reset(seed)

        observation = initial_observation["observation"]
        legal_actions = initial_observation["legal_actions"]
        done = False
        if execute_results:
            observation = execute_results[-1]["observation"]
            legal_actions = execute_results[-1]["legal_actions"]
            done = execute_results[-1]["done"]

        response_entries: list[tuple[ModelResponse, int, int, str, str, int]] = []
        prompt_strs: list[str] = []
        completions_strs: list[str] = []
        seqlens: list[int] = []
        step_logs: list[dict] = []
        process_rewards: list[float] = []
        trajectory_rewards: list[list[float]] = []

        format_reward = [0.05, 0.05]
        length_reward = [0.0, 0.0]

        turns = 0
        while not done and turns < self.max_turns:
            turns += 1
            current_player = env.current_player

            if current_player == self.player_id:
                prompt_messages = self._build_turn_prompt(
                    env, observation, legal_actions, self.player_id
                )
                resp, completion = await self._generate_action_text(
                    engine, self.tokenizer, prompt_messages, f"{rid}-agent-{turns}"
                )
                action_text = self._extract_action(completion)
                try:
                    action = env._string_to_action(action_text)
                except ValueError:
                    execute_results = env.get_losing_state(player_id=self.player_id)
                    done = True
                    step_index = len(trajectory_rewards)
                    trajectory_rewards.extend(
                        [result["rewards"] for result in execute_results]
                    )
                    format_reward[self.player_id] = -10.0 # Large penalty for format error
                    step_logs.append(
                        {
                            "player": self.player_id,
                            "action": action_text,
                            "invalid_action": True,
                        }
                    )
                else:
                    execute_results = env.step(action)
                    done = execute_results[-1]["done"]
                    step_index = len(trajectory_rewards)
                    trajectory_rewards.extend(
                        [result["rewards"] for result in execute_results]
                    )
                    step_logs.append(
                        {
                            "player": self.player_id,
                            "action": env._action_to_string(self.player_id, action),
                            "invalid_action": False,
                        }
                    )

                prompt_text = self.tokenizer.decode(resp.input_tokens)
                prompt_strs.append(prompt_text)
                completions_strs.append(completion)
                seqlens.append(len(resp.input_tokens) + len(resp.output_tokens))
                length_reward[self.player_id] = 0.5 * max(0, 1 - (len(resp.output_tokens) - 11) / (2048 - 11))
                response_entries.append(
                    (
                        resp,
                        self.player_id,
                        step_index,
                        prompt_text,
                        completion,
                        len(resp.input_tokens) + len(resp.output_tokens),
                    )
                )

                if not done:
                    observation = execute_results[-1]["observation"]
                    legal_actions = execute_results[-1]["legal_actions"]

            else:
                prompt_messages = self._build_turn_prompt(
                    env, observation, legal_actions, current_player
                )
                invalid_action = False
                if self.opp_api_key:
                    opp_resp, opp_completion = await self._generate_action_text_api(
                        prompt_messages,
                        self.opp_tokenizer or self.tokenizer,
                        f"{rid}-opp-{turns}",
                        self.opp_api_key,
                        self.opp_api_model,
                        self.opp_api_provider or "openai",
                        "_opp_api_session",
                    )
                elif self.opp_rollout:
                    opp_resp, opp_completion = await self._generate_action_text(
                        self.opp_rollout,
                        self.opp_tokenizer or self.tokenizer,
                        prompt_messages,
                        f"{rid}-opp-{turns}",
                    )
                else:
                    opp_resp, opp_completion = await self._generate_action_text(
                        engine,
                        self.tokenizer,
                        prompt_messages,
                        f"{rid}-selfplay-{turns}",
                    )

                opp_action_text = self._extract_action(opp_completion)
                try:
                    opp_action = env._string_to_action(opp_action_text)
                except ValueError:
                    execute_results = env.get_losing_state(player_id=current_player)
                    done = True
                    step_index = len(trajectory_rewards)
                    trajectory_rewards.extend(
                        [result["rewards"] for result in execute_results]
                    )
                    invalid_action = True
                    format_reward[1 - self.player_id] = -10.0 # Large penalty for format error
                else:
                    execute_results = env.step(opp_action)
                    done = execute_results[-1]["done"]
                    step_index = len(trajectory_rewards)
                    trajectory_rewards.extend(
                        [result["rewards"] for result in execute_results]
                    )

                prompt_text = self.tokenizer.decode(opp_resp.input_tokens)
                prompt_strs.append(prompt_text)
                completions_strs.append(opp_completion)
                seqlens.append(len(opp_resp.input_tokens) + len(opp_resp.output_tokens))
                length_reward[1 - self.player_id] = 0.5 * max(0, 1 - (len(opp_resp.output_tokens) - 11) / (2048 - 11))
                response_entries.append(
                    (
                        opp_resp,
                        current_player,
                        step_index,
                        prompt_text,
                        opp_completion,
                        len(opp_resp.input_tokens) + len(opp_resp.output_tokens),
                    )
                )
                step_logs.append(
                    {
                        "player": current_player,
                        "action": opp_action_text,
                        "invalid_action": invalid_action,
                    }
                )

                if not done:
                    observation = execute_results[-1]["observation"]
                    legal_actions = execute_results[-1]["legal_actions"]

        player_returns: dict[int, float] = {0: 0.0, 1: 0.0}
        if trajectory_rewards:
            cumulative = {0: 0.0, 1: 0.0}
            per_step_returns: list[dict[int, float]] = []
            for rewards in reversed(trajectory_rewards):
                # Calculate the returns at each step. Safety check in case of mismatch
                if isinstance(rewards, (list, tuple)) and len(rewards) >= 2:
                    cumulative[0] += float(rewards[0])
                    cumulative[1] += float(rewards[1])
                else:
                    cumulative[0] += float(rewards)
                    cumulative[1] += float(rewards)
                per_step_returns.append({0: cumulative[0], 1: cumulative[1]})
            per_step_returns.reverse()
            player_returns = dict(per_step_returns[0])
        else:
            per_step_returns = []

        stats_tracker.get("rollout").scalar(
            reward=player_returns[self.player_id], 
            reward_opp=player_returns[1-self.player_id], 
            format_reward=format_reward[self.player_id],
            length_reward=length_reward[self.player_id],
            num_turns=turns
        )
        logger.info(f"Rollout reward: {player_returns[self.player_id]} finished for player {self.player_id} with {turns} steps.")

        results = []
        for resp, player_id, step_index, _, _, _ in response_entries:
            # In agentic tasks, the reward shall in essence be the retrun at each step, not the step-wise reward.
            if player_id != self.player_id:
                continue
            if per_step_returns and step_index < len(per_step_returns):
                step_return = float(per_step_returns[step_index][player_id])
            else:
                step_return = player_returns[player_id]
            step_return += (format_reward[player_id] + length_reward[player_id])
            results.append(self._response_to_tensordict(resp, reward=step_return))

        return (
            results,
            prompt_strs,
            completions_strs,
            [player_returns[0], player_returns[1]],
            seqlens,
            step_logs,
        )

    async def arun_episode(
        self, engine: InferenceEngine, data: dict
    ) -> dict[str, torch.Tensor]:
        rid = uuid.uuid4().hex
        episodes = await asyncio.gather(
            *[
                self._run_one_episode(engine, data, rid)
                for _ in range(self.gconfig.n_samples)
            ]
        )

        results: list[dict[str, torch.Tensor]] = []
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
                    prompt_strs,
                    completion_strs,
                    reward_list,
                    seqlens,
                    step_logs,
                ) in enumerate(episodes):
                    record = {
                        "query_id": qid,
                        "episode_index": episode_idx,
                        "prompts": prompt_strs,
                        "completions": completion_strs,
                        "sequence_lengths": seqlens,
                        "returns": reward_list,
                        "steps": step_logs,
                        "timestamp": time.time(),
                        "model_version": version,
                    }
                    await f.write(json.dumps(_sanitize_for_json(record), indent=2) + "\n")

        return concat_padded_tensors(results)