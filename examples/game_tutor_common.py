from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import request


try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value.strip())
    return root


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none", "~"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:30000/v1"
    api_key: str = "EMPTY"
    model: str = "default"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout: int = 120


class OpenAIChatClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> str:
        return await asyncio.to_thread(self._complete_blocking, messages, max_tokens)

    def _complete_blocking(self, messages: list[dict[str, str]], max_tokens: int | None) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": int(max_tokens or self.cfg.max_tokens),
        }
        raw = json.dumps(payload).encode("utf-8")
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        req = request.Request(
            url,
            data=raw,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.cfg.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""


class ICLMemory:
    def __init__(
        self,
        *,
        enabled: bool,
        max_turns: int = 5,
        compress_every: int = 5,
        compressor: OpenAIChatClient | None = None,
    ):
        self.enabled = enabled
        self.max_turns = max(1, int(max_turns))
        self.compress_every = max(1, int(compress_every))
        self.compressor = compressor
        self.summary = ""
        self.recent: list[str] = []
        self.turns = 0

    def render(self) -> str:
        if not self.enabled:
            return ""
        parts = []
        if self.summary:
            parts.append(f"Compressed memory:\n{self.summary}")
        if self.recent:
            parts.append("Recent turns:\n" + "\n".join(self.recent[-self.max_turns :]))
        return "\n\n".join(parts)

    async def add(self, event: str) -> None:
        if not self.enabled:
            return
        self.turns += 1
        self.recent.append(event.strip())
        if len(self.recent) > self.max_turns:
            self.recent = self.recent[-self.max_turns :]
        if self.turns % self.compress_every == 0:
            await self.compress()

    async def compress(self) -> None:
        if not self.enabled or not self.recent:
            return
        source = "\n".join(self.recent)
        if self.compressor is None:
            self.summary = "\n".join((self.summary + "\n" + source).splitlines()[-24:]).strip()
            self.recent = []
            return
        prompt = (
            "Compress this game memory into concise tactical state for future decisions. "
            "Preserve hidden-information inferences, commitments, and score-relevant facts.\n\n"
            f"Previous compressed memory:\n{self.summary or 'None'}\n\nRecent turns:\n{source}"
        )
        self.summary = (
            await self.compressor.complete([{"role": "user", "content": prompt}], max_tokens=512)
        ).strip()
        self.recent = []


def strip_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else (text or "").strip()


def normalize_messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class EvalResult:
    total_score: float
    turns: int
    episodes: int
    invalid_actions: int
    traces: list[dict[str, Any]]

    @property
    def avg_score(self) -> float:
        return self.total_score / max(1, self.episodes)


class TeacherEvaluator:
    def __init__(
        self,
        *,
        env_factory: Callable[[dict[str, Any]], Any],
        prompt_builder: Callable[[Any, str, str, str, str], str],
        score_getter: Callable[[Any, Any], float],
        config: dict[str, Any],
    ):
        self.env_factory = env_factory
        self.prompt_builder = prompt_builder
        self.score_getter = score_getter
        self.config = config
        self.teacher = OpenAIChatClient(LLMConfig(**config["teacher_model"]))
        self.student = OpenAIChatClient(LLMConfig(**config["student_model"]))
        self.num_turns = int(config.get("num_turns", 300))
        self.max_episode_turns = int(config.get("max_episode_turns", 80))
        self.seed = int(config.get("seed", 1))
        self.icl_enabled = bool(config.get("icl_simulation", {}).get("enabled", False))
        self.icl_turns = int(config.get("icl_simulation", {}).get("turns", 5))
        self.compress_every = int(config.get("icl_simulation", {}).get("compress_every", self.icl_turns))

    async def run_pair(self) -> dict[str, Any]:
        baseline = await self.run_condition(use_teacher=False)
        teacher = await self.run_condition(use_teacher=True)
        return {
            "config": self.config,
            "baseline": baseline.__dict__ | {"avg_score": baseline.avg_score},
            "teacher": teacher.__dict__ | {"avg_score": teacher.avg_score},
            "improvement": {
                "total_score": teacher.total_score - baseline.total_score,
                "avg_score": teacher.avg_score - baseline.avg_score,
                "turns": teacher.turns - baseline.turns,
                "invalid_actions": teacher.invalid_actions - baseline.invalid_actions,
            },
        }

    async def run_condition(self, *, use_teacher: bool) -> EvalResult:
        turns_left = self.num_turns
        episode_idx = 0
        total_score = 0.0
        invalid_actions = 0
        traces: list[dict[str, Any]] = []

        while turns_left > 0:
            env = self.env_factory(dict(self.config.get("env_kwargs") or {}))
            episode_seed = self.seed + episode_idx
            obs, guide, info = await maybe_await(env.sreset(seed=episode_seed))
            memory = ICLMemory(
                enabled=self.icl_enabled,
                max_turns=self.icl_turns,
                compress_every=self.compress_every,
                compressor=self.teacher if use_teacher else self.student,
            )
            done = False
            episode_turns = 0
            episode_trace = []
            last_step: Any = None
            while not done and turns_left > 0 and episode_turns < self.max_episode_turns:
                memory_text = memory.render()
                teacher_advice = ""
                teacher_obs = info.get("teacher_observation", obs) if isinstance(info, dict) else obs
                if use_teacher:
                    teacher_policy = str(self.config.get("teacher_guidance_policy") or "").strip()
                    teacher_prompt = (
                        f"{teacher_obs}\n\nCurrent public observation:\n{obs}\n\nGuide:\n{guide}\n\n"
                        f"{memory_text}\n\nGive concise tutoring guidance for the acting player. "
                        "Do not choose the action unless necessary; explain the key consideration."
                    )
                    if teacher_policy:
                        teacher_prompt += f"\n\nAdditional tutoring policy:\n{teacher_policy}"
                    teacher_advice = await self.teacher.complete(
                        normalize_messages(
                            "You are a privileged game tutor evaluating how useful your guidance is.",
                            teacher_prompt,
                        )
                    )

                action_prompt = self.prompt_builder(env, obs, guide, teacher_advice, memory_text)
                action_text = await self.student.complete(
                    normalize_messages(
                        "You are the acting game player. Follow the required answer format exactly.",
                        action_prompt,
                    )
                )
                next_obs, next_guide, reward, done, _, info = await maybe_await(
                    env.step(("", [action_text]))
                )
                if not strip_answer(action_text):
                    invalid_actions += 1
                event = info.get("event", "") if isinstance(info, dict) else ""
                await memory.add(
                    f"Turn {episode_turns + 1}: advice={teacher_advice[:400]!r}; "
                    f"action={strip_answer(action_text)!r}; reward={reward}; event={event!r}"
                )
                episode_trace.append(
                    {
                        "turn": episode_turns + 1,
                        "teacher_advice": teacher_advice,
                        "action": action_text,
                        "parsed_action": strip_answer(action_text),
                        "reward": reward,
                        "event": event,
                    }
                )
                obs, guide = next_obs, next_guide
                episode_turns += 1
                turns_left -= 1
                last_step = reward

            score = float(self.score_getter(env, last_step))
            total_score += score
            traces.append(
                {
                    "episode": episode_idx,
                    "seed": episode_seed,
                    "condition": "teacher" if use_teacher else "baseline",
                    "turns": episode_turns,
                    "score": score,
                    "steps": episode_trace,
                }
            )
            episode_idx += 1

        return EvalResult(
            total_score=total_score,
            turns=self.num_turns - turns_left,
            episodes=episode_idx,
            invalid_actions=invalid_actions,
            traces=traces,
        )


async def maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


def write_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")


def parse_args(default_config: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--num-turns", type=int, default=None)
    parser.add_argument("--icl-turns", type=int, default=None)
    parser.add_argument("--enable-icl", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if args.num_turns is not None:
        updates["num_turns"] = args.num_turns
    if args.output is not None:
        updates["output_path"] = args.output
    icl_updates: dict[str, Any] = {}
    if args.icl_turns is not None:
        icl_updates["turns"] = args.icl_turns
    if args.enable_icl:
        icl_updates["enabled"] = True
    if icl_updates:
        updates["icl_simulation"] = deep_update(config.get("icl_simulation") or {}, icl_updates)
    return deep_update(config, updates)


def ensure_repo_on_path(file: str) -> None:
    examples_dir = Path(file).resolve().parents[1]
    repo_dir = examples_dir.parent
    for path in (examples_dir, repo_dir):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
