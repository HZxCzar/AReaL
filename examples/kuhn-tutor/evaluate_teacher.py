from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game_tutor_common import (
    TeacherEvaluator,
    apply_cli_overrides,
    load_config,
    parse_args,
    strip_answer,
    write_report,
)


try:
    from kuhn_poker_env import KuhnPokerConfig, KuhnPokerEnv  # noqa: E402
except Exception:  # pragma: no cover - fallback for lightweight local envs.
    KuhnPokerConfig = None
    KuhnPokerEnv = None


class KuhnTutorEnv:
    def __init__(self, **env_kwargs):
        self.player_id = int(env_kwargs.get("player_id", 0))
        self.last_returns = [0.0, 0.0]
        if KuhnPokerConfig is None or KuhnPokerEnv is None:
            self._fallback = SimpleKuhnTutorEnv(**env_kwargs)
            return
        self._fallback = None
        env_kwargs.pop("player_id", None)
        try:
            self.env = KuhnPokerEnv(KuhnPokerConfig(**env_kwargs))
        except Exception:
            self._fallback = SimpleKuhnTutorEnv(player_id=self.player_id, **env_kwargs)
            return

    async def sreset(self, seed=None):
        if self._fallback is not None:
            return await self._fallback.sreset(seed=seed)
        initial, executed = self.env.reset(seed=seed)
        obs = initial["observation"]
        legal = initial["legal_actions"]
        done = False
        if executed:
            obs = executed[-1]["observation"]
            legal = executed[-1]["legal_actions"]
            done = executed[-1]["done"]
            self.last_returns = list(executed[-1]["rewards"])
        return obs, self._guide(legal), {"teacher_observation": obs, "done": done}

    async def step(self, action):
        if self._fallback is not None:
            return await self._fallback.step(action)
        text = action[1][0] if isinstance(action, tuple) else str(action)
        parsed = strip_answer(text)
        try:
            env_action = self.env._string_to_action(parsed)
            executed = self.env.step(env_action)
        except Exception:
            executed = self.env.get_losing_state(player_id=self.player_id)
        last = executed[-1]
        self.last_returns = list(last["rewards"])
        obs = last.get("observation") or "Game over."
        legal = last.get("legal_actions") or {}
        return obs, self._guide(legal), self.last_returns, bool(last["done"]), False, {
            "event": f"Action: {parsed}",
            "teacher_observation": obs,
        }

    def _guide(self, legal_actions: dict[int, str]) -> str:
        actions = ", ".join(legal_actions.values()) or "<PASS>, <BET>"
        return (
            f"Legal actions: {actions}\n"
            "Respond exactly as <answer><PASS></answer> or <answer><BET></answer>."
        )


class SimpleKuhnTutorEnv:
    """Small Kuhn fallback used only when OpenSpiel/numpy are unavailable."""

    def __init__(self, **env_kwargs):
        self.player_id = int(env_kwargs.get("player_id", 0))
        self.last_returns = [0.0, 0.0]
        self.cards = ["J", "Q", "K"]
        self.history: list[str] = []
        self.private = ["J", "Q"]
        self.current_player = 0

    async def sreset(self, seed=None):
        import random

        rng = random.Random(seed)
        self.private = rng.sample(self.cards, 2)
        self.history = []
        self.current_player = 0
        self.last_returns = [0.0, 0.0]
        return self._obs(), self._guide(), {"teacher_observation": self._teacher_obs()}

    async def step(self, action):
        text = action[1][0] if isinstance(action, tuple) else str(action)
        parsed = strip_answer(text).upper()
        if "BET" in parsed:
            act = "BET"
        elif "PASS" in parsed:
            act = "PASS"
        else:
            self.last_returns = [-10.0, 0.0] if self.current_player == 0 else [0.0, -10.0]
            return "Game over.", self._guide(), self.last_returns, True, False, {"event": "invalid action"}
        self.history.append(act)
        done = self._is_terminal()
        if done:
            self.last_returns = self._returns()
            return "Game over.", self._guide(), self.last_returns, True, False, {"event": act, "teacher_observation": self._teacher_obs()}
        self.current_player = 1 - self.current_player
        return self._obs(), self._guide(), [0.0, 0.0], False, False, {"event": act, "teacher_observation": self._teacher_obs()}

    def _guide(self):
        return "Legal actions: <PASS>, <BET>\nRespond exactly as <answer><PASS></answer> or <answer><BET></answer>."

    def _obs(self):
        return f"You are player_{self.current_player}. Your private card is {self.private[self.current_player]}. History: {self.history or 'none'}."

    def _teacher_obs(self):
        return f"Privileged Kuhn state: player_0={self.private[0]}, player_1={self.private[1]}, history={self.history}."

    def _is_terminal(self):
        return self.history in (["PASS", "PASS"], ["BET", "PASS"], ["PASS", "BET"], ["BET", "BET"])

    def _returns(self):
        rank = {"J": 0, "Q": 1, "K": 2}
        if self.history == ["BET", "PASS"]:
            return [1.0, -1.0]
        if self.history == ["PASS", "BET"]:
            return [-1.0, 1.0]
        pot = 2.0 if self.history == ["PASS", "PASS"] else 4.0
        winner = 0 if rank[self.private[0]] > rank[self.private[1]] else 1
        return [pot / 2, -pot / 2] if winner == 0 else [-pot / 2, pot / 2]


def build_env(env_kwargs: dict[str, Any]) -> KuhnTutorEnv:
    return KuhnTutorEnv(**env_kwargs)


def build_prompt(env: KuhnTutorEnv, obs: str, guide: str, teacher_advice: str, memory: str) -> str:
    return (
        "You are playing Kuhn Poker.\n\n"
        f"Current state:\n{obs}\n\n"
        f"{memory}\n\n"
        f"Teacher guidance:\n{teacher_advice or 'No teacher guidance. Play from the current state.'}\n\n"
        f"{guide}"
    ).strip()


def score(env: KuhnTutorEnv, _last_reward: Any) -> float:
    return float(env.last_returns[env.player_id])


async def main() -> None:
    args = parse_args(str(Path(__file__).with_name("teacher_eval_config.yaml")))
    config = apply_cli_overrides(load_config(args.config), args)
    evaluator = TeacherEvaluator(
        env_factory=build_env,
        prompt_builder=build_prompt,
        score_getter=score,
        config=config,
    )
    report = await evaluator.run_pair()
    output_path = config.get("output_path", "examples/kuhn-tutor/outputs/teacher_eval.json")
    write_report(report, output_path)
    print(f"Wrote teacher evaluation report to {output_path}")
    print(f"Average score improvement: {report['improvement']['avg_score']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
