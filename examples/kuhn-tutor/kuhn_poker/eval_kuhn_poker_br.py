import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import pyspiel
from open_spiel.python import policy as openspiel_policy
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from areal.environment.kuhn_poker_env import KuhnPokerConfig, KuhnPokerEnv


# -----------------------
# Helpers: prompt + parsing
# -----------------------
def _build_turn_prompt(
    env: KuhnPokerEnv,
    observation: str,
    legal_actions: Dict[int, str],
    player_id: int,
) -> List[Dict[str, str]]:
    prefix = env.get_prompt(mode="prefix", player_id=player_id)
    actions_str = ", ".join(legal_actions.values())
    user_prompt = (
        f"{prefix['user']}"
        "CURRENT GAME STATE:\n"
        f"{observation}\n\n"
        "LEGAL ACTIONS:\n"
        f"{actions_str}\n\n"
        "Reply with your chosen action wrapped in <answer>...</answer>.\n"
        "Please keep your answers concise."
    )
    return [
        {"role": "system", "content": prefix["system"]},
        {"role": "user", "content": user_prompt},
    ]


def _extract_action(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def _compute_bets(history: List[int], first_to_act: int) -> List[int]:
    """
    OpenSpiel Kuhn history typically begins with 2 chance actions (dealing cards),
    followed by player actions (0/1 for pass/bet).
    This function reconstructs bets assuming 'first_to_act' is the player who
    takes the first non-chance action.
    """
    bets = [1, 1]
    # player actions start after 2 chance outcomes
    for t, action in enumerate(history[2:]):
        pid = (first_to_act + t) % 2
        # Kuhn: action is usually 0 (check/pass) or 1 (bet/call)
        bets[pid] += int(action)
    return bets


def _sample_one_action_text(
    model,
    tokenizer,
    prompt_messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: Optional[torch.device],
) -> str:
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = out[:, input_len:]
    txt = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return txt


def _normalize_action_probs(action_probs: Dict[int, float], legal_actions: List[int]) -> Dict[int, float]:
    filtered = {a: float(action_probs.get(a, 0.0)) for a in legal_actions}
    total = sum(filtered.values())
    if total <= 0:
        uniform = 1.0 / len(legal_actions)
        return {a: uniform for a in legal_actions}
    return {a: p / total for a, p in filtered.items()}


def _action_vector_to_probs(action_prob_vector: Sequence[float], legal_actions: List[int]) -> Dict[int, float]:
    if not legal_actions:
        return {}
    max_action = max(legal_actions)
    if len(action_prob_vector) <= max_action:
        raise ValueError(
            f"Action prob vector length {len(action_prob_vector)} is too short for action id {max_action}."
        )
    probs = {a: float(action_prob_vector[a]) for a in legal_actions if a < len(action_prob_vector)}
    return _normalize_action_probs(probs, legal_actions)


# -----------------------
# Policy cache: empirical opponent strategy per info-state
# -----------------------
@dataclass
class EmpiricalPolicyConfig:
    n_samples: int = 100
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9


class EmpiricalOpponentPolicyCache:
    """
    Computes and caches P(a | I_opponent) from LLM sampling,
    where key is opponent information_state_string(opponent_id).
    """

    def __init__(
        self,
        env: KuhnPokerEnv,
        model,
        tokenizer,
        opponent_id: int,
        cfg: EmpiricalPolicyConfig,
        device: Optional[torch.device],
        allow_inference: bool = True,
    ):
        self.env = env
        self.model = model
        self.tokenizer = tokenizer
        self.opponent_id = opponent_id
        self.cfg = cfg
        self.device = device
        self.allow_inference = allow_inference
        self.inference_performed = False
        self._cache: Dict[str, Dict[int, float]] = {}

    def load_cache(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        policy_blob = data.get("policy", data)
        cache: Dict[str, Dict[int, float]] = {}
        for info_state, probs in policy_blob.items():
            cache[info_state] = {int(a): float(p) for a, p in probs.items()}
        self._cache = cache

    def save_cache(self, path: Path) -> None:
        payload = {
            "opponent_id": self.opponent_id,
            "config": {
                "n_samples": self.cfg.n_samples,
                "max_new_tokens": self.cfg.max_new_tokens,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
            },
            "policy": {k: {str(a): p for a, p in v.items()} for k, v in self._cache.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def action_probabilities(self, state) -> Dict[int, float]:
        key = state.information_state_string(self.opponent_id)
        if key in self._cache:
            return self._cache[key]
        if not self.allow_inference:
            self._cache[key] = {}
            return {}

        # Ensure we're sampling at an opponent decision node.
        # If not, still return something safe (shouldn't happen if called correctly).
        legal_actions = state.legal_actions(self.opponent_id) if hasattr(state, "legal_actions") else state.legal_actions()
        if not legal_actions:
            self._cache[key] = {}
            return {}

        # Build observation/prompt *from opponent perspective*.
        # We reconstruct env internal state from the pyspiel state.
        # Important: set env.state and env.bets, then render.
        self.env.state = state.clone()

        # Determine who acts first in this game by looking at the first non-chance node.
        first_to_act = _first_non_chance_player(self.env._env)
        self.env.bets = _compute_bets(state.history(), first_to_act=first_to_act)

        obs = self.env.render()  # assumes render is consistent with env.state
        legal_action_dict = {a: self.env._action_to_string(self.opponent_id, a) for a in legal_actions}
        prompt_messages = _build_turn_prompt(self.env, obs, legal_action_dict, self.opponent_id)

        counts = {a: 0 for a in legal_actions}

        # Sample sequentially (lower VRAM than big num_return_sequences)
        for _ in range(self.cfg.n_samples):
            raw = _sample_one_action_text(
                self.model,
                self.tokenizer,
                prompt_messages,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                device=self.device,
            )
            sampled = _extract_action(raw)
            try:
                aid = self.env._string_to_action(sampled)
                if aid in counts:
                    counts[aid] += 1
            except Exception:
                pass

        total = sum(counts.values())
        if total == 0:
            probs = {a: 1.0 / len(counts) for a in counts}
        else:
            probs = {a: counts[a] / total for a in counts}

        self._cache[key] = probs
        self.inference_performed = True
        return probs


class MixedPolicy(openspiel_policy.Policy):
    def __init__(
        self,
        game,
        br_player: int,
        model_player: int,
        opp_policy: EmpiricalOpponentPolicyCache,
        br_policy_table: Dict[str, Dict[int, float]],
    ):
        super().__init__(game, [br_player, model_player])
        self.game = game
        self.br_player = br_player
        self.model_player = model_player
        self.opp_policy = opp_policy
        self.br_policy_table = br_policy_table

    def action_probabilities(self, state, player_id: Optional[int] = None) -> Dict[int, float]:
        if state.is_chance_node():
            return {a: float(p) for a, p in state.chance_outcomes()}

        cur = state.current_player() if player_id is None else player_id
        if cur == self.br_player:
            info_state = state.information_state_string(self.br_player)
            legal_actions = state.legal_actions(self.br_player)
            if info_state in self.br_policy_table:
                return _normalize_action_probs(self.br_policy_table[info_state], legal_actions)
            return _normalize_action_probs({}, legal_actions)

        probs = self.opp_policy.action_probabilities(state)
        legal_actions = state.legal_actions(cur)
        if probs:
            return _normalize_action_probs(probs, legal_actions)
        return _normalize_action_probs({}, legal_actions)


def _compute_exploitability(game, policy: openspiel_policy.Policy) -> float:
    # if hasattr(pyspiel, "exploitability"):
    #     return float(pyspiel.exploitability(game, policy))
    from open_spiel.python.algorithms import exploitability as exp

    return float(exp.exploitability(game, policy))


def action_vector_exploitability(
    game,
    state,
    br_player: int,
    model_player: int,
    opp_policy: EmpiricalOpponentPolicyCache,
    action_prob_vector: Sequence[float],
    base_br_policy: Dict[str, Dict[int, float]],
) -> float:
    info_key = state.information_state_string(br_player)
    legal_actions = state.legal_actions(br_player)
    br_policy_table = dict(base_br_policy)
    br_policy_table[info_key] = _action_vector_to_probs(action_prob_vector, legal_actions)
    policy = MixedPolicy(
        game=game,
        br_player=br_player,
        model_player=model_player,
        opp_policy=opp_policy,
        br_policy_table=br_policy_table,
    )
    return _compute_exploitability(game, policy)


# -----------------------
# Best response recursion (uses cached opponent policy)
# -----------------------
def br_value(state, br_player: int, opp_policy: EmpiricalOpponentPolicyCache) -> float:
    if state.is_terminal():
        return float(state.returns()[br_player])

    if state.is_chance_node():
        val = 0.0
        for a, p in state.chance_outcomes():
            nxt = state.clone()
            nxt.apply_action(a)
            val += float(p) * br_value(nxt, br_player, opp_policy)
        return val

    cur = state.current_player()
    if cur == br_player:
        best = -1e30
        for a in state.legal_actions():
            nxt = state.clone()
            nxt.apply_action(a)
            best = max(best, br_value(nxt, br_player, opp_policy))
        return best

    # opponent node: fixed empirical policy table (cached)
    probs = opp_policy.action_probabilities(state)
    if not probs:
        # fallback uniform over legal actions
        legal = state.legal_actions()
        p = 1.0 / len(legal)
        val = 0.0
        for a in legal:
            nxt = state.clone()
            nxt.apply_action(a)
            val += p * br_value(nxt, br_player, opp_policy)
        return val

    val = 0.0
    for a, p in probs.items():
        nxt = state.clone()
        nxt.apply_action(a)
        val += float(p) * br_value(nxt, br_player, opp_policy)
    return val


def br_best_action_at_state(state, br_player: int, opp_policy: EmpiricalOpponentPolicyCache) -> Tuple[int, float]:
    """
    Returns (best_action, best_value) for br_player at this state.
    """
    assert state.current_player() == br_player, (
        f"State is not a decision for br_player={br_player}. "
        f"current_player={state.current_player()}"
    )
    best_a = None
    best_v = -1e30
    for a in state.legal_actions():
        nxt = state.clone()
        nxt.apply_action(a)
        v = br_value(nxt, br_player, opp_policy)
        if v > best_v:
            best_v = v
            best_a = a
    return int(best_a), float(best_v)


# -----------------------
# Utility: enumerate info states for a given player
# -----------------------
def _collect_player_states(game, player_id: int) -> List:
    seen: Dict[str, object] = {}

    def _dfs(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                nxt = state.clone()
                nxt.apply_action(action)
                _dfs(nxt)
            return

        if state.current_player() == player_id:
            key = state.information_state_string(player_id)
            if key not in seen:
                seen[key] = state.clone()

        for action in state.legal_actions():
            nxt = state.clone()
            nxt.apply_action(action)
            _dfs(nxt)

    _dfs(game.new_initial_state())
    return [seen[k] for k in sorted(seen.keys())]


def _first_non_chance_player(game) -> int:
    s = game.new_initial_state()
    # advance chance nodes until first decision
    while s.is_chance_node():
        # apply the first chance action (structure doesn't matter for "who moves first")
        a, _ = s.chance_outcomes()[0]
        s.apply_action(a)
    return int(s.current_player())


def _uniform_policy_for_states(states: List, player_id: int) -> Dict[str, Dict[int, float]]:
    table: Dict[str, Dict[int, float]] = {}
    for state in states:
        if state.current_player() != player_id:
            continue
        legal = state.legal_actions(player_id)
        table[state.information_state_string(player_id)] = _normalize_action_probs({}, legal)
    return table


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute best-response actions in Kuhn Poker against an empirical LLM opponent policy."
    )
    parser.add_argument("--br-player", type=int, choices=[0, 1], required=True,
                        help="Which player to compute best response for.")
    parser.add_argument("--model-player", type=int, choices=[0, 1], default=1,
                        help="Which player the local model controls (opponent). Default: 1.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--model-policy-path",
        type=str,
        default=None,
        help="Optional path to a cached opponent policy JSON. If present, inference is skipped.",
    )

    parser.add_argument("--policy-samples", type=int, default=100,
                        help="Samples per opponent information state to estimate action frequencies.")
    parser.add_argument("--policy-max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Model dtype to reduce VRAM.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.br_player == args.model_player:
        raise ValueError("--br-player and --model-player must be different (model is the opponent).")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_policy_path = Path(args.model_policy_path) if args.model_policy_path else outdir / "model_policy.json"
    use_cached_policy = args.model_policy_path is not None and model_policy_path.exists()

    # device + dtype
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = None
    model = None
    if not use_cached_policy:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load with lower VRAM
        # Note: keep it simple and predictable: single-device load + dtype.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype if use_cuda else None,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

    env = KuhnPokerEnv(KuhnPokerConfig(built_in_opponent="none"))
    game = env._env

    # Configure empirical opponent policy (cached)
    pol_cfg = EmpiricalPolicyConfig(
        n_samples=args.policy_samples,
        max_new_tokens=args.policy_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    opp_policy = EmpiricalOpponentPolicyCache(
        env=env,
        model=model,
        tokenizer=tokenizer,
        opponent_id=args.model_player,
        cfg=pol_cfg,
        device=device,
        allow_inference=not use_cached_policy,
    )
    if model_policy_path.exists():
        print(f"Loading opponent policy cache from {model_policy_path}")
        opp_policy.load_cache(model_policy_path)

    # Collect BR-player information states
    br_states = _collect_player_states(game, args.br_player)
    print(f"Collected {len(br_states)} information states for br_player={args.br_player}.")
    uniform_br_policy = _uniform_policy_for_states(br_states, args.br_player)

    results = []
    for idx, s in enumerate(br_states):
        print(f"Collecting data for state {idx}...", flush=True)
        # Ensure this is indeed br_player node
        if s.current_player() != args.br_player:
            continue

        best_a, best_v = br_best_action_at_state(s, args.br_player, opp_policy)
        # convert to string via env mapping
        env.state = s.clone()
        first_to_act = _first_non_chance_player(game)
        env.bets = _compute_bets(s.history(), first_to_act=first_to_act)
        best_a_str = env._action_to_string(args.br_player, best_a)

        info_key = s.information_state_string(args.br_player)
        legal_actions = s.legal_actions(args.br_player)
        best_action_vector = [0.0] * (max(legal_actions) + 1)
        best_action_vector[best_a] = 1.0
        best_action_exploitability = action_vector_exploitability(
            game=game,
            state=s,
            br_player=args.br_player,
            model_player=args.model_player,
            opp_policy=opp_policy,
            action_prob_vector=best_action_vector,
            base_br_policy=uniform_br_policy,
        )

        results.append(
            {
                "idx": idx,
                "br_player": args.br_player,
                "model_player": args.model_player,
                "info_state": info_key,
                "best_action_id": int(best_a),
                "best_action_str": best_a_str,
                "best_value": float(best_v),
                "best_action_exploitability": float(best_action_exploitability),
            }
        )
        print(
            f"[{idx:02d}] best_action={best_a_str}  "
            f"value={best_v:.6f}  exploitability={best_action_exploitability:.6f}",
            flush=True,
        )

    # Save JSON
    with open(outdir / "br_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save a small summary too
    cnt = Counter([r["best_action_str"] for r in results])
    with open(outdir / "br_summary.txt", "w", encoding="utf-8") as f:
        for k, v in cnt.most_common():
            f.write(f"{k}\t{v}\n")

    if opp_policy.inference_performed or not model_policy_path.exists():
        opp_policy.save_cache(model_policy_path)
        print(f"Saved opponent policy cache: {model_policy_path}")

    print(f"Saved: {outdir / 'br_results.json'}")
    print(f"Saved: {outdir / 'br_summary.txt'}")


if __name__ == "__main__":
    main()


"""
Example:
python examples/kuhn_poker/eval_kuhn_poker_br.py --br-player 0 --model-player 1 \
    --model-path /storage/openpsi/models/Qwen__Qwen3-4B \
    --output-dir /storage/openpsi/experiments/logs/admin/xmy-kuhn/eval-kuhn-simple/step323-br-2
"""
