import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pyspiel
from open_spiel.python import policy as openspiel_policy


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
    if len(action_prob_vector) != 2:
        raise ValueError(
            "Action prob vector must be length 2: index 0 for <PASS>, index 1 for <BET>."
        )
    probs = {0: float(action_prob_vector[0]), 1: float(action_prob_vector[1])}
    return _normalize_action_probs(probs, legal_actions)


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


def _uniform_policy_for_states(states: List, player_id: int) -> Dict[str, Dict[int, float]]:
    table: Dict[str, Dict[int, float]] = {}
    for state in states:
        if state.current_player() != player_id:
            continue
        legal = state.legal_actions(player_id)
        table[state.information_state_string(player_id)] = _normalize_action_probs({}, legal)
    return table


class OpponentPolicyFromCache:
    def __init__(self, policy_table: Dict[str, Dict[int, float]]):
        self.policy_table = policy_table

    def action_probabilities(self, state) -> Dict[int, float]:
        info_state = state.information_state_string(state.current_player())
        legal_actions = state.legal_actions()
        probs = self.policy_table.get(info_state, {})
        return _normalize_action_probs(probs, legal_actions)


class MixedPolicy(openspiel_policy.Policy):
    def __init__(
        self,
        br_player: int,
        opp_policy: OpponentPolicyFromCache,
        br_policy_table: Dict[str, Dict[int, float]],
        game,
    ):
        super().__init__(game, [br_player, 1 - br_player])
        self.br_player = br_player
        self.opp_policy = opp_policy
        self.br_policy_table = br_policy_table

    def action_probabilities(self, state, player_id=None) -> Dict[int, float]:
        if state.is_chance_node():
            return {a: float(p) for a, p in state.chance_outcomes()}

        cur = state.current_player() if player_id is None else player_id
        if cur == self.br_player:
            info_state = state.information_state_string(self.br_player)
            legal_actions = state.legal_actions(self.br_player)
            if info_state in self.br_policy_table:
                return _normalize_action_probs(self.br_policy_table[info_state], legal_actions)
            return _normalize_action_probs({}, legal_actions)

        return self.opp_policy.action_probabilities(state)


def _compute_exploitability(game, policy: openspiel_policy.Policy) -> float:
    from open_spiel.python.algorithms import exploitability as exp

    return float(exp.exploitability(game, policy))


def _load_policy_cache(path: Path) -> Dict[str, Dict[int, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    policy_blob = data.get("policy", data)
    policy_table: Dict[str, Dict[int, float]] = {}
    for info_state, probs in policy_blob.items():
        policy_table[info_state] = {int(a): float(p) for a, p in probs.items()}
    return policy_table


def _load_action_vectors(path: Path) -> List[List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [list(vec) for vec in data]
    if isinstance(data, dict):
        ordered = [data[str(idx)] for idx in sorted((int(k) for k in data.keys()))]
        return [list(vec) for vec in ordered]
    raise ValueError("Action vectors file must be a JSON list or dict keyed by index.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score exploitability for per-state action vectors against a cached opponent policy.",
    )
    parser.add_argument("--br-player", type=int, choices=[0, 1], required=True)
    parser.add_argument("--model-player", type=int, choices=[0, 1], required=True)
    parser.add_argument("--model-policy-path", type=str, required=True)
    parser.add_argument("--action-vectors-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    if args.br_player == args.model_player:
        raise ValueError("--br-player and --model-player must be different.")

    game = pyspiel.load_game("kuhn_poker")
    br_states = _collect_player_states(game, args.br_player)
    base_br_policy = _uniform_policy_for_states(br_states, args.br_player)

    policy_table = _load_policy_cache(Path(args.model_policy_path))
    opp_policy = OpponentPolicyFromCache(policy_table)

    action_vectors = _load_action_vectors(Path(args.action_vectors_path))
    if len(action_vectors) < len(br_states):
        raise ValueError(
            f"Expected at least {len(br_states)} action vectors, got {len(action_vectors)}."
        )

    results = []
    for idx, state in enumerate(br_states):
        if state.current_player() != args.br_player:
            continue
        info_state = state.information_state_string(args.br_player)
        legal_actions = state.legal_actions(args.br_player)
        action_probs = _action_vector_to_probs(action_vectors[idx], legal_actions)
        br_policy_table = dict(base_br_policy)
        br_policy_table[info_state] = action_probs
        policy = MixedPolicy(
            br_player=args.br_player,
            opp_policy=opp_policy,
            br_policy_table=br_policy_table,
            game=game,
        )
        exploitability = _compute_exploitability(game, policy)
        results.append(
            {
                "idx": idx,
                "info_state": info_state,
                "action_vector": action_vectors[idx],
                "exploitability": float(exploitability),
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

"""
Example:
python examples/kuhn_poker/score_kuhn_poker_action_vectors.py --br-player 0 --model-player 1 \
    --model-policy-path /storage/openpsi/users/xmy/inclusionAI/AReaL-New/analysis/step323-br-3/model_policy.json \
    --action-vectors-path /storage/openpsi/users/xmy/inclusionAI/AReaL-New/analysis/trained_policy.json \
    --output-path /storage/openpsi/experiments/logs/admin/xmy-kuhn/eval-kuhn-simple/step323-br-exploit
"""