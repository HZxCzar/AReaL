import argparse
import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os, sys
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
sys.path.insert(0, ROOT)
from realhf.impl.environment.kuhn_poker_env import KuhnPokerConfig, KuhnPokerEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _build_turn_prompt(
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


def _extract_action(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def _compute_bets(history: list[int]) -> list[int]:
    bets = [1, 1]
    for idx, action in enumerate(history[2:]):
        player_id = idx % 2
        bets[player_id] += action
    return bets


def _collect_player_states(game, player_id: int) -> list:
    import pyspiel

    seen: dict[str, object] = {}

    def _dfs(state):
        if state.is_terminal():
            return
        if state.current_player() == pyspiel.PlayerId.CHANCE:
            for action, _ in state.chance_outcomes():
                next_state = state.clone()
                next_state.apply_action(action)
                _dfs(next_state)
            return

        if state.current_player() == player_id:
            info_state = state.information_state_string(player_id)
            if info_state not in seen:
                seen[info_state] = state.clone()

        for action in state.legal_actions():
            next_state = state.clone()
            next_state.apply_action(action)
            _dfs(next_state)

    _dfs(game.new_initial_state())
    return [seen[key] for key in sorted(seen.keys())]


def _sample_actions(
    model,
    tokenizer,
    prompt_messages: list[dict[str, str]],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[str]:

    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,              # <-- key change
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # outputs: [n_samples, input_len + gen_len]
    gen = outputs[:, input_len:]
    resp = tokenizer.batch_decode(gen, skip_special_tokens=True)
    print(resp)
    return resp


def _plot_action_counts(
    counts: dict[str, int],
    ne_action: str,
    title: str,
    observation: str,
    output_path: Path,
) -> None:
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    colors = []
    for label in labels:
        if label == "INVALID":
            colors.append("tab:red")
        elif label == ne_action:
            colors.append("tab:green")
        else:
            colors.append("tab:blue")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_ylim(0, max(values) + 1)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    wrapped_obs = textwrap.fill(observation, width=70)
    ax.text(
        0.5,
        -0.25,
        wrapped_obs,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.text(
        0.98,
        0.95,
        f"NE action: {ne_action}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Kuhn poker policies against CFR NE actions.",
    )
    parser.add_argument("--player-id", type=int, choices=[0, 1], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    env = KuhnPokerEnv(KuhnPokerConfig(built_in_opponent="none"))
    cfr_env = KuhnPokerEnv(KuhnPokerConfig(built_in_opponent="cfr"))
    cfr_policy = cfr_env.cfr_avg_policy

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game = env._env
    states = _collect_player_states(game, args.player_id)
    print(f"Collected {len(states)} player states for player_{args.player_id}.")

    for idx, state in enumerate(states):
        env.state = state.clone()
        env.bets = _compute_bets(state.history())

        observation = env.render()
        legal_actions = env.get_all_actions()
        prompt_messages = _build_turn_prompt(env, observation, legal_actions, args.player_id)

        completions = _sample_actions(
            model,
            tokenizer,
            prompt_messages,
            n_samples=args.samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        counts = Counter()
        for completion in completions:
            action_text = _extract_action(completion)
            try:
                action_id = env._string_to_action(action_text)
                action_str = env._action_to_string(args.player_id, action_id)
            except ValueError:
                action_str = "INVALID"
            counts[action_str] += 1

        for action_label in ["<PASS>", "<BET>", "INVALID"]:
            counts.setdefault(action_label, 0)

        state_policy = cfr_policy.action_probabilities(state)
        ne_action_id = max(state_policy, key=state_policy.get)
        ne_action = env._action_to_string(args.player_id, ne_action_id)

        info_state = state.information_state_string(args.player_id)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", info_state).strip("_")
        filename = f"player_{args.player_id}_state_{idx:02d}_{safe_name}.png"
        output_path = output_dir / filename

        title = f"Player {args.player_id} | {info_state}"
        _plot_action_counts(counts, ne_action, title, observation, output_path)
        print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()

"""
Example:
python examples/kuhn_poker/eval_kuhn_poker.py --player-id 0 \
    --model-path /storage/openpsi/experiments/checkpoints/admin/xmy-kuhn/train-qwen3-4b-vs-qwen3-4b-0-bs512-650k-smalllr2/default/epoch1epochstep67globalstep323 \
    --output-dir /storage/openpsi/experiments/logs/admin/xmy-kuhn/eval-kuhn-simple/step323
"""