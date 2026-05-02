import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@contextmanager
def all_seed(seed: int | None):
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    if seed is None:
        seed = 0

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)


@dataclass
class KuhnPokerConfig:
    seed: int = 42
    built_in_opponent: str = "cfr"
    opponent_player: int = 1
    include_opponent_turn: str = "action"

    # MCTS config
    uct_c: float = 2.0
    max_simulations: int = 100
    rollout_count: int = 10

    # CFR config
    cfr_iterations: int = 1000


class BaseEnv:
    """Abstract base class for text-based environments."""

    def reset(self, seed: int | None = None, **kwargs) -> Any:
        raise NotImplementedError

    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        raise NotImplementedError

    def render(self) -> Any:
        raise NotImplementedError


class BaseDiscreteActionEnv(BaseEnv):
    """Base class for environments with discrete action spaces."""

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        raise NotImplementedError

    def get_all_actions(self) -> List[int]:
        raise NotImplementedError


class KuhnPokerEnv(BaseDiscreteActionEnv):
    """Kuhn Poker game environment using OpenSpiel."""

    def __init__(self, config: KuhnPokerConfig | None = None):
        self.config = config or KuhnPokerConfig()
        self.built_in_opponent = self.config.built_in_opponent
        self.opponent_player = self.config.opponent_player
        self.include_opponent_turn = self.config.include_opponent_turn

        import pyspiel

        self._env = pyspiel.load_game("kuhn_poker")
        self.state = None
        self.bets = [1, 1]  # Initial ante for both players

        if self.built_in_opponent == "mcts":
            from open_spiel.python.algorithms import mcts

            random_state = np.random.RandomState(self.config.seed)
            evaluator = mcts.RandomRolloutEvaluator(
                self.config.rollout_count, random_state
            )
            self.mcts_bot = mcts.MCTSBot(
                self._env,
                self.config.uct_c,
                self.config.max_simulations,
                evaluator,
                solve=False,
                random_state=random_state,
            )
        elif self.built_in_opponent == "cfr":
            from open_spiel.python.algorithms import cfr
            import gzip
            import os
            import pickle

            self.cfr_solver = cfr.CFRSolver(self._env)
            cfr_checkpoint_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "cfr.pkl.gz"
            )
            if os.path.exists(cfr_checkpoint_path):
                with gzip.open(cfr_checkpoint_path, "rb") as f:
                    self.cfr_avg_policy = pickle.load(f)
            else:
                for _ in range(self.config.cfr_iterations):
                    self.cfr_solver.evaluate_and_update_policy()
                self.cfr_avg_policy = self.cfr_solver.average_policy()

    @property
    def current_player(self) -> int:
        if self.state is None or self.state.is_terminal():
            return 0
        return self.state.current_player()

    def reset(self, seed: Optional[int] = 0):
        """Reset the environment with a given seed.

        Seed mapping:
            0: J, Q (player_0 gets Jack, player_1 gets Queen)
            1: J, K (player_0 gets Jack, player_1 gets King)
            2: Q, K (player_0 gets Queen, player_1 gets King)
            3: Q, J (player_0 gets Queen, player_1 gets Jack)
            4: K, J (player_0 gets King, player_1 gets Jack)
            5: K, Q (player_0 gets King, player_1 gets Queen)
        """
        try:
            with all_seed(seed):
                card_0 = (seed // 2) % 3
                card_1 = ((seed - 3) // 2) % 3
                self.state = self._env.new_initial_state()
                self.state.apply_action(card_0)
                self.state.apply_action(card_1)
                self.bets = [1, 1]

                initial_observation = {
                    "observation": self.render(),
                    "legal_actions": self.get_all_actions(),
                }
                execute_results = []
                if self.built_in_opponent != "none":
                    done = self.state.is_terminal()
                    while self.current_player == self.opponent_player and not done:
                        current_player = self.current_player
                        opponent_action = self._opponent_step()
                        observation, rewards, done, info = self._step(opponent_action)
                        execute_results.append(
                            {
                                "current_player": current_player,
                                "action": self._action_to_string(
                                    current_player, opponent_action
                                ),
                                "rewards": rewards,
                                "done": done,
                                "info": info,
                                "next_player": self.current_player,
                                "observation": observation,
                                "legal_actions": self.get_all_actions(),
                            }
                        )
                return initial_observation, execute_results
        except (RuntimeError, RuntimeWarning):
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else 0
            return self.reset(next_seed)

    def step(self, action):
        execute_results = []
        current_player = self.current_player
        observation, rewards, done, info = self._step(action)

        execute_results.append(
            {
                "current_player": current_player,
                "action": self._action_to_string(current_player, action),
                "rewards": rewards,
                "done": done,
                "info": info,
                "next_player": self.current_player,
                "observation": observation,
                "legal_actions": self.get_all_actions(),
            }
        )
        if self.built_in_opponent != "none":
            while self.current_player == self.opponent_player and not done:
                current_player = self.current_player
                opponent_action = self._opponent_step()
                observation, rewards, done, info = self._step(opponent_action)
                execute_results.append(
                    {
                        "current_player": current_player,
                        "action": self._action_to_string(current_player, opponent_action),
                        "rewards": rewards,
                        "done": done,
                        "info": info,
                        "next_player": self.current_player,
                        "observation": observation,
                        "legal_actions": self.get_all_actions(),
                    }
                )
        return execute_results

    def _step(self, action):
        if isinstance(action, str):
            action = self._string_to_action(action)
        if self.state is None or self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        self.bets[self.current_player] += action
        self.state.apply_action(action)
        observation = self.render()
        rewards = self.state.rewards()
        done = self.state.is_terminal()
        info = self._get_info()
        return observation, rewards, done, info

    def _opponent_step(self) -> int:
        if self.built_in_opponent == "random":
            action = random.choice(list(self.get_all_actions().values()))
        elif self.built_in_opponent == "mcts":
            action = self.mcts_bot.step(self.state)
        elif self.built_in_opponent == "cfr":
            state_policy = self.cfr_avg_policy.action_probabilities(self.state)
            actions = list(state_policy.keys())
            probabilities = list(state_policy.values())
            action = int(np.random.choice(actions, p=probabilities))
        else:
            raise ValueError(f"Invalid built-in opponent: {self.built_in_opponent}")
        return action

    def get_prompt(self, mode: str = "prefix", think: bool = True, player_id: int = 0):
        if mode == "prefix":
            return self._get_prefix_prompt(think, player_id)
        raise ValueError(f"Invalid prompt mode: {mode}")

    def _get_prefix_prompt(self, think: bool = True, player_id: int = 0):
        system_prompt = (
            "You are an AI agent that makes optimal decisions to win in the game of Kuhn Poker."
        )
        rules = (
            "1. Kuhn poker is a two-player card game. The deck includes only three cards: "
            "King (K) > Queen (Q) > Jack (J).\n"
            "2. At the start of each game, both player_0 and player_1 place 1 chip into the pot as a blind ante.\n"
            "3. Each player is dealt a private card, and the third card is set aside unseen.\n"
            "4. The two players take turns acting, starting with player_0. A player can choose to:\n"
            "    a. <PASS>: place no additional chips into the pot.\n"
            "    b. <BET>: place 1 additional chip into the pot.\n"
            "5. If a player chooses to <PASS> after the other player's <BET>, the betting player wins the pot.\n"
            "6. If both players choose to <PASS> or both players choose to <BET>, the player with the higher card wins the pot."
        )
        information = (
            f"1. You are player_{player_id}. You are competing with player_{1-player_id}.\n"
            "2. In each of your turns:\n"
            "   a. The game state shows your private card and the betting history.\n"
            "   b. You need to choose an action based on your card and the current game state.\n"
            "   c. All legal actions for the current turn are provided in the format of `<PASS>` or `<BET>`."
        )
        format_prompt = "<answer>{your chosen action}</answer>"
        format_prompt_example = "<answer><PASS></answer>"
        instructions = (
            f"Always choose only one action from the legal actions and output `{format_prompt}` with no extra text "
            f"after you finish the thinking process. For example, `{format_prompt_example}`. "
            "Strictly follow the above format and keep your thinking process concise. "
            "Responses that do not follow the format will result in immediate loss of the game."
        )
        user_prompt = (
            f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\n{information}\n\n"
            f"RESPONSE INSTRUCTIONS:\n{instructions}\n\n"
        )
        return {"system": system_prompt, "user": user_prompt}

    def get_all_actions(self) -> Dict[int, str]:
        return self._get_legal_actions(self.current_player)

    def _get_legal_actions(self, player_id: int) -> Dict[int, str]:
        legal_actions: Dict[int, str] = {}
        if self.state is None:
            return legal_actions
        actions = self.state.legal_actions(player_id)
        for action in actions:
            legal_actions[action] = self._action_to_string(player_id, action)
        return legal_actions

    def _action_to_string(self, player_id: int, action):
        if isinstance(action, str):
            return action
        if action == 0:
            return "<PASS>"
        return "<BET>"

    def _string_to_action(self, action_str: str) -> int:
        action_str = action_str.strip().upper()
        if action_str == "<PASS>":
            return 0
        if action_str == "<BET>":
            return 1
        if "PASS" in action_str:
            return 0
        if "BET" in action_str:
            return 1
        raise ValueError(f"Invalid action string: {action_str}")

    def _get_info(self) -> Dict[str, Any]:
        if self.state is None:
            return {}

        if len(self.state.history()) == 2 and not self.state.is_terminal():
            card_0_idx = self.state.history()[0]
            card_1_idx = self.state.history()[1]
            return {
                "card_0_idx": float(card_0_idx),
                "card_1_idx": float(card_1_idx),
                "total_pot": float(sum(self.bets)),
                "player_0_bet": float(self.bets[0]),
                "player_1_bet": float(self.bets[1]),
            }

        if self.state.is_terminal():
            returns = self.state.returns()
            winner = int(np.argmax(returns)) if returns[0] != returns[1] else -1
            deck = ["Jack (J)", "Queen (Q)", "King (K)"]
            card_0_idx = self.state.history()[0]
            card_1_idx = self.state.history()[1]
            return {
                "card_0": deck[card_0_idx],
                "card_1": deck[card_1_idx],
                "player_0_return": returns[0],
                "player_1_return": returns[1],
                "winner": winner,
                "player_0_lose_for_wrong_format": 0,
                "player_1_lose_for_wrong_format": 0,
                "player_0_lose_for_overlong_response": 0,
                "player_1_lose_for_overlong_response": 0,
                "player_0_lose_for_overlong_sequence": 0,
                "player_1_lose_for_overlong_sequence": 0,
                "player_0_success": winner == 0,
                "player_1_success": winner == 1,
                "draw": winner == -1,
                "total_pot": float(sum(self.bets)),
                "player_0_bet": float(self.bets[0]),
                "player_1_bet": float(self.bets[1]),
            }
        return {
            "total_pot": float(sum(self.bets)),
            "player_0_bet": float(self.bets[0]),
            "player_1_bet": float(self.bets[1]),
        }

    def get_losing_state(
        self,
        player_id: int = 0,
        overlong_response: bool = False,
        overlong_sequence: bool = False,
    ):
        done = True
        if player_id == 0:
            reward = [-self.bets[0] - 10, 0]
            info = {
                "player_0_return": -self.bets[0],
                "player_1_return": self.bets[0],
                "winner": 1,
                "player_0_lose_for_wrong_format": 1,
                "player_1_lose_for_wrong_format": 0,
                "player_0_lose_for_overlong_response": 1 if overlong_response else 0,
                "player_1_lose_for_overlong_response": 0,
                "player_0_lose_for_overlong_sequence": 1 if overlong_sequence else 0,
                "player_1_lose_for_overlong_sequence": 0,
                "player_0_success": False,
                "player_1_success": True,
                "draw": False,
            }
        else:
            reward = [0, -self.bets[1] - 10]
            info = {
                "player_0_return": self.bets[1],
                "player_1_return": -self.bets[1],
                "winner": 0,
                "player_0_lose_for_wrong_format": 0,
                "player_1_lose_for_wrong_format": 1,
                "player_0_lose_for_overlong_response": 0,
                "player_1_lose_for_overlong_response": 1 if overlong_response else 0,
                "player_0_lose_for_overlong_sequence": 0,
                "player_1_lose_for_overlong_sequence": 1 if overlong_sequence else 0,
                "player_0_success": True,
                "player_1_success": False,
                "draw": False,
            }
        execute_results = [
            {
                "current_player": player_id,
                "action": "",
                "rewards": reward,
                "done": done,
                "info": info,
                "next_player": None,
                "observation": None,
                "legal_actions": None,
            }
        ]
        return execute_results

    def render(self) -> str:
        if self.state is None:
            return "Game not started"

        history = ["1. Blind ante: both player_0 and player_1 place 1 chip into the pot."]
        deck = ["Jack (J)", "Queen (Q)", "King (K)"]

        info_state = self.state.information_state_tensor(self.current_player)
        card_idx = np.argmax(info_state[2:5])
        card = deck[card_idx]
        history.append(f"2. Deal: your card is {card}.")

        action_set = ["<PASS>", "<BET>"]
        if len(self.state.history()) > 2:
            num_turns = len(self.state.history()) - 2
            for idx in range(num_turns):
                player_id = idx % 2
                action_idx = self.state.history()[2 + idx]
                action = action_set[action_idx]
                history.append(
                    f"{idx + 3}. Turn {idx + 1}: player_{player_id} chooses to {action}."
                )
        return "\n".join(history)

    def close(self):
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()