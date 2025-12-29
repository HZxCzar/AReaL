import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from realhf.api.core.env_api import EnvironmentService, register_environment
from realhf.base import logging


logger = logging.getLogger("HanabiEnv")


COLORS = ["red", "yellow", "green", "blue", "white"]
COLOR_TO_LETTER = {"red": "R", "yellow": "Y", "green": "G", "blue": "B", "white": "W"}
LETTER_TO_COLOR = {v: k for k, v in COLOR_TO_LETTER.items()}
RANK_COUNTS = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}


@dataclass(frozen=True)
class Card:
    color: str
    rank: int

    def short(self) -> str:
        return f"{COLOR_TO_LETTER[self.color]}{self.rank}"


class HanabiEnv(EnvironmentService):
    """A light-weight text environment for the cooperative card game Hanabi."""

    def __init__(
        self,
        num_players: int = 2,
        hand_size: Optional[int] = None,
        repeat_rules: bool = True,
        max_info_tokens: int = 8,
        max_fuse_tokens: int = 3,
        scenario: str = "full",
    ):
        if num_players < 2:
            raise ValueError("Hanabi requires at least two players.")
        if hand_size is None:
            hand_size = 5 if num_players <= 3 else 4

        global COLORS
        global RANK_COUNTS
        if scenario == "full":
            self.rules = (
                "You are playing Hanabi, a fully cooperative, turn-based card game.\n"
                "Goal: Build 5 color stacks (red (R), yellow (Y), green (G), blue (B), white (W)) strictly in rank order from 1 to 5.\n"
                "There are 50 cards in total, each color has 3 rank1 cards, 2 rank2–4 cards, and 1 rank5 card.\n"
                "Maximum score is 25; partial stacks score their highest completed rank.\n\n"

                "Information model:\n"
                "- You see all public state: current stacks, discard pile, remaining deck size, "
                "information tokens, fuse tokens, and the full hands of all OTHER players.\n"
                "- You NEVER see your own cards.\n"
                "- You have plausible knowledge of your own cards based on previously provided hints and public information.\n"

                "Actions:\n"
                "1) Play a card: succeeds only if it is the next required rank of its color (e.g. Red Stack should go from R1 -> R2 -> R3 -> R4 -> R5); "
                "otherwise a fuse token is lost and the card is discarded.\nCompleting a stack to rank 5 grants +1 information token if any are missing.\n"
                "2) Discard a card: removes it, draws a new card if available, "
                "and restores +1 information token (up to the maximum).\n"
                "3) Give a hint: spend 1 information token to name EXACTLY ONE color OR rank "
                "to a single teammate; the hint must mark ALL and ONLY matching cards in their hand.\n"
                "The game starts with 8 information tokens. If none remains, you can not give a hint.\n\n"

                "Additional rules:\n"
                "- Misplays permanently remove that copy from the game.\n"
                "- When the deck emptjies, each player (including the one who drew last) gets exactly one final turn.\n"
                "- If all fuse tokens are lost, the score becomes 0 and the game ends immediately.\n"
                "- If an illegal action is proposed, you will skip this turn."
            )
        elif scenario == "simple":
            COLORS = COLORS[:3]
            RANK_COUNTS = {1: 3, 2: 1}
            hand_size = 5
            max_info_tokens = 8
            max_fuse_tokens = 3
            self.rules = (
                "You are playing Hanabi, a fully cooperative, turn-based card game.\n"
                "Goal: Build 3 color stacks (red (R), yellow (Y), green (G)) strictly in rank order from 1 to 2.\n"
                "There are 12 cards in total, each color has 3 rank1 cards and 1 rank2 card.\n"
                "Maximum score is 6; partial stacks score their highest completed rank.\n\n"

                "Information model:\n"
                "- You see all public state: current stacks, discard pile, remaining deck size, "
                "information tokens, fuse tokens, and the full hands of all OTHER players.\n"
                "- You NEVER see your own cards.\n"
                "- You have plausible knowledge of your own cards based on previously provided hints and public information.\n"

                "Actions:\n"
                "1) Play a card: succeeds only if it is the next required rank of its color (e.g. Red Stack should go from R1 -> R2); "
                "otherwise a fuse token is lost and the card is discarded.\nCompleting a stack to rank 5 grants +1 information token if any are missing.\n"
                "2) Discard a card: removes it, draws a new card if available, "
                "and restores +1 information token (up to the maximum).\n"
                "3) Give a hint: spend 1 information token to name EXACTLY ONE color OR rank "
                "to a single teammate; the hint must mark ALL and ONLY matching cards in their hand.\n"
                "The game starts with 8 information tokens. If none remains, you can not give a hint.\n\n"

                "Additional rules:\n"
                "- Misplays permanently remove that copy from the game.\n"
                "- When the deck empties, each player (including the one who drew last) gets exactly one final turn.\n"
                "- If all fuse tokens are lost, the score becomes 0 and the game ends immediately.\n"
                "- If an illegal action is proposed, you will skip this turn."
            )
        elif scenario == "mini":
            COLORS = COLORS[:2]
            RANK_COUNTS = {1: 3, 2: 1}
            hand_size = 3
            max_info_tokens = 3
            max_fuse_tokens = 3
            self.rules = (
                "You are playing Hanabi, a fully cooperative, turn-based card game.\n"
                "Goal: Build 2 color stacks (red (R), yellow (Y)) strictly in rank order from 1 to 2.\n"
                "There are 8 cards in total, each color has 3 rank1 cards and 1 rank2 card.\n"
                "Maximum score is 4; partial stacks score their highest completed rank.\n\n"

                "Information model:\n"
                "- You see all public state: current stacks, discard pile, remaining deck size, "
                "information tokens, fuse tokens, and the full hands of all OTHER players.\n"
                "- You NEVER see your own cards.\n"
                "- You have plausible knowledge of your own cards based on previously provided hints and public information.\n"

                "Actions:\n"
                "1) Play a card: succeeds only if it is the next required rank of its color (e.g. Red Stack should go from R1 -> R2); "
                "otherwise a fuse token is lost and the card is discarded.\nCompleting a stack to rank 5 grants +1 information token if any are missing.\n"
                "2) Discard a card: removes it, draws a new card if available, "
                "and restores +1 information token (up to the maximum).\n"
                "3) Give a hint: spend 1 information token to name EXACTLY ONE color OR rank "
                "to a single teammate; the hint must mark ALL and ONLY matching cards in their hand.\n"
                "The game starts with 3 information tokens. If none remains, you can not give a hint.\n\n"

                "Additional rules:\n"
                "- Misplays permanently remove that copy from the game.\n"
                "- When the deck empties, each player (including the one who drew last) gets exactly one final turn.\n"
                "- If all fuse tokens are lost, the score becomes 0 and the game ends immediately.\n"
                "- If an illegal action is proposed, you will skip this turn."
            )
        print("[DEBUG] initialize environment: ", scenario, COLORS, RANK_COUNTS, flush=True)

        self.num_players = num_players
        self.hand_size = hand_size
        self.max_info_tokens = max_info_tokens
        self.max_fuse_tokens = max_fuse_tokens
        self.repeat_rules = repeat_rules
        self.scenario = scenario

        self.players: List[str] = [f"player{i+1}" for i in range(num_players)]
        self.deck: List[Card] = []
        self.hands: Dict[str, List[Card]] = {}
        self.knowledge: Dict[str, List[Dict[str, Optional[str | int]]]] = {}
        self.discard_pile: List[Card] = []
        self.fireworks: Dict[str, int] = {c: 0 for c in COLORS}
        self.info_tokens = self.max_info_tokens
        self.fuse_tokens = self.max_fuse_tokens
        self.final_turns_remaining: Optional[int] = None

        self.agent_player: str = ""
        self.current_player_idx = 0
        self.turn_count = 0
        self.last_event: str = ""
        self.answer_format_record: List[float] = [70.0, 70.0]
        self.firework_score_multiplier: float = 1.0
        self.trajectory: List[str] = []

        

        self.guide = (
            "Plan a cooperative move that fits both the public information and long-term plan.\n"
            "Valid actions are: {actions}.\n"
        )
        self.format_req = "Format your response exactly as \"<think>your reasoning</think> <answer>your chosen action</answer>\".\n"

    # ------------------------------------------------------------------
    # Setup helpers
    def _build_deck(self) -> List[Card]:
        deck: List[Card] = []
        print("[DEBUG] initialize deck: ", COLORS, RANK_COUNTS, flush=True)
        for color in COLORS:
            for rank, count in RANK_COUNTS.items():
                deck.extend(Card(color, rank) for _ in range(count))
        random.shuffle(deck)
        return deck

    def _deal_initial_hands(self) -> None:
        self.hands = {p: [] for p in self.players}
        self.knowledge = {p: [] for p in self.players}
        for _ in range(self.hand_size):
            for p in self.players:
                self._draw_card(p)

    def _draw_card(self, player: str) -> None:
        if not self.deck:
            if self.final_turns_remaining is None:
                # Trigger final round: every player (including current) gets one more turn
                self.final_turns_remaining = self.num_players
            return
        card = self.deck.pop(0)
        self.hands[player].append(card)
        self.knowledge[player].append({"color": None, "rank": None})

    # ------------------------------------------------------------------
    async def reset(self, seed=None, options=None):
        obs, guide, _ = await self.sreset(seed=seed, options=options)
        return f"{obs} {guide}", {}

    async def sreset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)

        self.deck = self._build_deck()
        self.discard_pile = []
        self.fireworks = {c: 0 for c in COLORS}
        self.info_tokens = self.max_info_tokens
        self.fuse_tokens = self.max_fuse_tokens
        self.final_turns_remaining = None
        self.current_player_idx = 0
        self.turn_count = 0
        self.last_event = "Game start."
        self.answer_format_record = [70.0, 70.0]
        self.firework_score_multiplier: float = 1.0
        self.trajectory = []

        self._deal_initial_hands()
        self.agent_player = self.players[self.current_player_idx]

        obs = self._build_observation(self.agent_player)
        guide = self._build_guide(self.agent_player)
        teacher_obs = self._build_teacher_observation(self.agent_player)
        state = self._build_state(self.agent_player)
        self.trajectory.append(self._snapshot_state("Initial setup"))

        return obs, guide, {"teacher_observation": teacher_obs, "state": state}

    # ------------------------------------------------------------------
    # Observation helpers
    def _format_fireworks(self) -> str:
        return ", ".join(f"{COLOR_TO_LETTER[c]}:{lvl}" for c, lvl in self.fireworks.items())

    def _format_discard(self) -> str:
        if not self.discard_pile:
            return "empty"
        counts: Dict[str, int] = {}
        for card in self.discard_pile:
            key = card.short()
            counts[key] = counts.get(key, 0) + 1
        return ", ".join(f"{k}x{v}" for k, v in sorted(counts.items()))

    def _visible_hands(self, viewer: str) -> str:
        desc = []
        for p in self.players:
            cards = self.hands[p]
            if p == viewer:
                knowledge = []
                for idx, card_info in enumerate(self.knowledge[p]):
                    known_color = card_info.get("color")
                    color_repr = COLOR_TO_LETTER[known_color] if known_color else "?"
                    known_rank = card_info.get("rank")
                    rank_repr = str(known_rank) if known_rank else "?"
                    knowledge.append(f"{idx + 1}:{color_repr}{rank_repr}")
                desc.append(f"{p}'s inferred hand: {' '.join(knowledge) if knowledge else 'empty'}")
            else:
                desc.append(f"{p}'s hand: {' '.join(card.short() for card in cards) if cards else 'empty'}")
        return "; ".join(desc)

    def _full_hands(self) -> str:
        parts = []
        for p in self.players:
            cards = " ".join(card.short() for card in self.hands[p]) or "empty"
            parts.append(f"{p}: {cards}")
        return "; ".join(parts)

    def _build_observation(self, player: str) -> str:
        score = sum(self.fireworks.values()) * self.firework_score_multiplier
        obs_parts = []
        if self.repeat_rules:
            obs_parts.append(self.rules)
        obs_parts.append(
            "\n".join(
                [
                    "=== Public Status ===",
                    f"Player: {player}",
                    f"Score: {score} | Info tokens: {self.info_tokens}/{self.max_info_tokens} | Fuse tokens: {self.fuse_tokens}/{self.max_fuse_tokens}",
                    f"Deck remaining: {len(self.deck)}",
                    f"Fireworks: {self._format_fireworks()} | Discards: {self._format_discard()}",
                    f"Recent event: {self.last_event}",
                ]
            )
        )
        obs_parts.append("\n=== Visible Hands & Self Knowledge ===\n" + self._visible_hands(player))
        return "\n".join(obs_parts)

    def _build_teacher_observation(self, player: str) -> str:
        score = sum(self.fireworks.values()) * self.firework_score_multiplier
        teacher_info = [
            "# Rules\n",
            self.rules,
            "\n\n# Instruction\n",
            "You are a privileged Hanabi observer with perfect information.",
            "=== Turn Context ===",
            f"Active player: {player}",
            f"Score: {score}",
            f"Info tokens: {self.info_tokens}/{self.max_info_tokens} | Fuse tokens: {self.fuse_tokens}/{self.max_fuse_tokens}",
            f"Deck remaining: {len(self.deck)}",
            "=== Board ===",
            f"Fireworks: {self._format_fireworks()}",
            f"Discards: {self._format_discard()}",
            "=== Hands ===",
            f"All hands: {self._full_hands()}",
        ]
        if self.last_event:
            teacher_info.append(f"Most recent event: {self.last_event}")
        return "\n".join(teacher_info)

    def _build_state(self, player: str) -> str:
        state = self._build_teacher_observation(player).replace("You are a privileged Hanabi observer with perfect information.", "Below is the global state at the current turn of Hanabi game.")
        return state  

    def _build_guide(self, player: str) -> str:
        actions = ", ".join(self._get_valid_actions(player))
        guide = self.guide.format(actions=actions)
        if self.fuse_tokens <= 1:
            guide += f"You only have {self.fuse_tokens} fuse tokens left. Plan your next move carefully, since losing all fuse tokens will end the game with score 0.\n"
        guide += self.format_req
        return guide

    def _snapshot_state(self, prefix: str) -> str:
        return (
            f"{prefix}: score={sum(self.fireworks.values()) * self.firework_score_multiplier}, info={self.info_tokens}, fuse={self.fuse_tokens}, "
            f"deck={len(self.deck)}, fireworks={self._format_fireworks()}, discard={self._format_discard()}, "
            f"hands={self._full_hands()}"
        )

    # ------------------------------------------------------------------
    def _get_valid_actions(self, player: str) -> List[str]:
        actions: List[str] = []
        hand = self.hands[player]
        for card in hand:
            card_name = card.short()
            actions.append(f"play {card_name}")
            actions.append(f"discard {card_name}")
        if self.info_tokens > 0:
            for teammate in self.players:
                if teammate == player:
                    continue
                colors = sorted({card.color for card in self.hands[teammate]})
                ranks = sorted({card.rank for card in self.hands[teammate]})
                for color in colors:
                    actions.append(f"hint {teammate} color {color}")
                for rank in ranks:
                    actions.append(f"hint {teammate} rank {rank}")
        return actions or ["wait"]

    def _resolve_card_identifier(self, player: str, identifier: str) -> Optional[int]:
        token = identifier.strip()
        if not token:
            return None

        # Allow legacy numeric indices for robustness.
        if token.replace(" ", "").isdigit():
            try:
                idx = int(token.replace(" ", "")) - 1
            except ValueError:
                idx = -1
            if 0 <= idx < len(self.hands[player]):
                return idx

        normalized = token.lower().replace(" ", "")
        for idx, card in enumerate(self.hands[player]):
            short_name = card.short().lower()
            long_name = f"{card.color}{card.rank}"
            if normalized in {short_name, long_name}:
                return idx

        # Support formats like "red-1" or "R1" with punctuation removed.
        normalized = "".join(ch for ch in normalized if ch.isalnum())
        for idx, card in enumerate(self.hands[player]):
            variants = {
                card.short().lower(),
                f"{card.color}{card.rank}",
                f"{COLOR_TO_LETTER[card.color].lower()}{card.rank}",
            }
            if normalized in variants:
                return idx

        return None

    def _consume_format_reward(self, text: str) -> None:
        self.answer_format_record[1] += 0.2
        if "<answer>" in text and "</answer>" in text:
            self.answer_format_record[0] += 0.2
        elif "<think>" in text and "</think>" in text:
            self.answer_format_record[0] += 0.1

    def _parse_action(self, text: str) -> str:
        import re

        match = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match[-1].strip().lower()
        return ""

    def _advance_player(self) -> None:
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        self.agent_player = self.players[self.current_player_idx]

    def _apply_play(self, player: str, idx: int) -> Tuple[str, float]:
        if idx < 0 or idx >= len(self.hands[player]):
            return "Invalid play action.", -0.5

        card = self.hands[player].pop(idx)
        self.knowledge[player].pop(idx)
        expected_rank = self.fireworks[card.color] + 1
        if card.rank == expected_rank:
            self.fireworks[card.color] = card.rank
            reward = 3.0
            msg = f"{player} successfully played {card.short()} onto the {card.color} stack."
            if card.rank == 5 and self.info_tokens < self.max_info_tokens:
                self.info_tokens += 1
                msg += " Completed stack grants an information token."
            if sum(self.fireworks.values()) == 25:
                msg += " All stacks complete!"
        else:
            self.discard_pile.append(card)
            self.fuse_tokens -= 1
            reward = -1.0
            msg = (
                f"{player} misplayed {card.short()} (needed {COLOR_TO_LETTER[card.color]}{expected_rank}). "
                "Fuse token lost."
            )
        self._draw_card(player)
        return msg, reward

    def _apply_discard(self, player: str, idx: int) -> Tuple[str, float]:
        if idx < 0 or idx >= len(self.hands[player]):
            return "Invalid discard action.", -0.2

        card = self.hands[player].pop(idx)
        self.knowledge[player].pop(idx)
        self.discard_pile.append(card)
        msg = f"{player} discarded {card.short()}."
        reward = 0.0
        if self.info_tokens < self.max_info_tokens:
            self.info_tokens += 1
            msg += " Recovered one information token."
            reward += 0.1
        self._draw_card(player)
        return msg, reward

    def _apply_hint(self, player: str, target: str, hint_type: str, value: str) -> Tuple[str, float]:
        if self.info_tokens <= 0:
            return "No information tokens remain.", -0.2
        if target not in self.hands:
            return "Invalid hint target.", -0.2
        if target == player:
            return "Cannot hint yourself.", -0.2

        matches: List[int] = []
        if hint_type == "color":
            color = value.lower()
            if color in LETTER_TO_COLOR:
                color = LETTER_TO_COLOR[color]
            if color not in COLORS:
                return "Unknown color for hint.", -0.2
            for idx, card in enumerate(self.hands[target]):
                if card.color == color:
                    self.knowledge[target][idx]["color"] = color
                    matches.append(idx)
        elif hint_type == "rank":
            try:
                rank = int(value)
            except ValueError:
                return "Rank hint must be a number between 1 and 5.", -0.2
            if rank not in RANK_COUNTS:
                return "Rank hint must be in [1, 5].", -0.2
            for idx, card in enumerate(self.hands[target]):
                if card.rank == rank:
                    self.knowledge[target][idx]["rank"] = rank
                    matches.append(idx)
        else:
            return "Hint must specify color or rank.", -0.2

        if not matches:
            return "Hint provided no information and is invalid.", -0.2

        self.info_tokens -= 1
        positions = ", ".join(str(i + 1) for i in matches)
        if hint_type == "color":
            msg = f"{player} hinted {target} that positions {positions} are {COLOR_TO_LETTER[color]} cards."
        else:
            msg = f"{player} hinted {target} that positions {positions} are rank {value}."
        return msg, 0.5

    # ------------------------------------------------------------------
    async def step(self, action) -> Tuple[str, str, float, bool, bool, Dict]:
        qid, acts = action if isinstance(action, tuple) and len(action) == 2 else ("", action)
        text = acts[0] if isinstance(acts, list) and acts else ""
        self._consume_format_reward(text)
        player = self.agent_player
        parsed_action = self._parse_action(text)
        msg = ""
        prev_score = sum(self.fireworks.values()) * self.firework_score_multiplier
        # reward = 0.0

        logger.debug("%s submits action text: %s", player, text)

        if parsed_action.startswith("play "):
            idx = self._resolve_card_identifier(player, parsed_action[len("play "):])
            if idx is None:
                msg = "You gave an unidentifiable play command."
                # reward = -0.2
            else:
                msg, reward = self._apply_play(player, idx)
        elif parsed_action.startswith("discard "):
            idx = self._resolve_card_identifier(player, parsed_action[len("discard "):])
            if idx is None:
                msg = "You gave an unidentifiable discard command."
                # reward = -0.2
            else:
                msg, reward = self._apply_discard(player, idx)
        elif parsed_action.startswith("hint "):
            parts = parsed_action.split()
            if len(parts) >= 4:
                target = parts[1]
                hint_type = parts[2]
                value = " ".join(parts[3:])
                msg, reward = self._apply_hint(player, target, hint_type, value)
            else:
                msg = "You gave an incomplete hint command."
                # reward = -0.2
        else:
            msg = "You preformed and invalid or empty action."
            # reward = -0.2

        self.turn_count += 1
        self.last_event = msg
        self.trajectory.append(f"Turn {self.turn_count}: {player} -> {parsed_action} | {msg}\n{text}")
        done = self._check_termination()

        if not done:
            if self.final_turns_remaining is not None:
                self.final_turns_remaining -= 1
                if self.final_turns_remaining <= 0:
                    done = True

        if self.fuse_tokens <= 0:
            # Losing the final fuse immediately ends the game and resets the reward
            self.fireworks = {c: 0 for c in COLORS}
            done = True

        if not done:
            self._advance_player()
            obs = self._build_observation(self.agent_player)
            guide = self._build_guide(self.agent_player)
        else:
            obs = self._terminal_observation()
            guide = "Game over."

        info = {
            "event": msg,
            "score": sum(self.fireworks.values()) * self.firework_score_multiplier,
            "info_tokens": self.info_tokens,
            "fuse_tokens": self.fuse_tokens,
            "deck_remaining": len(self.deck),
            "teacher_observation": self._build_teacher_observation(self.agent_player) if not done else obs,
            "state": self._build_state(self.agent_player) if not done else obs,
        }
        current_score = info["score"]
        reward = current_score - prev_score

        return obs, guide, reward, done, False, info

    def _terminal_observation(self) -> str:
        score = sum(self.fireworks.values())
        outcome = "All fireworks completed!" if score == 25 else "Game over."
        if self.fuse_tokens <= 0:
            outcome = "Fuse tokens depleted. The team loses."
        elif self.final_turns_remaining is not None and self.final_turns_remaining <= 0:
            outcome = "Deck exhausted. Final round finished."
        return "\n".join(
            [
                f"{outcome}",
                f"Final score: {score * self.firework_score_multiplier}",
                f"Fireworks: {self._format_fireworks()}",
                f"Discard pile: {self._format_discard()}",
            ]
        )

    def _check_termination(self) -> bool:
        if self.fuse_tokens <= 0:
            return True
        if all(level == 5 for level in self.fireworks.values()):
            return True
        # If all players have empty hands and deck empty
        if not self.deck and all(len(hand) == 0 for hand in self.hands.values()):
            return True
        return False

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, int]:
        return {
            "score": sum(self.fireworks.values()) * self.firework_score_multiplier,
            "turns": self.turn_count,
            "info_tokens": self.info_tokens,
            "fuse_tokens": self.fuse_tokens,
            "deck_remaining": len(self.deck),
        }

    def get_trajectory(self) -> List[str]:
        return list(self.trajectory)


register_environment("hanabi_env", HanabiEnv)