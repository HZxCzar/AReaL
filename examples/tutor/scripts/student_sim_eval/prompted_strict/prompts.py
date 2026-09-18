"""Compose student instructions from the same criteria used by the external gate."""

import json
import random
from pathlib import Path

INSTRUCTION = (
    "Before responding, check whether the tutor's latest message meets the "
    "following learning preference. Keep your judgment internal. "
    "If it does, respond as a student and work on the mathematics. "
    "Otherwise, reply only with: {complaint_reply}"
)


def student_instruction(gate_path: Path, preference: str, complaints_path: Path) -> str:
    """Preserve the gate criterion verbatim; do not add preference-specific rules."""
    preferences = json.loads(gate_path.read_text())["personalities"]
    if preference not in preferences:
        raise ValueError(f"No gate criterion for student preference: {preference}")
    complaints = json.loads(complaints_path.read_text())["bare"]
    if not complaints or any(
        not isinstance(x, str) or not x.strip() for x in complaints
    ):
        raise ValueError("Expected a nonempty bare complaint pool")
    return (
        f"{INSTRUCTION}\n\nLearning preference:\n"
        f"{preferences[preference]['preference']}"
    )


def turn_instruction(instruction, complaints, seed, state):
    """Draw reproducibly by problem text and turn, independent of scheduling."""
    turn_idx = sum(turn["role"] == "student" for turn in state.public_history.turns)
    rng = random.Random(json.dumps([seed, state.task, turn_idx], ensure_ascii=False))
    reply = rng.choice(complaints)
    return instruction.replace("{complaint_reply}", json.dumps(reply), 1)


def with_student_instruction(original, protocol):
    """Append only to the current student request, never mutate saved dialogue."""
    instruction = protocol.get("student_system_prompt", "").strip()
    if not instruction.startswith(INSTRUCTION):
        raise ValueError("Expected a compiled prompted_strict student instruction")
    if not protocol["free_chat"]["enabled"]:
        raise ValueError("prompted_strict requires the free-chat student protocol")
    if protocol["personality"]["gate_sample_rate"] != 0 or any(
        axis["personalities"] != ["none"] for axis in protocol["student_axes"]
    ):
        raise ValueError("External preference gating must be disabled")
    from examples.tutor.scripts.student_sim_eval.runner import REPO

    complaints = json.loads(
        (REPO / protocol["personality"]["complaints_path"]).read_text()
    )["bare"]
    seed = int(protocol["seed"])

    def messages(self, state):
        result = [dict(message) for message in original(self, state)]
        if not result or result[-1]["role"] != "user":
            raise ValueError(
                "Expected the current teacher reply in the final user message"
            )
        reminder = turn_instruction(instruction, complaints, seed, state)
        result[-1]["content"] = f"{result[-1]['content']}\n\n{reminder}"
        return result

    return messages
