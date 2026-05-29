from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from game_tutor_teacher_training import run_game_teacher_training


if __name__ == "__main__":
    run_game_teacher_training(sys.argv[1:], game="werewolf")
