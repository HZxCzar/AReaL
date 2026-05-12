# Game Tutor Evaluation Examples

This directory contains teacher-evaluation variants of four game environments:

- `werewolf-tutor`
- `hanabi-tutor`
- `kuhn-tutor`
- `hidden-rule-tutor`

The common objective is to evaluate whether a teacher/tutor model improves a student/player model. The scripts run a baseline condition without teacher guidance and a teacher-guided condition over comparable seeds, then report score improvement.

## Pipeline Overview

Each evaluation has the same high-level loop:

1. Create a game environment.
2. Run a baseline student/player for a fixed turn budget.
3. Run the same student/player with teacher guidance enabled.
4. Track per-turn traces, game scores, and final improvement.
5. Write a JSON report to the configured `output_path`.

The default turn budget is `num_turns: 300`. For multi-turn episodic games, the evaluator keeps starting new episodes until this turn budget is consumed.

The teacher and student endpoints are OpenAI-compatible HTTP endpoints. By default, configs assume:

- Teacher: `http://127.0.0.1:30000/v1`
- Student: `http://127.0.0.1:30001/v1`

Run from the repo root:

```bash
bash examples/werewolf-tutor/start_teacher_eval.sh
bash examples/hanabi-tutor/start_teacher_eval.sh
bash examples/kuhn-tutor/start_teacher_eval.sh
bash examples/hidden-rule-tutor/start_teacher_eval.sh
```

Override common settings:

```bash
bash examples/hanabi-tutor/start_teacher_eval.sh --num-turns 100 --icl-turns 3
bash examples/werewolf-tutor/start_teacher_eval.sh --output /tmp/werewolf_eval.json
```

## In-Context Learning Memory

All evaluators support an ICL memory block. This simulates short-horizon tutoring memory across recent turns and periodically compresses it.

Config:

```yaml
icl_simulation:
  enabled: true
  turns: 5
  compress_every: 5
```

Fields:

- `enabled`: Whether to include memory in prompts.
- `turns`: Number of recent turns kept verbatim. Default is `5`.
- `compress_every`: How often to compress recent memory into a concise state summary. Usually set equal to `turns`.

Use a smaller `turns` value for cheaper/faster evaluation. Use a larger value when the game requires longer strategic memory.

## Shared Config Fields

Most `teacher_eval_config.yaml` files use these fields:

```yaml
num_turns: 300
max_episode_turns: 70
seed: 1
output_path: examples/<env>/outputs/teacher_eval.json
```

- `num_turns`: Total evaluation budget per condition. Baseline and teacher-guided conditions each receive this many turns.
- `max_episode_turns`: Safety cap per episode. If an episode does not terminate, it stops after this many turns.
- `seed`: First seed used for episode generation. Later episodes increment from this seed.
- `output_path`: JSON report location.

Model endpoint config:

```yaml
teacher_model:
  base_url: http://127.0.0.1:30000/v1
  api_key: EMPTY
  model: default
  temperature: 0.2
  top_p: 1.0
  max_tokens: 2048
  timeout: 120

student_model:
  base_url: http://127.0.0.1:30001/v1
  api_key: EMPTY
  model: default
  temperature: 0.7
  top_p: 0.95
  max_tokens: 2048
  timeout: 120
```

- `base_url`: OpenAI-compatible server base URL.
- `api_key`: Bearer token. Use `EMPTY` for local vLLM/SGLang servers that ignore auth.
- `model`: Served model name.
- `temperature`: Sampling temperature. Lower teacher temperature is usually better for consistent guidance.
- `top_p`: Nucleus sampling parameter.
- `max_tokens`: Maximum generation length.
- `timeout`: HTTP timeout in seconds.

## Werewolf Tutor

Path:

```bash
examples/werewolf-tutor
```

Game rule summary:

- Social deduction game with villagers and werewolves.
- Villagers win by eliminating all werewolves.
- Werewolves win by eliminating everyone else.
- Phases include night actions, discussion, day voting, and optional hunter action.
- Roles may include villager, werewolf, witch, foreseer, and hunter.

Evaluation focus:

- Baseline: student sees only its player observation and acts.
- Teacher-guided: teacher sees privileged state, including hidden roles, and gives concise guidance before the student acts.
- Score rewards villager-side success by default: villager win, correct votes, correct witch/hunter actions, and penalties for wrong votes or werewolf win.

Important `env_kwargs`:

```yaml
env_kwargs:
  repeat_rules: true
  num_villagers: 3
  num_werewolves: 1
  num_witches: 0
  num_foreseers: 0
  num_hunters: 0
```

- `repeat_rules`: Include full game rules in every observation. Set `false` to reduce prompt length.
- `num_villagers`: Number of ordinary villager-side players.
- `num_werewolves`: Number of werewolf players.
- `num_witches`: Number of witches. Witch can heal/poison.
- `num_foreseers`: Number of foreseers. Foreseer can inspect roles.
- `num_hunters`: Number of hunters. Hunter can shoot when killed.

Suggested settings:

- Small/easy eval: `num_villagers: 3`, `num_werewolves: 1`, special roles all `0`.
- Harder eval: add witch, foreseer, hunter, and increase `max_episode_turns`.

## Hanabi Tutor

Path:

```bash
examples/hanabi-tutor
```

Game rule summary:

- Cooperative card game.
- Players build colored stacks in rank order.
- A player sees teammates' hands but not their own.
- Legal actions are play, discard, or hint.
- Misplays consume fuse tokens. Completing stacks increases score.

Evaluation focus:

- Baseline: student acts from public/self-hidden observation.
- Teacher-guided: teacher sees privileged hidden hands/global state and gives guidance.
- Score is final Hanabi score from completed stacks.

Important `env_kwargs`:

```yaml
env_kwargs:
  num_players: 2
  scenario: simple
  repeat_rules: true
```

- `num_players`: Number of Hanabi players. Minimum `2`.
- `scenario`: Game size preset.
  - `mini`: smallest setting, useful for smoke tests.
  - `simple`: reduced colors/ranks, good default.
  - `full`: full Hanabi-like setting, more expensive and harder.
- `repeat_rules`: Include full rules in observations. Set `false` to reduce prompt length.

Suggested settings:

- Smoke test: `scenario: mini`, `num_turns: 30`.
- Main eval: `scenario: simple`, `num_turns: 300`.
- Stress eval: `scenario: full`, higher `max_episode_turns`.

## Kuhn Poker Tutor

Path:

```bash
examples/kuhn-tutor
```

Game rule summary:

- Two-player simplified poker.
- Deck contains Jack, Queen, King.
- Each player antes one chip and receives one private card.
- Legal actions are `<PASS>` and `<BET>`.
- If both pass or both bet, higher card wins. If one bets and the other passes, bettor wins.

Evaluation focus:

- Baseline: student chooses poker actions directly.
- Teacher-guided: teacher receives current/privileged state and gives strategy guidance before action.
- Score is return for configured `player_id`.

Important `env_kwargs`:

```yaml
env_kwargs:
  player_id: 0
  built_in_opponent: cfr
  opponent_player: 1
  include_opponent_turn: action
  cfr_iterations: 1000
```

- `player_id`: Which player the evaluation score tracks. Usually `0`.
- `built_in_opponent`: Opponent source.
  - `cfr`: OpenSpiel CFR opponent if dependencies are installed.
  - `mcts`: OpenSpiel MCTS opponent.
  - `random`: Random built-in opponent.
  - `none`: No built-in opponent.
- `opponent_player`: Which player is treated as opponent.
- `include_opponent_turn`: How opponent turns are represented in observations.
- `cfr_iterations`: Number of CFR iterations if CFR policy must be built.

Dependency note:

- If `numpy`/OpenSpiel are unavailable, the tutor evaluator falls back to a tiny local Kuhn implementation for smoke testing.
- For real experiments, install the original Kuhn dependencies and use `built_in_opponent: cfr` or `mcts`.

## Hidden Rule Tutor

Path:

```bash
examples/hidden-rule-tutor
```

Game rule summary:

- Environment samples a hidden Boolean rule over strings.
- Student asks for evidence and tries to infer the rule.
- Tutor knows the true rule and can provide labeled examples or hints.
- Student is evaluated on held-out string classification accuracy and rough rule match.

Evaluation focus:

- Baseline: default deterministic tutor provides labeled examples.
- Teacher-guided: OpenAI-compatible teacher provides examples plus concise hints without revealing the rule verbatim.
- Score improvement measures whether teacher guidance improves held-out accuracy/reward or reduces examples/turns needed.

Important config:

```yaml
rounds: 8
num_turns: 300
seed: 7
examples_per_episode: 48
success_threshold: 0.95
```

- `rounds`: Max tutor-student rounds per episode.
- `num_turns`: Total turn budget. The evaluator converts this into `floor(num_turns / rounds)` episodes.
- `seed`: Rule sampling seed.
- `examples_per_episode`: Number of examples sampled for each hidden rule.
- `success_threshold`: Held-out accuracy threshold used for success rate.

Teacher-only config:

```yaml
teacher_model:
  base_url: http://127.0.0.1:30000/v1
  api_key: EMPTY
  model: default
  temperature: 0.2
  top_p: 1.0
  max_tokens: 512
  timeout: 120
```

Hidden-rule uses the deterministic `MemoryStudent` by default, so it does not need a `student_model` endpoint.

## Report Format

Each evaluator writes a JSON report. Common top-level keys:

- `config`: Effective evaluation config.
- `baseline`: Metrics/traces without teacher guidance.
- `teacher`: Metrics/traces with teacher guidance.
- `improvement`: Difference between teacher and baseline metrics.

For Werewolf/Hanabi/Kuhn:

- `total_score`
- `avg_score`
- `turns`
- `episodes`
- `invalid_actions`
- `traces`

For Hidden Rule:

- `mean_reward`
- `mean_heldout_accuracy`
- `success_rate`
- `avg_examples_used`
- `avg_turns_taken`
- `traces`

## Practical Setup

1. Start the teacher endpoint.
2. Start the student endpoint if the game config uses `student_model`.
3. Edit the relevant `teacher_eval_config.yaml`.
4. Run the bash launcher.
5. Inspect `outputs/teacher_eval.json`.

Example:

```bash
export CUDA_VISIBLE_DEVICES=0
# Start teacher server separately on :30000.
# Start student server separately on :30001.

bash examples/werewolf-tutor/start_teacher_eval.sh --num-turns 300
```

For cheaper smoke tests:

```bash
bash examples/kuhn-tutor/start_teacher_eval.sh --num-turns 20 --icl-turns 2
bash examples/hidden-rule-tutor/start_teacher_eval.sh --num-turns 40
```

## Choosing Settings

- Use low teacher temperature, usually `0.0` to `0.3`.
- Use higher student temperature if you want to test whether guidance stabilizes weaker/variable behavior.
- Increase `num_turns` for more stable metrics.
- Disable `repeat_rules` if context is too long.
- Reduce `icl_simulation.turns` if the model context or runtime cost is too high.
- Increase `max_episode_turns` only when episodes often hit the cap before natural termination.
