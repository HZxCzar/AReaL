# Interactive Hidden Rule Induction

This is a small research scaffold for an interactive tutor-student game:

- The environment samples a hidden Boolean rule over strings.
- The tutor can access many positive and negative examples for that rule.
- The student interacts with the tutor over several rounds and tries to infer the rule.
- The project includes a deterministic memory baseline, a vLLM-backed LLM student, and a simple LoRA SFT trainer.

The current code is intentionally minimal and easy to extend toward meta-RL. A natural next step is to replace the SFT trainer with a policy-gradient loop where each episode reward is based on final rule-identification and held-out classification accuracy.

## Layout

```text
hidden_rule_game/
  environment.txt        Linux/server setup notes
  requirements.txt       Python dependencies
  hidden_rule_game/
    rules.py             Rule classes and string generators
    data_generator.py    JSONL dataset generator
    env.py               Tutor-student episode environment
    students.py          Memory and LLM student agents
    inference_engine.py  vLLM wrapper
    game.py              CLI for interactive episodes
    trainer.py           Simple LoRA SFT trainer
```

## Smoke Test

```bash
cd hidden_rule_game
python -m hidden_rule_game.data_generator --out data/examples.jsonl --rules 20 --examples-per-rule 40
python -m hidden_rule_game.game --student memory --rounds 8 --episodes 3
```

## Complete Run Script

```bash
# No GPU required; verifies data generation and interactive evaluation.
QUICK=1 STUDENT=memory bash scripts/run_complete_game.sh

# vLLM base-model game evaluation.
STUDENT=llm MODEL=meta-llama/Llama-3.1-8B-Instruct bash scripts/run_complete_game.sh

# Full data generation, LoRA training, then vLLM LoRA evaluation.
STUDENT=lora MODEL=meta-llama/Llama-3.1-8B-Instruct bash scripts/run_complete_game.sh

# Local downloaded model path also works for training and vLLM.
STUDENT=llm MODEL=/models/Llama-3.1-8B-Instruct bash scripts/run_complete_game.sh

# W&B logging. The launch script defaults to offline W&B logs.
WANDB_PROJECT=hidden-rule-game WANDB_RUN_NAME=local-smoke \
  QUICK=1 STUDENT=memory bash scripts/run_complete_game.sh

# Online W&B logging.
WANDB_MODE=online WANDB_PROJECT=hidden-rule-game WANDB_RUN_NAME=gpu-lora \
  STUDENT=lora MODEL=/models/Llama-3.1-8B-Instruct bash scripts/run_complete_game.sh
```

## Training Setup

Use these four launch-script toggles for the normal two-model setup:

```bash
# 1. Student base model: the model that will receive LoRA training.
export STUDENT_MODEL=/models/student-base

# 2. Teacher model: the tutor model used during the interactive game.
# Leave empty to use the deterministic code tutor.
export TEACHER_MODEL=/models/teacher-instruct

# 3. Student LoRA: optional existing adapter to continue training.
# Leave empty to create a new LoRA in OUT_DIR.
export STUDENT_LORA_PATH=
export OUT_DIR=runs/my_student_lora/outputs/lora-student

# 4. Start training and evaluate.
export STUDENT=lora
export SKIP_TRAIN=0
bash scripts/run_complete_game.sh
```

To evaluate an existing student LoRA without additional training:

```bash
STUDENT=lora SKIP_TRAIN=1 STUDENT_MODEL=/models/student-base \
  STUDENT_LORA_PATH=/models/adapters/student-lora \
  TEACHER_MODEL=/models/teacher-instruct bash scripts/run_complete_game.sh
```

The shell script only holds easy-to-edit toggles. The maintained pipeline logic lives in `hidden_rule_game/run_complete_game.py`, which you can also call directly:

```bash
python -m hidden_rule_game.run_complete_game --student memory --quick
python -m hidden_rule_game.run_complete_game --student lora --model meta-llama/Llama-3.1-8B-Instruct
```

## vLLM Server

The game uses embedded vLLM by default. If you also want an OpenAI-compatible vLLM server for manual probing, the server launcher accepts the same local model path style:

```bash
MODEL=/models/Llama-3.1-8B-Instruct SERVED_MODEL_NAME=student-base \
  bash scripts/start_vllm_server.sh

MODEL=/models/Llama-3.1-8B-Instruct LORA_PATH=runs/my_run/outputs/lora-student \
  SERVED_MODEL_NAME=student-lora bash scripts/start_vllm_server.sh
```

Artifacts are written to `runs/<timestamp>/` by default:

- `data/train.jsonl`
- `data/eval.jsonl`
- `outputs/lora-student/`
- `logs/game_eval_*.jsonl`
- `logs/metrics.json`
- `logs/metrics.txt`
- `run_config.json`

## Data Format

Each JSONL record is one training prompt for rule induction:

```json
{
  "rule_id": "even_vowels",
  "rule_description": "A string is valid if it contains an even number of vowels.",
  "examples": [{"x": "abed", "y": true}],
  "prompt": "...",
  "answer": "..."
}
```

Use `rule_description` only for supervised training or evaluation. The tutor/student game does not reveal it unless the episode has ended.
