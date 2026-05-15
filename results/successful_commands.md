# Successful Commands

Recorded on 2026-05-13.

## Model Servers

Student endpoint:

```bash
source /data/xmy/miniconda3/etc/profile.d/conda.sh
conda activate areal
export CUDA_VISIBLE_DEVICES=1
python examples/tutor/openai_transformers_server.py \
  --model-path /data/xmy/models/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30001 \
  --served-model-name default \
  --dtype bfloat16 \
  > results/logs/transformers_student_30001.log 2>&1
```

Teacher endpoint:

```bash
source /data/xmy/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export CUDA_VISIBLE_DEVICES=2
python -m vllm.entrypoints.openai.api_server \
  --model /data/xmy/models/Qwen3.5-2B \
  --host 127.0.0.1 \
  --port 30000 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --served-model-name default \
  --gpu-memory-utilization 0.35 \
  --max-model-len 8192 \
  --enforce-eager \
  > results/logs/vllm_teacher_30000.log 2>&1
```

## Evaluations

Hidden rule:

Current config for this run uses `rounds: 8` and `num_turns: 64`, which gives 8 episodes.

```bash
source /data/xmy/miniconda3/etc/profile.d/conda.sh
conda activate areal
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
python examples/hidden-rule-tutor/evaluate_teacher.py \
  --config examples/hidden-rule-tutor/teacher_eval_config.yaml \
  --output results/tutoring/hidden_rule.json \
  2>&1 | tee results/logs/hidden_rule_eval.log
```

Hanabi:

```bash
source /data/xmy/miniconda3/etc/profile.d/conda.sh
conda activate areal
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
python examples/hanabi-tutor/evaluate_teacher.py \
  --config examples/hanabi-tutor/teacher_eval_config.yaml \
  --num-turns 10 \
  --icl-turns 2 \
  --output results/tutoring/hanabi.json \
  2>&1 | tee results/logs/hanabi_eval.log
```

Werewolf:

Current config for this run uses `num_turns: 120`, `max_episode_turns: 40`, and forced safe teacher guidance to avoid role/status leaks or direct action instructions.

```bash
source /data/xmy/miniconda3/etc/profile.d/conda.sh
conda activate areal
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
python examples/werewolf-tutor/evaluate_teacher.py \
  --config examples/werewolf-tutor/teacher_eval_config.yaml \
  --output results/tutoring/werewolf.json \
  2>&1 | tee results/logs/werewolf_eval.log
```
