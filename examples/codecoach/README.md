# CodeCoach TutorInf-Style Workflow

This example runs the CodeCoach script-writing task with the TutorInf training pattern:

- multi-turn teacher/student interaction
- one training sample per teacher turn
- CodeCoach rule-based evaluator for task progress and correctness
- optional pairwise reward between the current LoRA teacher and a lagged reference LoRA
- no leak check

The teacher is the trainable AReaL actor. The student is configured through
`auxiliary_model`: `mode: api` calls an external OpenAI-compatible student service, while
`mode: self` uses the actor base model with LoRA disabled.

## Dataset

Prepare a HuggingFace dataset on disk from a manifest:

```bash
python3 examples/codecoach/prepare_dataset.py \
  --manifest /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AgentGym-RL/examples/codecoach/circle_packing_manifest.json \
  --train-ids /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AgentGym-RL/AgentGym-RL/AgentItemId/codecoach_train.json \
  --test-ids /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AgentGym-RL/AgentGym-RL/AgentItemId/codecoach_test.json \
  --output /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/codecoach/circle_packing_dataset
```

Each dataset sample contains:

- `id`
- `task_markdown`
- `initial_code`
- `evaluator_path`
- `target_score`
- `entry_function`
- `eval_timeout_sec`
- `metadata`

## Train

```bash
python3 examples/codecoach/train.py \
  --config examples/codecoach/config.yaml \
  scheduler.type=local
```

For an external student service, update `auxiliary_model.base_url`,
`auxiliary_model.model`, and request parameters in `auxiliary_model.request_params`.
For base-model student calls, set `auxiliary_model.mode=self`.
