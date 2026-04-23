# CodeCoach AgentWorkflow

This example ports the old `agentenv-codecoach` loop into an AReaL-native
`AgentWorkflow`. The teacher model is trained through AReaL's OpenAI proxy. The student
model remains external and the evaluator runs locally.

## Dataset

Prepare a HuggingFace dataset on disk from a manifest:

```bash
python3 examples/codecoach/prepare_dataset.py \
  --manifest examples/codecoach/demo/manifest.json \
  --output examples/codecoach/demo_dataset
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

Update `student_base_url` and `student_model` in the config to point at the external
student service.

The example reads shared endpoint parameters from:

- `examples/codecoach/api_params_config.json`

If needed, override:

- `api_params_config_path`
- optional `api_params_key`

The workflow will merge `default` and endpoint-specific entries, including `extra_body`.
