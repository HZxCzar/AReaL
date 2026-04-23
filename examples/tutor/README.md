# Tutor AgentWorkflow

This example ports the old `agentenv-tutor` logic into an AReaL-native `AgentWorkflow`.
Only the teacher model is trained. The student, leak checker, and transfer generator are
external auxiliary calls.

## Dataset

Prepare a HuggingFace dataset on disk from a manifest:

```bash
python3 examples/tutor/prepare_dataset.py \
  --manifest examples/tutor/demo/manifest.json \
  --output examples/tutor/demo_dataset
```

Each dataset sample contains:

- `id`
- `task`
- `ground_truth`
- `metadata`

## Train

```bash
python3 examples/tutor/train.py \
  --config examples/tutor/config.yaml \
  scheduler.type=local
```

Update `aux_base_url` and `aux_model` in the config to point at the external student /
leak-check / generator model service.

The example reads shared endpoint parameters from:

- `examples/tutor/api_params_config.json`

If needed, override:

- `api_params_config_path`
- optional `api_params_key`

The resolution order matches the old tutor filter script:
`default` -> inferred or explicit endpoint key -> workflow fields.
