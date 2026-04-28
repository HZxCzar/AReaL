# Tutor AgentWorkflow

This example ports the old `agentenv-tutor` logic into an AReaL-native `AgentWorkflow`.
Only the teacher model is trained. The student, leak checker, and transfer generator are
external auxiliary calls.

## Dataset

Prepare a HuggingFace dataset on disk from a manifest:

```bash
python3 examples/tutor/prepare_dataset.py \
  --manifest /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AgentGym-RL/examples/tutor/aime_manifest.json \
  --train-ids /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AgentGym-RL/AgentGym-RL/AgentItemId/tutor_train.json \
  --test-ids /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AgentGym-RL/AgentGym-RL/AgentItemId/tutor_test.json \
  --output /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/aime_dataset
```

Each dataset sample contains:

- `id`
- `task`
- `ground_truth`
- `metadata`

If `--train-ids` and `--test-ids` are provided, the split follows the old
`AgentItemId/tutor_train.json` and `AgentItemId/tutor_test.json` files instead of using
the last `N` samples as test data.

## Filter pre-solved tasks

Before training, optionally remove train rows that the auxiliary student can solve
without tutor feedback:

```bash
python3 examples/tutor/filter_pre_solved.py \
  --config examples/tutor/config.yaml \
  --input /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/aime_dataset \
  --output /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/aime_dataset_no_pre_solve \
  --splits train \
  --attempts 1 \
  --overwrite
```

The output dataset keeps unselected splits unchanged, so both `train_dataset.path` and
`valid_dataset.path` can point to the filtered dataset directory. Increase `--attempts`
to drop a row if any sampled initial student attempt solves it. The script also writes a
`*_pre_solve_filter_report.json` file with kept, dropped, and error ids.

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
