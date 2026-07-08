# Tutor AgentWorkflow

This example ports the old `agentenv-tutor` logic into an AReaL-native `AgentWorkflow`.
Only the teacher model is trained. The student and leak checker are external
auxiliary calls; public history is maintained locally as visible student/tutor
transcript text.

## Dataset

Prepare a HuggingFace dataset on disk from an AIME manifest:

```bash
python3 examples/tutor/scripts/prepare_dataset.py \
  --format aime \
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

To convert the checked-in MATH JSONL files under `examples/tutor/raw_data/`:

```bash
python3 examples/tutor/scripts/prepare_dataset.py \
  --format math \
  --train-jsonl examples/tutor/raw_data/math_train.jsonl \
  --test-jsonl examples/tutor/raw_data/math_test.jsonl \
  --output examples/tutor/data/math_dataset
```

To download and convert Polaris:

```bash
python3 examples/tutor/scripts/prepare_dataset.py \
  --format polaris \
  --output examples/tutor/data/polaris_dataset
```

Then point training/eval at that saved dataset and switch the dataset type:

```bash
python3 examples/tutor/train.py \
  --config examples/tutor/config.yaml \
  dataset_type=math \
  train_dataset.path=examples/tutor/data/math_dataset \
  valid_dataset.path=examples/tutor/data/math_dataset \
  scheduler.type=local
```

`answer_scorer=auto` follows `dataset_type`; explicit scorers must match the dataset
type. `dataset_type=polaris` uses the Polaris boxed-answer rule judge and is
incompatible with leak checks, so set `leak_handling_mode=disabled` for Polaris runs.
Pass the same `dataset_type` and dataset path overrides to
`filter_task.py`, `manual_tutor.py`, or `demo_run.py` when using MATH rows.

## Filter Tasks

Before training, regenerate the train/test task filter with the configured LLM answer
judge fallback. The filter first drops rows that the external student can solve without
tutor feedback, then keeps only rows that the external teacher can solve independently.
For the full old 8B self-filter and current 4B+8B runbooks, see
[`FILTER_RUNBOOK.md`](FILTER_RUNBOOK.md).
Use explicit student and teacher endpoints when the two models differ:

```bash
python3 examples/tutor/scripts/filter_task.py \
  --config examples/tutor/configs/math/staged_leak/qwen8b-nonthinking-qwen4b-remote-overfit-8-generalize-staged-leak.yaml \
  --input examples/tutor/data/math_dataset \
  --output examples/tutor/data/math_dataset_filter_task \
  --splits train test \
  --teacher-base-url http://127.0.0.1:30008/v1 \
  --teacher-model default \
  --thinking off \
  --student-attempts 1 \
  --teacher-attempts 1 \
  --overwrite
```

Student defaults come from `auxiliary_model` in the config. Teacher defaults come from
`gconfig` plus `--teacher-base-url`. Use `--student-*` or `--teacher-*` flags only for
intentional role-specific overrides. The script writes a `*_filter_task_report.json`
report with the resolved endpoints and request parameters.

After the LLM judge filter, prune known bad task ids if needed:

```bash
python3 examples/tutor/scripts/filter_wrong_tasks.py \
  --input examples/tutor/data/math_dataset_filter_task \
  --output examples/tutor/data/math_dataset_filter_task_pruned \
  --splits train \
  --overwrite
```

The output dataset keeps unselected splits unchanged, so both `train_dataset.path` and
`valid_dataset.path` can point to the filtered dataset directory.

## Manual human tutor probe

Use the same auxiliary student and filtered training rows, but type tutor feedback by
hand:

```bash
python3 examples/tutor/manual_tutor.py \
  --config examples/tutor/config.yaml \
  --dataset examples/tutor/aime_dataset_no_pre_solve \
  --random
```

The script prints the task, the student's initial answer, and the configured judge
result. Each tutor turn is checked for answer leakage by default. Leak detection only
affects reward accounting: leaked tutor turns still remain visible to the student, but
receive the configured leak penalty. Pass `--skip-leak-check` only when you want to
disable that penalty signal.

## Train

```bash
python3 examples/tutor/train.py \
  --config examples/tutor/config.yaml \
  scheduler.type=local
```

Update `aux_base_url` and `aux_model` in the config to point at the external student
and leak-check service.

Set auxiliary API call parameters directly under `auxiliary_model` in YAML. Common
parameters use dedicated fields such as `max_tokens`, `temperature`, and `top_p`; put
additional OpenAI request kwargs under `request_params`, including backend-specific
`extra_body` values.

```bash
cd /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL

python examples/tutor/scripts/estimate_multiturn_difficulty.py   --config /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/configs/math/baseline-overfit-1.yaml   --input /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/data/math_dataset_train_llm_judge   --output /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/data/math_dataset_train_llm_judge_multiturn_difficulty   --report /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/report/math_dataset_train_llm_judge_multiturn_difficulty_report.json   --csv /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/report/math_dataset_train_llm_judge_multiturn_difficulty.csv   --splits train   --base-url http://127.0.0.1:30008/v1   --model default   --attempts 3   --partial-every 10   --overwrite

python examples/tutor/scripts/estimate_multiturn_difficulty.py   --config /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/configs/math/baseline-overfit-1.yaml   --input /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/data/math_dataset_test_llm_judge   --output /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/data/math_dataset_test_llm_judge_multiturn_difficulty   --report /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/report/math_dataset_test_llm_judge_multiturn_difficulty_report.json   --csv /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/report/math_dataset_test_llm_judge_multiturn_difficulty.csv   --splits test   --base-url http://127.0.0.1:30008/v1   --model default   --attempts 3   --partial-every 10   --overwrite
```
