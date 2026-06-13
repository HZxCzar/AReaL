# Tutor AgentWorkflow

This example ports the old `agentenv-tutor` logic into an AReaL-native `AgentWorkflow`.
Only the teacher model is trained. The student and leak checker are external
auxiliary calls; public history is maintained locally as visible student/tutor
transcript text.

## Dataset

Prepare a HuggingFace dataset on disk from an AIME manifest:

```bash
python3 examples/tutor/prepare_dataset.py \
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
python3 examples/tutor/prepare_dataset.py \
  --format math \
  --train-jsonl examples/tutor/raw_data/math_train.jsonl \
  --test-jsonl examples/tutor/raw_data/math_test.jsonl \
  --output examples/tutor/data/math_dataset
```

Then point training/eval at that saved dataset and switch the answer scorer:

```bash
python3 examples/tutor/train.py \
  --config examples/tutor/config.yaml \
  answer_scorer=math \
  train_dataset.path=examples/tutor/data/math_dataset \
  valid_dataset.path=examples/tutor/data/math_dataset \
  scheduler.type=local
```

`answer_scorer=aime` uses the AIME exact-match scorer, while `answer_scorer=math`
uses the lm-eval/Hendrycks MATH boxed-answer extraction and string-normalized exact
match. Both scorers extract the student's final answer from the last `\boxed{...}` or
`\fbox{...}` in the student response.
Pass the same `answer_scorer=math` and dataset path overrides to
`filter_pre_solved.py`, `manual_tutor.py`, or `demo_run.py` when using MATH rows.

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

For MATH rows, use the MATH dataset path and scorer. The filter calls the auxiliary
student with `auxiliary_model` settings from the resolved config (`base_url`, `model`,
`max_tokens`, `temperature`, `top_p`, timeout, API params, and concurrency):

```bash
python3 examples/tutor/filter_pre_solved.py \
  --config examples/tutor/config.yaml \
  --input examples/tutor/data/math_dataset \
  --output examples/tutor/data/math_dataset_no_pre_solve \
  --splits train \
  --attempts 1 \
  --overwrite \
  answer_scorer=math \
  train_dataset.path=examples/tutor/data/math_dataset \
  valid_dataset.path=examples/tutor/data/math_dataset
```

The output dataset keeps unselected splits unchanged, so both `train_dataset.path` and
`valid_dataset.path` can point to the filtered dataset directory. Increase `--attempts`
to drop a row if any sampled initial student attempt solves it. The script also writes a
`*_pre_solve_filter_report.json` file with kept, dropped, and error ids, plus the
resolved answer scorer and auxiliary request config used for the run.

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

python examples/tutor/scripts/filter_with_llm_judge.py \
    --config examples/tutor/configs/math/baseline.yaml \
    --input examples/tutor/data/math_dataset \
    --output examples/tutor/data/math_dataset_8b_llm_judge_filter \
    --splits train test \
    --base-url http://127.0.0.1:30008/v1 \
    --model default \
    --overwrite
  
 python examples/tutor/scripts/filter_with_llm_judge.py \
    --config examples/tutor/configs/math/baseline.yaml \
    --input examples/tutor/data/math_dataset \
    --output examples/tutor/data/math_dataset_train_llm_judge \
    --report examples/tutor/report/math_dataset_train_llm_judge_report.json \
    --splits train \
    --base-url http://127.0.0.1:30008/v1 \
    --model default \
    --overwrite

python examples/tutor/scripts/filter_with_llm_judge.py \
    --config examples/tutor/configs/math/baseline.yaml \
    --input examples/tutor/data/math_dataset \
    --output examples/tutor/data/math_dataset_test_llm_judge \
    --report examples/tutor/report/math_dataset_test_llm_judge_report.json \
    --splits test \
    --base-url http://127.0.0.1:30008/v1 \
    --model default \
    --overwrite