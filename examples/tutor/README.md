# Tutor AgentWorkflow

This example ports the old `agentenv-tutor` logic into an AReaL-native `AgentWorkflow`.
Only the teacher model is trained. Students can be sampled from an external API pool,
while leak and answer judges remain fixed auxiliary calls; public history is maintained
locally as visible student/tutor transcript text.

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
Pass the same `dataset_type` and dataset path overrides to `filter_task.py`,
`manual_tutor.py`, or `demo_run.py` when using MATH rows.

## Filter Tasks

Before training, regenerate the train/test task filter with the configured LLM answer
judge fallback. The filter first drops rows that the external student can solve without
tutor feedback, then keeps only rows that the external teacher can solve independently.
For the full old 8B self-filter and current 4B+8B runbooks, see
[`FILTER_RUNBOOK.md`](FILTER_RUNBOOK.md). Use explicit student and teacher endpoints
when the two models differ:

```bash
uv run python examples/tutor/scripts/filter_task.py
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

`auxiliary_model` is the fixed leak/answer judge. To train against a mix of API
students, configure any number of entries under `student_models`:

```yaml
auxiliary_model:
  mode: api
  base_url: https://your-openai-compatible-endpoint.example/v1
  model: qwen3-8b
  api_key: ${oc.env:INF_API_KEY}

student_models:
  - name: qwen3-4b
    weight: 0.5
    base_url: ${auxiliary_model.base_url}
    model: qwen3-4b
    api_key: ${auxiliary_model.api_key}
    max_tokens: 2048
    temperature: 0.7
    top_p: 0.8
    request_params:
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
  - name: qwen3-8b
    weight: 0.5
    base_url: ${auxiliary_model.base_url}
    model: qwen3-8b
    api_key: ${auxiliary_model.api_key}
    max_tokens: 1024
    temperature: 0.0
    top_p: 1.0
    request_params:
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
```

Training samples one student per episode according to `weight` and keeps that student
for the initial answer, all tutor turns, and generalization probes. Evaluation ignores
the weights and runs the complete validation set for every
configured student; `evaluator.average_rollouts` is applied independently to each
student. An empty `student_models` list preserves the legacy behavior where
`auxiliary_model` is also the student.

Repeated evaluation also reports per-task binary stability for both tutoring
`solved` and overall `final_correct` under `eval-rollout/repeat/<outcome>/...`.
Metrics include `variance`, `std`, pairwise `agreement`/`disagreement`, `all_equal`,
`any_success`, `all_success`, the strict all-repeat `success_set_jaccard`, and
`pairwise_success_set_jaccard` over all repeat pairs. With one rollout these remain
defined (`variance=0`, `agreement=1`). `variance` is computed within each task before
being averaged over the test set; the Jaccard metrics are computed from globally
aggregated intersections and unions.

`evaluator.average_rollouts` controls repeats for the clean base student prompt.
Additional seen and held-out student instruction prompts use
`evaluator.student_prompt_average_rollouts`; leaving it unset preserves the previous
behavior by reusing `average_rollouts`.

Per-student metrics are emitted automatically under `rollout/student/<name>/...` and
`eval-rollout/student/<name>/...`, including `selected`, `solved`, `pre_solved`,
`reward`, `turns`, and `call_failed`. Existing overall rollout metrics remain unchanged.
The complete two-student example is
`configs/math/july/baseline-overfit-1-generalize-001020-lora-batch128-rebn-nomean-5-mixed-students.yaml`.

Set common API parameters through their dedicated fields and place backend-specific
OpenAI request values under `request_params.extra_body`.

```bash
bash examples/tutor/run_official.sh examples/tutor/configs/math/july/baseline-overfit-1-generalize-001020-lora-batch128-rebn-nomean-5-mixed-students.yaml
```

```bash
cd /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL

python examples/tutor/scripts/estimate_multiturn_difficulty.py   --config /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/configs/math/baseline-overfit-1.yaml   --input /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/data/math_dataset_train_llm_judge   --output /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/data/math_dataset_train_llm_judge_multiturn_difficulty   --report /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/report/math_dataset_train_llm_judge_multiturn_difficulty_report.json   --csv /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/report/math_dataset_train_llm_judge_multiturn_difficulty.csv   --splits train   --base-url http://127.0.0.1:30008/v1   --model default   --attempts 3   --partial-every 10   --overwrite

python examples/tutor/scripts/estimate_multiturn_difficulty.py   --config /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/configs/math/baseline-overfit-1.yaml   --input /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/data/math_dataset_test_llm_judge   --output /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/data/math_dataset_test_llm_judge_multiturn_difficulty   --report /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/report/math_dataset_test_llm_judge_multiturn_difficulty_report.json   --csv /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/report/math_dataset_test_llm_judge_multiturn_difficulty.csv   --splits test   --base-url http://127.0.0.1:30008/v1   --model default   --attempts 3   --partial-every 10   --overwrite
```
