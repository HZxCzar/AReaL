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

Set `teacher_pre.verify=false` to generate one private teacher draft without judging
or retrying it; the rollout continues even if that draft is incorrect.

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

The historical selector draws each problem group independently. To make every rollout
input batch follow the configured student proportions, enable stratification:

```yaml
student_sampling:
  strategy: stratified
```

Quotas follow positive `student_models[].weight` values. Equal weights give equal
counts whenever the batch size is divisible by the number of students; otherwise the
remainder rotates across batches. All `gconfig.n_samples` rollouts of a problem still
share one student, and evaluation remains unchanged.

Repeated online evaluation reports `eval-rollout/solved` and
`eval-rollout/final_correct` as separate test scores. Per-task stability is logged
only for `final_correct`:

- `repeat/final_correct/mean_task_sample_variance` computes the unbiased sample
  variance within each task before averaging over the test set.
- `repeat/final_correct/pairwise_success_jaccard` aggregates success-set
  intersections and unions over every repeat pair.

With one rollout, the two stability metrics are still emitted as degenerate values
(`variance=0`; successful tasks contribute `Jaccard=1`) and should not be interpreted
as measured stability.

When `debug_trace_dir` is configured, compact per-task outcomes are appended under
`eval/repeat_outcomes/*.jsonl`. Each row stores only `task_id`, `lora_version`, and the
binary `final_correct` list, so detailed statistics can be reconstructed without
adding redundant online metrics or duplicating prompts and responses.

`evaluator.average_rollouts` controls repeats for the clean base student prompt.
Additional seen and held-out student instruction prompts use
`evaluator.student_prompt_average_rollouts`; leaving it unset preserves the previous
behavior by reusing `average_rollouts`.

Per-student metrics are emitted automatically under `rollout/student/<name>/...` and
`eval-rollout/student/<name>/...`, including `selected`, `solved`, `pre_solved`,
`reward`, `turns`, and `call_failed`. Existing overall rollout metrics remain unchanged.
The complete two-student example is
`configs/math/july/baseline-overfit-1-generalize-001020-lora-batch128-rebn-nomean-5-mixed-students.yaml`.

For train-only response-level behavior diversity, configure a weighted JSON pool:

```yaml
prompt_pool:
  student_turn_behavior:
    enabled: true
    path: examples/tutor/prompt_pools/student_turn_behaviors_v1.json
```

The behavior file must contain exactly one clean entry with an empty instruction, and
all probabilities must sum to `1.0`. A new behavior is sampled before the initial
student answer and before every later student response. The draw is deterministic for
the same seed, task, and response index. Evaluation always disables turn behaviors,
so repeated test runs keep the clean student prompt and are not affected by behavior
sampling. Training metrics record response counts and fractions under
`rollout/student_turn_behavior/<name>/...`, and debug traces store the selected
behavior for the initial response and every subsequent turn.

To reward whether the teacher follows a sampled student request, enable the optional
request judge together with turn behaviors:

```yaml
prompt_pool:
  student_turn_behavior:
    enabled: true
    path: examples/tutor/prompt_pools/student_request_behaviors_v1.json

reward:
  student_request_judge:
    enabled: true
    weight: 0.5
    behavior_names:
      - ask_question
```

For each listed behavior, the judge compares the actual student reply with the next
teacher reply. Scores `-1/0/1` mean respectively: the student asked but the teacher
did not answer appropriately, the student did not ask a question, or the student
asked and the teacher answered appropriately. The score adds
`-weight/0/+weight` directly to that teacher turn under the
`student_request_fulfillment` reward component. Leaked turns are skipped. Like turn
behavior sampling, this judge is always disabled for evaluation. Metrics are reported
under `rollout/student_request_judge/...`; the selected behavior, aligned student
reply, judge verdict, and applied reward are retained in debug traces.

The ready-to-run MATH example is
`configs/math/0723/2gpu/qwen8b-train-qwen1.7b-student-request-judge-eval3-math-pre-aleak.yaml`.

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
