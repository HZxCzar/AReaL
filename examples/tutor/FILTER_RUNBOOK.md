# Tutor Filter 操作手册

这份文档记录两套仍然支持的 tutor task filter 流程：

- 旧 8B self-filter：Student/Judge/Teacher 都调用同一个 Qwen3-8B endpoint
- 当前 4B Student + 8B Teacher：Student/Judge 调远端 Qwen3-4B-Instruct，Teacher 调本地 Qwen3-8B

统一使用：

```text
examples/tutor/scripts/filter_task.py
```

不要再使用已经删除的老版本分阶段脚本。

## Filter 到底做什么

`filter_task.py` 对每道题跑两道门：

1. Student pre-solve
   初始 Student 在没有 tutor feedback 的情况下直接做题。如果做对，这题太简单，丢掉。

2. Teacher solve
   Teacher 独立解同一道题。如果 Teacher 能做对，这题保留；Teacher 也做不对就丢掉。

评分逻辑跟训练里一致：先走 raw answer scorer；raw 判错后，如果
`answer_judge_enabled=true`，再走 LLM judge fallback。

attempt 逻辑也保留原来的做法：

- 默认 `student_attempts=1`，`teacher_attempts=1`
- 如果 Student 有多次 attempt，只要任意一次做对，就算 pre-solved，这题丢掉
- 如果 Teacher 有多次 attempt，只要任意一次做对，就算 teacher-solved，这题保留
- 只有所有 attempt 都失败，才算该阶段失败

脚本会输出两个东西：

- 过滤后的 HuggingFace dataset
- `*_filter_task_report.json` 报告，里面有最终解析出来的 endpoint、model、attempt 次数和 request 参数

每次跑完先看 report，再进入下一步。

## 通用准备

在 repo 根目录执行：

```bash
cd /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL
source .venv/bin/activate
```

如果要用远端 4B API，还需要加载 `.env`，里面提供 `INF_API_KEY`：

```bash
source .env
```

检查本地 8B endpoint：

```bash
curl http://127.0.0.1:30008/v1/models \
  -H "Authorization: Bearer EMPTY"
```

检查远端 4B endpoint 时，先把 `QWEN4B_BASE_URL` 设成当前 config 里的
`auxiliary_model.base_url`：

```bash
curl "$QWEN4B_BASE_URL/models" \
  -H "Authorization: Bearer $INF_API_KEY"
```

## 旧 8B Self-Filter 流程

这个流程用于复现原来的 self-filter 思路：Student/Judge/Teacher 都用同一个 8B。

注意说人话版本：这个离线 filter 脚本不会启动训练 engine。这里的 self-filter
意思是“这几个角色都打同一个 OpenAI-compatible 8B API endpoint”，通常就是：

```text
http://127.0.0.1:30008/v1
```

参考 config：

```text
examples/tutor/configs/math/staged_leak/baseline-overfit-8-generalize-staged-leak.yaml
```

这套旧 config 的关键默认值是：

- Student/Judge：同一个 8B endpoint，non-thinking，`max_tokens=1024`，`temperature=0.0`
- Teacher：同一个 8B endpoint，按 config 开 thinking，`max_new_tokens=16384`，`temperature=0.6`，`top_p=1.0`
- attempt：默认一次

### 1. 跑 train filter

```bash
uv run python examples/tutor/scripts/filter_task.py \
  --config examples/tutor/configs/math/staged_leak/baseline-overfit-8-generalize-staged-leak.yaml \
  --input examples/tutor/data/math_dataset \
  --output examples/tutor/data/math_dataset_qwen8b_self_filter_train \
  --splits train \
  --base-url http://127.0.0.1:30008/v1 \
  --model default \
  --api-key EMPTY \
  --student-attempts 1 \
  --teacher-attempts 1 \
  --overwrite
```

### 2. 跑 test filter

可以和 train 放在两个 tmux window 里并行跑，但 output 目录必须不同。

```bash
uv run python examples/tutor/scripts/filter_task.py \
  --config examples/tutor/configs/math/staged_leak/baseline-overfit-8-generalize-staged-leak.yaml \
  --input examples/tutor/data/math_dataset \
  --output examples/tutor/data/math_dataset_qwen8b_self_filter_test \
  --splits test \
  --base-url http://127.0.0.1:30008/v1 \
  --model default \
  --api-key EMPTY \
  --student-attempts 1 \
  --teacher-attempts 1 \
  --overwrite
```

### 3. 合并 train/test

```bash
uv run python - <<'PY'
from datasets import DatasetDict, load_from_disk

train_ds = load_from_disk("examples/tutor/data/math_dataset_qwen8b_self_filter_train")
test_ds = load_from_disk("examples/tutor/data/math_dataset_qwen8b_self_filter_test")

DatasetDict(
    {
        "train": train_ds["train"],
        "test": test_ds["test"],
    }
).save_to_disk("examples/tutor/data/math_dataset_qwen8b_self_filter")
PY
```

### 4. 跑真实 multi-turn estimate

这一步不是 filter 本身，而是后续难度估计：严格跑完整 multi-turn tutor 流程，统计三次尝试里成功几次。

```bash
uv run python examples/tutor/scripts/estimate_multiturn_difficulty.py \
  --config examples/tutor/configs/math/staged_leak/baseline-overfit-8-generalize-staged-leak.yaml \
  --input examples/tutor/data/math_dataset_qwen8b_self_filter \
  --output examples/tutor/data/math_dataset_qwen8b_self_filter_multiturn \
  --report examples/tutor/report/math_dataset_qwen8b_self_filter_multiturn_report.json \
  --csv examples/tutor/report/math_dataset_qwen8b_self_filter_multiturn.csv \
  --splits train test \
  --attempts 3 \
  --base-url http://127.0.0.1:30008/v1 \
  --model default \
  --api-key EMPTY \
  --partial-every 10 \
  --overwrite
```

## 当前 4B Student + 8B Teacher 流程

这个流程用于当前实验：用更轻的 4B 做 Student/Judge，用 8B non-thinking 做 Teacher。

参考 config：

```text
examples/tutor/configs/math/staged_leak/qwen8b-nonthinking-qwen4b-remote-overfit-8-generalize-staged-leak.yaml
```

当前调用参数：

- Student/Judge：远端 Qwen3-4B-Instruct，non-thinking
- Student/Judge 参数：`max_tokens=2048`，`temperature=0.7`，`top_p=0.8`，`top_k=20`，`min_p=0`，`seed=42`
- Teacher：本地 Qwen3-8B，`http://127.0.0.1:30008/v1`，non-thinking
- Teacher 参数：`max_new_tokens=2048`，`max_tokens=24576`，`temperature=0.7`，`top_p=0.8`，`top_k=20`
- attempt：filter 阶段默认一次

远端 4B 的 API key 来自 `INF_API_KEY`，所以先 `source .env`。

### 1. 跑 train filter

```bash
uv run python examples/tutor/scripts/filter_task.py \
  --config examples/tutor/configs/math/staged_leak/qwen8b-nonthinking-qwen4b-remote-overfit-8-generalize-staged-leak.yaml \
  --input examples/tutor/data/math_dataset \
  --output examples/tutor/data/math_dataset_qwen4b_qwen8b_filter_train \
  --splits train \
  --teacher-base-url http://127.0.0.1:30008/v1 \
  --teacher-model default \
  --teacher-api-key EMPTY \
  --student-request-params '{"extra_headers":{"x-inspire-inference-key":"tutor-filter-train-qwen4b"}}' \
  --thinking off \
  --student-attempts 1 \
  --teacher-attempts 1 \
  --overwrite
```

### 2. 跑 test filter

```bash
uv run python examples/tutor/scripts/filter_task.py \
  --config examples/tutor/configs/math/staged_leak/qwen8b-nonthinking-qwen4b-remote-overfit-8-generalize-staged-leak.yaml \
  --input examples/tutor/data/math_dataset \
  --output examples/tutor/data/math_dataset_qwen4b_qwen8b_filter_test \
  --splits test \
  --teacher-base-url http://127.0.0.1:30008/v1 \
  --teacher-model default \
  --teacher-api-key EMPTY \
  --student-request-params '{"extra_headers":{"x-inspire-inference-key":"tutor-filter-test-qwen4b"}}' \
  --thinking off \
  --student-attempts 1 \
  --teacher-attempts 1 \
  --overwrite
```

`x-inspire-inference-key` 不是模型参数。它只是远端 4B 网关的路由标签。相同标签的请求倾向于落到同一个后端 worker；train/test 用不同标签，是为了让两个并行 job 更容易分开跑。

### 3. 合并 train/test

```bash
uv run python - <<'PY'
from datasets import DatasetDict, load_from_disk

train_ds = load_from_disk("examples/tutor/data/math_dataset_qwen4b_qwen8b_filter_train")
test_ds = load_from_disk("examples/tutor/data/math_dataset_qwen4b_qwen8b_filter_test")

DatasetDict(
    {
        "train": train_ds["train"],
        "test": test_ds["test"],
    }
).save_to_disk("examples/tutor/data/math_dataset_qwen4b_qwen8b_filter_task")
PY
```

### 4. 跑真实 multi-turn estimate

这一步会跑完整训练式 multi-turn 流程，记录每题 3 次 attempt 里成功几次，以及 turn 数、reward、leak 等指标。

Train split：

```bash
uv run python examples/tutor/scripts/estimate_multiturn_difficulty.py \
  --config examples/tutor/configs/math/staged_leak/qwen8b-nonthinking-qwen4b-remote-overfit-8-generalize-staged-leak.yaml \
  --input examples/tutor/data/math_dataset_qwen4b_qwen8b_filter_task \
  --output examples/tutor/data/math_dataset_qwen4b_qwen8b_filter_task_multiturn_train \
  --report examples/tutor/report/math_dataset_qwen4b_qwen8b_filter_task_multiturn_train_report.json \
  --csv examples/tutor/report/math_dataset_qwen4b_qwen8b_filter_task_multiturn_train.csv \
  --splits train \
  --attempts 3 \
  --tutor-base-url http://127.0.0.1:30008/v1 \
  --tutor-model default \
  --tutor-api-key EMPTY \
  --aux-request-params '{"extra_headers":{"x-inspire-inference-key":"tutor-estimate-train-qwen4b"}}' \
  --thinking off \
  --partial-every 10 \
  --overwrite
```

Test split：

```bash
uv run python examples/tutor/scripts/estimate_multiturn_difficulty.py \
  --config examples/tutor/configs/math/staged_leak/qwen8b-nonthinking-qwen4b-remote-overfit-8-generalize-staged-leak.yaml \
  --input examples/tutor/data/math_dataset_qwen4b_qwen8b_filter_task \
  --output examples/tutor/data/math_dataset_qwen4b_qwen8b_filter_task_multiturn_test \
  --report examples/tutor/report/math_dataset_qwen4b_qwen8b_filter_task_multiturn_test_report.json \
  --csv examples/tutor/report/math_dataset_qwen4b_qwen8b_filter_task_multiturn_test.csv \
  --splits test \
  --attempts 3 \
  --tutor-base-url http://127.0.0.1:30008/v1 \
  --tutor-model default \
  --tutor-api-key EMPTY \
  --aux-request-params '{"extra_headers":{"x-inspire-inference-key":"tutor-estimate-test-qwen4b"}}' \
  --thinking off \
  --partial-every 10 \
  --overwrite
```

## 后续处理

filter 完之后，用合并后的 dataset 路径填到对应训练 config 的 `train_dataset.path` 和 `valid_dataset.path`。

当前 4B+8B config 里的 dataset path 是故意先空着的，因为最终数据集要等 filter 和 estimate 结果出来之后再定。

不要在这一轮 filter 里加 student generalization variants。泛化变种是后面人工设计变种题、再单独评估的时候用的。

