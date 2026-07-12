# Tutor Filter 操作手册

这份文档记录三套 tutor task filter 流程：

- 旧 8B self-filter：Student/Judge/Teacher 都调用同一个 Qwen3-8B endpoint
- 当前 1.7B Student + 8B Teacher：Student 调 Qwen3-1.7B non-thinking API，Teacher/Judge 调 Qwen3-8B non-thinking API
- 上一版 8B Student + 4B Teacher：保留用于复现实验

统一使用：

```text
examples/tutor/scripts/filter_task.py
```

不要再使用已经删除的老版本分阶段脚本。

## Filter 到底做什么

`filter_task.py` 对每道题跑两道门：

1. Student pre-solve
   初始 Student 在没有 tutor feedback 的情况下直接做题。如果做对，这题太简单，丢掉。

1. Teacher solve
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

加载远端 8B API key，并清掉当前 shell 中会干扰该 API 的代理变量：

```bash
source .env
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy

curl https://chj8bobdbm9acbj8kkcqgemj8pj8e9gp.openapi-qb-ai.sii.edu.cn/v1/models \
  -H "Authorization: Bearer $INF_API_KEY" \
  -H "x-inspire-inference-key: tutor-filter-qwen8b"
```

启动 4B Teacher 后检查它的 endpoint：

```bash
curl http://127.0.0.1:30004/v1/models \
  -H "Authorization: Bearer EMPTY"
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

## 当前 1.7B Student + 8B Teacher 流程

当前目标是筛出：Qwen3-1.7B non-thinking Student 初始做不出，但
Qwen3-8B non-thinking Teacher 能独立做对的题。Answer Judge 也使用较强的 8B，
不让 1.7B 自己判断答案等价性。

这里的 Student 是官方后训练版 `Qwen/Qwen3-1.7B`，通过
`enable_thinking=false` 关闭 thinking；不是缺少 instruction/chat 能力的
`Qwen3-1.7B-Base`。

两边调用参数都按 Qwen3 non-thinking 推荐值：

- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `min_p=0`
- `enable_thinking=false`
- `max_completion_tokens=2048`：有意与后续 tutor rollout 的 2048 token
  budget 对齐，而不是使用官方通用 API 示例中的 8192
- attempt 默认一次，Student/Teacher 并发默认都是 4

### 1. 在云端下载并部署 1.7B

下载脚本使用统一 Hugging Face cache
`/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/hdd/.cache/huggingface/hub`，
保持标准的 `models--Qwen--Qwen3-1.7B/snapshots/<revision>` 布局：

```bash
bash /inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/hdd/deploy/download-Qwen3-1.7B.sh
```

八卡前台启动；脚本使用 `TP=1, DP=8`，默认端口 `30017`，不使用 nohup：

```bash
bash /inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/hdd/deploy/deploy-Qwen3-1.7B.sh
```

### 2. 开始 filter

Student、Teacher 和 Answer Judge 都通过同一个 router endpoint：

```text
https://chj8bobdbm9acbj8kkcqgemj8pj8e9gp.openapi-qb-ai.sii.edu.cn/v1
```

endpoint、两个 model name、non-thinking 参数和输出路径均已
写入 `filter_task.py` 默认值。加载 `.env` 中的 `INF_API_KEY` 后一行启动：

```bash
uv run python examples/tutor/scripts/filter_task.py
```

默认输出：

```text
examples/tutor/data/math_dataset_qwen1.7b_student_qwen8b_teacher_filter_task
```

筛选语义是：1.7B Student 做对即丢弃；Student 做错后调用 8B Teacher；8B
Teacher 做对才保留。Raw scorer 无法确认答案时，再由 8B Answer Judge 判断。

## 上一版 8B Student + 4B Teacher 流程

这个流程用于当前实验：用已有的 Qwen3-8B non-thinking API 做 Student/Judge，
临时启动 Qwen3-4B-Instruct-2507 做 Teacher。filter 结束后可以关闭 4B 服务；
后续训练中的 4B teacher 由训练 engine 自己运行。

参考 config 仅提供 math scorer、prompt 和 generation 默认值；历史文件名中的
student/teacher 角色不再适用，下面的命令会显式覆盖 endpoint、model 和请求参数：

```text
examples/tutor/configs/math/qwen4&8/qwen8b-qwen4b-remote-overfit-8-generalize-staged-leak.yaml
```

当前调用参数：

- Student/Judge：已有的 Qwen3-8B API，显式 `enable_thinking=false`
- Teacher：Qwen3-4B-Instruct-2507 API
- 两者参数：`max_completion_tokens=2048`，`temperature=0.7`，`top_p=0.8`，`top_k=20`，`min_p=0`，`seed=42`
- attempt：filter 阶段默认一次
- 并发起始值：Student/Judge 4，Teacher 4

在两张空闲 GPU 上启动两个 4B replica：

```bash
DEPLOY_DIR=/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/hdd/deploy
bash "$DEPLOY_DIR/deploy-Qwen3-4B-Instruct-2507-2GPU.sh"
```

脚本固定使用 `GPU 0,1`、`PORT 30004`、`TP=1`、`DP=2`，并将 4B context
length 设为 40960。4B 是直连 SGLang 服务，因此 `--teacher-model` 必须使用
`/v1/models` 返回的完整 model id，不能写 `default`。

### 1. 开始 filter

所有参数都已写入默认值。脚本自动加载 repo 根目录的 `.env`，同时过滤 train
和 test，并覆盖默认输出目录：

```bash
uv run python examples/tutor/scripts/filter_task.py
```

默认输出：

```text
examples/tutor/data/math_dataset_qwen8b_student_qwen4b_teacher_filter_task
```

筛选语义是：Student 做对即丢弃；Student 做错后调用 Teacher；Teacher 做对才保留。

### 2. 跑真实 multi-turn estimate

这一步会跑完整训练式 multi-turn 流程，记录每题 3 次 attempt 里成功几次，以及 turn 数、reward、leak 等指标。

Train split：

```bash
uv run python examples/tutor/scripts/estimate_multiturn_difficulty.py \
  --config 'examples/tutor/configs/math/qwen4&8/qwen8b-qwen4b-remote-overfit-8-generalize-staged-leak.yaml' \
  --input examples/tutor/data/math_dataset_qwen8b_student_qwen4b_teacher_filter_task \
  --output examples/tutor/data/math_dataset_qwen8b_student_qwen4b_teacher_filter_task_multiturn_train \
  --report examples/tutor/report/math_dataset_qwen8b_student_qwen4b_teacher_filter_task_multiturn_train_report.json \
  --csv examples/tutor/report/math_dataset_qwen8b_student_qwen4b_teacher_filter_task_multiturn_train.csv \
  --splits train \
  --attempts 3 \
  --tutor-base-url http://127.0.0.1:30004/v1 \
  --tutor-model /inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tutor-api-key EMPTY \
  --aux-base-url https://chj8bobdbm9acbj8kkcqgemj8pj8e9gp.openapi-qb-ai.sii.edu.cn/v1 \
  --aux-model qwen3-8b \
  --tutor-request-params '{"seed":42,"extra_body":{"top_k":20,"min_p":0}}' \
  --aux-request-params '{"seed":42,"extra_headers":{"x-inspire-inference-key":"tutor-filter-qwen8b"},"extra_body":{"top_k":20,"min_p":0}}' \
  --thinking off \
  --partial-every 10 \
  --overwrite
```

Test split：

```bash
uv run python examples/tutor/scripts/estimate_multiturn_difficulty.py \
  --config 'examples/tutor/configs/math/qwen4&8/qwen8b-qwen4b-remote-overfit-8-generalize-staged-leak.yaml' \
  --input examples/tutor/data/math_dataset_qwen8b_student_qwen4b_teacher_filter_task \
  --output examples/tutor/data/math_dataset_qwen8b_student_qwen4b_teacher_filter_task_multiturn_test \
  --report examples/tutor/report/math_dataset_qwen8b_student_qwen4b_teacher_filter_task_multiturn_test_report.json \
  --csv examples/tutor/report/math_dataset_qwen8b_student_qwen4b_teacher_filter_task_multiturn_test.csv \
  --splits test \
  --attempts 3 \
  --tutor-base-url http://127.0.0.1:30004/v1 \
  --tutor-model /inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tutor-api-key EMPTY \
  --aux-base-url https://chj8bobdbm9acbj8kkcqgemj8pj8e9gp.openapi-qb-ai.sii.edu.cn/v1 \
  --aux-model qwen3-8b \
  --tutor-request-params '{"seed":42,"extra_body":{"top_k":20,"min_p":0}}' \
  --aux-request-params '{"seed":42,"extra_headers":{"x-inspire-inference-key":"tutor-filter-qwen8b"},"extra_body":{"top_k":20,"min_p":0}}' \
  --thinking off \
  --partial-every 10 \
  --overwrite
```

## 后续处理

filter 完之后，用合并后的 dataset 路径填到对应训练 config 的 `train_dataset.path` 和 `valid_dataset.path`。

当前 4B+8B config 里的 dataset path 是故意先空着的，因为最终数据集要等 filter 和 estimate 结果出来之后再定。

不要在这一轮 filter 里加 student generalization variants。泛化变种是后面人工设计变种题、再单独评估的时候用的。
