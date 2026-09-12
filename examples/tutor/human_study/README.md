# Human study: standardized next-teacher-response generation

独立增量工具，不导入或修改现有 benchmark runner、评分器、训练配置。
这里的 eval **只生成，不打分**；不会启动学生模拟器或 subagent judge。

## Protocol v1

- 数据：MathTutorBench 的 standard + hard 原始 JSON，全量处理，不抽样、不去重。
- 原始记录的最后一条教师发言是参考回复，**不传给模型或评审**；参考解答也不传入。
- 原题目与其余历史文字原样保留（包括空格、换行与原有错误）。
- 输入：一个 user message，标准 Scaffolding 教师指令 + 题目 + 历史，末尾严格为
  `Teacher (maximum two sentences): `（含末尾空格）。普通版和 Hard 均保留两句要求。
- 无 presolve、训练格式或 XML 输出要求。保留原始回复，不通过停止词截断已生成的
  文本；人工评审不采用 benchmark 的自动评分后处理。生成输入必须与标准 benchmark 一致。
- 默认 temperature=0、top_p=1、seed=42、max_tokens=2048，Qwen native thinking 关闭。
  参数和提示在 `configs/study.yaml` 中对所有模型共用；改动后使用新结果目录。
- 一条 trajectory = 原始历史 + 一条新教师回复。不是完整多轮 rollout。
- 原始输出不清洗、不截断、不因格式异常重生成。空回复、长度截断、训练标签单独标记。
  `reasoning_content` 保存在完整 API 响应中，但不冒充学生可见的回复或导出给评审。
- 第一版只做原始续写。追加学生需求的实验需另外明确构造规则并准备新数据版本；
  不会在本版本隐式添加 preference。

此前误删两句要求的已生成结果属于旧的非标准提示实验，不能作为标准 Scaffolding
结果使用。本次恢复只修正提示，不改写历史请求或回复。再次采集必须使用新的输出目录；
eval 的配置/代码指纹检查会拒绝把新提示混入旧运行。

## 1. Prepare ALL data (no API calls)

以下命令从仓库根目录运行；只依赖已有环境的 Python + PyYAML。

```bash
.venv/bin/python -m examples.tutor.human_study.prepare_data \
  --source-dir examples/math_tutor_bench/.runtime/upstream/datasets \
  --output examples/tutor/human_study/data/ready-v1.json
```

只执行两条固定规则，没有人工审核、待定状态或规则 3/4：

1. `problem` 必须是去掉首尾空白后非空的字符串。
2. `dialog_history` 必须是列表；每条发言角色是 Teacher/Tutor/Student，文本是去掉
   首尾空白后非空的字符串。

检查时使用 strip，但保存原文不 strip。不做语义判断，不因学生或老师答错排除。
拿掉末尾参考教师回复只是原 benchmark 的输入构造，不是额外筛选条件。

产物 schema v2 包含 `cases`、`excluded`、`summary`、固定规则和源文件 SHA-256。
每条排除记录保存原始记录、ID、下标及全部触发的原因；重叠原因不重复删除记录。
case ID 基于 split、原始下标及内容哈希；不同历史使用同一道题时仍全部保留。
源文件仅以只读方式打开，过滤结果另存于本目录的 `data/`，不影响原 benchmark。

实际全量结果：1,477 条 → 排除 212 条 → 保留 **1,265 条**（standard 1,000、hard 265）。
rule1 命中 210 条、rule2 命中 6 条，其中 4 条重叠。`ready-v1.json` 可直接供 eval 使用。
任何输出文件存在时拒绝覆盖。旧 `mathdialbridge-draft-v1.json` 保留作为历史草稿，
不再使用，eval 会拒绝旧 schema，防止把未过滤的草稿混入正式实验。

## 2. Generate each model using the SAME evaluator

### Local dedicated eight-GPU node (four GPUs per model)

`run_local_8gpu.sh` starts two private SGLang TP=4 servers, then calls the same
`eval.py` concurrently for both models. It stops only its own processes on exit.
No public endpoints or API keys are needed; servers bind to loopback. Use eight
allocated/free GPUs. It does not reserve scheduler resources or stop existing jobs.

```bash
bash examples/tutor/human_study/run_local_8gpu.sh \
  --base-model /path/to/local/Qwen3-8B/snapshot \
  --adapter /path/to/20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu/default/epoch31epochstep42globalstep1499 \
  --gpus 0,1,2,3,4,5,6,7 \
  --output /path/to/short/human_study
```

Use `--dry-run` to validate paths/data and print deployment commands without any
server, API call, GPU allocation, or output writes. `--base-port` defaults to 33100
(two consecutive ports); `--workers` defaults to 16 per model. The common study
settings remain unchanged. Repeating the identical invocation resumes existing
results; `--retry-incomplete` explicitly permits retrying uncertain/failed calls.

The output directory is explicit and may be outside the code repository. Its `logs/` contains
`server-0.log`, `server-1.log`, `eval-qwen3_8b.log`, `eval-trained_1500.log`.
Its `qwen3_8b/` and `trained_1500/` subdirectories contain full raw trajectories.
Hugging Face, PyTorch, Triton, CUDA and FlashInfer cache paths are redirected into
its `runtime/`; temporary files and IPC sockets use its shorter `tmp/` subdirectory.
Startup validates the Unix socket pathname length before launching GPU processes.
Use a short output path; no symlinks or automatic external temporary directories
are created. Python bytecode writes are disabled.
Local weights are read-only inputs; no merging, copying or downloading is done.
This does not control driver/system logs or undocumented library scratch paths.
For complete scheduler-output isolation, also set the scheduler's stdout/stderr
paths to this human-study directory.

### Already deployed endpoints

凭据通过环境变量提供，不放 config、不保存 Authorization header。
运行前在当前 shell 配置 `HUMAN_STUDY_BASE_URL`、`HUMAN_STUDY_TRAINED_URL`
（OpenAI-compatible URL，包含 `/v1`）和 `INF_API_KEY`。不自动读取或执行 `.env` 文件。
支持标准 urllib 代理环境变量；内网端点需要时配置 `NO_PROXY`。

```bash
bash examples/tutor/human_study/run_eval.sh \
  --data examples/tutor/human_study/data/ready-v1.json \
  --study examples/tutor/human_study/configs/study.yaml \
  --model-config examples/tutor/human_study/configs/models/qwen3_8b.yaml \
  --output examples/tutor/human_study/results/v1/qwen3_8b \
  --workers 2

bash examples/tutor/human_study/run_eval.sh \
  --data examples/tutor/human_study/data/ready-v1.json \
  --study examples/tutor/human_study/configs/study.yaml \
  --model-config examples/tutor/human_study/configs/models/trained_1500.yaml \
  --output examples/tutor/human_study/results/v1/trained_1500 \
  --workers 2
```

`PYTHON` 可以覆盖启动脚本使用的解释器。新模型只需新增 YAML；model/LoRA 路由不能在
study 的生成参数中覆盖。SGLang 配置核对 base 模型、adapter 注册及 checkpoint 路径，
trained **显式发送 `lora_path: step1499`**。普通非 SGLang 服务可配置
`identity: {kind: models}` 来检查 `/models` 中的模型 ID，但这不是权重内容的证明。
运行期间应保持服务模型不变；服务端身份检查不能发现原路径下权重被偷偷替换。

每模型结果目录：

```text
manifest.json                 # 数据/配置/代码哈希、endpoint、服务身份快照（私有）
dataset.json                  # 完整冻结数据版本及筛选记录（私有）
status.json                   # 是否全部完成
records/<case-id>/
  attempt-<uuid>/request.json # 实际完整 JSON 请求，不含凭据
  attempt-<uuid>/response.json# 完整 API 响应；错误另存类型与HTTP状态，不保存敏感错误正文
  result.json                # 输入哈希、完整请求/响应、原样content、finish_reason、异常标记
```

原命令再次运行即可续跑：已有结果不重新调用；若响应已保存而 result 尚未写入，会从
响应恢复。输入、配置、代码或服务器身份改变时拒绝混入同一目录。一个目录只能有一个进程。

失败或只有 request 的不确定请求**不静默重试**；检查后显式加 `--retry-incomplete`。
远端请求在断线时可能已执行，显式重试可能产生额外调用/费用，无法保证远端 exactly-once。
正常返回的空回复/length 回复视为已完成，不因质量重试。不同目录是独立运行。

## 3. Export for human judges (no automatic judge)

```bash
.venv/bin/python -m examples.tutor.human_study.export_judge \
  --left examples/tutor/human_study/results/v1/qwen3_8b \
  --right examples/tutor/human_study/results/v1/trained_1500 \
  --output examples/tutor/human_study/results/v1/pairwise \
  --seed 42
```

仅允许相同冻结数据、提示、生成参数及 runner 版本的完整配对。每个 split 内 A/B
位置尽量平衡，样本顺序随机，分配与输出质量无关。缺任何回复都会报错；空回复不会被筛掉。

- **只分享 `pairwise/human/`**：`REVIEW.md`（原文阅读材料）、`items.jsonl`
  （题目、历史、A/B 原文）、`judgments.csv`（空白标注表）。
- **不分享 `pairwise/private/` 或模型结果目录**：包含真实身份、映射和异常标记。
- 评审说明只问：根据题目和历史，哪条下一轮教师回复更适合学生？选 A/B/tie/neither，
  简述理由。不提供教学风格 prior、参考教师答案、自动评分或模型身份。
- 匿名化仅隐藏元信息，不改写回复中模型自己说出的内容。正式评审前需检查源数据隐私。
- 该导出不是多评审分发平台；人数、分配、同意流程及一致性统计由后续 human-study 协议确定。

## Offline verification

```bash
.venv/bin/python -m pytest tests/test_human_study.py -q
bash -n examples/tutor/human_study/run_eval.sh
```

测试使用合成记录及 fake HTTP，不访问端点、不生成真实回复、不加载 GPU。
