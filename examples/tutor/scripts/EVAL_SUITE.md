# 两模型八卡串行评估套件

入口：`bash examples/tutor/scripts/run_eval_suite.sh run`。

固定 checkpoint：

- PedRL：`0906-pedagogical-rl-qwen3-8b-lr5e-5-8gpu@globalstep999`。
- Ours：`20260908_0901-reward-v4-all-id-fork775-8gpu@globalstep999`。

所有任务只评估，绝不执行训练更新。每次使用独立新结果目录，不复用以前三次评估的数据。

## 执行顺序

| 阶段 | 顺序 | 数据量 | 协议 |
|---|---|---|---|
| 1 | PedRL-1 → Ours-1 → PedRL-2 → Ours-2 → PedRL-3 → Ours-3 | 每次 528 题 | 我们的 none student，presolve 开，leak gate 开 |
| 2 | Ours → PedRL | 每模型 528 × 2 = 1056 段 | PedRL prompt／流程，GUIDED + ATTEMPTED，none，统一 XML，无 presolve；原生 judge 只判定，最后测试 |
| 3 | PedRL-1 → Ours-1 → PedRL-2 → Ours-2 → PedRL-3 → Ours-3 | 每次 528 题 | 与阶段 1 相同，仅关闭 presolve |

总共 14 次模型评估、8448 段对话。重复次数之间严格串行；单次 tutor 评估内部用 4 对 teacher/student 分片并行。
PedRL 协议复用已有八卡 launcher，`total_train_steps=0`；它会初始化 actor，但不会更新模型。
教师输出上限、prompt、采样等沿用各自既有评估配置。三次都保留原 seed=42，测重复运行波动，不是不同 seed 的实验。

## 运行

```bash
set -euo pipefail
cd /inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent
cd AReaL.worktrees/dev-unified
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash examples/tutor/scripts/run_eval_suite.sh run
```

程序打印 `[suite] /.../output_hdd/eval_suites/<时间>/`。也可用 `--output /一个全新目录` 指定目录。
只查看任务顺序用 `plan`；不启动 GPU 的配置校验用 `preflight`（会依次校验两模型、两种 presolve 模式与 PedRL 配置）。

## 实时结果和轨迹

- `progress.md`：每完成一次立即更新分数、状态、待补数量。
- `suite_state.json`：任务命令对应的固定 checkpoint、代码摘要、进度和错误。
- `repeat_statistics.json`：按模型和 presolve 模式计算已有重复的均值、样本标准差（数值为 0–1，乘 100 为百分点）。
- `logs/<任务>.log`：每项完整启动及评估日志；总控每 20 秒打印当前日志路径。
- `<任务>/cells/<模型>/qwen3-1.7b-text-original/shards/*/traces/<presolve_on或presolve_off>/`：tutor 完整轨迹，包含新的 leak 原始返回／错误记录。
- `ped_protocol/attempts/<模型>-<尝试号>/eval/`：PedRL 完整轨迹。完成后 `ped_protocol/ours` 和 `ped_protocol/pedrl` 链接到成功的尝试目录。
- `ped_protocol/comparison.md`、`comparison.json`：两个模型在 PedRL 协议下的单独对比，不和 tutor 协议分数混算。

完成数必须完整，轨迹不能缺。少量被记录的调用/检查异常仍按 evaluator 当前口径保留，标记 `done_with_diagnostics` 和待补数量后继续；不自动丢样本、不改判、不无限补跑。PedRL 的原生 format 失败仍可能不执行最后测试、计零，保持既有规则。

## 任务切换和中断恢复

- 每次子任务使用独立进程 session，并带随机所有权标记；结束或中断时只清理该任务拥有的进程，不按 GPU 或模型名乱杀进程。
- 下一项启动前确认 8 张指定 GPU 没有计算进程，且 30001、37000/37001 到 37030/37031 端口空闲。
- 等待最多 120 秒；资源没有释放就停止套件并报告占用，不冒险继续。不可同时在这些 GPU/端口运行其他任务。
- 同一输出目录加排他锁，防止两个总控同时运行。失败即停，不静默跳过未完成任务。

中断后在相同代码配置、相同 GPU 编号下：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash examples/tutor/scripts/run_eval_suite.sh resume --output /第一次打印的suite目录
```

已完成任务（包括已记录诊断异常的完整任务）直接跳过。未完成的 tutor 任务使用其原目录续评；未完成的 PedRL 单模型评估没有可靠逐样本续评机制，因此从新 attempt 目录重跑该模型，旧轨迹保留、不混入汇总。另一个已完成模型不会重跑。
代码/配置变化时拒绝混合续跑，需开新套件目录。这个大套件不会修改 `analysis/results.md` 或旧评估目录。
