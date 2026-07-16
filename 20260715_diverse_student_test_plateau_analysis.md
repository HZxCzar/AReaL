# Diverse Student 实验为何在 clean test 上接近：证据审计

> 只读分析；未修改训练或评测代码。主体统计冻结在 test model-version
> `150 / 60 / 50 / 40 / 310`，共同横向比较固定使用 model-version 40
> （对应 metrics `global_step=39`）。运行中的实验之后新增的点不回写这张固定对照表。

## 结论先行

目前的数据**不能支持“不同方法已经收敛到同一个本质能力上限”**。更准确的结论是：

1. 五条 run 根本没有跑到共同 horizon，更没有完成计划训练。计划约 750 updates，两个长
   run 分别只到约 152/316，三个新 run 在本次审计时只到约 40–70。
1. Student/persona/model 的差异确实进入了 rollout，训练目标不是 no-op；但 test 被统一投影到
   **clean Qwen3-1.7B、无 persona** 这一维，根本没有直接测 7B 或 persona 目标上的差异。
1. 当前 `teacher_success/solved` 既不是最终正确率，也不是条件教学成功率。约 22–25% 初始就
   答对的题被记为 `pre_solved=1, solved=0`，另有约 3–5% teacher-pre 自解失败。因此 raw
   `solved` 的结构上限只有约 71–74%。两个较长 run 的约 0.53 平台，对 eligible 题其实约为
   73% 的教学成功率；若把 pre-solved 加回来，总成功率约为 76%。
1. 单次 test 是 528 题、每题 **1 次** rollout、Teacher/Student temperature 都是 0.7，不是
   pass@2 eval。`p≈0.5` 时单点评估标准误约 2.18 pp，95% 误差约 ±4.3 pp。共同 step 40 的
   4.17 pp 全跨度尚未达到任何一组配对显著性。
1. 同分绝不等于同一策略。共同 step 40 任意两条 run 有 197–226 题的 raw 成败状态相反；
   failure composition 也明显不同。更强的参数证据是：共同 step 49 的有效 LoRA 更新
   `Δ(BA)` norm 几乎一样，但两两 cosine 只有 0.042–0.085，方向接近正交。
1. 真正存在的共同约束主要是：
   - raw metric 的结构性 ceiling 和单次随机评测；
   - test 只看共同 clean marginal，抹掉训练时的 persona/model 维度；
   - teacher-pre + pre-solved 对训练支持集的 censor；
   - episode 几乎被压成 `+1/-1`，ReBN 再做 batch mean/std normalization；
   - `leak terminate` 与 `max_turns=10` 形成“保守但教不完”对“直接但泄题”的 Pareto trade-off；
   - train 759 题实际上每个 dataloader cycle 只覆盖前 640 题，尾部 119 题永久未进入 rollout。
1. 没有证据表明 PPO clip、KL、reward clip、gradient clip 或 context length 把不同目标强制
   训成了同一个解。

所以这不是“纯粹巧合”，但也不是“所有方法撞上同一个模型容量墙”。它更像是：**不同参数
方向和不同 failure trade-off，被一个有 ceiling、噪声很大且只测共同 marginal 的标量压到了
同一条 iso-score 带上。**

## 1. 首先纠正“已经最终收敛”的前提

五条配置都是 `total_train_epochs=150`。train dataloader 在当前配置下每个逻辑 epoch 为
5 updates，因此计划约 750 updates。

| 简称 | Trial                                      | 审计时已有 test version | 训练状态                      |
| ---- | ------------------------------------------ | ----------------------: | ----------------------------- |
| A    | `20260715_014404_...pre-aleak`             |                   0–150 | 停在约 update 152；无完成标记 |
| P2   | `20260715_183426_...student5-aleak-pre`    |      0–60（之后仍在跑） | 运行中                        |
| M    | `20260715_183456_...mix177-pre`            |      0–50（之后仍在跑） | 运行中                        |
| P1   | `20260715_185250_...student5-aleak-pre-v1` |      0–40（之后仍在跑） | 运行中                        |
| N    | `20260713_075318_...pre`                   |                   0–310 | 停在约 update 316；无完成标记 |

两个停止的长 run 也只走了计划的约 20%/42%，而且日志停在一次 step 中途，不是正常训练完成。
三个新 run 更早。尤其 P2 在 step 9–59 的简单趋势仍约为每 10 updates `+3.1 pp`，不能称为
平台。不同 run 的 peak 也不能直接比：N 有 32 次抽奖机会，P1 只有 5 次；在单点 SD 约
2 pp 的情况下，取最大值有明显 winner's curse。

## 2. 公平的共同 step 40：总分接近，但失败原因不同

下面固定比较 test model-version 40，即 metrics 的 `global_step=39`。Test 一共 528 题；各列
恰好构成互斥分解：

\[
solved+pre_solved+teacher_pre_skipped+leak+max_turn+context=1.
\]

| Run |     solved | pre-solved | teacher-pre skip |  leak-stop |   max-turn | context |
| --- | ---------: | ---------: | ---------------: | ---------: | ---------: | ------: |
| A   | **49.43%** |     23.30% |            4.17% |  **9.47%** | **13.64%** |   0.00% |
| P2  |     48.11% |     24.05% |            3.98% | **18.37%** |  **5.30%** |   0.19% |
| M   |     45.27% |     24.43% |            4.92% |     15.15% |     10.04% |   0.19% |
| P1  |     45.83% |     23.67% |            4.73% |     15.15% |     10.61% |   0.00% |
| N   |     47.92% |     23.48% |            5.30% |     14.77% |      8.52% |   0.00% |

最直接的反例是 A vs N：raw solved 只差 1.52 pp，但 A 少了 5.30 pp leak，却多了
5.11 pp max-turn，几乎精确抵消。A vs P2 也类似：一个更保守、较少泄题但更容易 10 轮仍未
完成；另一个对话更短但更容易触发 leak termination。它们不是同一行为，只是落在相同标量
等高线上。

同一步的 eligible 平均轮数也不同：A/P2/M/P1/N 分别约为
`3.60 / 2.78 / 3.53 / 3.33 / 3.06`。这进一步说明标量接近没有抹掉交互风格差异。

## 3. raw `solved` 的口径本身制造了约 26–29 pp ceiling

Workflow 中存在两个 early return：

1. 当前 Teacher 先生成 private solution，最多三次都答错时直接跳过样本，不产生训练序列；
1. Student 初始就答对时也直接返回，不产生 Teacher action。

指标中的 `solved` 只认 tutoring turn 之后首次答对；初始答对另记为 `pre_solved`。源码位置：

- `examples/tutor/workflow.py:1186-1301`
- `examples/tutor/workflow.py:2840-2869`

因此至少应并列报告三个量：

\[
\\text{total final success}=pre_solved+solved
\]

\[
\\text{eligible tutoring success}=\\frac{solved}{1-pre_solved-teacher_pre_skipped}
\]

以及 `leak/max-turn` failure partition。

用两个较长 run 的后期多个 eval 点做平均，而不是挑 peak：

| Window             | raw solved | pre-solved | pre-skip | solved + pre | solved / eligible |   leak | max-turn |
| ------------------ | ---------: | ---------: | -------: | -----------: | ----------------: | -----: | -------: |
| A, version 90–150  |     53.11% |     23.38% |    4.30% |   **76.49%** |        **73.43%** | 11.61% |    7.55% |
| N, version 170–310 |     52.30% |     23.45% |    4.53% |   **75.74%** |        **72.62%** | 13.89% |    5.83% |

这两个窗口确实接近，但它们接近的是“在共同 clean Student、共同终止规则下，约四分之三
eligible episode 成功”的水平，不是“Teacher 只能解决一半 test”。剩余 eligible failure 几乎
完全由 leak 与 max-turn 组成。

这构成一个真实的**评测诱导边界**：提示越直接越容易教会，也越容易被 leak detector 立即
终止；提示越保守越安全，却更容易在 10 轮预算内教不完。它更像 Pareto frontier，不像所有
模型都不会同一批题。

## 4. `aleak` 起步快主要是 test-time instruction scaffold，不是训练后才学到

A/P1/P2/M 的 anti-leak flag 会把“不要泄露答案”的 instruction 直接附加到 Teacher system
prompt；N 没有。相同初始 LoRA 下，aleak runs 的 version 0 solved 约 23–25%，而 N 只有
11.55%；N 的 leak-stop 为 60.42%，aleak baseline 为 46.59%。差异在任何 update 之前已经
存在。

所以：

- 若比较端到端方法，native prompt 的差异可以保留；
- 若想归因“训练学到了什么”，A 与 N 并不处在相同 eval protocol。需要对同一 checkpoint
  做 instruction on/off 的 2×2 cross-eval。

到后期，N 从 leak penalty 中也逐渐学会少泄题；A 的一部分低-leak收益又转成更多 max-turn，
于是 raw solved 接近。这正是“起步快、后期总分靠拢”的直接机制。

此外，resolved configs 显示早期两条 run 与三个新 run 使用了不同的 auxiliary/student API
endpoint。报告不记录 endpoint 或 credential，但这意味着“都开 pre”并不足以保证严格的
服务版本/随机性一致。

## 5. 不同训练目标确实生效了

前 40 updates 的 train rollout 平均值：

| Run | assisted solved | pre-solved |       leak |   max-turn | raw reward |    turns |
| --- | --------------: | ---------: | ---------: | ---------: | ---------: | -------: |
| A   |          39.46% |     25.10% |     18.93% |      9.82% |     +0.097 |     2.27 |
| P2  |          34.38% |     27.57% |     28.90% |      2.97% |     +0.020 |     1.51 |
| M   |          34.53% | **38.98%** |     16.59% |      3.98% |     +0.135 |     1.54 |
| P1  |      **26.69%** |     19.44% | **31.13%** | **16.20%** | **−0.220** | **3.24** |
| N   |          35.65% |     24.39% |     26.14% |      7.00% |     +0.022 |     1.99 |

更细的直接证据：

- P1 Quick persona：pre-solved 约 0.8%，assisted solved 约 1.7%，平均约 7.1 turns；
- P1 Skeptical：assisted solved 约 16–17%；
- P1 Receptive：assisted solved 约 40%；
- Mix 的 7B：pre-solved 约 53.8%，assisted solved 约 27%；
- Mix 的 1.7B：pre-solved 约 24.6%，assisted solved 约 42%。

Persona suffix 确实附加到了 Student system prompt，多 Student 也确实按权重抽样：
`examples/tutor/workflow.py:795-830, 890-931, 1062-1065, 1852-1866`。

所以不能用“配置没有进训练”解释结果。

## 6. 但名义 Student 权重不是实际 gradient 权重

训练只保留：

\[
\\text{Teacher-pre 能自解}\\cap\\text{Student 初始失败}
\]

强 Student 的主要优势恰好会被 censor 掉：7B 约 53.8% 初始答对，这些 episode 不产生任何
Teacher token。因此 M 虽然名义上 1.7B/7B 是 50/50，按实际 Teacher-turn proxy 已变成约：

- 1.7B：70.4%
- 7B：29.6%

P1 也被轨迹长度重加权：Quick 名义只有约 1/6，却贡献约 37.4% Teacher turns，因为它大量
跑到长失败轨迹。每个 Teacher turn 是一条独立训练 sequence，loss 又按有效 Teacher output
token 加权（`examples/tutor/core/tensors.py:8-68`，`areal/trainer/ppo/actor.py:438-459`）。

因此这些目标的确不同，但没有配置名字看起来那么正交：

- M 的有效 Teacher 数据仍主要来自与 test 相同的 1.7B；
- Persona runs 仍包含约 1/6 clean base，且五种 persona 都是同一个 1.7B 底座；
- 所有 run 共享同一个 math task support、Teacher base model 和 reward definition。

对 M 当前 1,792 条 sampled train debug traces 的检查只能描述 outcome/task support，不能据此
判断行为是否新。由于 debug sampler 的步长，只覆盖了 64 个反复出现的 train index；在这
64 题上：

- 7B 的 pre-solved 为 56.9%，1.7B 为 19.0%；实际 Teacher turns 因而约 28% 来自 7B、72%
  来自 1.7B；
- 7B 曾失败过的每一题，1.7B 也至少失败过一次；未观察到新的失败题目 support；
- 49/64 题上 7B 的 pre-solve rate 更高，只有 8/64 更低；
- 在两种 Student 都产生过非空错误 boxed answer 的 49 题中，30 题（61.2%）至少共享一个完全
  相同的错误最终答案；
- 进入 tutoring 后，7B 条件成功率反而更高（72.9% vs 62.0%），平均轮数更短（2.54 vs
  3.41）。

这些数字说明 7B 过滤了更多 easy failures，其 Teacher token 权重也被明显稀释；但**不能推出
Qwen2.5-7B 的 behavior 与 Qwen3-1.7B 相似或冗余**。两种 Student 可以在同一道题上给出同一
最终错答，却有完全不同的 reasoning、人格、固执程度、反馈吸收方式和后续状态转移。当前
trace 没有在“相同 task、相同初始状态、相同 Teacher feedback”条件下比较 Student 的下一步
response distribution，所以 behavioral novelty 仍未被检验。当前能成立的结论只有：独特 7B
信号至多占约 28% Teacher turns，而且 clean test 不测 7B；它可能是冗余信号，也可能是真实但
被稀释且完全落在 test projection 之外的信号。

### 直接 trajectory 行为审计

进一步不再使用 task correctness 或最终错答作为 behavior proxy，而直接读取首次有效 Teacher
feedback 后的 Student response。`repeat wrong answer` 只在首轮仍答错且前后都有可提取 boxed
answer 时统计；`question` 表示首轮 Student response 是否主动提出问题。虽然 Teacher feedback
未被严格缓存成完全相同的文本，Student/model/persona 在训练任务中是随机抽样的，因此这些
大样本条件频率能直接显示实际 rollout 中的行为差异。

Mix 的两种 Student：

| Student    | pre-solved | 首轮成功 | 重复原错误答案 | 主动提问 | 最终教学成功 | max-turn | 平均轮数 |
| ---------- | ---------: | -------: | -------------: | -------: | -----------: | -------: | -------: |
| Qwen3-1.7B |      19.3% |    36.1% |          55.2% |     1.0% |        62.6% |    12.5% |     3.41 |
| Qwen2.5-7B |      57.1% |    36.5% |          44.8% |    12.8% |        73.6% |     3.1% |     2.55 |

7B 并非只是相同 correctness label：它在收到 feedback 后主动提问的概率高约 13 倍，更少坚持
原错误答案，后续更容易修正，也更少进入 10 轮失败。首轮成功率几乎相同但最终成功率相差 11
pp，说明差异恰好发生在 multi-turn response dynamics，而不是初始对错上。

强 persona v1 的差异更极端：

| Persona           | 首轮成功 | 重复原错误答案 | 主动提问 | 最终教学成功 | max-turn | 平均轮数 |
| ----------------- | -------: | -------------: | -------: | -----------: | -------: | -------: |
| BASE              |    41.7% |          59.7% |     0.9% |        56.8% |    14.4% |     3.29 |
| QuickHelpSeeker   |     0.0% |            N/A |    98.9% |         1.5% |    48.2% |     7.40 |
| ReceptiveFollower |    38.0% |          72.9% |     0.8% |        56.4% |    15.7% |     3.33 |
| SkepticalDefender |    11.2% |          83.3% |     2.8% |        19.9% |    24.1% |     4.49 |

v2 的 persona 指令较软，差异较小但仍可见：ThinkAloudCollaborator 首轮成功 49.1%、重复原
错误 45.0%、平均 2.12 轮；SkepticalDefender 分别为 39.6%、65.7%、2.77 轮；
QuickHelpSeeker 的主动提问率为 5.0%，BASE 只有 1.1%。

因此现有数据已经排除“Student/persona 实际行为相同，所以 test 相同”这个解释。真正未回答
的是 Teacher 是否学会了这些 behavior-specific adaptation：当前所有 test 都关闭 persona，并
只测 clean Qwen3-1.7B，无法观察该能力。

按 task id 将 train traces 分为四个时间段后，可以继续区分“共享能力学习”和“独特行为学习”。
首段到末段的条件教学成功率变化为：

| 训练目标                  | 首段 | 末段 |   变化 |
| ------------------------- | ---: | ---: | -----: |
| Mix Qwen3-1.7B            |  53% |  64% | +11 pp |
| Mix Qwen2.5-7B            |  66% |  76% | +10 pp |
| P2 BASE                   |  42% |  62% | +20 pp |
| P2 ReceptiveFollower      |  39% |  65% | +26 pp |
| P2 ThinkAloudCollaborator |  48% |  69% | +21 pp |
| P1 AdaptiveExplorer       |  31% |  68% | +37 pp |
| P1 QuickHelpSeeker        |   2% |   0% |  −2 pp |
| P1 SkepticalDefender      |  20% |  24% |  +4 pp |

因此训练不是 no-op：mix 的两种模型和多数中等 persona 都随 Teacher 更新明显改善。但强制
persona 中最独特、最难的两类并没有被学会；Quick 始终只提问而几乎不提交答案，Skeptical
持续捍卫旧解。clean test 的上升可以完全由 BASE、普通 1.7B 和可修正 persona 的共享数学纠错
能力驱动，而无需 extreme behavior-specific objective 得到改善。相似 clean plateau 因而更像
“共同能力被测到，独特能力被关掉或没学会”，不是 Student diversity 不存在。

## 7. ReBN 压平的是更新尺度，不是更新方向

当前 reward 近似为 episode 级离散标签：

- tutoring 成功：`+1`
- leak：`-1`
- 10 轮未解：`-1`
- early bonus：0
- turn penalty：关闭
- student generalization level 1/2 reward：0

接近正确但没提交 boxed answer、明显进步但 10 轮未完成、完全错误，最终都可能得到同一个
`-1`。这是 objective 人工制造的 right/wrong cliff，不是题目天然存在一个离散能力断层。

源码路径：

- `examples/tutor/core/rewards.py:86-236`
- `areal/trainer/ppo/actor.py:254-286`

ReBN 在 `turn_discount=1` 时把最终 `±1` 传播给同 trajectory 的所有 Teacher turns，再对
turn return 做 batch mean/std normalization。前 40 updates 中，97–99% episode total reward
恰为 `±1`。不同 run 的正 return 比例从 P1 的约 17% 到 M 的约 44% 差异很大，但 normalization
会给稀有正样本更大的 z-score。实测：

| Run | raw rollout reward | grad norm |
| --- | -----------------: | --------: |
| A   |             +0.097 |    0.0658 |
| P2  |             +0.020 |    0.0697 |
| M   |             +0.135 |    0.0671 |
| P1  |             −0.220 |    0.0666 |

原始 reward 差 0.35，梯度 norm 却都约 0.067。这是成功率绝对尺度被 normalize 的明确证据。

但它没有把不同目标变成同一个方向。共同保存的 step 49 checkpoint，以 LoRA 的有效权重
`BA`（而不是有 gauge ambiguity 的原始 A/B 因子）计算相对相同 initial LoRA 的更新：

| Run | `||Δ(BA)||` |
|\---|---:|
| A | 2.036 |
| P2 | 2.116 |
| M | 2.078 |
| N | 2.139 |

两两 cosine 只有 **0.042–0.085**。作为校准，同一 run 的 step 50/100/150 更新 cosine 为
约 **0.57–0.85**，而 A 与 N 在相同步数的 cosine 始终只有约 **0.05**。

这意味着：不同方法走了近似相同距离，但方向完全不同。相似 clean-test 标量是宽阔 level set
上的相似函数值，不是参数收敛到同一个 basin。

## 8. Test 主动把不同训练目标投影回同一个 clean marginal

Persona v1/v2 都配置 `test_persona: false`，所以 test 不带任何 persona suffix。Mix 虽然训练
1.7B + 7B，但 evaluator 明确只测 `qwen3-1.7b`。相关逻辑在：

- `examples/tutor/configs.py:190-203`
- `examples/tutor/train.py:169-257`
- `examples/tutor/workflow.py:890-970`

于是不同训练目标

\[
J_i(\\theta)=\\mathbb E\_{x,s\\sim P_i}\[R(\\theta;x,s)\]
\]

最后都只用同一个一维切片

\[
V(\\theta)=\\mathbb E\_{x\\sim test,s=clean\\ 1.7B}\[success\]
\]

打分。`argmax J_i` 不同，并不推出 `V(argmax J_i)` 必须不同。高维策略空间里，同一个
`V=c` 的等高集合非常大；上面的 LoRA cosine 正好观察到了这一点。

Teacher prompt 也不显式提供 Student model/persona identity，只包含 task、history、judge feedback
和 round。若不同 latent Student 在相同 observable history 下需要不同下一步，当前共享 policy
无法立即分支，只能先学 mixture-average 策略，再从后续行为猜测。

## 9. 单次评测噪声足以掩盖 2–4 pp 方法差

所有 run 都是：

- `evaluator.average_rollouts=1`
- `eval_gconfig.n_samples=1`
- Teacher temperature 0.7
- Student temperature 0.7
- test N=528

run 名中的 `pass@2` 是数据构造/筛选名，不代表在线 test 每题采样两次。

共同 step 40 的 Wilson 95% 区间：

| Run |   count |  score |      Wilson 95% CI |
| --- | ------: | -----: | -----------------: |
| A   | 261/528 | 49.43% | \[45.19%, 53.69%\] |
| P2  | 254/528 | 48.11% | \[43.87%, 52.37%\] |
| M   | 239/528 | 45.27% | \[41.07%, 49.53%\] |
| P1  | 242/528 | 45.83% | \[41.63%, 50.10%\] |
| N   | 253/528 | 47.92% | \[43.69%, 52.18%\] |

10 个 pairwise exact McNemar test 全部 `p>=0.147`。这不是证明它们相等，而是说明当前一次
rollout 的统计功效不足以分辨这一级别的差异。

更直接的 noise control：P1/P2/M 的 initial LoRA 文件 hash 完全相同，version 0 eval 又都是
clean 1.7B、无 persona。三者在任何训练之前仍分别得到 119/124/133 个 solved，pairwise
solved-set Jaccard 仅 0.217–0.300。也就是说，即使 Teacher 权重相同，随机 Student 初答、
Teacher generation、teacher-pre 与 judge 路径也会让大量逐题二元状态翻转。

所以逐题 set 差异必须这样解释：**相同总分肯定不代表同一批题；但在 n=1 eval 下，也不能把
全部 set flip 都归因于 learned policy 差异。**

### 为什么逐题集合剧烈变化，但总分仍然稳定

这两个现象并不矛盾。A 与 M 的 initial LoRA 完全相同；version 0 分别成功 121 和 124 题，
其中共同成功 51 题、A-only 70 题、M-only 73 题。也就是说逐题有 143 个状态不同，但正常
总分里 `+70` 与 `-73` 几乎完全抵消，所以两次总分只差 3/528，即 0.57 pp。两次取 Oracle
时则不再抵消，而是把两边的单独成功全部保留，因此直接得到 `51+70+73=194`，即 36.74%。

一般地，若第 i 题单次成功概率为 `p_i`，单次总分是 528 个 Bernoulli 的平均：

```text
Var(score) = sum_i p_i(1-p_i) / 528^2
```

但两次 rollout 的预期逐题 disagreement 是：

```text
2 * sum_i p_i(1-p_i)
```

所以大量题目翻转只会以平方根速度影响总分。step 0 的 143 个 observed flips 对应的总分单次
标准差量级约 1.6 pp，而不是 27 pp。四个相同初始 LoRA 的实测单次分数为 22.54%–25.19%，
样本标准差只有 1.17 pp；它们的逐题集合却差很多。

后期数据也符合这个量级：A 的 version 80–150 分数均值 52.91%、标准差 2.49 pp；N 的
version 160–310 均值 52.26%、标准差 1.81 pp。这与 528 道 Bernoulli 在 `p≈0.5` 时约 2.18
pp 的单次标准误一致。训练带来的约 25–30 pp 均值移动远大于约 2 pp 的评测噪声，因此曲线
可以稳定显示学习趋势，同时每次具体成功题集合仍大量变化。

若只看真正进入 tutoring 后以 leak/max-turn/其他原因失败的题，共同 step 40 的失败集合也并不
基本重合。A 有 122 题失败，N 有 123 题失败，但交集只有 51 题：A-only 71 题、N-only 72
题，failure-set Jaccard 只有 0.263。五个 run 的 10 组两两比较中，交集为 50–62 题，Jaccard
为 0.251–0.308。因此“同一批失败题只是在 leak 与 max-turn 之间换标签”不符合这次单次
rollout 的逐题观测；最多只有交集部分可能发生这种转换。这里仍需保留上面的随机性警告：低
重合不等于这些方法具有稳定且不同的失败题集合。

### A 与 mix 的 A-only success 个案审计

共同 step 40 中，按 raw `solved` 集合计算有 116 题是 A-only success，但其中只有 45 题是
“A 教成功、M 也实际进入 tutoring 后失败”。另外 71 题在 M 中没有 Teacher rollout，属于
`pre_solved` 或 teacher-pre skipped，不能称为 M Teacher 教学失败。反方向则有 46 题是
“M 教成功、A 实际教学失败”，和 45 几乎完全对称。

随机性基线同样强：A 与 M 的 initial LoRA SHA256 完全相同；在 version 0，A 成功/M 失败有
44 题，M 成功/A 失败有 41 题。到 version 40 仍是 45 对 46，说明这种大规模逐题分叉在训练
产生策略差异之前就存在。对 version 40 的 45 道 A-only 真失败题，查看其他六个共同 eval
version；只统计双方都进入 tutoring 的轨迹，A-only success 出现 36 次，M-only success 出现
34 次，双方都成功 65 次，双方都失败 63 次。按题统计，16 题在其他版本更偏 A、15 题持平、
14 题反而更偏 M；强 A 倾向和强 M 倾向各 6 题。因此不能把 version 40 的大部分 A-only
样本解释成稳定的 learned-policy 优势。

仍有少数可信的行为模式候选。例如 test index 491 的三角函数乘积题，在七个共同版本中 A
五次成功而 M 失败，另外两次双方失败，M 从未反胜。step 40 中 A 第二轮给出与当前表达式
直接匹配的三正弦恒等式，Student 随即解出；M 连续十轮重复一个形式不匹配、部分表述错误的
通用乘积公式，未能让 Student 前进。这更像真实的“具体可执行纠错”对“抽象重复提示”策略
差异。

相反，test index 38 的复利题是典型随机翻转：step 40 中两边初始 Student 都把周期利率误当
年利率；A 一轮明确指出 `1%` 是每周期利率后成功，M 先给模糊提示、第二轮给出具体式子后因
leak 终止。但其他共同版本里 M 三次单独成功，A 没有稳定优势。test index 168 也主要来自
Student sample：两边都已找到边长 `17,17,16`，A 的 Student 下一轮算出 50，M 的 Student
却反复坚持 `17+17+16=49`，最后 Teacher 直接说 50 而触发 leak。这类样本不能归因于 Teacher
知识或策略的稳定差异。

## 10. 有共同题目难度结构，但没有一堵固定的“不可解题墙”

为了降低单次二元噪声，使用两个较长 run 的多个后期 eval：A version 80–150 共 8 次，N
version 160–310 共 16 次。只在该题真正进入 tutoring 时计算条件成功率。

- 两个 run 的 per-item 条件成功率 Spearman 相关约 0.61；要求双方有更多 attempt 后约
  0.66–0.69。说明确实有共享的题目难度 landscape。
- 但合并 24 次 eval 后，528 题全部至少进入过一次 tutoring，**524 题至少成功过一次**。
  只有 4 题一次都没成功，其中只有 2 题有至少 12 次 eligible attempt 仍为 0。
- Intermediate Algebra 的条件教学成功率最低，约 64.5–64.9%；Precalculus 约 67.7–68.8%；
  Geometry 约 79.2–81.2%。这是平滑的类别/题目概率梯度，不是“做对组”和“永远做不对组”
  之间的巨大断层。

因此用户问“是不是没做对的题和做对的题之间 gap 太大”：

- **对训练 label：是。** 当前 reward 把连续进步人为压成 `+1/-1` cliff。
- **对题目本身：不是。** 重复评测表明绝大多数题都能在某次轨迹中成功，单次成败是概率
  事件；只有极少数题显示持久困难。

## 11. 一个额外的真实共同 support bottleneck：759 题只反复训练前 640 题

train arrow 有 759 题；配置为 `batch_size=128, shuffle=false`，而 `drop_last` 使用默认 true。
`create_dataloader` 每次 cycle 只产生 5 个完整 batch，即 640 个 index。`cycle_dataloader`
重启时 shuffle 仍为 false，所以索引 640–758 的 119 题永久不被 yield。

相关位置：

- `examples/tutor/configs/math/july/pass@2/qwen8b-qwen1.7b-math-pre.yaml:230-237`
- `areal/api/cli_args.py:2385-2430`
- `areal/utils/dataloader.py:229-285`
- `areal/utils/data.py:1415-1427`

这不是纯理论推断：N 的 5926 条抽样 train debug traces 只命中 dataset index 0–630，0 条
命中 `>=640`；task id 与实际 train index 满足 `task_id % 640`。前 40 updates 的 rollout
文件显示各 run 已经覆盖前 640 中的 632–639 题，却没有可能看到尾部 119 题。

尾部也不是随机类别：

| 实际 support       | Prealgebra | Precalculus | 其他类型 |
| ------------------ | ---------: | ----------: | -------: |
| 前 640（会训练）   |         45 |       **0** |      595 |
| 尾 119（永久丢弃） |         34 |      **85** |        0 |

所以所有方法都在反复优化同一个 640-task support，且完全没有训练 Precalculus。它是一个明确
的共同 data coverage ceiling，也可能促成共同 test 平台。不过它不是平台的充分解释：后期
Precalculus 条件成功率约 68%，而有训练数据的 Intermediate Algebra 还略低。Precalculus 在
test 中占 56/528（10.6%）；成熟 run 目前已做对约 27–29 题，因此即使假设补齐该类训练能把
剩余题全部做对，量级上也只回收约 5.1–5.5 pp。它足以吞掉几个百分点的方法差，却不可能
单独解释 raw `solved≈0.53` 或全部 failure。因此应把它视为高优先级共享约束，而不能单独
宣称为全部因果。

Train/test 的 exact task 文本交集为 0，未发现数据 split 的 exact leakage；这里的 `aleak`
是 anti-answer-leak instruction，不是 train/test 数据泄漏。

## 12. 排除的共同优化器瓶颈

日志和配置基本排除：

- reward clip：阈值 20，实际信号几乎都在 `±1`；
- KL：`kl_ctl=0`；
- PPO clip：前 40 steps clip ratio 约 0.14–0.18%；
- behavior importance cap：实际被 cap/mask 的 token 极少；
- gradient clip：阈值 1.0，实际 norm 约 0.07；
- context limit：test 几乎为 0；
- Student API call failure：不是跨 run 的普遍现象。

这些机制没有把几个目标“夹”成同一个更新。真正被压平的是 reward/advantage 的**尺度**和
最后观察到的 clean-test **投影**。

## 13. 对原问题的直接回答

### 是纯粹这么巧一样了吗？

不是纯巧合。它们共享 base model、task support、reward、censor、ReBN、clean 1.7B test 和
终止规则；单次 eval 又有约 ±4 pp 的 95% 不确定性。多个不同策略落到相似 scalar band 是
相当自然的。但当前数字也没有“完全一样”：共同 step 40 仍跨 4.17 pp，动态 latest/peak 因
horizon 不同可跨约 9–10 pp。

### 是不是存在本质瓶颈？

存在明确的**协议/数据/信息瓶颈**：metric ceiling、n=1 eval、clean marginal projection、
640-task support、binary reward、teacher-pre/pre-solved censor、latent Student identity 和
leak-vs-max-turn frontier。

但没有证据证明存在一个 Qwen3-8B 的本质能力上限，使所有训练目标必然停在 0.53。要下这个
结论，至少要先移除评测噪声和 projection，跑到共同 horizon，并看到 per-task success
probability 也真正重合。

### 为什么不同方法学出来还能这么接近？

它们只是在**同一个 clean-test scalar 上接近**：

- 参数更新方向并不接近；
- train success/reward/turns 并不接近；
- leak 与 max-turn trade-off 并不接近；
- 单次逐题成败集合也不接近；
- 当前 test 根本没测 persona 和 7B 维度。

所以正确表述不是“学出来一样”，而是“不同策略在一个噪声较大的共同 marginal 上取得了相似
平均值”。

## 14. 最小可证伪实验矩阵

不需要先改训练算法；先把评测做成能回答问题的实验：

1. 固定共同 checkpoint/version，不比较动态 latest 或各自 peak。
1. 缓存同一份 initial Student answer、teacher-pre draft 和 judge 结果，或至少使用 task-fixed
   seeds；每题做 8–16 rollouts，估计 per-task success probability。
1. 同时报 raw solved、`solved+pre`、`solved|eligible`、leak、max-turn、solve-turn。
1. 做完整 `student model × persona` eval matrix：clean 1.7B、五 persona、2.5-7B。若差异只在
   这些格子出现，说明当前 clean scalar 丢失了目标差异。
1. 只改 eval boundary 做 `max_turns 10→20`、`leak terminate→feedback/reward_only` 的诊断；
   若方法排序因此拉开，说明当前平台主要是 protocol frontier。
1. 对 A/N checkpoint 做 anti-leak instruction on/off 的 2×2 cross-eval，拆开 test-time
   scaffold 与 learned weights。
1. 用 shuffle 或确保 remainder 被轮换覆盖的对照，验证 640-task support 是否是共同 ceiling；
   同时保持所有服务 endpoint 完全一致。

只有在固定随机环境、多 rollout、完整 target matrix、共同训练 horizon 下，多个方法的
per-task probability 和 trajectory behavior 仍重合，才有足够证据称为“本质共同能力瓶颈”。
