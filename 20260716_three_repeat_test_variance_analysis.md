# 三次重复 Test：逐题方差与成功集合稳定性审计

> 分析快照：2026-07-16 约 04:30 UTC。三条 run 仍在运行，训练日志约到 step 28；当前只有
> model version 0、10、20 三个完整 eval 点。version 30 正在生成，本文不能回答真正后期或
> 最终收敛行为，只回答目前的早期变化。

## 实验

- Persona：`20260716_020249_qwen8b-qwen1.7b-math-pass@2-student5-aleak-pre`
- Mix：`20260716_021244_qwen8b-mix-qwen1.7b-qwen2.5-7b-math-pass@2-baseline-aleak-mix177-pre`
- Baseline：`20260716_020139_qwen8b-qwen1.7b-math-pass@2-pre-aleak`

三条 run 都使用 `average_rollouts=3`，且当前 test 都是 clean Qwen3-1.7B。Persona run 明确
`test_persona=false`；Mix eval 也通过 `student_model_names=[qwen3-1.7b]` 只测 1.7B。

## 指标口径

每题三次最终结果可写成成功次数 `k=0,1,2,3`：

- `final_correct`：Student 初始答对或经 Teacher 教学后答对；这是端到端可靠性指标；
- `solved`：只计算 Teacher tutoring 成功，初始答对仍记为 0；它不适合直接衡量最终稳定性；
- `variance`：先在每题的三个 0/1 结果内计算 `p(1-p)`，再对 528 题平均；
- `all_success`：三次都正确，即 `k=3`；
- `any_success`：至少一次正确，即 `k>=1`；
- `pairwise_success_set_jaccard`：三组两两成功集合的平均交并比；
- `success_set_jaccard`：三组共同成功集合 / 三组成功并集，即最严格的三路 Jaccard。

对三次重复，`k=0/3` 的题方差为 0，`k=1/2` 的题方差为 `2/9`。因此 raw variance 下降只
表示题目更确定；必须同时看它们是变成三次都对，还是三次都错。

## 三次平均后，test 均值确实更可靠

由每题 repeat variance 估算，step 20 的三次平均标准误约 0.91–0.97 pp，对应 95% rollout
不确定性约 ±1.8–1.9 pp：

| Run      | step 20 `final_correct` | 约 95% rollout 区间 |
| -------- | ----------------------: | ------------------: |
| Persona  |                  66.35% |            ±1.90 pp |
| Mix      |              **74.31%** |            ±1.79 pp |
| Baseline |                  67.74% |            ±1.91 pp |

Mix 比 Baseline 高 6.57 pp、比 Persona 高 7.96 pp，远大于三次平均后的噪声；这是当前最
可靠的 run 间差异。Persona 与 Baseline 只差 −1.39 pp，仍无法可靠区分。

三条 run 的 version 0 `final_correct` 只在 46.65–47.60% 之间，说明 Mix 的 step 20 优势不是
初始 checkpoint 差异。

## 逐题稳定性确实在改善，Mix 最明显

### Step 20 横向比较

| Run      |   平均正确 |   三次都对 | 只对 1/3 或 2/3 |  三次都错 |   平均方差 | 两路 Jaccard | 三路 Jaccard |
| -------- | ---------: | ---------: | --------------: | --------: | ---------: | -----------: | -----------: |
| Persona  |     66.35% |     42.99% |          44.51% |    12.50% |     0.0989 |       63.45% |       49.13% |
| Mix      | **74.31%** | **52.08%** |      **39.58%** | **8.33%** | **0.0880** |   **69.84%** |   **56.82%** |
| Baseline |     67.74% |     44.32% |          45.08% |    10.61% |     0.1002 |       63.69% |       49.58% |

换算成 528 道题：

| Run      |    0/3 |    1/3 | 2/3 |     3/3 |
| -------- | -----: | -----: | --: | ------: |
| Persona  |     66 |    100 | 135 |     227 |
| Mix      | **44** | **66** | 143 | **275** |
| Baseline |     56 |    105 | 133 |     234 |

Mix 的优势不是只靠某一次 rollout 撞对：它有 275 题三次都正确，比 Baseline 多 41 题；三次
都错只有 44 题，比 Baseline 少 12 题。

### Version 0 到 20 的变化

| Run      |  平均正确变化 | 方差相对变化 |  三次都对变化 | 两路 Jaccard 变化 |
| -------- | ------------: | -----------: | ------------: | ----------------: |
| Persona  |     +19.70 pp |       −17.8% |     +22.92 pp |         +19.26 pp |
| Mix      | **+26.70 pp** |   **−30.1%** | **+32.77 pp** |     **+26.63 pp** |
| Baseline |     +20.64 pp |       −17.9% |     +24.05 pp |         +19.68 pp |

三个实验都出现“均值上升 + 方差下降 + 成功集合重合度上升”。这正是 Teacher 不仅提高平均
成功率，而且对同一题的多种随机 Student trajectory 更稳定的证据。Mix 在三个维度上都最强。

## 成功题集合到底重不重合

`final_correct` 的成功集合重合度：

| Run      | Version | 两次 repeat 平均 Jaccard | 三次 repeat 严格 Jaccard |
| -------- | ------: | -----------------------: | -----------------------: |
| Persona  |       0 |                   44.20% |                   27.04% |
| Persona  |      10 |                   57.93% |                   42.17% |
| Persona  |      20 |               **63.45%** |               **49.13%** |
| Mix      |       0 |                   43.21% |                   25.44% |
| Mix      |      10 |                   60.55% |                   45.34% |
| Mix      |      20 |               **69.84%** |               **56.82%** |
| Baseline |       0 |                   44.02% |                   26.95% |
| Baseline |      10 |                   57.99% |                   42.08% |
| Baseline |      20 |               **63.69%** |               **49.58%** |

所以 version 0 时“第一遍对这些、第二遍对另外一些”的问题确实严重：任意两遍的成功并集中
只有约 43–44% 是共同成功，三遍共同成功只占三遍并集约 25–27%。到 version 20，普通两遍
重合提高到约 63–70%，三遍严格重合提高到约 49–57%。它仍不是确定性系统，但稳定性已经发生
很大的真实改善。

## 为什么不要直接看 `repeat/solved/variance`

三条 run 的 raw `solved` variance 从 version 0 到 20 反而大致从 0.10 升到 0.13。这不代表
Teacher 更不稳定，原因有两个：

1. `solved` 均值从约 0.23 上升到 0.43–0.50；Bernoulli 的 `p(1-p)` 在 p 接近 0.5 时机械地
   变大；
1. Student 初始答对时 `solved=0`，所以同一题在“这次初始会”和“这次经 Teacher 教会”之间
   切换也会制造 solved variance。

本问题应主要看 `repeat/final_correct/*`。如果要严格回答“在初始错误的前提下，Teacher 是否
三次都能教会”，当前还缺少 `tutoring_success_given_initial_wrong` 的 repeat 分组指标；现有
`solved` 和 `final_correct` 分别在两侧混入了 pre-solved 的影响。

## Mix 的当前优势来自哪里

Step 20 的 failure partition：

| Run      | Teacher solved | pre-solved | teacher-pre skip |       leak |  max-turn | eligible tutoring success |
| -------- | -------------: | ---------: | ---------------: | ---------: | --------: | ------------------------: |
| Persona  |         42.61% |     23.74% |            4.42% |     21.15% |     8.08% |                    59.31% |
| Mix      |     **49.94%** |     24.37% |        **3.79%** | **15.40%** |     6.50% |                **69.51%** |
| Baseline |         43.75% |     23.99% |            4.67% |     21.97% | **5.62%** |                    61.33% |

Mix 相比 Baseline 的主要收益是 leak 减少约 6.57 pp；max-turn 只增加 0.88 pp，净 failure
明显下降。因此 Mix 当前学到的不是单纯“更敢直接给答案”，而是更少触发泄题终止，同时保持
较高教学成功率。这和它更高的三次稳定成功率一致。

## 当前 Takeaways

1. 三次平均是有效的：均值的 rollout 不确定性从单次约 ±4 pp 降到约 ±1.8–1.9 pp；现在
   6–8 pp 的差异已经可以可信地区分。
1. Teacher 训练不只提高平均分，也在降低逐题随机性。三条 run 的三次共同成功集合都扩大，
   混合的 1/3、2/3 边界题比例下降。
1. Mix 当前明显领先：step 20 clean final accuracy 74.31%，三次都对 52.08%，两路 Jaccard
   69.84%；三个指标都优于另外两条。
1. Persona 与 Baseline 在 clean test 上仍基本相同。由于 Persona test 关闭，这不能判断
   persona-specific adaptation 是否学到，只能说明它目前没有提高 clean 1.7B 的均值或稳定性。
1. 低 variance 必须和 `all_success`/`never_success` 一起读。Persona 从 step 10 到 20 的方差
   下降同时伴随三次都错从 9.28% 回升到 12.50%；它有一部分是向稳定失败极化，不全是稳定
   成功。Mix 和 Baseline 在同期则三次都错继续下降，改善更干净。
1. 目前只有 0/10/20，不能声称后期 variance 已经收敛。后续应持续追踪相同四格分布
   `0/3,1/3,2/3,3/3`；若均值平台后 `1/3+2/3` 继续下降并主要流向 `3/3`，才是用户关心的
   “同题面对不同随机 Student 都能稳定教会”。
