# 0808 / full

**The objective: an episode where the student is still wrong after two
explanations should still end solved.** Today 33.8% of teaching episodes get
that far and only **46.3%** of them are won. With the hazard not decaying that
is **74.7%** — **+9.6 pp** on overall success, more than anything else on the
table.

## Layout

    2gpu/base.yaml    self-contained. Nothing outside this folder is inherited.
    2gpu/*.yaml       four arms, `defaults: [base@_global_]`
    4gpu/base.yaml    inherits ../2gpu/base, overrides the allocation only
    4gpu/*.yaml       the same four arms

Same shape as `math/0723/2gpu` + `math/0723/4gpu`. Trial names are `0808f2-*`
for 2 GPUs and `0808f-*` for 4.

| arm | instruction | where | covers | question |
| --- | --- | --- | --- | --- |
| `baseline` | — | — | — | control |
| `decompose` | decompose | prompt | depth 3–10 | what is a sentence worth after training? |
| `opd-decompose` | decompose | distilled | depth 3–10 | can the same effect live in the weights? |
| `opd-decompose-cap2` | decompose | distilled | depth 3–4 | is reaching *all* the stuck turns what matters, or just reaching them? |

All four. `cap2` is not optional: the 0808d runs already showed gate 2 beating
gate 0 across three separate instructions (14/15 shared eval steps — repair
+0.329, decompose +0.234, handback +0.133), so the only untested increment in
`opd-decompose` is removing the per-episode cap, and `cap2` is its control.

Read `opd-decompose` against `decompose`, not `baseline` — appending a sentence
to a prompt is free.

## GPU budget — read before launching

`CUDA_VISIBLE_DEVICES` is `0..7`, eight GPUs.

- **2gpu: four arms fit exactly**, and that combination has run (56–73 metric
  rows each on 20260809_0513).
- **4gpu: two arms at a time.** Three at once asks for 12 GPUs and is what
  killed 20260809_1920 — all three arms produced 0 metric rows, with
  `eval-rollout` workers failing to bind their forked ports
  (`Address already in use`, then `Readiness timeout`). Stagger launches by a
  minute: forked-worker ports are allocated per run with no coordination between
  runs, and `d2` doubles the number of forks each run needs.

4 GPUs per arm does not shorten the experiment on its own — it halves the
per-arm wall-clock and halves how many arms run concurrently.

## What differs from the 0808 arms

| | 0808 | here |
| --- | --- | --- |
| split | 132 / 118 | **759 / 528** |
| student probes | retest + level1 + level2 (12 calls/ep) | **retest only (4)** |
| unparseable teacher turn | penalise −0.5, hand student `""`, continue | **terminate, dropped from the loss** |
| teacher pre-solve at eval | verified | **unverified** |
| eval | 118×3 every 10 steps | 528×1 every 25 steps |

Reasoning for each is in the header of `2gpu/base.yaml`. Two that matter most:

**Format errors terminate.** A malformed turn is not a recoverable state: the
empty message is written back into the tutor's *own* history as a blank
assistant turn, and P(next malformed | this malformed) is 87.8% healthy / 98.1%
drifted. Every episode starts well-formed and none is malformed throughout — the
failure is absorbing. The student solves 0.3% of the time after a malformed turn
against 18.3% after a well-formed one, so the remainder was never data. The
episode is **dropped from the loss**, not scored — `format_error_penalty` is 0.0.
Charging −1.0 was tried and measured against the previous handling on this same
split (baseline, no OPD, steps ≥40): reward +0.098 → +0.073 and solved 0.390 →
0.364, both inside run-to-run spread, while format errors per episode went 0.034
→ 0.196. It bought nothing. Dropping is what the reference systems do — they
terminate *and* remove the invalid trajectory — and what DAPO does with
truncated samples. **Cost:** dropped rows shrink GRPO groups, so watch
`stop/format_error`; it is now the only thing keeping this visible.

**Eval runs unverified.** Training still requires the teacher to solve the
problem itself before teaching it; eval no longer does, because nothing checks
that at deployment. **This makes eval numbers incomparable with every earlier
run** — the `teacher_pre_skipped` bucket (2.6–4.4% of eval episodes, the ones
the teacher could not solve) disappears and those episodes now count. Expect a
downward level shift that is not a regression. Compare within this group only.

## Where the stuck episodes are lost

Depth = how many times the student had already answered wrong when the turn was
generated. 1520 baseline episodes:

| depth | turns | resolve | repeats previous | mean overlap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1107 | 38.2% | — | — |
| 2 | 614 | 28.0% | 8.4% | 0.218 |
| 3–4 | 628 | 18.5% | 24.2% | 0.321 |
| 5–7 | 479 | 10.0% | 39.2% | 0.463 |
| 8–10 | 323 | 2.8% | 61.4% | 0.631 |

Monotone and opposite; reproduces on three other runs. Within-(task, depth) —
same problem, same depth, some rollouts repeat and some do not — the
non-repeating turn resolves **10.5%** more often. Repetition is about 40% of the
gap; the rest is the tutor running out of material, which is what the decompose
clause targets.

    gate 0 + cap 2  -> depth 1-2    1721 turns   54.6%   the healthy end
    gate 2 + cap 2  -> depth 3-4     628 turns   19.9%   opd-decompose-cap2
    gate 2 + cap 0  -> depth 3-10   1430 turns   45.4%   opd-decompose

Uncapping also makes the placement comparison clean: `prompt_instruction` has no
per-episode cap, so at gate 2 it covers the same 45.4%.

## What to watch

**The objective.** Counts — divide the batch means, never average a per-episode
rate.

    depth/stuck_episode_solved / depth/stuck_episode      46.3% today, 74.7% ceiling

**Where it moves.** Per bucket `d1 d2 d3_4 d5_7 d8_10 deep`:

    depth/{b}/resolved        / depth/{b}/turns            hazard -- the objective
    depth/{b}/repeats         / depth/{b}/repeat_scored    repeat rate
    depth/{b}/containment_sum / depth/{b}/repeat_scored    mean overlap, threshold-free

Read the mean overlap first: no threshold, and it moved 0.218 → 0.631 across
depth where the thresholded rate moved 8.4% → 61.4%. `core/repetition.py`
documents how it is computed; the same function produces these metrics and the
offline census.

**Stability.** `opd-decompose` collapsed at step 44 on its 2-GPU run
(20260809_051324). The leading indicators, in order of usefulness:

- `opd_reverse_kl/avg` — ran 0.005 → 0.133 without ever plateauing, while
  `cap2` settled at 0.018. **Above 0.05 and still climbing, expect a break
  within ~10 steps.**
- `ppo_actor/update/clipped_tokens` — drifted 337 → 955 → 2460 while `cap2`
  stayed at 87–241.
- `stop/format_error` — new, and now a terminal state. A few percent and
  climbing means the drift is back; error handling will not save it, structural
  supervision would.
- `opd_advantage/min` and `/max` inside ±0.2; `rollout/format_errors` low.
- `prompt_instruction/share_of_turns` ≈ `opd/selected_ratio` ≈ **45.4%**
  (~19.9% for cap2). If they disagree the arms are not comparable.
- The line `student_generalize <split> split: kept N/M rows` should **not**
  appear. If it does, a transfer level is on and the split is being cut.

Teaching quality on the well-formed turns was unchanged throughout that collapse
(17.7% → 18.3%), so it was structural, not a loss of capability.

## Do not decide this on final_correct

The depth collapse is worth about +9.6 pp overall, the repeat part about +4 pp.
On 528 test problems the 95% interval is about ±4.3 pp.
`depth/stuck_episode_solved` has ~374 episodes and the per-depth hazards ~1430
turns *per training step*, and they move in tens of points. Decide on those.

## Schedule

100 steps is about 2.1 epochs here against about 12 on the filtered split, so
each train problem is seen roughly twice rather than twelve times. Intended, but
a shorter per-problem exposure than any earlier run. If the curves are still
climbing at 100, raise `total_train_steps` rather than shrinking the dataset.
