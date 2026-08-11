# 0810 / freechat

**The teacher opens, the pair talks for a fixed budget, and the only thing scored
is whether the student can then solve the problem alone.**

Nothing inside the conversation is judged. The student is not told the task, not
told the subject, and not told to produce an answer; its entire system prompt is
`You are a student talking with a teacher.` What the conversation is about is the
teacher's decision, and the student sees the problem statement for the first time
at the re-test.

## Why

In the answer-attempt loop the student was told on every turn to solve the task
and to box an answer, every reply was judged, and the episode stopped the moment
one was correct. Three things follow from that, and all three are removed here:

| | answer-attempt loop | here |
| --- | --- | --- |
| student's instruction | "solve the task", every turn, boxed | "you are a student talking with a teacher" |
| every student reply is | an answer attempt | whatever a student would say |
| episode length | 2.1 rounds measured, set by the student's luck | exactly `budget` |
| reward | binary, the episode ended solved or not | fraction of 4 solo re-test attempts |
| who ends it | the environment, on a correct answer | the budget |

The reward having five levels instead of two is not cosmetic. Under GRPO an
all-fail or all-pass group has zero advantage and contributes no gradient, and the
hardest problems — the ones worth teaching — are exactly where the binary outcome
was constant across all 8 rollouts.

## Layout

Parallel to `math/0808`, same shape as `math/0808/full`.

    2gpu/base.yaml             inherits math/0808/full/2gpu/base, overrides the rollout
    2gpu/leak-reward.yaml      leak_handling_mode: reward_only
    2gpu/leak-terminate.yaml   leak_handling_mode: terminate
    4gpu/base.yaml             inherits ../2gpu/base, overrides the allocation only
    4gpu/*.yaml                the same two arms

Trial names are `0810fc2-*` at 2 GPUs and `0810fc-*` at 4, so the two allocations
cannot collide on an output directory or an nfs name_resolve record.

## GPU budget -- read before launching

`CUDA_VISIBLE_DEVICES` is `0..7`, eight GPUs.

- **4gpu: the two leak arms are exactly the whole node.** Nothing else fits
  beside them. Stagger the two launches by a minute -- forked-worker ports are
  allocated per run with no coordination between runs, and `d2` doubles the forks
  each run needs. Three concurrent 4-GPU arms is what killed 20260809_1920: 12
  GPUs requested against 8, all three arms produced 0 metric rows with
  `eval-rollout` workers failing to bind (`Address already in use`, then
  `Readiness timeout`).
- **2gpu: both arms plus two more.** 4 GPUs per arm halves the per-arm wall clock
  and halves how many arms run at once; it does not shorten the experiment.

## Launching

Two arms, one minute apart:

    bash examples/tutor/run_official.sh       examples/tutor/configs/math/0810/4gpu/leak-reward.yaml
    sleep 60
    bash examples/tutor/run_official.sh       examples/tutor/configs/math/0810/4gpu/leak-terminate.yaml

`run_official.sh` is what sets `PYTHONPATH`, sources `.env` and puts HF/wandb in
offline mode; a bare `python examples/tutor/train.py` picks up whichever worktree
the shared editable install points at and fails on the missing endpoints.

Before launching, and after any config edit:

    python tests/test_tutor_free_chat.py

It loads all six configs, checks that the two leak arms differ in exactly one key
at both allocations, that 4gpu changed nothing but the allocation, and that the
guard rails still refuse a `max_turn_penalty != 0` or an unrewarded re-test.

## The episode

    teacher 1        no student input at all -- system prompt only
    student 1
    ...
    teacher 5
    student 5
    ---------------- always ends here ----------------
    re-test x4       fresh branch, transcript replayed, task shown, solve alone
    reward = correct / 4

## Reward

    retest        retest_reward x (n_correct / 4)   on the last completed round
    leak          leak_penalty                     episode-aggregated
    format_error  format_error_penalty              per malformed turn
    max_turn      0.0  -- REQUIRED, the run refuses to start otherwise

`max_turn_penalty` has to be zero because reaching the budget is now the normal
ending; the inherited `-1.0` would charge every single episode. The workflow
raises at startup rather than letting that happen quietly.

With `retest_reward` 1.0 against `leak_penalty` -1.0, a teacher that leaks and
scores 4/4 lands on 0.0 and loses to an honest teacher scoring 1/4.

ReBN runs with `turn_discount: 1.0`, so the re-test reward propagates back to
every round of the episode with equal weight. That is the intended default here —
all five rounds jointly produced the outcome and nothing in the episode says which
one mattered.

## The two leak arms

| | `leak-reward` | `leak-terminate` |
| --- | --- | --- |
| conversation | runs the full budget | stops at the leaking turn |
| re-test transcript | full, including the leak | **excludes the leaking turn** |
| reward | `retest + leak_penalty` | `retest(truncated) + leak_penalty` |

Terminate is the harsher arm twice over: the penalty, plus the loss of whatever
the remaining rounds would have taught, and the leaked answer buys nothing because
the student never sees it. A leak on turn 1 leaves no completed round, so the
re-test is skipped rather than run on an empty transcript and the episode is worth
the penalty alone.

## What to watch

**The objective.** Two numbers, off the same eval rollout.

    generalize/test/student_original_success          IN THE WILD  <- headline
    generalize/test/student_original_preleak_success  train-consistent

`evaluator.leak_terminate: false` makes the eval conversation run the full budget
however much the teacher gives away. Terminating on a leak is a *training policy*:
it is there to make leaking expensive while learning, and nothing truncates a real
conversation, so it must not decide what gets measured. The leak judge still runs,
so `rollout/leaks` tells you whether it leaked -- you get the leak rate and the
untruncated outcome, instead of one number that confounds them.

The second is what the terminate arm would have scored: the same re-test on the
transcript through the last completed round before the first leak. It is derivable
from the untruncated rollout because that prefix is identical either way, so the
two numbers are paired on one sample rather than on two eval passes -- much less
noise between them. It is never rewarded, and it only appears at eval on an arm
whose training mode is `terminate`. A leak on round 1 leaves no prefix, and that
episode scores 0 for the train-consistent number.

Both are already per-episode means over the 4 replays. `solved`, `final_correct`
and `student/*/solved` all report the in-the-wild number too, so nothing reads as a
flat zero any more.

NOTE this changes what `student_original_success` means for the terminate arm at
eval: it used to be the truncated transcript, it is now the whole conversation.
Eval rows logged before this are not comparable with the ones after. leak-reward is
unaffected, since it never truncated.

A version-0 pass on 20260810_2341 measured 0.620 for leak-reward and 0.518 for
leak-terminate, but that run used `teacher_history_tags: stripped` and is
superseded: format errors ran at 0.62 per episode there, so those numbers are a
baseline for a teacher that was malformed a third of the time. Re-measure under
masked before treating anything as the reference.

`student_original_success_binary` tracks attempt 1 only and is the noisier series;
read the fraction.

**Whether there is any gradient at all.** The reason for k=4 is group variance, so
check it directly: the spread of `student_original_success` across the 8 rollouts
of a problem. If groups are still constant, the finer reward did not buy anything
and the budget or the split is the thing to change.

**Teacher format drift.** `rollout/format_errors` and `stop/format_error`. Under
`format_handling_mode: continue` a malformed turn hands the student an empty
message and the episode keeps going, so this shows up as reward noise rather than
as a stop.

This is the number that forced `teacher_history_tags: masked`. Measured per depth
on 143 episodes of the stripped run:

    depth      1      2      3      4      5
    malformed  6.3%   8.0%  23.9%  34.1%  41.7%

and per episode, same model, three settings:

    0808 stripped, answer-attempt, 2.1 turns avg    0.26 -> ~0.10 by step 12
    0808 masked,   answer-attempt, 2.1 turns avg    0.15 -> ~0.06 by step 12
    0810 stripped, free chat, 5 forced turns        0.62, flat over 6 steps

The fixed budget is what makes it bite: every episode now reaches the depths where
the rate is high instead of stopping at 2.1 turns. If masked does not bring this
under ~0.2 within ten steps, shorten the budget before touching anything else.

Note that `terminate` is now *safe* here in a way it was not in `math/0808/full`:
there an early format exit dodged `max_turn_penalty -1.0`, which made it a cheap
way out of a losing episode. With `max_turn_penalty 0.0` a format-terminated
episode collects `-0.5` and loses its re-test, which is strictly worse than the
worst honest outcome of `0.0`. If format drift becomes the dominant failure, this
is the knob.

**Leaks.** `rollout/leaks`, `stop/leak`. Compare the two arms.

**Series that are identically zero by construction**, because there is no per-turn
judge: `solved`, `solve_turn`, `final_correct`, and every `depth/*/resolved` and
`depth/stuck_episode_solved`. The repetition half of the depth metrics
(`depth/*/repeats`, `depth/*/containment_sum`) still works — it only reads the
teacher's own text — so tutor repetition is still measurable across depth.

## Cost

Every episode runs the full budget instead of the measured 2.1 turns, so this is
roughly **2x** an `0808/full` episode: 5 teacher generations, 5 student calls, 4
re-test calls, 4 re-test judge calls, and up to 5 leak checks.

**Watch the student endpoint.** A training step is 16 problems x 8 rollouts = 128
episodes x 4 re-tests = **512 student calls that all arrive at the end of the
rollout wave**, against `student_models[0].max_concurrent_calls` of 16. That
number was set for a rollout that made one student call per turn, spread out over
the episode. If steps stall at the tail, this is why.

## Not comparable with anything earlier

The reward is a different quantity, the episode length is fixed, and the student is
answering a question it was never shown during the conversation. Compare within
this group only.
