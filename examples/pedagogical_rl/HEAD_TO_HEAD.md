# Tutor vs PedagogicalRL: two protocols x two scorers, over an SFT floor

**Each method trains its own way. At evaluation both are measured on both
protocols and both scorers, so the eight-cell table separates the teacher from
the protocol it was trained in. A third arm distils the teacher's answers into
the student with no teaching at all, which is the floor the other two have to
clear.**

## The three arms

| | ours | theirs | SFT floor |
| --- | --- | --- | --- |
| config | `examples/tutor/configs/math/0810/4gpu/leak-local-stable-xeval.yaml` | `examples/pedagogical_rl/configs/qwen3_8b_qwen3_1_7b_math_pass2_baseline_xeval.yaml` | `examples/tutor/configs/sft/qwen3-1.7b-distill.yaml` |
| launcher | `examples/tutor/run_official.sh` | `examples/pedagogical_rl/run_official.sh` | `examples/math/gsm8k_sft.py` (stock) |
| what trains | the teacher | the teacher | **the student** |
| dialogue | teacher opens, 5 fixed rounds, nothing judged in between, student never shown the problem | GUIDED/ATTEMPTED classroom, up to 10 teacher turns, student answers in-band | none |
| reward / loss | the delayed solo re-test, 4 replays, five levels | their four native terms, whole-dialogue judges, 8 final attempts | cross-entropy on the teacher's own solutions |
| leak gate | AReaL's turn-level rawbase judge, terminating, penalty local to the leaking turn | their whole-dialogue judge | n/a — the answer is the target |
| teacher pre-solve | verified draft + row filter | off | n/a |
| GPUs | 4 | 2 | 2, and only for minutes |
| steps | 500 | 750 | 3 epochs |

The two RL arms train the same base teacher (Qwen3-8B, rank-16 LoRA) against the
same frozen student (Qwen3-1.7B) on the same data (`math_1.7b_8b/math_pass@2`),
and one outer step is 16 problems x 8 rollouts = 128 episodes on both, so the
W&B x-axis is directly comparable without rescaling.

The SFT arm is the odd one out on purpose: it is the only arm that changes the
student. It asks what the teacher can transfer with no interaction at all, so
whatever the taught arms score above it is what teaching bought over plain
distillation.

## The cross

At every eval pass each arm runs the OTHER protocol's dialogue with its own
teacher, then scores both transcripts with both scorers.

    xeval/free_chat/retest/success       our rollout,   our score
    xeval/free_chat/interview/success    our rollout,   their score
    xeval/classroom/retest/success       their rollout, our score
    xeval/classroom/interview/success    their rollout, their score

Both arms emit those same four names, and the SFT arm fills a fifth column with
the same two scorers on an empty transcript:

    xeval/no_dialogue/retest/success       no teaching, our score
    xeval/no_dialogue/interview/success    no teaching, their score

so the table is:

|  | free_chat + retest | free_chat + interview | classroom + retest | classroom + interview | no_dialogue + retest | no_dialogue + interview |
| --- | --- | --- | --- | --- | --- | --- |
| **our teacher** | | | | | | |
| **their teacher** | | | | | | |
| **base student** | | | | | ✓ | ✓ |
| **SFT student** | | | | | ✓ | ✓ |

The diagonal cells are each method measured the way its own paper measures it.
The off-diagonal cells are what the comparison is for: a teacher that only wins
under its own protocol and its own scorer has not been shown to teach better. The
no_dialogue column is the floor — `base student` is what the problem is worth
unaided, `SFT student` is what distillation alone buys, and a taught cell that
does not clear the SFT row has not shown teaching to be worth its cost.

An empty transcript is a complete prompt for both scorers, not a degenerate one:
ours puts the task in the final turn and theirs in the system prompt, so neither
needs a dialogue to be well-formed.

Also per cell: `success_any`, `incomplete`, and `success_math` (interview only —
the same 8 samples under AReaL's normalizing math scorer). Per protocol:
`turns`, `leak/turn`, `leak/native`, and `format_errors`.

## What the two axes actually are

**Protocol** is the dialogue. Ours gives the teacher a fixed five-round budget
and never tells the student what the conversation is about; theirs runs a
classroom where the student attempts the problem in-band and an ATTEMPTED
episode opens with its own attempt.

**Scorer** is what happens afterwards. Ours replays the transcript on a fresh
branch, shows the problem for the first time, takes four independent solo
attempts and judges them with AReaL's exact math scorer followed by its LLM
answer judge. Theirs has the student re-read the dialogue in their
student-perspective form and produce eight solutions from ONE n-choice request,
scored by a lowercased string compare on the last `\boxed{}` span.

The scorers are not the same quantity even on one transcript. Theirs calls
`\boxed{\frac{1}{2}}` wrong against a ground truth of `1/2`; ours calls it
right. That is why `success_math` exists — it re-scores their eight samples our
way, so a ranking that only survives one scorer is visible as such.

## One instrument, not two

`examples/pedagogical_rl/cross_eval.py` holds both protocols and both scorers and
is imported by both arms. Each arm supplies plain async callables for its own
teacher, student and judge clients; nothing about the protocol or the scoring is
reimplemented on either side. The configuration is likewise one dataclass tree,
`examples/pedagogical_rl/cross_eval_config.py`, imported by both config modules,
and the two YAML `cross_eval` blocks are byte-identical.

`tests/test_ped_2x2_configs.py` is what enforces that. Nothing at runtime can
notice the two arms drifting apart — each reads its own YAML and neither can see
the other's — so these are the checks that stand in for it:

    the two cross_eval blocks are identical
    our arm changes nothing about its own training vs leak-local-stable
    their arm changes only freq_steps and average_rollouts vs their baseline
    the crossed classroom protocol equals the one they train with
    retest.replays equals our student_generalize.replays
    interview.attempts equals their number_student_attempts
    both arms evaluate on the same steps and the same data
    each arm still trains its own way

Run it, plus the instrument, adapter and SFT suites, after any edit:

```bash
set -a && . ./.env && set +a && PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_cross_eval.py tests/test_cross_eval_arms.py tests/test_ped_2x2_configs.py tests/test_sft_distill_arm.py -q
```

## Launching

Six of the eight GPUs, so both arms fit at once. Stagger them: forked-worker
ports are allocated per run with no coordination between runs.

```bash
bash examples/tutor/run_official.sh examples/tutor/configs/math/0810/4gpu/leak-local-stable-xeval.yaml
```

```bash
bash examples/pedagogical_rl/run_official.sh examples/pedagogical_rl/configs/qwen3_8b_qwen3_1_7b_math_pass2_baseline_xeval.yaml
```

## The SFT floor, in three steps

Minutes of GPU, not days, and it can run before or after the RL arms.

**1. Generate.** The teacher answers the 759 training problems under **the
zero-dialogue re-test prompt** — the exact input the student is shown in
`xeval/no_dialogue/retest` — and the same judge both RL arms score with accepts or
rejects each attempt. The builder and the scorer both call
`cross_eval.retest_messages(transcript=[], ...)`, so the training input cannot
drift from the eval input, and two tests assert they are equal. Do not write under
`examples/tutor/data` — that is a symlink to the shared data directory every
worktree reads.

```bash
set -a && . ./.env && set +a && PYTHONPATH=$PWD .venv/bin/python examples/tutor/scripts/build_sft_distill_dataset.py --out /inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/sft/data/distill_qwen3_8b
```

Up to 3 tries per problem, stopping at the first accepted solution, so **every row
is correct and every problem contributes exactly one row**. A problem the teacher
never solves in 3 tries is skipped rather than trained on with a wrong target.

One row per problem is the fair choice: keeping every correct sample would give a
problem the teacher finds easy several rows and a hard one a single row, training
the student mostly on what was already easy — the part of the split where teaching
has least to add. It is also the unit the RL arms train on, so "all three arms
train on the same 759 rows" is literally true. `--per-task 0` builds the larger,
imbalanced set for comparison.

Expect ~750 examples. On a 30-problem smoke run: 30/30 solved, coverage 1.0, mean
1.07 attempts per problem, per-attempt pass rate 0.94 — and 0.90 under the neutral
math template, so wearing the student persona costs the teacher nothing. Early
stopping also makes this cheaper than a fixed sample count.

It writes `samples.jsonl` (every attempt with its verdict and whether it was
kept), `correct/` (the tokenized dataset), and `summary.json` with `coverage`,
`problems_skipped`, `teacher_pass_rate` and `mean_attempts_used`. **Quote
`coverage` next to whatever this arm scores** — a floor built on 95% of the split
is a floor on 95% of the split.

**2. Train.** No tutor-specific training code; the stock SFT entry point reads the
config.

```bash
PYTHONPATH=$PWD .venv/bin/python examples/math/gsm8k_sft.py --config examples/tutor/configs/sft/qwen3-1.7b-distill.yaml
```

`saver.freq_epochs: 1` keeps all three epochs, so where to stop is measured in
step 3 rather than assumed.

**3. Score,** with the same two scorers the RL arms use.

```bash
set -a && . ./.env && set +a && PYTHONPATH=$PWD .venv/bin/python examples/tutor/scripts/eval_sft_student.py --checkpoint <saver output> --out /inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/sft/results.json
```

**Quote the base-to-SFT difference, not the raw number.** The RL arms reach their
student over the INF endpoint; this script serves a checkpoint locally with
SGLang, and the two are not bit-identical — different batching, different kernels,
and the endpoint's pinned seed does not pin sampling. That is why the script
measures the base student through the same harness by default: the difference is
within-harness and the absolutes are not directly comparable with the
endpoint-measured columns.

## Cost

A crossed problem costs roughly 50 external calls against the ~24 an ordinary
episode makes: one extra dialogue (up to 10 teacher and 10 student calls), 4
student plus up to 4 judge calls per re-test, one n-choice request per interview,
and up to 15 leak-judge calls.

`cross_eval.sample_rate` is **1.0** in both configs — the whole 528-problem eval
split, no subsampling. This is the measurement the comparison rests on, so it is
not estimated from part of the split. That is about 25 h of extra evaluation per
arm over 500 steps at `freq_steps: 25`, and about 2.2 pp standard error per cell
in a single pass (about 3.1 pp on a difference between two cells) before the ~20
passes are pooled.

The knob still exists for a smoke run: below 1.0 it selects on a stable hash of
the problem text, so both arms would cross exactly the same subset without
agreeing on a seed or an ordering. If you use it, change it on both arms or the
columns stop being paired. For a real run that is too long, lower
`evaluator.freq_steps` on both arms instead — that costs points on the curve
rather than precision at each point.

`cross_eval.run_other_protocol: false` keeps only the two scorers on the arm's
own transcript. That is the cheap half: it answers the scoring question and not
the protocol one, and it collapses the table to four cells.

If eval passes stall, the student endpoint is the first place to look. The ped
arm's `student_model.max_concurrent_calls` is **4** against **32** on the tutor
side.

## Reading it, and what it cannot tell you

**A free cross-check first.** `xeval/free_chat/retest/success` on our arm is our
own headline re-measured through the shared instrument, so it should track
`generalize/test/student_original_success`. It is computed independently rather
than copied, precisely so it can disagree. If those two series separate, the
instrument is not reproducing what our arm is rewarded on and no column of the
table is trustworthy — fix that before reading anything else.

**Leaks.** Each arm trains against one leak judge, so each arm read under its own
judge flatters itself. `leak/turn` and `leak/native` are both recorded on both
transcripts of both arms, as metrics only — neither terminates a rollout at eval
and neither touches a reward. `success_and_leak/*` is logged only on a successful
rollout, so P(leak | success) is the ratio of those two means; it is deliberately
not logged as a per-rollout ratio, which is undefined for a failed rollout and
wrong to average.

**An `incomplete` cell is missing data, not a zero.** A dead student call, a
mismatched n-choice response, or a teacher call that failed mid-dialogue all mark
the cell incomplete and score 0.0, and the rate is reported. Read `success`
alongside `incomplete`; a cell with a high incomplete rate is not a low score.

**What the table cannot separate.** Our arm trains with a verified private
teacher draft and the row filter that comes with it; their baseline has neither,
because their method does not have that stage. That is what "each method trains
its own way" means here, and it means a win is a win for the method as
configured, not for the rollout design alone. `stop/teacher_pre_skipped` has run
between 1.6% and 12% across runs — those rows never reach our teacher and do
reach theirs. `qwen3_8b_qwen3_1_7b_math_pass2_teacher_pre_verified.yaml` is the
arm that would measure it, at the cost of a third training run.

**One asymmetry is deliberate and cannot be removed.** Their student's system
prompt contains the problem statement; ours does not. So the interview scorer
always tells the student what it is working on and the retest scorer only tells
it at the end, on both arms and on both transcripts. Normalising it would mean
scoring neither arm the way its own paper does.

**The two RL arms stop at different steps.** 500 for ours, 750 for theirs. Neither
is truncated to the other; read each curve against step and stop comparing where
the shorter one ends.

**The SFT arm is aligned to our scorer and not to theirs.** It is trained on the
zero-dialogue re-test input byte for byte, which is deliberately the best case for
our column: if teaching cannot beat a student fine-tuned directly on the target
format, the floor was never in doubt. Their interview scorer prompts the student
its own way, so `xeval/no_dialogue/interview` is measured under a prompt this
student never saw and is the weaker of its two cells by construction. Read the
retest cell as the floor for our column and the interview cell as a transfer
number — not as the floor for theirs. `--train-prompt math` builds the neutral,
symmetric version if the even-handed floor is wanted alongside.

**The floor covers only the problems the teacher can solve.** Problems it misses
in 3 tries are skipped rather than distilled wrong, so `coverage` in
`summary.json` bounds what the row means. If coverage is well under 1.0, the
skipped problems are by definition the hard ones — the same ones where the taught
arms have the most room — so a taught arm beating this floor on the full split is
partly beating it on problems the floor never trained for.

**The SFT arm's teacher is the base teacher**, not either trained checkpoint. That
is what makes it the no-teaching control: the same Qwen3-8B both RL arms start
from, before any teaching-specific training. Distilling a *trained* teacher is a
different question, and it would need that checkpoint served.

**Every row already carries a human-written solution** in `metadata['solution']`,
and training on that would be a different and probably stronger floor. It is
deliberately unused: this arm is about what the teacher in this experiment can
transfer, not about what the best available data can.
