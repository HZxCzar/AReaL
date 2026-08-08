# 0808 — what should the teacher be told, and does it survive training?

Four arms, run together, differing in exactly one thing each.

| config | told | training | deployment | one-line question |
| --- | --- | --- | --- | --- |
| `baseline.yaml` | — | — | — | control |
| `prompt.yaml` | repair | in the prompt | in the prompt | what is a sentence in the prompt worth *after* training? |
| `opd.yaml` | repair | distilled | nothing in the prompt | can the same effect be moved into the weights? |
| `opd-handback.yaml` | handback | distilled | nothing in the prompt | was repair the wrong thing to say? |

The first three are one experiment about *placement*, holding the sentence fixed.
The fourth changes the sentence, holding placement fixed. Read `opd-handback`
against `baseline` for "did it help" and against `opd` for "was repair the wrong
target".

All four inherit `math/0805/v2/…-g8k4-leakt` unchanged. Cross-directory
inheritance works via `hydra.searchpath`, which the loader honours; without it
Hydra's search path is the config file's own directory.

## What is being tested

Measured on the current policy, adding one sentence to the teacher's system
prompt from turn 3 onward — *"your previous message did not get through: the student is still
answering incorrectly. Do not restate what you have already said. Explain the next
step in more detail and at a finer grain than you did before"* — is worth
**+7.9% [+4.3, +11.5]** on stuck states, via better message content plus an
11-point drop in answer leakage. It is not a change of teaching move: the move mix
barely shifts.

Two things that measurement cannot say, and that these runs are for:

1. **Does it still pay after RL?** Every number above is one application to the
   *current* policy at a fixed state. RL might find the same behaviour on its own,
   in which case the sentence only raises the starting point.
2. **Does it have to be in the prompt?** Noticing across a long context that the
   same thing keeps failing looks like a capability, and a sentence in the prompt
   does not install a capability. If that is right, `prompt` lands near
   `baseline` while `opd` does not.

## The fourth arm, and why the sentence might be wrong

Everything above targets one failure: the tutor repeating itself while the student
stays stuck. The 30-task collection dump says the expensive failure is a different
one.

Same 30 tasks, same 1.7b student, five teachers, counting a task as won only if
the student got it *and* the teacher never stated the answer:

| | success | leaked | **clean win** | turns |
| --- | --- | --- | --- | --- |
| gemini | 100% | 17% | **83%** | 1.93 |
| qwen_untrained | 93% | 47% | 50% | 2.30 |
| trained_leak1 | 100% | 57% | 43% | 2.30 |
| leakt | 57% | 23% | 47% | 5.43 |
| trained_leak0 | 100% | 73% | **27%** | 2.30 |

Paired on the same tasks, gemini is +33.3% [+13.3, +53.3] over `qwen_untrained`,
+40.0% [+20.0, +60.0] over `trained_leak1`, +36.7% [+16.7, +56.7] over `leakt` and
+56.7% [+40.0, +73.3] over `trained_leak0`. The three qwen arms at 100% success
are buying it by leaking. (`Leak count` is in the header of the trained dumps;
gemini and `qwen_untrained` have no such field, so the column is recomputed the
same way for all five — every numeric atom of the ground truth appearing in one
teacher message — and agrees with the header on 24-27 of 30 where a header
exists.)

`leakt` shows the reward can already suppress leaking: terminating the episode on
a leak takes the rate to 23%. It also takes success to 57%. What the reward cannot
supply is the thing gemini puts in the leak's place. On first messages gemini
hands the work back on 93% of tasks against 33-47%, asserts the fix on 30% against
80-83%, brings in 6.2 distinct mathematical objects against 2.9-3.5 of which only
44% were already in the student's own work against 53-62%, and ends on a question
57% of the time against 3-17%. `TEACHER_HANDBACK_INSTRUCTION` is those three
numbers written as a sentence.

Note what this is *not* evidence for. At the level of coarse move category gemini
is more uniform than the qwen teachers, not more varied — shape entropy 2.05
against 2.33-2.73. It plays one move almost everywhere. What varies per task is
which idea it drags in, which is a level below any label we have. That is the same
conclusion as "Not here" below, reached from the other direction.

## Reading the result

```
opd ~ baseline                        the distillation did nothing
prompt ~ baseline                     the sentence only ever raised the starting
                                      point; drop this line of work
opd ~ prompt > baseline               the prompt's effect is now in the weights
prompt > opd > baseline               only partly transferred; raise opd.loss_weight
opd > prompt > baseline               distilling beats prompting outright
```

`prompt` is the arm that makes any of this a claim rather than an anecdote.
Appending a sentence costs nothing, so OPD has to beat *that*, not beat nothing.

## Where the sentence goes

At the end of the **system prompt**, added per turn once the gate fires, on top of
an otherwise identical `teacher_system_prompt`. Not at the end of the message
list: the teacher's prompt is a real conversation -- its own turns assistant, the
student's user -- so a directive appended there is attributed to the student, and
from turn 3 on some student turns would carry one while earlier ones do not.

This differs from where the +7.9% was measured. That harness rendered the whole
state as a single user message, where the tail is the end of a state description
rather than someone's utterance, and tail placement was followed more reliably
than the system prompt there. Generation here uses the chat format, so that
comparison does not transfer and the effect size measured under it is not
guaranteed to carry. If the run comes back flat, re-measuring the instruction in
the chat format is the first thing to check, not the last.

## Metrics that decide whether the run is even valid

- `opd/selected_ratio` — share of turns OPD supervised. On the first attempt this
  was 8.8% of trainable tokens. Near zero means the turn gate is too strict for
  how short episodes actually are; lower `opd.min_prior_failed_turns` to 1.
- `opd_reverse_kl` — per-token reverse KL in nats, negative while the instructed
  teacher still prefers something the policy does not. It should rise toward 0 as
  the instruction is absorbed. Flat and far from 0 means it is not being absorbed
  and `opd.loss_weight` is the knob.
- `prompt_instruction/share_of_turns` — must land close to `opd/selected_ratio`,
  otherwise the two arms are not seeing the instruction at comparable rates and
  the comparison is not clean.
- The usual: `final_correct`, `leaks`, grad norm, entropy.

For `opd-handback` specifically, `opd/selected_ratio` should be much *higher* than
in `opd`: the turn gate is off, so it is bounded by `max_turns_per_episode: 2` and
by the leak skip rather than by turn 3 being reached at all. If it comes back
starved, `skip_leaked_rows` is the knob — leaked rows are exactly the ones this
instruction disagrees with most, and they are excluded only to avoid stacking an
unclipped KL on tokens that already carry a terminal penalty.

## Not here

`guided_slots` exists in the code but no arm uses it. Its premise was that which
teaching move suits a problem is learnable and transferable, worth +7.2% at a
one-turn horizon. At the three-turn horizon that measurement reverses to
**−3.1% [−6.1, +0.0]** over 491 states: choosing a move per task is *worse* than
always playing the single best one, and no diagnosis-conditioned rule beats
DECOMPOSE in any subgroup. Since 41.2% of successes arrive after the first turn,
three turns is the horizon that counts. The mechanism stays in the tree, off by
default; the evidence for running it does not exist.

That result is scoped to a six-label move taxonomy. It says the *category* of move
carries no conditional information — not that adaptation is worthless. Whether to
go finer, and how much finer, lives inside a category, and that is what OPD acts
on.

The collection dump agrees independently. The best teacher on that set plays the
same coarse move nearly everywhere (handback on 93% of tasks, category entropy
2.05 against 2.33-2.73 for the qwen teachers) and varies underneath it. A
slot mechanism that prescribes a category is the wrong instrument for that, and a
token-level one is the right shape.
