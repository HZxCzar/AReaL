# 0808 — does the repair instruction survive training, and where does it live?

Three arms, run together, differing in exactly one thing each.

| config | training | deployment | one-line question |
| --- | --- | --- | --- |
| `baseline.yaml` | — | — | control |
| `prompt.yaml` | instruction in the prompt | instruction in the prompt | what is a sentence in the prompt worth *after* training? |
| `opd.yaml` | distilled into the weights | nothing in the prompt | can the same effect be moved into the weights? |

All three inherit `math/0805/v2/…-g8k4-leakt` unchanged. Cross-directory
inheritance works via `hydra.searchpath`, which the loader honours; without it
Hydra's search path is the config file's own directory.

## What is being tested

Measured on the current policy, appending one sentence to the end of the teacher
prompt — *"your previous message did not get through: the student is still
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
