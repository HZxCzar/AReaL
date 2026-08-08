# Guided slots + repair OPD — implementation plan

Branch `feat/guided-opd`, worktree
`/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL.worktrees/feat-guided-opd`,
forked from `feat/diversity` @ f161d194.

Baseline config:
`examples/tutor/configs/math/0805/v2/qwen8b-train-qwen1.7b-eval3-math-pre-aleak-generated-g8k4-leakt.yaml`
(G=8, max_turns=10, `rebn` + `group_baseline: episode` + leave-one-out,
`use_decoupled_loss: true`, `behave_imp_weight_cap: 5.0`, `eps_clip: 0.4`).

## 1. What is being built

**Across-task (point 1).** The trained policy plays one move: PINPOINT 86.3% of
turns, entropy 1.77 → 1.16 bits over training. It is not a capability loss —
checkpoint step199 is indistinguishable from the base model on per-move quality,
leak rate and instruction compliance. It is a propensity collapse, and it starves
the group of the comparisons GRPO needs: 8 free rollouts contain the state's best
move only 37.7% of the time (32 rollouts: 66.6%). Reserving 3 of the 8 slots for
move-prescribed generation lifts coverage to 78% and best-in-group value from
29.1% to 40.7% of the available gap. Which move is best is a **task**-level
property that transfers to unseen problems (+7.2% [+3.8, +10.8] over one fixed
move; cross-task kNN +7.9% [+2.1, +13.7]; not explained by leak avoidance,
+8.1% among never-leaking actions).

**In-task (point 2).** Per-state move choice adds nothing over per-task move
choice, even with an oracle (−1.4% [−5.0, +2.2]). So in-task adaptation is not
about *switching strategy*; it is about *message content*. Re-generating the turn
with one added sentence — "your last message did not get through, do not restate
it, explain the next step at a finer grain" — is worth **+7.3% [+3.7, +10.9]**
(+8.9% on repeat-heavy states). Mechanism is better content plus an 11-point leak
reduction, not a different move (move mix 14.2% → 13.6%). Meanwhile instructions
that *prescribe a strategy* are worth nothing on average (+1.7% [−4.8, +8.2])
despite moving the move mix from 1.1% to 27.9% — they pay off only through group
coverage, which is exactly what point 1 uses them for.

## 2. Two mechanisms, not one

They look similar and are not. Keeping them separate is what makes a result
attributable.

**Guided slots (point 1)** generate the turn with a move instruction appended and
then train on the instruction-free prompt. That is deliberately off-policy, and
three pieces of existing machinery carry it:

| need | existing machinery |
| --- | --- |
| generate on prompt A, train on prompt B | `input_tokens_override` in `response_to_tensordict` (`examples/tutor/core/tensors.py:47`), already used to strip the prompt-pool suffix |
| off-policy correction for that mismatch | `use_decoupled_loss: true` — `prox_logp` is recomputed by a forward pass on the **training** `input_ids`, and `behave_imp_weight = exp(prox_logp - logprobs)` is capped by `behave_imp_weight_cap: 5.0` (`areal/trainer/ppo/actor.py:1788-1860`) |
| guided and free rollouts sharing one baseline | `GroupedRolloutWorkflow` returns all G episodes as one trajectory dict; `_compute_episode_group_baseline` averages over episodes in a group (`areal/trainer/ppo/actor.py:578`) |

So point 1 adds no loss term at all. The learning signal is the ordinary GRPO
advantage; the instruction only moves where the samples come from.

**OPD (point 2)** does not touch the rollout. The prompt the policy generates
from is unmodified and on-policy; the instruction is given only to a teacher that
scores those same tokens after the fact. Nothing about it is off-policy, and it
does add a loss term. See §5.

Because the two differ in exactly this way, OPD skips any turn that carried a
guided instruction (`opd.skip_guided_rows`): a row with both is a row where
neither effect is attributable.

## 3. Design decision: no prefix-sharing branching in phase 1

"Branching GRPO" as originally sketched meant: run a rollout to turn *t*, clone
the state, fork *k* continuations. That is not what buys the coverage. The payoff
we measured is **task**-level, and the G=8 rollouts of a group already share the
task. Reserving slots inside the existing group therefore gets the full
+7.2%-shaped signal with none of the state-cloning machinery.

Prefix sharing remains worth doing later, but for a different reason —
variance, not coverage: 26.9% of episodes contribute zero trainable turns, 14.9%
end in a leak, 5.2% of groups are degenerate, and episode-level credit is ~10×
noisier than state-matched credit. Deferred to phase 3.

## 4. Changes, file by file  (implemented; see `git diff` on this branch)

### 4.1 `areal/infra/remote_inf_engine.py` — slot index into the group

`GroupedRolloutWorkflow.arun_episode` (line 96) currently fans out G identical
calls. Pass the index, opt-in so the other worktrees' workflows are unaffected:

```python
wants_index = bool(getattr(self.workflow, "wants_group_index", False))
results = await asyncio.gather(*[
    self.workflow.arun_episode(
        engine,
        {**data, "group_index": i, "group_size": self.group_size} if wants_index else data,
    )
    for i in range(self.group_size)
])
```

`TutorAgentWorkflow` sets `wants_group_index = True` only when guidance is
enabled, so behaviour is byte-identical when the feature is off.

### 4.2 `examples/tutor/prompts.py` — the instruction texts

Add verbatim from the validated measurement code
(`analysis/hazard_20260806/common.py` `FORCED_MOVES`,
`analysis/hazard_20260806/gate1_supervisor.py` `REPAIR_INSTRUCTION`):

- `TEACHER_MOVE_INSTRUCTIONS: dict[str, str]` with the six parallel-form entries
  PINPOINT / DECOMPOSE / REFRAME / HINT / PROBE / WORKED.
- `TEACHER_REPAIR_INSTRUCTION: str`.

Do not paraphrase. Every number above was measured against these exact strings.

### 4.3 `examples/tutor/configs.py` — config surface

Two independent blocks on `TutorConfig`, so either can run alone:

```yaml
guided_slots:
  enabled: false
  slots: 3                            # of gconfig.n_samples = 8
  moves: [DECOMPOSE, REFRAME, PROBE]  # provisional, see §7
  turns: [1]                          # where the group still shares a state
  rotate_by_task: true                # offset move->slot by a task hash

opd:
  enabled: false
  loss_weight: 0.05
  instruction: ""                     # empty = the validated repair wording
  min_prior_failed_turns: 2           # i.e. turn 3 onward
  max_turns_per_episode: 0            # 0 = uncapped
  reward_clip: 5.0                    # nats, per token
  skip_guided_rows: true
  skip_leaked_rows: true
```

Validation refuses, with a message explaining why rather than just what:
`slots >= gconfig.n_samples` (no free rollout left to compare against), unknown
move names, `n_samples < 2`, `guided_slots` without `actor.use_decoupled_loss`
(nothing would correct the rewritten prompt), non-positive weights or clips.

### 4.4 `examples/tutor/workflow.py` — the injection

1. **`TutorTurnState`** — add `guidance: TeacherGuidance | None = None` where
   `TeacherGuidance` is a small frozen dataclass `(kind, name, instruction)`,
   `kind ∈ {"move", "repair"}`.

2. **`_run_episode`** (line 1571) — read `data.get("group_index")` once; inside
   the turn loop (line 1810) call a new `_select_guidance(group_index, turn_idx,
   consecutive_failures)` and pass the result into `TutorTurnState`. Returns
   `None` when: guidance disabled, `workflow_context.get().is_eval`, this slot is
   not reserved, or the turn is not eligible. Move-vs-repair precedence: repair
   wins when its trigger fires, because at turn ≥2 the repair instruction is the
   one with a measured effect.

3. **`_build_tutor_messages`** (line 3994) — add `include_guidance: bool = True`.
   When a guidance is present and included, append to the **last** message if it
   is a `user` turn, else append a new `user` turn:

   ```
   \n\nInstruction for this reply:\n{instruction}\n
   ```

   The position matters and is not cosmetic: the same text placed in the system
   prompt is followed far less often than when it sits immediately before
   generation. This is where the measurement put it.

4. **`_clean_tutor_messages`** (line 4038) — today it reuses
   `artifact.tutor_messages[1:]` verbatim, which would keep the injected
   instruction in the training prompt. Rebuild instead:

   ```python
   return self._build_tutor_messages(artifact.tutor_state, clean=True,
                                     include_guidance=False)
   ```

   Deterministic from the state, so the prefix still matches what the model
   generated from, minus the guidance.

5. **`_run_episode` line 2053 — the gate that will silently break this.**
   `clean_tutor_inputs` is computed only
   `if artifact.tutor_state.teacher_prompt_selection is not None`. With the
   prompt pool off and guidance on, the override would be `None` and the
   instruction would be trained on. Change the condition to
   `... is not None or artifact.tutor_state.guidance is not None`.

6. **`_clean_teacher_input_token_reserve`** (line 4152) — returns
   `max(0, clean_len - rollout_len)`. The clean prompt is *shorter* under
   guidance, so the reserve stays 0 and no change is needed; but it early-returns
   on `selection is None`, so pass guidance through and keep returning 0
   explicitly rather than relying on the early return.

7. **`_log_rollout_stats`** (line 4188) and `_maybe_dump_debug_trace` — record
   `guidance/kind`, `guidance/name`, `guidance/slot`, and per-move
   `selected / solved / reward`, mirroring the existing
   `prompt_source/{warmup_full,pool,pool_base}/…` block at line 4280.

### 4.5 New config

`examples/tutor/configs/math/0805/v3/qwen8b-…-guided3.yaml`, inheriting the v2
baseline and setting only the `teacher_guidance` block. Keep
`reward.teacher_diversity` and `reward.teacher_context` **off**: the bge-m3
cosine cannot detect restatement (r=+0.024 vs a judge's r=−0.136) and the
position-swap logprob scores non-adaptive policies just as highly.

## 5. Point 2: on-policy distillation, as the reference defines it

Verified against the Thinking Machines blog post and
`tinker_cookbook/distillation/train_on_policy.py`. The algorithm:

```
reverse_kl = log pi_sampled(a_t) - log pi_teacher(a_t)     # sampled token only
advantages = advantages - coef * reverse_kl                # discount 0
<ordinary importance-sampling / PPO loss, unchanged>
```

Four points, all of which the first draft got wrong and are now fixed:

1. **Sampled token only, not the full vocabulary.** The reference queries the
   teacher with `compute_logprobs` on the student's trajectory and gets back the
   teacher's log-prob for the tokens the student actually produced. No top-k, no
   full-vocab KL. So no engine work is needed here — what `compute_logp` already
   returns is exactly what the reference uses.
2. **The student side is the sampling-time log-prob**, not a recomputed
   current-policy one. It is data, fixed for the whole update.
3. **It enters as a per-token advantage penalty**, not as a term in the loss.
   That is what lets the distillation signal inherit the PPO ratio, the clipping
   and the behaviour-importance weighting for free, and it is also what makes the
   gradient treatment correct without any special handling: by the time the
   penalty is applied, both log-probs are plain data, so nothing differentiates
   through the sampling distribution.
4. **Discount factor zero.** The penalty lands on the token that produced it and
   is not accumulated forward, so it must be added *after* the task advantage is
   complete and must never enter GAE. The blog is explicit that this is chosen for
   empirical reasons rather than mathematical correctness.

Defaults follow the reference: `loss_weight` is the reference's
`kl_penalty_coef`, default **1.0** (its scale comes from the KL being in nats, so
it is not comparable to a loss weight). `reward_clip` defaults to **0 = off**; the
reference does not clip, and the knob exists only in case a single outlier token
is seen dominating `opd_advantage`.

Two deliberate departures from the reference, both forced by this codebase:

- The reference runs OPD as a standalone distillation phase. Here it runs jointly
  with GRPO, so `coef` trades off against a task advantage rather than being the
  whole signal. No precedent for the balance; 1.0 is the reference default and a
  starting point, not a tuned value.
- The reference's teacher is a separate, stronger model. Here the teacher is the
  same weights conditioned on an instruction the student never sees — **on-policy
  context distillation**. That is a recognized setup (it is a listed Tinker
  project idea), but it means "the teacher is better" rests entirely on the
  measured +7.3% from the repair instruction, not on model capacity.

Implementation: `_attach_opd_teacher_logps` (`rl_trainer.py`) mirrors the existing
`_attach_teacher_context_logps` to run the instructed-teacher forward, then
realigns its log-probs onto the training layout — the two prompts have different
lengths, so the same output tokens sit at different offsets.
`_compute_opd_advantages` (`actor.py`) applies the penalty.

## 5b. Is the train/inference prompt mismatch dangerous?

Worked through the actual loss (`ppo_actor_loss_fn` in
`areal/utils/functional/functional.py`, called from `actor.py:1841`). Write `s`
for the clean state, `I` for the appended instruction, `a` for the sampled turn.

- **The PPO ratio never sees `I`.**
  `ratio = exp(logprobs - proximal_logprobs) = π_θ(a|s) / π_θprox(a|s)` — both
  terms are forward passes on the *training* (clean) `input_ids`. Clipped to
  [0.6, 1.4] by `eps_clip: 0.4`. Structurally identical to a normal on-policy
  step.
- **`I` enters exactly one place**, the behaviour importance weight
  `w = π_θprox(a|s) / π_θold(a|s+I)`, per token, with
  `behave_imp_weight_mode: token_mask` and cap 5.0: tokens with `w > 5` are
  **dropped**, the rest are scaled by `w`. Gradient magnitude is bounded by
  `5 × 1.4 × |A|`.
- **The direction is shrinkage, not amplification.** The tokens the instruction
  actually caused are, by construction, ones the clean policy finds *less*
  likely, so `w < 1` there. Nothing in this path can blow a token up; the cap
  only bites in the other direction, on tokens the instruction made *less* likely
  than baseline, which are rare and uninformative.
- **So the realistic failure is the feature doing nothing**, not divergence: the
  most informative tokens of a guided turn get scaled toward 0 and the guided
  gradient quietly vanishes. That is what to monitor, not instability.
- **Autoregression limits the damage.** The divergence is concentrated in the
  first tokens that commit to a move; once the message has opened with
  "let's isolate a smaller step", the remaining few hundred tokens are continued
  just as happily by the clean policy, so `w ≈ 1` over most of the sequence.
- **Off-policy rows are a small minority.** Only the guided *turn* carries `I`;
  every later turn of the same episode is generated with no instruction, so its
  `old_logp` and `prox_logp` share a prompt and `w ≈ 1`. With 3 of 8 slots and
  ~3 turns per episode, roughly 12% of trainable rows are off-policy.

Three honest caveats:

1. **This path is not new, but the placement is.** The teacher prompt pool is the
   same generate-on-A/train-on-B mechanism, and three July configs used it
   (`configs/math/july/baseline-overfit-1-leakt-batch128-rebn-nomean-5-{prompt,teacher}-pools{,-wp}.yaml`).
   But those put the extra text at the *front*, in the system prompt. Guidance
   has to go at the *end*, immediately before generation — a system-prompt
   placement is followed far less often, which is why the measurement put it
   last. End placement means a stronger behavioural effect and therefore a larger
   `w` gap. The July runs are partial evidence, not proof.
2. **The current baseline does not enable the prompt pool at all**, so
   `clean_tutor_inputs` is `None` throughout and this path is unexercised in the
   configuration we are branching from.
3. **The safe fallback is not free.**
   `valid_turn_mask = loss_mask.sum(dim=-1) > 0` (`actor.py:1143`), so zeroing a
   guided row's `loss_mask` also removes it from the group baseline — destroying
   the one thing it was kept for. Implementing `train_on_guided: false` properly
   needs an `advantage_only` column that keeps the row in `valid_turn_mask` and
   zeroes `loss_mask` only after `compute_advantages`. ~20 lines in `actor.py`.

Monitoring that settles it in the first few hundred steps: log
`behave_imp_weight` and `behave_mask` **restricted to guided rows**. Both already
exist in the loss `stat` dict; they are just not grouped by row provenance yet.

## 6. Third, cheap item

Immediate-turn repetition costs 20.2% → 8.4% student success, and "same ask as
last turn" rises from 14% to 70% by turn 9. A direct penalty is worth roughly
+2 points — less than Variant A's +7.3%, but it needs no extra generation and can
ride the existing `batch_centered_penalty_score/weight` channel
(`tensors.py:112`) with a judge-based score in place of the bge-m3 cosine.
Phase 4, only if A/B underdeliver.

## 7. Status

Implemented and unit-tested on `feat/guided-opd`. Configs, all siblings of the v2
baseline in `examples/tutor/configs/math/0805/v2/`:

| config suffix | arms |
| --- | --- |
| `-guided3` | across-task only |
| `-opd` | in-task only |
| `-guided3-opd` | both |

Tests, both runnable without a GPU:

- `tests/test_tutor_guided_opd.py` — 30 checks. The two that matter are that the
  instruction never survives into the training prompt, and that the OPD
  log-probabilities land on the right tokens after the prompt-length change (with
  a check that a one-token shift would be caught).
- `tests/test_tutor_guided_opd_configs.py` — resolves all three configs through
  the loader `train.py` uses and asserts on the result, including that three bad
  configs are rejected.

One open value: which three moves fill the slots. The 491-state / 3-turn run
(`analysis/hazard_20260806/stage1_full.jsonl`, tmux `tutor-full`) settles it. It
is a config list, not code — the 1-turn ranking put REFRAME first, but at 3 turns
DECOMPOSE overtakes it and PROBE goes 7.5% → 26.5%, so the committed list is
provisional.

Not done, and deliberately: prefix-sharing branching (variance, not coverage —
§3), the immediate-repetition penalty (§6), and anything for across-*student*
diversity, which needs a genuinely different set of students before it means
anything (§9).

## 8. What to watch on the first run

- `guidance/*/selected` and `*/solved` per move — is the group actually getting
  coverage, and is any move systematically wasted?
- Behaviour-importance-weight distribution **restricted to guided rows**. These
  are the off-policy ones; if they saturate the 5.0 cap the guided gradient is
  being clipped away and `train_on_guided: false` becomes the honest setting.
- Move entropy over the six classes. The failure mode to catch is the policy
  learning to imitate the guidance style without ever choosing it unprompted.
- Leak rate on guided rows: WORKED leaked 58% of the time when forced, which is
  why it is not in the default slot list.
- Eval must stay clean. Assert `guidance is None` whenever
  `workflow_context.get().is_eval`.

## 9. Scope limits carried over from the measurements

Every number here is against the 6-move taxonomy, one student (qwen3-1.7b), and
math. The in-task null is an oracle-level result within that taxonomy — it does
not say a teacher cannot adapt in-context, it says choosing among these six moves
per-state buys nothing over per-task. Across-*student* diversity is untested and
cannot be tested with the current environment, which has no genuinely
behaviourally different students; that needs latent student types before any
persona-vector work is meaningful.
