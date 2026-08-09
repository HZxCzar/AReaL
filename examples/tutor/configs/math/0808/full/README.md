# 0808 / full

**The objective: an episode where the student is still wrong after two
explanations should still end solved.** Today 33.8% of teaching episodes get
that far and only **46.3%** of them are ever won. If the tutor stopped losing
ground with depth that would be **74.7%**, which is **+9.6 pp** on overall
success — far more than anything else on the table.

Three things change together against the `0808` arms: the whole 759/528 split
instead of the transfer-filtered 132/118, the original re-test as the only
student probe, and OPD supervising the entire stuck tail instead of the first
two turns.

| config | instruction | where | covers | question |
| --- | --- | --- | --- | --- |
| `baseline.yaml` | — | — | — | control |
| `decompose.yaml` | decompose | prompt | depth 3–10 | what is a sentence worth after training? |
| `opd-decompose.yaml` | decompose | distilled | depth 3–10 | can the same effect live in the weights? |
| `opd-decompose-cap2.yaml` | decompose | distilled | depth 3–4 | **ablation** — was it reaching the stuck turns, or all of them? |

Run the first three. `opd-decompose-cap2` only has a question to answer if
`opd-decompose` beats `decompose`.

Read `opd-decompose` against `decompose`, not against `baseline`: appending a
sentence to a prompt is free, so the distilled arm has to beat *that*.

## Where the stuck episodes are lost

Depth = how many times the student had already answered incorrectly when the
tutor produced the turn. 1520 baseline episodes, the 1-in-10 dump:

| depth | turns | resolve | repeats previous | mean overlap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1107 | 38.2% | — | — |
| 2 | 614 | 28.0% | 8.4% | 0.218 |
| 3–4 | 628 | 18.5% | 24.2% | 0.321 |
| 5–7 | 479 | 10.0% | 39.2% | 0.463 |
| 8–10 | 323 | 2.8% | 61.4% | 0.631 |

Monotone and opposite, and the shape reproduces on three other runs. The policy
has about two distinct things to say. Past that the hazard collapses.

Repetition is **part** of why, not all of it. Within-(task_id, depth) — same
problem, same depth, some rollouts repeat and some do not — the non-repeating
turn resolves 10.5% more often, and eliminating repeats entirely is worth about
+4 pp of overall success against the +9.6 pp above. So roughly 40% of the gap is
the tutor recycling words, and the rest is it running out of material. The
decompose clause is aimed at the material, not at the wording.

## OPD covers depth 3–10

`workflow.py:4314` walks turns in order with a running counter, so a cap takes
the FIRST n eligible turns rather than a sample:

    gate 0 + cap 2  -> depth 1-2     1721 turns   54.6%   the healthy end
    gate 2 + cap 2  -> depth 3-4      628 turns   19.9%   opd-decompose-cap2
    gate 2 + cap 0  -> depth 3-10    1430 turns   45.4%   opd-decompose

Uncapping also makes the placement comparison clean. `prompt_instruction` has no
per-episode cap, so at gate 2 it covers depth 3–10 — the same 45.4%. In `0808`
the prompt arm was gate 0 and uncapped (100% of turns) against an OPD arm at
19.9%, comparing two coverages as well as two placements.

## The split, and the probes

`student_generalize.source: generated` was selecting the transfer bank *and*
filtering both splits to the ids that have verified variants. Transfer is not
being measured and its rewards were already `0.0`.

    generated + both levels    132 train / 118 test
    re-test only               759 train / 528 test

The probes are three switches now — `retest_original`, `level1_enabled`,
`level2_enabled`, the last two new and defaulting to true, so the existing arms
are unaffected (`0808/opd-decompose-t3` still resolves to 132/118). Missing
variants **filter rather than raise**, for every source: level1 alone keeps rows
with variant1, level2 alone rows with variant2, both keeps the intersection,
re-test only keeps everything. Losing every row is still an error.

These arms run re-test only, so nothing filters and probe cost drops from 12
student calls per completed episode to 4. `evaluator.average_rollouts` drops
3 → 1 to pay for the bigger test split; 528×1 is cheaper than 118×3 would scale
to and a better estimator, since problems differ from each other far more than
repeat rollouts of one problem do.

## What to watch

**The objective.** Counts, so divide the batch means — never average a
per-episode rate.

    depth/stuck_episode_solved / depth/stuck_episode      46.3% today, 74.7% is the ceiling

**Where it moves.** For each bucket `d1 d2 d3_4 d5_7 d8_10 deep`:

    depth/{b}/resolved       / depth/{b}/turns            the hazard -- the objective
    depth/{b}/repeats        / depth/{b}/repeat_scored    the repeat rate
    depth/{b}/containment_sum/ depth/{b}/repeat_scored    mean overlap, threshold-free

Read the mean overlap before the repeat rate: it needs no threshold, and it
moved 0.218 → 0.631 across depth where the thresholded rate moved 8.4% → 61.4%.
`core/repetition.py` documents how overlap is computed and what it does and does
not catch; the same function produces these metrics and the offline census, so
the two are directly comparable.

**Sanity.**

- `prompt_instruction/share_of_turns` ≈ `opd/selected_ratio` ≈ **45.4%**, and
  ~19.9% for the cap2 ablation. If they disagree the arms are not seeing the
  instruction at comparable rates and neither number means anything.
- `opd_advantage/min` and `/max` inside ±0.2. About half the supervised turns
  now come from episodes ending at `max_turns`, where the task advantage sits
  near −1.0 and the KL rides on top. First place to look if it destabilises;
  the fallback is `opd-decompose-cap2`, not a return to gate 0.
- `rollout/format_errors` at 0, `opd_reverse_kl/avg` trending toward 0.
- The startup line `student_generalize <split> split: kept N/M rows` should
  **not** appear. If it does, a transfer level is on and the split is being cut.

## Do not decide this on final_correct

The whole depth collapse is worth about +9.6 pp overall and the repeat part of
it about +4 pp. On 528 test problems the 95% interval is about ±4.3 pp, so even
the optimistic number is barely two standard errors and the realistic one is
inside the noise. `depth/stuck_episode_solved` has ~374 episodes and the
per-depth hazards ~1430 turns *per training step*, and they move in tens of
points rather than single digits. Decide on those.

## Caveat on the schedule

100 steps is about 2.1 epochs here against about 12 on the filtered split, so
each train problem is seen roughly twice rather than twelve times. Intended, but
a shorter per-problem exposure than any earlier run. If the curves are still
climbing at step 100, raise `total_train_steps` rather than shrinking the
dataset back.
