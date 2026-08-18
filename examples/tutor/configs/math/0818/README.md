# 0818

The default arm, and the allocations of it. Replaces `0810` as the group new work
starts from.

## Layout

Arms live in `base/`. The allocation directories carry GPU numbers and nothing else.

    base/default.yaml        every setting of the default arm
    base/<arm>.yaml          an arm: inherits default, changes settings only
    <n>gpu/alloc.yaml        the <n>-GPU allocation, written down once
    <n>gpu/<arm>.yaml        composition: base/<arm> + alloc, plus the name

So a launchable config is always *an arm plus an allocation*, and neither states the
other. Adding an arm means one file in `base/` and three three-line wrappers; adding
a GPU size means one `alloc.yaml` and one wrapper per arm.

`0810` kept its settings in `2gpu/base.yaml` and had the other allocations inherit
from it, so that file was both the shared configuration and one allocation of it --
editing a setting there read as editing the 2-GPU arm.

`test_tutor_arm_wiring.py` enforces the split rather than trusting it: every
`<n>gpu/<arm>.yaml` is diffed against `base/<arm>.yaml` and may differ only in
`cluster.n_gpus_per_node`, the two backends, `rollout.max_concurrent_rollouts`,
`ref.backend` (which is `${actor.backend}` and follows by interpolation), and the
names derived from `trial_name`. Anything else fails the build. `alloc.yaml` is an
overlay with no `defaults` list, so it is deliberately not loadable on its own and
the test skips it.

## Arms

| arm | what it changes |
| --- | --- |
| `default` | nothing; the settings below |
| `full-students` | all 8 `(behavior, information)` cells trained, eval off |

## The defaults

| | |
| --- | --- |
| `leak_handling_mode` | `terminate` |
| `format_handling_mode` | `terminate` |
| `reward.turn_local_components` | `[leak, format_error]` |
| `teacher_pre` | on, **verified** in training |
| `evaluator.teacher_pre_enabled` / `_verify` | on / **not** verified |
| `student_models` | one student, `(text, unmasked)` |
| `evaluator.freq_steps` | 25 |
| `free_chat.transfer_prompts` | off — the re-test problem is the dialogue's problem |
| `teacher_history_tags` | `unmasked` |

Both terminates are training policy only: `evaluator.leak_terminate` and
`evaluator.format_terminate` are `false`, so the headline number runs the full budget
however much the teacher gives away or mis-tags. The rates are still reported by
`rollout/leaks` and `rollout/format_errors`, and the train-consistent number comes
off the same rollout as a second re-test on the pre-leak prefix.

Format terminate is safe **only** because `reward.max_turn_penalty` is `0.0`. In a
rollout that charged for reaching the budget, terminating early would be the cheap
way out of a losing episode; here a format-terminated episode collects `-0.5` and
loses its re-test, which is strictly worse than the worst honest outcome of `0.0`.

Two axes share the word "unmasked" and are unrelated. `teacher_history_tags:
unmasked` is how the **teacher** sees its own earlier replies. The student's
`mask.mode: full` is the **information** axis — what the student sees of the
dialogue. The other student axis is `mode: text | code`, the behavior.

## Not set here

`student_generalize.turn_credit` and `free_chat.no_teaching_baseline` are both off,
so the whole episode reward lands on the last turn and `retest/improvement` is not
reported. Turn credit requires the baseline; both are one line in an arm that wants
them.

## Launching

    bash examples/tutor/run_official.sh examples/tutor/configs/math/0818/4gpu/base.yaml

Two 4-GPU arms are the whole node — stagger the launches by a minute, since
forked-worker ports are allocated per run with no coordination between runs. One
8-GPU arm is the whole node, so it cannot be half of a paired comparison; use
`4gpu/` for anything paired.

Before launching, and after any config edit:

    python tests/test_tutor_arm_wiring.py
    python tests/test_tutor_free_chat.py

## Not comparable with 0810

Every 0810 arm ran `leak_handling_mode: reward_only`, or `format_handling_mode:
continue`, or both, and most ran `teacher_history_tags: masked`. Read 0818 against
0818.

## Eval cost scales with the number of students

The evaluator duplicates the validation set once per student **name**, and
`evaluator.max_samples` truncates **before** that expansion
([train.py:464-471](../../../train.py)), so

    eval episodes per pass = min(max_samples, len(valid)) x number of eval students

Every student runs every problem as a full episode. A pass measured 1650-1840 s at
one student on the 528-problem split:

| eval students | episodes/pass | wall clock |
| --- | --- | --- |
| 1 | 528 | ~30 min |
| 4 | 2112 | ~2 h |
| 8 | 4224 | ~3.5-4 h |

At `freq_steps: 25` over 500 steps that is twenty passes, so eight students would
spend more wall clock on evaluation than on the training it measures. Three levers,
in order of preference:

1. **Turn eval off** and read the training-side per-student series, which is what
   `full-students` does and what `0810/masked-students` settled on at four students.
2. **`max_samples`** — a per-student cap, since it applies before expansion.
   `max_samples: 66` with eight students is 528 episodes, one of today's passes, at
   a ~6 pp standard error per student instead of ~2 pp.
3. **`evaluator.student_model_names`** — name a subset to evaluate and leave the rest
   to training.

Raising `freq_steps` also works and is orthogonal to all three.

## Offline evaluation of the checkpoints

An arm with eval off still writes checkpoints: `saver.freq_steps` is 50, so a
500-step run leaves 10 LoRA adapters (~87 MB each) at

    checkpoints/root/tutor-math-baseline/<trial>/default/epoch*epochstep*globalstep*/

They are under **`default/`**, not `actor/` — `actor/` holds only `initial_lora`,
the adapter shipped to the rollout engine at startup, which is the untrained one.
The global step is one *below* the step you would name, because the save fires at the
end of it: `freq_steps: 50` gives `globalstep49`, `99`, `149`.

`scripts/eval_checkpoints.py` sweeps them. It does not reimplement the protocol — the
evaluation is `evaluate_api_teacher.py` driving the real `TutorAgentWorkflow` — and
adds discovery, the sweep, and a liveness probe.

    python examples/tutor/scripts/eval_checkpoints.py \
        --trial-dir .../checkpoints/root/tutor-math-baseline/<trial> \
        --base-url http://localhost:30000/v1 --model qwen3-8b \
        --config examples/tutor/configs/math/0818/base/default.yaml \
        --output-root .../offline_eval/<trial> --max-samples 128

It attaches to an endpoint you already have; it does not start one. The endpoint must
serve the **base** model with LoRA enabled (`--enable-lora`, and `--lora-paths` on
builds that require pre-registration).

**Why the liveness probe is the point of the script.** Serving a checkpoint through
the `model` field does not work — `model="step199"` silently returns the base model,
and only `extra_body.lora_path` applies the adapter. Nothing errors, so a whole sweep
can come back with plausible, internally consistent numbers that are all the
untrained teacher, with a flat line across checkpoints as the only hint and "training
did nothing" as the natural misreading. Before spending hours the script therefore
checks, in this order:

1. the endpoint answers a plain request — first, because a connection error is
   indistinguishable from a refusal and would otherwise be read as a pass;
2. a **bogus** `lora_path` is refused — if it is accepted, the field is being ignored;
3. the first checkpoint differs from the base under greedy decoding;
4. the earliest and latest checkpoints differ from each other.

Any of those failing aborts rather than warns. `--dry-run` runs discovery and the
probe and stops; `--steps 49,199` picks a subset; `--skip-liveness` exists but the
numbers mean nothing without independent proof the adapter applied.
