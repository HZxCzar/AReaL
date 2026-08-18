# 0818

The default arm, and the allocations of it. Replaces `0810` as the group new work
starts from.

## Layout

    base/default.yaml   every setting. Also launchable, at the 2-GPU allocation.
    2gpu/base.yaml      GPU count, the two backends, rollout concurrency
    4gpu/base.yaml      the same three keys
    8gpu/base.yaml      the same three keys

`0810` kept its settings in `2gpu/base.yaml` and had the other allocations inherit
from it, so that file was both the shared configuration and one allocation of it —
editing a setting there read as editing the 2-GPU arm. Splitting `base/` out makes
the difference explicit, and `test_tutor_arm_wiring.py` enforces it: each allocation
file may differ from the base only in the GPU count, the two backends, rollout
concurrency, and the names derived from `trial_name`. Anything else fails.

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
