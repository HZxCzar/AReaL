# PedagogicalRL baseline in AReaL

This example ports the PedagogicalRL **method** into AReaL while leaving both projects'
infrastructure untouched.

## What is aligned

- Qwen3-8B LoRA teacher, frozen Qwen3-1.7B API student, Math Pass@2 data.
- Native PedagogicalRL teacher/student prompts, GUIDED/ATTEMPTED state machine, ten
  teacher turns, two whole-dialogue judges, eight final training attempts, and all four
  native reward terms.
- 16 problems x 8 rollouts = 128 episodes per outer step, group-wise reward
  normalization, and two full optimizer updates per rollout batch (mu=2).
- 750 outer rollout steps / 96,000 episodes / 1,500 optimizer updates.

There is no student pre-solve filter. Training selects one leak gate with
`generation.leak_judge_mode`: `pedagogical_rl` uses the native whole-dialogue leak gate,
while `turn` uses AReaL's rawbase judge after every teacher output. The native
whole-dialogue pedagogical-values gate remains enabled in both modes.

Evaluation follows the native GUIDED/ATTEMPTED classroom state machine. Every teacher
output is checked by the rawbase judge without terminating the rollout; after the full
dialogue, both native whole-dialogue judges run, and eight final student answers are
always generated. This gives raw student accuracy plus turn-gated and native-leak-gated
accuracy from the same rollout. Evaluation runs three repeats per test problem.

Optional `teacher_pre` privately asks the trainable teacher for a solution draft before
both training and evaluation dialogues. With `verify: true`, up to `attempts` drafts are
checked and a problem group is rejected unless a correct draft is found. With
`verify: false`, exactly one unjudged draft is used. The accepted draft is teacher-only:
it is absent from the public transcript and from both leak judges' inputs.

Comparable evaluation metrics use the tutor W&B schema: `final_correct`, `pre_solved`
(always zero because student pre-solve is disabled), `solved`, `leaks`, `turns`,
`stop/leak`, and the two `repeat/final_correct/*` stability metrics. Turn-level and
native whole-dialogue judge diagnostics remain separate under `turn_leak/*` and
`native_leak/*`; failed verified teacher presolves use `teacher_pre/rejected`.

The W&B step is the AReaL outer rollout step. Evaluation every 10 displayed steps
therefore means every 20 optimizer updates; do not divide the W&B x-axis.

## Ablations

The baseline is the complete config. The other three configs inherit it and override
only their ablation switches, so model, data, optimizer, LoRA, and sampling settings
stay identical. Pass the selected config to the unified launcher:

```bash
bash examples/pedagogical_rl/run_official.sh examples/pedagogical_rl/configs/qwen3_8b_qwen3_1_7b_math_pass2_baseline.yaml
bash examples/pedagogical_rl/run_official.sh examples/pedagogical_rl/configs/qwen3_8b_qwen3_1_7b_math_pass2_turn_leak.yaml
bash examples/pedagogical_rl/run_official.sh examples/pedagogical_rl/configs/qwen3_8b_qwen3_1_7b_math_pass2_teacher_pre_verified.yaml
bash examples/pedagogical_rl/run_official.sh examples/pedagogical_rl/configs/qwen3_8b_qwen3_1_7b_math_pass2_teacher_pre_unverified.yaml
```

The launcher defaults to the baseline config. An optional second argument overrides
`trial_name`.

## Local links and offline execution

The untracked `data`, `models/teacher`, `.venv`, and repository `.env` links point to
shared files already prepared on the CPU host. The GPU host does not need external
network access. The launch script sources the linked `.env`, forces Hugging Face and W&B
offline, and unsets proxy variables only inside the submitted job.

The INF endpoint is aligned with the tutor experiment and stored in the YAML. The API
key is loaded from the linked AReaL `.env`, so the one submitted command is:

```bash
bash examples/pedagogical_rl/run_official.sh
```

## Head-to-head with the tutor method

`_eval_episode` also emits its numbers under `ped_eval/*`. The tutor workflow
runs this same post-dialogue protocol when `pedagogical_eval.enabled=true`, so
the two arms are compared on one metric, `ped_eval/final_correct`:

```bash
bash examples/pedagogical_rl/run_official.sh examples/pedagogical_rl/configs/qwen3_8b_qwen3_1_7b_math_pass2_baseline.yaml
```

```bash
bash examples/tutor/run_official.sh examples/tutor/configs/math/0723/2gpu/qwen8b-train-qwen1.7b-math-pre-aleak-generated-pedeval.yaml
```

See the PedagogicalRL comparison section of `examples/tutor/README.md` for what
the protocol changes on the tutor side and which asymmetries remain.
