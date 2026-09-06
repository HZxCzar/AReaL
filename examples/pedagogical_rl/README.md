# PedagogicalRL comparison arm

This directory runs the PedagogicalRL method on AReaL without changing tutor or
core AReaL code. The comparison configs are under `configs/comparison/`.

## Training contract

The comparison preserves the parts that define PedagogicalRL:

- native PedagogicalRL teacher and student prompts;
- deterministic GUIDED/ATTEMPTED classroom assignment;
- no preference and no teacher pre-solve during training;
- two attempts from each native whole-dialogue hard judge;
- eight final student attempts, scored with PedagogicalRL's exact final-box
  string match;
- one scalar reward per episode, normalized across the eight trajectories for
  the same problem and broadcast to every teacher token;
- PPO clipping `epsilon=0.2`, `mu=2` full-batch policy updates, and
  `beta=0.001` sampled forward KL in the token loss;
- teacher temperature 1.0 and frozen student/judge temperature 0.6.

The controlled substitutions shared with our arm are Qwen3-8B rank-16 LoRA as
teacher, Qwen3-1.7B as student, the same Math Pass@2 split, ten teacher actions,
and the common teacher action envelope. The launcher serves Qwen3-1.7B locally;
the frozen judges use the rollout engine's Qwen3-8B base model with LoRA
disabled. No external inference endpoint is used.

```xml
<reasoning>private reasoning</reasoning><output>non-empty student-visible reply</output>
```

or:

```xml
<reasoning>private reasoning</reasoning><end></end>
```

PedagogicalRL has no generic output-format reward. One of its four reward helpers
is a `<think>`-tag helper, which is zero in its published non-thinking setting. Under
the shared XML interface that inactive slot is replaced by the agreed format
rule: malformed or empty output ends the episode before another student call or
final test and contributes `-0.5`; a valid action contributes zero. The native
final-answer/hard-rejection reward, `+0.1` early-end reward, and `-0.5`
max-length penalty are unchanged.

The budget is 1,000 rollout batches x 16 problems x 8 trajectories = 128,000
episodes. Native `mu=2` is retained, so this is 2,000 optimizer updates over the
same sampled data.

## Train on eight GPUs

From the `dev-unified` worktree:

```bash
bash examples/pedagogical_rl/run_comparison.sh
```

The launcher uses all eight visible GPUs, starts a data-parallel local student
server on the rollout GPUs, and writes a checkpoint every 25 rollout steps. An
explicit trial name or Hydra override may follow the config:

```bash
bash examples/pedagogical_rl/run_comparison.sh \
  examples/pedagogical_rl/configs/comparison/8gpu.yaml \
  my-pedagogical-run
```

## PedagogicalRL-protocol cross-evaluation

The evaluation config accepts a rank-16 teacher adapter trained by either
method. It evaluates every test problem under both full GUIDED and full
ATTEMPTED protocols and all seven students (`none` plus six V3 preferences).
Thus the complete split is 528 x 2 x 7 = 7,392 dialogues.

For each row it first samples eight official no-tutor attempts, then runs the
classroom and samples eight final attempts. ATTEMPTED also has its separate
student-first attempt inside the conversation; that initial student action is
never preference-gated. After each teacher action, preference students use the
same V3 binary gate and complaint pool as our protocol. A failed action and its
scripted complaint remain visible to the teacher but are hidden from the frozen
student, native whole-dialogue judges, and final attempts.

No tutor rawbase leak gate is inserted. PedagogicalRL's native whole-dialogue
leak judge is diagnostic and never stops evaluation. Results include both raw
improvement and leak-aware improvement, where a natively leaked dialogue is
assigned zero improvement:

- `ped_eval/<guided|attempted>/<preference>/improvement_raw`
- `ped_eval/<guided|attempted>/<preference>/improvement_leak_aware`
- matching final accuracy, leak, gate-compliance, turn, and format metrics.

Run it after training, or on one of our adapters:

```bash
bash examples/pedagogical_rl/run_comparison.sh \
  examples/pedagogical_rl/configs/comparison/eval_8gpu.yaml \
  actor.init_lora_path=/absolute/path/to/checkpoint \
  trial_name=ped-protocol-eval-name
```

`total_train_steps=0` makes this evaluation-only: it loads the adapter, performs
the version-0 validation matrix, and exits without a policy update.

The reverse cross uses the existing tutor 0901 full-matrix evaluator, unchanged,
through a local wrapper:

```bash
bash examples/pedagogical_rl/run_tutor_protocol_eval.sh \
  /absolute/path/to/pedagogical-rl-checkpoint
```

That target protocol therefore owns its prompts, ten-turn dialogue, unverified
evaluation-time pre-solve, leak/preference masking, seven students, and retest;
none of those semantics is reimplemented in this example.

## Local links

This worktree already has the shared repository `.venv` and `.env` symlinks.
`examples/pedagogical_rl/data` points to the same prepared offline data tree.
The comparison launcher sets placeholder API credentials only for the local
student server. No source or configuration outside `examples/pedagogical_rl`
is modified by this comparison arm.
