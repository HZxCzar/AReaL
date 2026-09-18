# Student simulation evaluation

A configuration-driven factorial evaluation of **student identity × teaching
strategy**, using the existing **prepare → tutoring → test** evaluation flow.
All teachers use Luna by default with a request-only strategy reminder. Endpoints,
credentials and exact deployed model IDs are environment variables. No server
deployment, proxy activation, or cluster-specific paths belong in this package.

## Protocol and methods

`protocol.yaml` is a standalone snapshot of the checkpoint evaluation protocol,
not an inheritance link to mutable training defaults. `configs/settings.yaml`
contains the common execution/teacher settings. Change scientific settings there
for all methods, or supply your own configuration with relative `extends`.

| Config | Student rows | Preference implementation |
| --- | --- | --- |
| `ours` | None + six preferences | Existing v3 gate, complaints, and teacher-only visibility for rejected turns |
| `prompt-only` | Same seven rows, same base model | Natural-language student system prompt; no gate or injected complaints |
| `different-models` | Configured distinct model IDs | No preference prompt or gate |

Every row is crossed with seven teacher strategies, including unrestricted None.
None appears once on each axis: seven cells in its row, not repeated evaluation
passes. The first two matrices contain 49 cells each. The example different-models
config provides three deployment slots (21 cells); choose and document the models
yourself. It does not assign those models artificial preference labels.

The snapshot's student/aux request parameters target Qwen-compatible servers.
For another API, set `request_params: {}` (or its supported parameters) in the
student slot / `roles.student` / `roles.judge`. This replaces the complete request
parameter mapping, without changing common temperature or output-token budgets.
Configure native thinking explicitly where supported and document it per model.

The strategy instructions live in `strategies.json`; student prompt-only profiles
live in `student_preferences.json`. They express the six existing categories
without instructing the student to deliberately fail. The teacher never receives
the row label, but still receives ordinary dialogue, including our method's
existing preference complaints. The reminder is an instruction, not enforcement;
teacher adherence must be checked in saved traces before making strong claims.

In this ablation, `personality.explain_ratio: 0.0` selects only the shared bare
complaint pool, never a preference-specific explanation or suggested strategy.
Examples include "I am still confused by that response." and "I am still stuck.
Please try again." The existing pool also includes a generic request to try a
different approach, without naming a strategy. Prompt-only and different-models
still inject no complaints; their ordinary model-generated replies are unchanged.
This setting is local to the ablation, not training or main-table evaluation.
Previously collected explanatory-complaint outputs must remain separate.

Each of the six fixed styles has a short positive description of the teaching
method. The transient user reminder contains
"Teaching the student using the following method.", a blank line, and
that description, without a style label or an additional persistence contract.
When anti-leak prompting is enabled, it ends with "Please do not directly reveal
the final answer or an equivalent expression." This also applies to None.
The system message is reused from training, including the student-awareness
paragraph asking the teacher to infer student characteristics and adapt teaching.
The temporary user reminder supplies the specific method. The separate adaptive
instruction in the training opening user message is not used here. This applies
uniformly across the suite, without changing training or main-table evaluation.
None remains the unconstrained teacher control, with no fixed-style contract.
This strengthens prompting; it is not a hard enforcement gate. Do not discard
low-performing or non-adherent episodes to manufacture a clean diagonal. Audit
style adherence separately, especially after complaints. Old output directories
cannot be resumed across this prompt revision because their scientific identity
has changed.

Shared defaults: 100 seeded-random questions, one episode per cell/question, seed 42, ten-turn
free chat, teacher presolve (three attempts), record-only leak checking,
no-teaching baseline, eight original-question retests, and length retry (three
generation attempts). Teacher settings match the existing Luna API-run preset:
OpenAI-compatible provider, medium reasoning, thinking-model output format,
provider-default sampling and output limits. The shared API adapter therefore
omits explicit teacher output caps and sampling seeds; seed 42 does not guarantee
deterministic remote teacher outputs. Thinking-format is the teacher output
contract, not a change to the prepare/tutoring/test stages. Student and auxiliary
settings retain the shared protocol. The student is a black box to the teacher:
only its ordinary replies/feedback are visible, not its implementation or row label.

All methods use `leak_handling_mode: reward_only` with `reward.leak_penalty: 0.0`:
leaks are annotated after tutoring, without masking a teacher message, terminating
the dialogue, or injecting leak feedback. The teacher's anti-leak instruction
remains enabled. Only the preference gate controls preference-based message
acceptance in `ours`; the two control methods have no preference gate. Existing
length/format handling and stopping rules are unchanged. This deliberately differs
from training/main-table masked-continue leak handling. Use a new output directory
for this protocol; do not mix it with earlier masked-continue runs.

Prompt-only uses the **same preference prompt** for conversation, no-teaching
baseline, and independent retest branches. `evaluate.py` adds it at the shared
free-chat prompt resolver in its own process, because ordinary
`student_system_prompt` is otherwise ignored by free chat. It changes no core or
existing evaluation files. All other behavior, including budget accounting,
provider handling, API retries and result recording, is reused from `api_run`.

The ablation has its own teacher request builder and reuses the training teacher
system-message builder. The generated `teacher_system_prompt` field stores the method
description for compatibility; the adapter places it in a transient **user**
message, not the system message. Each request is:

1. The training free-chat system message containing the budget, teaching goal,
   student-awareness paragraph, and problem. There is no separate opening user
   message repeating the problem or adding output-format instructions.
2. The teacher-visible conversation with its original teacher/student text and
   roles, including this method's ordinary gate feedback.
3. One temporary user message asking the teacher to continue using the specified
   method, followed by the anti-leak sentence when enabled. None receives a
   generic request to continue tutoring with the same anti-leak sentence.

The temporary reminder is created for the current request and never appended to
the conversation history. Later requests reconstruct it once at the tail; student
and retest histories never receive it. API request logs retain it for auditing.
Prepare and test still run, but the private presolve exchange and answer are not
included in teaching requests. All simulation methods use this same construction.
This is a protocol change: start a new output directory rather than resuming old
system-prefix or training-context runs.

## Run

From the repository root, inspect the matrix without environment variables,
network calls, output writes, or GPU access:

```bash
bash examples/tutor/scripts/student_sim_eval/run.sh ours
```

Fill a private environment file using `.env.example`. Set real provider prices
in `SIM_TEACHER_PRICES`, as a JSON array. API keys may be `EMPTY` only for servers
that do not require authentication. Model IDs must match deployed API IDs.
For another teacher provider, override `teacher.provider`, `endpoint_env` and
`key_env` in a config; the environment variable `SIM_TEACHER_MODEL` stays reusable.

By default leave `SIM_QUESTION_IDS` unset. The runner reads the local dataset's
528-question test split, sorts its stable IDs, and samples 100 without replacement
with `sampling.seed: 42`. Every student and teacher strategy uses the same selected
IDs. These IDs and the seed are frozen in `matrix.json`; selection never depends on
teaching outcomes. `--limit 2` uses two IDs from that sample for a smoke test, not
the first two dataset rows. Seeded sampling is not a claim of manual quality review.

`ours-full` selects the entire 528-question test split. It uses the same protocol
as `ours`, changing only `expected_questions`. Supply `TUTOR_DATASET` to select a
different local dataset and update the declared population size explicitly.

Optionally set `SIM_QUESTION_IDS` to a JSON array of exactly `expected_questions`
unique dataset IDs to reuse a curated/frozen list. Missing or duplicate IDs fail;
changed selections reject resume. Unset a previous 100-ID override before using
`ours-full`. A 7 × 7 matrix has 4,900 teaching episodes for 100 questions, or 25,872
episodes for all 528 questions, plus student baseline and retest API calls.

First run a small end-to-end pilot across the matrix:

```bash
bash examples/tutor/scripts/student_sim_eval/run.sh ours \
  --env-file /path/to/private.env --output-dir /path/to/pilot-ours --limit 2 --run
```

Full runs (use separate output directories):

```bash
bash examples/tutor/scripts/student_sim_eval/run.sh ours \
  --env-file /path/to/private.env --output-dir /path/to/full-ours --run
bash examples/tutor/scripts/student_sim_eval/run.sh prompt-only \
  --env-file /path/to/private.env --output-dir /path/to/full-prompt --run
bash examples/tutor/scripts/student_sim_eval/run.sh different-models \
  --env-file /path/to/private.env --output-dir /path/to/full-models --run
```

Run cells sequentially; default episode concurrency is two, and each student/aux
caller is capped at two. These are process-local limits, not a cross-job global
rate limiter. Multiple matrices launched concurrently multiply the load; external
wrappers must divide capacity or provide a shared relay. This package never starts
a proxy. Supply proxy environment variables in your private launch wrapper only
when needed (`execution.proxy: direct` disables their use).

**Cost:** `execution.budget_usd` is a teacher-only budget **per cell**. At the
default $2, one 49-cell matrix permits up to $98 in teacher usage (plus in-flight
request overshoot); student and auxiliary costs are not included. This safety
default is not an estimate of full-run cost. An exhausted budget stops the run;
do not silently drop remaining questions. Choose the budget in a pilot before
launching the full matrix. To raise this safety cap when resuming, add
`--budget-usd 10` (for example: $10 per cell, not $10 per matrix). Budget changes
are recorded separately and do not alter scientific identity.

## Resume and outputs

Rerun the exact command with the same output and limit to resume via the shared
API evaluator. Matrix manifests reject changed protocol, model, endpoints,
prompts, or source code; interrupted cells keep their original output directory.
API/diagnostic failures are not counted as student failures and block a complete
report. Diagnose failures before retrying; do not move a pilot into a full output.

```text
output/
  matrix.json
  student-row/teacher-strategy/
    config.yaml
    protocol.yaml
    question_ids.json
    evaluation.experiment.json
    evaluation.log
    evaluation/
      run_config.json
      results.jsonl
      traces/
      teacher_usage.jsonl
  report.json
  heatmap.csv
  heatmap.svg
```

Each heatmap cell is `100 × mean(retest score − no-teaching baseline)`, in
percentage points. Reports also include baseline/retest accuracy, episode-level
standard error, gate acceptance when applicable, and leak rate. All cells must
contain the same question IDs and attempt indices, complete eight-replay results,
and no unresolved diagnostic errors. Different-model rows have no diagonal
interpretation. SVGs share a fixed −100 to +100 pp color scale across methods.

Regenerate a report without API calls:

```bash
bash examples/tutor/scripts/student_sim_eval/run.sh ours \
  --output-dir /path/to/full-ours --summarize
```

Use the CSV for publication plotting. This package does not select per-student
best strategies on the test set or claim human-student fidelity. Saved traces and
API records may contain tasks and private deployment metadata: audit them before
publication. No existing results markdown is modified automatically.
