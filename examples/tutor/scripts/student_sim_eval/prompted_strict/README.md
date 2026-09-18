# Strict prompted student

This setting implements the simulator's decision rule **inside the student model**.
It is separate from `configs/prompt-only.yaml`, which merely describes a preference
in the student system prompt. Existing settings are unchanged.

## Protocol

- Preserve the original student system prompt and raw teacher/student history.
- Append the shared instruction in `prompts.py` and the exact selected criterion
  from the configured personality gate JSON to the **last user message**, after
  the current teacher response. Reconstruct it afresh on each tutoring request.
- The student internally judges preference compliance, then either answers normally
  or uses the single verbatim reply supplied for that turn from the shared
  complaint file's `bare` pool. Code samples one reply using the protocol seed
  (42 by default), problem text, and turn index; concurrency and resume cannot
  change the draw. The prompt never lists the entire pool.
  The `explain` pool is excluded, matching ours with `explain_ratio: 0`.
  No code validates or replaces its response.
- The added instruction is not saved in dialogue history or shown to the teacher.
  The original teacher content remains visible even if the student complains.
- External preference gating is disabled (`personalities: [none]`, sampling zero).
  There is no online gate audit. The auxiliary model still serves answer judging
  and the unchanged, record-only leak audit.
- Preparation and independent baseline/retest prompts do **not** receive the
  added student instruction. The existing 10-turn, 8-baseline/8-retest protocol,
  length retry and sampling settings are inherited unchanged.
- `teacher_strategies.json` freezes the short teacher descriptions. The existing
  strict-style reminder and anti-leak reminder remain in the teacher request.

The existing protocol field `student_system_prompt` stores the compiled reminder
for hashing and resume. **This setting's evaluator applies that field to tutoring
user messages only.** Always use this setting's entrypoint, not the generic
prompt-only entrypoint.

## Usage

From the repository root, inspect the plan without API calls:

```bash
bash examples/tutor/scripts/student_sim_eval/prompted_strict/run.sh config
```

Default coverage is 6 × 6, the same 50 sampled questions in every cell, seed 42.
For exact comparison to an existing run, set `SIM_QUESTION_IDS` to its saved
`question_ids.json` (exactly 50 IDs). This is preferred to resampling implicitly.

Models, endpoint URLs, keys and teacher prices use the shared environment variables:
`SIM_TEACHER_MODEL`, `SIM_TEACHER_PRICES`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
`SIM_STUDENT_MODEL`, `STUDENT_BASE_URL`, `STUDENT_API_KEY`, `SIM_AUX_MODEL`,
`AUX_BASE_URL`, `AUX_API_KEY`, and optional local dataset/tokenizer overrides.
See the shared `configs/settings.yaml` and API-run documentation. No private
endpoint, credential, proxy address or cluster path is embedded here.

Only after choosing an output directory and budget, explicitly enable calls:

```bash
bash examples/tutor/scripts/student_sim_eval/prompted_strict/run.sh config \
  --env-file /path/to/private.env --output-dir /path/to/new-output --run
```

The standard matrix runner is sequential across cells. `--budget-usd` is **per
cell**, not a total-matrix cap. Concurrent execution and a shared spending cap
belong in an external private wrapper, which must invoke this folder's `cell`
module. Resume uses the same output and verifies settings/source identities.

The main comparison is student learning gain by teacher strategy. Because the
student makes its own decision, complaint frequency is not an external gate
accuracy measure. Any later offline gate audit must remain non-intervening.
