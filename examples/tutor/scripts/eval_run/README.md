# Checkpoint evaluation

One public entrypoint, one frozen protocol, one small YAML per experiment.
This package evaluates **already deployed OpenAI-compatible teacher, student and
auxiliary endpoints**. It does not start GPU services, install packages, choose
cluster paths or activate proxies. Use `api_run` for paid API-teacher experiments
that require its pricing and budget accounting; this runner has no spending cap.

```text
eval_run/
  run.sh                 portable entrypoint
  runner.py              config validation, preflight, manifests, locks, resume
  summarize.py           checked shard merge and preference metrics
  protocol.yaml          standalone evaluation protocol; no training inheritance
  configs/
    base.yaml            common model-role/execution/comparison defaults
    all-legacy-1500.yaml  trained teacher, adapter selected through environment
    subgoal-1500.yaml     trained teacher, adapter selected through environment
    untrained.yaml        plain text protocol, native thinking OFF, no adapter
  .env.example           placeholders only
```

## Running

From the repository root, fill in a **private** env file using `.env.example`.
Each role has an independent endpoint and key. Explicitly use `EMPTY` for a local
server without authentication. Environment variables take precedence over the
env file. `TUTOR_PYTHON` selects the installed Python environment.

```bash
# Configuration-only check: no API calls and no output writes.
bash examples/tutor/scripts/eval_run/run.sh untrained --dry-run

# Check all three model catalogs, without generation or output writes.
bash examples/tutor/scripts/eval_run/run.sh all-legacy-1500 --preflight

# Smoke evaluation: one base question, expanded across the seven preferences.
# This makes real model calls; keep it separate from the full run.
bash examples/tutor/scripts/eval_run/run.sh untrained \
  --limit 1 --output-dir output/eval-run/untrained-smoke

# Full evaluation; repeat the identical command to resume/backfill diagnostics.
bash examples/tutor/scripts/eval_run/run.sh all-legacy-1500
bash examples/tutor/scripts/eval_run/run.sh untrained
bash examples/tutor/scripts/eval_run/run.sh subgoal-1500

# Another experiment: same entrypoint, another config.
bash examples/tutor/scripts/eval_run/run.sh /path/to/experiment.local.yaml
```

The catalog check proves connectivity/model naming, not checkpoint identity or
judgment accuracy. For trained teachers, `TEACHER_ADAPTER` is the path or adapter
identifier understood by the teacher server (`extra_body.lora_path`). The runner
does not upload/load adapters. A local adapter directory's JSON and safetensors
files are hashed; a remote registered selector can only be recorded by selector
hash and the declared checkpoint identity. Verify remote deployments separately.

## Fixed paper protocol

The included presets share these settings:

- Seven preferences, **one attempt each, including None once**; 528 questions,
  3,696 episodes per teacher, 11,088 across the three included teachers.
- Comparison ID = None / Attempt / Subgoal / Contrast. These labels do not imply
  that an untrained teacher or a specialist trained on all four preferences.
- Current v3 gate prompts, explain ratio 1, gate each turn; rawbase leak checks.
- Presolve enabled, verification disabled; ten turns; teacher output limit 2,048;
  temperature 1, top-p 1; eight original-question retests and no-teaching baseline.
- Length retry enabled: **three total teacher generation attempts** when a reply
  ends on the length limit. This is distinct from whole-episode infrastructure
  retries. It is an explicitly approved change from the historical all-legacy
  evaluator, not an auxiliary-only historical reproduction.
- Fixed auxiliary Qwen3.8-27B-FP8, native thinking off; gate/leak output cap 1,024,
  answer-judge output cap 256. The student remains Qwen3-1.7B.
- Trained teachers use `non_thinking` response format (reasoning/output XML).
  Untrained uses `format: thinking` (**plain text or standalone `<end>`**) while
  `enable_thinking: false`. Response protocol and native reasoning are independent.

`protocol.yaml` is a flattened snapshot, not an import of mutable training
defaults. Model presets cannot inject arbitrary Hydra overrides. To intentionally
change experimental semantics, create a versioned protocol and point a config's
`protocol` at it; use a new output directory. To change roles, preferences or
execution limits, change those explicit fields in an experiment config. Unknown
fields, inheritance cycles and missing credentials are rejected before generation.

## Parallelism and private deployment

```bash
# Four processes: run once for each INDEX in 0,1,2,3, with a distinct output.
bash examples/tutor/scripts/eval_run/run.sh all-legacy-1500 \
  --shard-count 4 --shard-index 0 \
  --output-dir output/eval-run/all-legacy-1500/shard-0
```

Sharding reuses the existing train-aligned evaluator and partitions the expanded
dataset by global row index, not independent random subsamples. The index/count
are immutable on resume. Each shard retains all seven preferences. Four shards
cover the same 3,696 episodes exactly once; every process has its own output and
lock. Do not run multiple writers against one shard directory.

`execution.concurrency` limits episodes per process. Student and auxiliary
concurrency settings limit individual API callers, **not aggregate load from
all processes**. A private wrapper can deploy GPU pairs, set role endpoints,
schedule shards and place a shared concurrency-limited relay in front of the
auxiliary endpoint. No such deployment assumptions belong in public configs.
The current private four-pair wrapper uses a shared cap of 64 upstream requests;
other jobs outside that relay are not included in its limit.

`execution.proxy: environment` respects the caller's routing; `direct` removes
proxy variables for the evaluation child. Neither starts a proxy. Set proxy and
`NO_PROXY` externally as appropriate. The runner uses the existing evaluator;
there are no new per-model workflow monkey patches.

## Results, resume and publication

Each output directory contains:

```text
experiment.json      sanitized config + protocol/source/endpoint/adapter hashes
protocol.yaml        exact portable protocol passed to the evaluator
invocations.jsonl    execution/resume events
completion.json     expected coverage and pending-diagnostic status
console.log
evaluation/          owned exclusively by the evaluator (empty on first launch)
  run_config.json    evaluator signature, including dataset identity
  results.jsonl      raw episodes and retry replacements
  summary.json       evaluator summary
  pending_backfill.jsonl
  traces/
  api_requests.jsonl actual serialized requests/responses, excluding auth headers
```

Wrapper metadata must remain outside `evaluation/`: the evaluator requires an
empty output directory on first launch and a matching `run_config.json` on resume.
Layout version 2 does not adopt failed version-1 output directories; retain those
for diagnosis and start with a new output directory.

Resume rejects changed settings, endpoints, adapter assets, protocol or evaluator
source. Existing historical output directories cannot be silently adopted.
The exception is `execution.concurrency`: episode scheduling parallelism may be
changed on resume and is recorded in invocation history. It does not alter the
generated protocol. Per-caller limits, timeout/retry policy and scientific
settings remain strict. The included default is eight episodes per shard.
Completed episodes are retained; diagnostic retries do not select low-scoring
episodes. A zero evaluator exit code alone is insufficient: missing coverage or
pending diagnostics makes the public runner exit nonzero. Repeat the same command
after investigating failures; no automatic protocol relaxation is performed.

For a checked report, supply every shard exactly once (one directory for an
unsharded run):

```bash
python -m examples.tutor.scripts.eval_run.summarize \
  output/eval-run/all-legacy-1500/shard-{0,1,2,3} \
  --output output/eval-run/all-legacy-1500/report.json
```

The report uses latest records per key, rejects overlap/incomplete diagnostics,
and reports per-preference improvement, retest, gate micro-compliance and episode
leak rate. ID/OOD/Overall improvement and retest are unweighted preference means.
It never changes `analysis/results.md` automatically.

Do not publish private wrappers/env files or raw artifacts without auditing them.
The formal manifest omits credential/endpoint literals; legacy evaluator logs,
run signatures and traces can contain deployment paths, endpoint URLs and benchmark
content. A config's model ID is not proof that a remote endpoint loaded its claimed
weights. Small synthetic judge checks are not estimates of real-benchmark accuracy.

Tests: `pytest -q tests/test_eval_run.py`.
