# Paper API teacher experiments

One runner, one shared evaluation protocol, one small YAML per teacher model.
No model-specific Python or dated launch scripts are required.

```text
api_run/
  run.sh                 # portable shell entrypoint; no proxy activation
  runner.py              # config, preflight, locking, resume, logging
  protocol.yaml          # standalone paper evaluation protocol
  configs/
    base.yaml            # shared generation and execution defaults
    gemini-3.8-flash.yaml
    gpt-5.6-luna.yaml
    gpt-5.6-terra.yaml
  .env.example           # names/placeholders only; no real endpoints or secrets
```

## Run

From the repository root, with dependencies installed in `.venv`:

```bash
# Resolve the full configuration without API calls or output-file creation.
bash examples/tutor/scripts/api_run/run.sh gemini-3.8-flash --dry-run

# Full Gemini run, explicitly allowing a $100 estimated budget.
bash examples/tutor/scripts/api_run/run.sh gemini-3.8-flash --budget-usd 100

# Other presets use the same runner. Terra is configured, not live-validated.
bash examples/tutor/scripts/api_run/run.sh gpt-5.6-luna --budget-usd 25
bash examples/tutor/scripts/api_run/run.sh gpt-5.6-terra --budget-usd 200

# A private/copied model config works identically.
bash examples/tutor/scripts/api_run/run.sh path/to/model.local.yaml --dry-run
```

The default budget remains **$2**. Nothing starts automatically. The commands above
explicitly authorize larger per-directory estimated budgets when you execute them.
These are spending thresholds, not provider billing caps: already-sent concurrent
requests may finish after the threshold. Prices are configurable estimates; check
your provider's current pricing/contract before running. Missing usage stops a
formal run instead of silently counting a failed request as free.

After investigating a missing-usage failure, `--unknown-request-reserve-usd 0.10`
can explicitly reserve an estimated amount for each unknown call in the same
ledger. This is not proof of the actual charge or a bound on it. The default
remains zero (stop on unknown usage); reservations count toward the budget and
are recorded in invocation events. Do not use this to ignore recurring failures.

The public entrypoint never starts or selects a proxy and has no dependency on
machine-specific shell functions. Without proxy environment variables, requests
connect directly. If needed, configure standard proxy variables externally or
use a private, git-ignored `*.local.sh` wrapper; do not publish that wrapper.
The default `execution.proxy: environment` preserves HTTP(S) proxy settings; `direct` clears
them for the child process. Student and judge hosts bypass the proxy in both modes.
Use `TUTOR_PYTHON` to select another Python environment.

## Private deployment configuration

The default private file is repository-root `.env`; alternatively use
`--env-file path/to/private.env`. Process environment takes precedence. Fill in
the names shown in `.env.example`. Never commit the real file. The included
`.gitignore` protects local env files and `*.local.yaml` configs in this folder.

The public YAML stores **environment variable names**, never endpoint URLs or
keys. Teacher credentials and the two local-role credentials are selected
independently. The base preset uses `INF_API_KEY` for both local roles to match
the current deployment; replace each role's `key_env` name if your servers use
separate credentials. Teacher model IDs are provider/gateway-specific: the GPT
presets use the gateway's `openai/` prefix; adjust for another provider's catalog.

Use `TUTOR_TOKENIZER` for a pinned Qwen3-8B tokenizer snapshot and `TUTOR_DATASET`
for the paper's benchmark directory. The defaults are a public model identifier
and a repository-relative dataset path. Reproducing the paper requires the same
tokenizer revision, dataset contents, student/judge weights and prompt pools;
an accessible endpoint alone does not establish matching model weights.

## Protocol and extension

`protocol.yaml` is a flattened snapshot of the reward-v4 seven-preference
evaluation. It does not inherit any dated experiment config. Deployment-specific
headers and absolute cluster paths have been removed; regression tests compare
all remaining config fields and expanded student settings with the historical
baseline. Do not change prompts or scoring independently in a model preset.

Common defaults: medium native thinking, provider-default sampling, no teacher
seed, plain text or standalone `<end>`, and no explicit teacher output-token cap
(including presolve). Student/judge sampling and caps are unchanged. Full eval
uses 528 questions × seven preferences, one trajectory each, ten-turn maximum,
presolve, no-teaching baseline, and eight original-question retests. Concurrency
defaults to 16 and is configurable; it is not a measured provider capacity limit.

To add a teacher, copy a model YAML, retain `extends: base.yaml`, and set an exact
model ID, environment-variable names and four token prices. Supported transport
is **OpenAI-compatible Chat Completions**, including Gemini's compatible endpoint.
Claude/DeepSeek can use a compatible gateway if its reasoning parameters match;
native Anthropic/Gemini/Responses adapters are not implemented. Do not treat an
untested preset as provider validation. There are no per-model runtime patches.

The runner calls the shared `examples.tutor.evaluate_teacher_api` adapter, which
in turn uses `scripts/evaluate_api_teacher.py` and the tutor workflow. Keep those
shared framework files, `core/api_eval_budget.py`, prompt pools and benchmark
assets when publishing. This folder is an experiment interface within AReaL,
not a standalone evaluator detached from the repository.

## Resume, audit and publication

Repeat the same command to resume automatically. Completed trajectories are kept;
all previous billed calls remain in the same ledger. `--backfill` explicitly
retries infrastructure/diagnostic failures, not low scores or format errors.
For a smoke test use `--limit 1 --output-dir output/api-run/my-smoke`; a full run
must use its own directory. Never reset a ledger to bypass a budget stop.

Default artifacts:

```text
output/api-run/<run_name>/                   # results, traces, usage, budget
output/api-run/<run_name>.experiment.json    # resolved public settings + hashes
output/api-run/<run_name>.invocations.jsonl  # runtime budget/concurrency/resume log
output/api-run/<run_name>.log                # console log
output/api-run/<run_name>.lock               # prevent concurrent writers
```

Formal runs also write `api_requests.jsonl` at the HTTP transport boundary, after
SDK serialization. It contains the actual full `messages` array and generation
parameters for teacher (including presolve), student and judge, paired with
responses by `request_id`, `episode_key` and `execution_try`. Headers, endpoints
and credentials are not logged. Native thinking not returned by the provider
cannot be reconstructed. This file contains full prompts/benchmark content and
must be reviewed before publication. Old traces cannot retroactively prove what
was sent; request capture applies only to new calls.

Every record is flushed and fsynced immediately, rather than waiting for episode
completion. Successful responses retain their full JSON in `payload`, including
provider-returned reasoning fields and metadata, separately from student-facing
content. Failed HTTP responses record their status, not their potentially sensitive
error body. A request without a paired response records an interrupted/failed call,
not evidence that the provider returned no content.

The experiment manifest prevents mixing models, prices, protocols, data/tokenizer
locations or endpoints on resume. Raw endpoint values and credentials are omitted
from that manifest; endpoint/path hashes detect changes. Existing evaluator
signatures additionally track dataset contents and adapter code. Budget,
concurrency and proxy routing may change explicitly; invocation events record them.

Old Luna/Gemini launchers, output directories and published measurements are left
untouched. The formal runner refuses to adopt a legacy output directory without
its manifest; continue an old run with its original entrypoint. New formal runs
use the new namespace and do not overwrite old results.

Publish the new config/code/templates, **not** real `.env` files. Existing legacy
scripts may still contain private URLs; this change deliberately does not delete
them. Raw evaluator logs/run configs/traces may contain internal paths, endpoints
and benchmark text: audit/redact them separately before releasing artifacts.

Tests: `pytest -q tests/test_api_run.py tests/test_api_eval_budget.py tests/test_external_teacher_api.py`.
