# API teachers on MathTutorBench

This is a separate entrypoint; existing checkpoint evaluation scripts are unchanged.
It separates CPU/API generation on an internet-connected host from local official
Ped-RM scoring on an offline GPU host. No student server is required.

## Prerequisites

Use the repository Python environment (or set `PYTHON` to an equivalent environment).
Stage the parent directory's pinned `.runtime/` assets as described by its README:
upstream revision `6faed173ec2bef55cb899b2a3e0f93982f9cb176`, private dependencies,
and dataset caches. On another host, copy this directory together with `.runtime/`;
generation does not download data or weights. The repository's existing
`prepare.py` stages source/dependencies, but does not replace missing HF dataset
caches. Offline checks fail if assets are missing.

Fill an external private env file using `.env.example`, or export the variables.
The preset stores variable names only. No credentials or deployment URLs belong
in public YAML. No proxy is started; standard HTTP(S)_PROXY / NO_PROXY may be
configured outside the official entrypoint. For this machine, run `clashon` in
your shell before the commands if needed; that is not a publishing dependency.

## Connected machine: estimate, smoke, generate

Run from the repository root:

```bash
# Offline estimate; no API calls, result files, model downloads or GPU use.
bash examples/math_tutor_bench/api_run/run.sh estimate

# Optional smoke: ONE example PER TASK (nine requests, not one total).
bash examples/math_tutor_bench/api_run/run.sh generate --env-file .env \
  --limit 1 --budget-usd 2 --concurrency 1 \
  --output-dir examples/math_tutor_bench/results/api/gemini-smoke

# Full run; explicit estimated spending allowance. Nothing starts automatically.
bash examples/math_tutor_bench/api_run/run.sh generate --env-file .env \
  --budget-usd 50 --concurrency 8
```

Default output: `examples/math_tutor_bench/results/api/gemini-3.8-flash-medium/`.
Repeat the exact command to resume. A different sample limit, model, prompt/data
content, endpoint, prices or adapter code requires a different directory. Budget,
concurrency and retry settings may be changed explicitly; invocations are audited.
To add another compatible teacher, copy the small YAML and pass `--config FILE`.
The only supplied/tested-offline preset is Gemini; real endpoint access and
behavior have not been smoke-tested by this implementation.

The default budget is $2. `--budget-usd` applies to cumulative usage in the output
directory, not one invocation. Every request is logged before dispatch, including
retries. Missing usage or interrupted requests reserve `--unknown-reserve-usd`
(default $0.10) each; use 0 to stop on unknown usage. Reserves are estimates, not
guaranteed charges or bounds. Already-running requests can exceed the threshold.
Successful responses include all returned usage and reasoning metadata; output
cost includes thinking via `max(completion_tokens, total_tokens - prompt_tokens)`.
Cache discounts apply only when the response supplies cache counts.

## Offline GPU machine: score

Copy the **entire result directory**, including `run.json`, generated task files
and `generation_complete.json`, to the GPU host. Copy/stage `.runtime/` and the
official reward model beforehand; no network is needed during scoring.

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/math_tutor_bench/api_run/run.sh score \
  --output-dir /path/to/copied/gemini-3.8-flash-medium \
  --pedrm-model /path/to/Qwen2.5-1.5B-pedagogical-rewardmodel
```

One GPU is sufficient. This runs the existing official Ped-RM implementation on
the four pedagogy tasks and then the existing summary generator. It does not
re-call Gemini or deploy a chat endpoint. Inspect `summary.json`, `summary.yaml`,
and `pedrm/pedrm_metrics.json`. GPU execution is a separate integration check.

For eight-GPU scoring, use the same entrypoint with explicit physical GPU IDs:

```bash
bash examples/math_tutor_bench/api_run/run.sh score --output-dir /path/to/completed/run --pedrm-model /path/to/Qwen2.5-1.5B-pedagogical-rewardmodel --gpu-ids 0,1,2,3,4,5,6,7
```

This runs two shards per task concurrently using the existing Ped-RM worker and
merger. It does not reparse or overwrite `tasks/*/predictions.jsonl`, original
`generations.json`, or API request/response logs. Without `--gpu-ids`, the previous
scorer behavior is unchanged; a single explicit GPU ID selects one device.
The run lock prevents simultaneous scoring/generation through this API entrypoint.
Worker failure or Ctrl-C cleans up live workers before exiting. Repeat the same
command to resume completed shards, whose cache identity includes model identity,
generation inputs, shard count and scorer code. Logs and intermediate shards are
stored in `pedrm/shards/<identity>/`. No API keys, network or proxy are required.

The new report selects Solution Correctness **F1** and Mistake Location
**micro-F1**, matching the official leaderboard. This corrects the legacy summary
field selection for new API runs only; historical summaries are not rewritten.

## Protocol and audit

- Nine pinned official task configs, prompts, examples, parsers and metrics are
  used. Four tasks require official Ped-RM scores after generation.
- All API tasks send the **unchanged rendered official prompt** as one user chat
  message. Completion-mode tasks therefore have an explicit transport difference
  from local-Qwen evaluation; no Qwen empty-thinking suffix is appended.
- Gemini uses Medium reasoning, provider-default sampling, no seed, no explicit
  output cap. This is NOT decoding-identical to the existing Qwen runner's
  temperature-0 / seed-42 / 2048-token policy. Report these differences in a paper.
- Official stop strings are applied after extracting visible text, using the same
  adapter as local evaluation. Hidden reasoning is never a fallback answer.
- There is one response per example, not an interactive student rollout. No
  previous thinking signatures are injected, no explicit cache objects created.
- `api_requests.jsonl` is the durable source: exact submitted JSON and complete
  successful response JSON, keyed by task/example/request ID, flushed and fsynced
  per event. Native reasoning fields remain separate from `message.content`;
  unavailable internal thinking cannot be recovered. HTTP failure bodies are
  omitted to avoid echoed credentials; status codes remain recorded.
- `tasks/*/predictions.jsonl` contains inputs, raw/visible answers and predictions;
  `generations.json` matches official Ped-RM input. These derived files can be
  rebuilt from successful response records without new paid generation.
- `pending.json` records incomplete examples. Model format/incorrect-answer
  outcomes are scored, not silently retried for better answers. Transport failures
  receive bounded retries; other failures remain visible for resume.
- Do not publish private env files or raw benchmark/request logs without review.

Estimate output scenarios include visible output **plus thinking**, with input
length approximated as characters/4 and no cache discount. They are planning
scenarios, not measured Gemini costs or guaranteed bounds. Ped-RM adds local GPU
time, not Gemini token charges.

The locally staged full benchmark contains **10,602 examples** across nine tasks.
The September 12 offline estimate is approximately 3.17M input tokens (characters/4).
At the preset's $0.75/M input and $3.75/M output rates, average combined
output+thinking lengths of 256 / 512 / 1024 / 2048 tokens imply approximately
**$12.56 / $22.73 / $43.09 / $83.80**, before retries and unknown-call reserves.
These estimates do not transfer measured tutoring-trajectory lengths to this
different benchmark. Inspect actual usage after an explicitly authorized smoke.
