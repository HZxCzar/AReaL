# Full API teacher evaluation

For new paper experiments, use the configuration-driven
[api_run entrypoint](scripts/api_run/README.md). The instructions below describe
the earlier launchers and smoke-cost audit; they remain available for resuming
their existing output directories.

The shared entrypoint is `evaluate_teacher_api.py`; the full-run wrapper is
`scripts/eval_api_full.py`. Neither starts training or deploys model servers.
Keep the shared `scripts/evaluate_api_teacher.py` evaluator when cleaning up
dated experiment scripts.

## Gemini 3.8 Flash

From the repository root, in your tmux window:

```bash
bash examples/tutor/scripts/eval_gemini_full.sh --budget-usd 100
```

This explicitly authorizes a $100 estimated stopping threshold for the full run.
Without `--budget-usd`, the safe default remains $2. The completed $2 smoke test
is a separate directory and cost ledger; its results are not merged into full eval.

The script loads `clashon` from the user's interactive Bash setup. Teacher traffic
uses its HTTP(S) proxy; student and judge hosts are added to `NO_PROXY`. Endpoints
and secrets are read from the ignored repository `.env`:

- `GEMINI_BASE_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`
- `STUDENT_BASE_URL`, `AUX_BASE_URL`, `INF_API_KEY`

The Gemini shell preset pins `gemini-3.8-flash`. To use another model/provider,
invoke the shared Python wrapper with explicit provider, model and prices.

Defaults match the Luna API protocol: medium native thinking, provider-default
sampling, no teacher seed, plain text or standalone `<end>`, and no explicit
teacher output cap (including presolve). Student, judge, preferences, gates and
eight original retests retain the comparison YAML settings. All 528 base items
are evaluated across seven preferences: 3,696 episodes. Default concurrency is
16; override using `--concurrency`. This concurrency has not been load-tested on
Gemini, and the seven-example smoke test does not establish quota capacity.

## Inspection, resume and failures

```bash
# Offline validation: no API calls (the shell still appends its console log).
bash examples/tutor/scripts/eval_gemini_full.sh --budget-usd 100 --dry-run

# Repeat the SAME command after interruption: auto-resume, same cumulative spend.
bash examples/tutor/scripts/eval_gemini_full.sh --budget-usd 100

# Explicitly backfill infrastructure/diagnostic failures, not low scores.
bash examples/tutor/scripts/eval_gemini_full.sh --budget-usd 100 --backfill
```

Output: `output/api-eval/gemini-3.8-flash-full-medium/`.
Console log: `gemini-3.8-flash-full-medium.log`.
`teacher_usage.jsonl` preserves raw usage; `budget_status.json` contains estimated
spend. Resume accounts for all previous calls, including replaced attempts. A
file lock prevents concurrent writers to the same output directory.

The guard stops on missing usage; investigate before resuming. Do not delete the
usage ledger or choose a fresh directory to bypass a budget stop. Already-sent
requests may exceed the threshold, so this is not a provider billing hard cap.
Budget accounting uses the larger of `completion_tokens` and
`total_tokens - prompt_tokens`: Gemini's compatible API can include thinking only
in the latter, while OpenAI already includes it in completions.

Preset prices (USD/M tokens): input 0.75, cached input 0.075, output including
thinking 3.75. Cache-write fallback is charged at ordinary input rate; the script
does not create explicit caches or incur cache-storage charges. Verify prices
before runs after December 31, 2026:
<https://ai.google.dev/gemini-api/docs/pricing>.

## Smoke-based cost estimate (September 11, 2026)

One question was tested once under each preference, sequentially. Successful
final trajectories contain 46 teacher calls including seven presolves. An
interrupted, replaced partial trajectory adds two more successful calls; the
initial location-rejected request has unknown usage. Grouping by sequential
presolve boundaries excludes those two replaced calls from the extrapolation.

| Preference | Teacher turns including end | Input tokens | Output incl. thinking | Estimated cost for 528 items |
| --- | ---: | ---: | ---: | ---: |
| None | 2 | 2,645 | 1,366 | $3.75 |
| Attempt diagnosis | 2 | 2,377 | 1,217 | $3.35 |
| Subgoal decomposition | 5 | 7,611 | 1,770 | $6.52 |
| Contrastive comparison | 8 | 17,200 | 4,082 | $14.89 |
| Causal justification | 7 | 12,872 | 4,098 | $13.21 |
| Step demonstration | 7 | 9,793 | 2,925 | $9.67 |
| Independent verification | 8 | 16,425 | 4,268 | $14.95 |

Clean sample cost: $0.12566475; multiply by 528 = **$66.350988**.
Total observed smoke spend including replaced calls: $0.130329, plus the
separately recorded $0.10 reservation for unknown usage (not actual billed spend).
No cached-token counts were returned, so this estimate assumes no cache discount;
it does not reuse Luna's cache hit rate. Thinking is included, not added twice.
Student/judge GPU costs are excluded. **This is one question, not a representative
sample or statistical confidence interval.** $100 is a planning allowance, not a
guarantee of full completion within that budget.
