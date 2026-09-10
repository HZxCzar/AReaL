# Teacher response format

Existing configs default to `teacher_response_format: non_thinking`; no experiment
YAML needs changing. This keeps the existing explicit reasoning/output protocol
and the existing paired end tag.

To opt in for a native-thinking teacher, set these top-level evaluation fields:

```yaml
teacher_response_format: thinking
teacher_api_request_params:
  reasoning_effort: medium
```

Thinking teaching replies are direct, non-empty student-facing text without
wrapper tags, or exactly `<end>` (only when early ending is enabled). Empty replies,
explicit reasoning/output tags and mixed text/end replies are format errors.
The task instructions stay the same; only the response-format instructions change.
Teacher history uses plain visible text, without wrappers or private reasoning,
regardless of the `teacher_history_tags` setting in thinking mode.
Pre-solve, student and judge protocols are unchanged.

The format does not turn native reasoning on or off. `enable_thinking` controls
the local backend's thinking setting; external API reasoning is configured through
`teacher_api_request_params`. Choose options supported by your actual endpoint.
These request parameters apply only to the teacher, including teacher pre-solve.
API output-token budgets may include internal reasoning tokens.

External teacher API evaluation defaults to omitting output-length parameters,
for both teaching and pre-solve. It also disables the local training-sample token
cap and does not truncate visible replies. Service-side output/context limits,
workflow context guards, turn limits and timeouts still apply. Student/judge
budgets and training/checkpoint evaluation defaults are unchanged.
Use `--teacher-output-limit config` to restore configured teacher budgets;
`--teacher-thinking-token-reserve` is only applicable with that option.

`evaluate_teacher_api.py` uses the config format for every provider by default.
`--teacher-format thinking` explicitly overrides it. `--reasoning-effort` overrides
the config effort; the config effort takes precedence over environment defaults.
No credentials belong in YAML: use the ignored `.env` or process environment.

For an old YAML without these keys, Hydra command-line overrides need `+`:

```bash
-- +teacher_response_format=thinking +teacher_api_request_params.reasoning_effort=medium
```

Existing experiments using the default `enable_thinking: false` retain their
protocol. If migrating an older explicitly `enable_thinking: true` experiment,
select the new response format deliberately: the former bare-text protocol is
replaced by this explicit format selector.
