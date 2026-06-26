# json_reasoning-10iter-spark Report

Experiment: `prompt-loop` optimization for AReaL tutor (8-row rawbase config), 10-iteration budget.

## Setup
- AReaL root: `/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL`
- Data/config: `/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/examples/tutor/configs/math/baseline-overfit-8-rawbase.yaml`
- Teacher mode used: `json_reasoning`
- Teacher hidden-mode flag: `--teacher-output-mode json_reasoning --teacher-enable-thinking false`
- Student/judge flow: real API (default for `prompt_loop.py run`)
- Environment: `.env` sourced from AReaL root before API calls.
- Proxy: default prompt-loop behavior (cleared unless `--keep-proxy`).
- Quota handling: `result.json` and `analysis.md` are symlinks to `/tmp`; only compact artifacts are kept in experiment directory.

## Commands
- Subset driver pattern:
  - `set -a; source /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL/.env; set +a`
  - `uv run python /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/prompt-loop/scripts/prompt_loop.py run --teacher-mode api --areal-root ... --config examples/tutor/configs/math/baseline-overfit-8-rawbase.yaml --prompt-file iter_###/prompt.md --result iter_###_result.json --teacher-output-mode json_reasoning --teacher-enable-thinking false --limit 3 --max-turns 10 --early-stop-min-rows 3 --early-stop-failures 2 --early-stop-success-rate 0.34`
- Analyzer pattern used per run:
  - `uv run python .../prompt_loop.py analyze --result <tmp result> --prompt-file <iter>/prompt.md --analysis-out <iter>/analysis.md --next-prompt-out <iter>/next_prompt.md`

## Iteration Outcomes (limit=3 subset)
All 10 json_reasoning iterations were run on subset only due poor early metrics.  
No iteration met success/quality threshold for immediate promotion to full 8-row run.

| Iter | success/total | pre_solved | leaked_rows | stopped_early |
| --- | --- | --- | --- | --- |
| 1 | 0/3 | 0 | 0 | true |
| 2 | 0/3 | 0 | 0 | true |
| 3 | 0/3 | 0 | 0 | true |
| 4 | 0/3 | 0 | 0 | true |
| 5 | 0/3 | 0 | 0 | true |
| 6 | 0/3 | 0 | 0 | true |
| 7 | 0/3 | 0 | 0 | true |
| 8 | 0/3 | 0 | 0 | true |
| 9 | 0/3 | 0 | 0 | true |
| 10 | 0/3 | 0 | 0 | true |

Full result summaries are stored under `/tmp` and linked from the manifest.

## Plain teacher-mode comparison
- Ran one comparison run with:
  - `--teacher-output-mode plain --teacher-enable-thinking true`
- Result: `success=0/3`, `pre_solved=0`, `leaked_rows=0`, stopped early.
- This was not better than `json_reasoning` on comparable 3-row subset.

## Best prompt
- No metric improvement over the 10-budget; all prompts tied at `0/3` subset success with zero leakage.
- Selecting a deterministic best path: `iter_001/prompt.md`.

## Outputs in this experiment directory
- Per-iteration artifacts: `prompt.md`, `next_prompt.md`, `analysis.md`, `result.json` (symlink), `iteration_summary.json`.
- Shared files: `manifest.json`, `report.md`.
- Full raw `result*.json` traces are in `/tmp`; not duplicated here.

## Observation summary
- Dominant failure patterns remained:
  - repeated/checkpoint errors without converging,
  - repeated wrong extracted answers being reworked instead of invalidated,
  - occasional wrong arithmetic in tutor-proposed repaired branches.
- No prompt in this budget escaped repeated max-turn, low-success regime.

## Full result artifact map
- See `manifest.json` for all `/tmp` result/analysis/summary/next-prompt paths per iteration.
