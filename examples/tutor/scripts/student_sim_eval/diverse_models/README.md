# Diverse student models

Rows are four unconditioned student models; columns are six fixed teaching
strategies. The result is a **4 × 6 improvement heatmap**, not a gate heatmap.
There is no privileged diagonal: model identity does not specify a preferred style.

The shared evaluator preserves preparation, tutoring (10 turns), independent
baseline/retest (8 samples each), length retry, non-thinking student requests,
answer judging and record-only leak detection. All cells use the same questions.
No student preference instruction, scripted complaint or external preference gate
is injected. A student may still complain naturally. Student system prompts and
dialogue construction are unchanged. Teacher descriptions and the strict-style
reminder match the approved short-prompt experiments.

## Configuration and execution

`config.yaml` inherits the common settings. `.env.example` lists the four model
IDs and private deployment variables. A single `STUDENT_BASE_URL` can route by
model; configurations may also assign separate endpoint variables to each row.
No private endpoints or proxy configuration are embedded in the public entrypoint.

From the repository root, inspect the 24-cell / 1,200-episode plan without API calls:

```bash
bash examples/tutor/scripts/student_sim_eval/diverse_models/run.sh config
```

Supply a private environment file and reuse the **exact saved 50 question IDs**
from ours via `SIM_QUESTION_IDS`. `--limit 2` takes the same first two IDs for every
cell; omit it for all 50. API calls require explicit `--run`:

```bash
bash examples/tutor/scripts/student_sim_eval/diverse_models/run.sh config \
  --env-file /path/to/private.env --output-dir /path/to/output --limit 2 --run
```

This public entrypoint executes cells sequentially. Its budget is **per cell**,
not a matrix-wide spending cap. A private parallel wrapper must enforce shared
teacher/judge caps and per-student caps across all six strategies for that model.
Do not give each cell an independent full-endpoint concurrency allowance.

## Results

The shared summarizer produces `report.json`, `heatmap.csv` and `heatmap.svg`:

```bash
bash examples/tutor/scripts/student_sim_eval/diverse_models/run.sh config \
  --output-dir /path/to/output --summarize
```

The figure displays `100 × mean(retest score − baseline score)` in percentage
points, with the same fixed −100 to +100 scale as the other methods. The report
also preserves baseline, retest, question-level improvement standard error and
leak rate. Gate pass is null; no gate figure is produced. The shared summarizer
rejects incomplete/diagnostically failed records rather than silently dropping
them. Preserve failures for review before reporting a final matrix.

Compare strategies **within each model row**, and report each model's baseline
because their initial mathematics ability is not matched. Each model answers its
own baseline and retest; never subtract Qwen's baseline from another model's score.
