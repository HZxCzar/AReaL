# MathTutorBench evaluation

This directory is a self-contained runner for evaluating an AReaL LoRA checkpoint
on the complete official MathTutorBench leaderboard. It does not modify code,
configuration, environments, or data outside this directory. Runtime files are
written to `.runtime/`; evaluation outputs are written to `results/`.

The runner is pinned to upstream MathTutorBench commit
[`6faed173`](https://github.com/eth-lre/mathtutorbench/commit/6faed173ec2bef55cb899b2a3e0f93982f9cb176).
The required upstream Python/YAML/JSON files, two private Python packages, and
official GSM8K/StepVerify data are staged under `.runtime/`. The evaluation runner
forces Hugging Face offline mode: it never contacts the Hub and fails immediately
if a staged asset is missing. It also never downloads a base model or reward model;
both must already exist in the local cache or be supplied as local paths.

## Full run on eight GPUs

```bash
cd /path/to/AReaL.worktrees/dev-unified
GPU_IDS=0,1,2,3,4,5,6,7 \
  bash examples/math_tutor_bench/run.sh /path/to/checkpoint
```

`/path/to/checkpoint` may be either a LoRA checkpoint containing
`adapter_config.json` and `adapter_model.safetensors`, or a complete Hugging Face
base-model snapshot. For LoRA evaluation, the base-model path is read from
`adapter_config.json`.

The default run is the full benchmark. Eight independent SGLang replicas are
started, one per GPU, and the nine official configs are balanced across them.
After generation, all replicas are stopped and the official
`eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel` is run on the first GPU for the
four open-ended pedagogy tasks. Results from an interrupted run are resumable by
running the same command again.

Useful optional environment variables:

```bash
RUN_DIR=/path/inside/this/folder/results/custom-run  # deterministic default otherwise
BASE_PORT=32100                                      # default: 32100
REQUEST_CONCURRENCY=16                               # requests per SGLang replica
MAX_TOKENS=2048                                      # official documented default
MAX_SAMPLES=0                                        # 0 means full benchmark
PED_RM_MODEL=/local/path/or/cached-hub-id
SKIP_PED_RM=1                                        # generation metrics only
```

The output directory contains:

- `summary.yaml` and `summary.json`: consolidated leaderboard metrics;
- `tasks/<task>/metrics.json`: official per-task metrics;
- `tasks/<task>/predictions.jsonl`: raw and visible responses for auditing;
- `tasks/<pedagogy-task>/generations.json`: official Ped-RM input schema;
- `pedrm/`: official candidate-vs-human-reference scores and enriched examples;
- `logs/`: one server log and one task log per worker.

## Evaluation fidelity

### StepVerify scoring corrections (`stepverify-v2`, 2026-09-12)

- Mistake Correction keeps the complete visible teacher reply. `Problem:` and
  `Student:` are not used to truncate it: these can occur in ordinary headings
  or quoted student work. Native Qwen thinking cleanup still applies; the
  upstream numerical-answer parser and accuracy metric are unchanged. This
  scores the returned completion, without heuristic dialogue-boundary cuts.
- Solution Correctness uses the last explicit Yes/No judgment: an answer-labelled
  or standalone line-start Yes/No, allowing Markdown emphasis. If none exists,
  the last whole-word Yes/No is used; if no judgment exists, the upstream
  `incorrect=True` fallback remains. Explanatory quotations do not supersede an
  explicit answer. This intentionally permits a final correction of an earlier
  judgment and does not measure consistency or concise format compliance.
- Other tasks retain their existing processing. The summary writer reports
  Solution Correctness **F1** and Mistake Location **Micro-F1**, not Accuracy
  and Macro-F1.
- Both local and external-API runners use the shared processing functions.
  Revised scores are a local protocol variant, not unmodified official scores.

Existing outputs can be rescored on CPU with no dataset loading, generation,
or reward-model calls:

```bash
python examples/math_tutor_bench/rescore_stepverify.py /path/to/existing/run
```

This creates a sibling `<run>-stepverify-v2` directory and refuses to overwrite
it. Original runs are not modified. `rescore.json` records source hashes and
before/after metrics; new prediction files retain identical raw responses.
The new summary combines the two rescored tasks with the unchanged metrics
from the original run, explicitly recording that provenance. For full-generation
results already truncated at the server, missing text cannot be recovered by
rescoring; these must not be conflated with the recoverable postprocessing issue.

The downloaded official task configs, prompts, datasets, and metric
implementations are used, with the versioned local scoring exceptions below.
Decoding follows the official Qwen evaluation
setup: temperature 0, seed 42, native Qwen thinking disabled, completion mode for
the first four tasks, and chat mode for the five dialogue/pedagogy tasks.

Only native Qwen `<think>` traces / residual `</think>` tags are cleaned before
task processing. Training-specific `<reasoning>`, `<output>`, and `<end>` tags
are not interpreted: if returned, they remain ordinary response text for the
task parser. The unused training-XML adapter was removed on 2026-09-12.
Official stop strings are then applied, except for Mistake Correction under
`stepverify-v2`. Benchmark prompts and targets are unchanged.

A leading `Teacher:` role label is stripped before applying the official stop
strings. Later occurrences of `Teacher:` remain stop boundaries. This prevents a
harmless repeated role label from erasing the complete teacher response.

Existing raw generations can be reparsed and rescored across eight GPUs without
running the teacher model again. Each of the four Ped-RM tasks is split over two
GPUs:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
  bash examples/math_tutor_bench/rescore.sh /path/to/result-directory
```
