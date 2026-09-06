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

The downloaded official task configs, prompts, datasets, task parsers, and metric
implementations are used directly. Decoding follows the official Qwen evaluation
setup: temperature 0, seed 42, native Qwen thinking disabled, completion mode for
the first four tasks, and chat mode for the five dialogue/pedagogy tasks.

Our trained teacher may emit
`<reasoning>...</reasoning><output>...</output>`. Hidden reasoning is removed and
only `<output>` is passed to the official task parser and Ped-RM. Official stop
strings are then applied to that visible output. A bare `<end></end>` therefore
becomes an empty answer and receives no special credit. If the model emits no XML,
its response is passed through unchanged. This adapter prevents hidden reasoning
or XML syntax from being scored as the teacher response; it does not alter the
benchmark prompt, target, or metric.

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
