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

Response processing is `teacher-student-boundary-v1` (2026-09-14), shared by
local and API runs across all nine tasks:

- Keep existing native Qwen thinking cleanup. Training XML is not interpreted.
- Remove an optional initial `Teacher:` label.
- Stop only at subsequent line-start `Teacher:` or `Student:` labels, ignoring
  case and accepting indentation and ASCII/full-width colons.
- Preserve paragraph breaks, newlines, and other task stop strings such as
  `Problem:`, `Question:`, `Q:` and `Explanation:`. Inline role labels are kept.
  `Tutor:` is not a boundary in this protocol.
- Apply these rules after generation; no task stop list is sent to the server.

Solution Correctness retains `stepverify-v2`: use the last explicit Yes/No
judgment, then the last whole-word Yes/No if needed, then the upstream
`incorrect=True` fallback. All other answer parsers are unchanged, including
Mistake Location's first-number parser. The summary reports Solution Correctness
F1 and Mistake Location Micro-F1. These are local protocol adaptations, not
unmodified official leaderboard scores.

See [appendix notes](../../results/appendix_notes.md) for the integrated protocol
and [current comparison](../../results/mathtutorbench.md) for rescored results.
The previous correction-only exemption and task-specific text stops are no
longer the production behavior. Older `stepverify-v2` / `full-response-v1`
reports must not be mixed with the current table.

Existing raw outputs are sufficient for migration: reparse all tasks and rerun
Ped-RM on the extracted teaching responses in a separate output directory.
The historical two-task `rescore_stepverify.py` is not a complete migration to
the current protocol. Preserve original reports. The local launcher records
`response_processing` and rejects an incompatible existing `RUN_DIR`; the API
runner additionally fingerprints processing code. Use a new result directory
when changing protocol, rather than reusing old scoring caches.

Prompts, datasets, targets and generation settings are unchanged. Local Qwen
uses temperature 0, seed 42, native thinking disabled, completion mode for the
first four tasks and chat mode for the other five. API settings are documented
in the appendix. Raw replies and finish reasons are retained independently of
the extracted text.
