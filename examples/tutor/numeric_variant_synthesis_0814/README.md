# Numeric-only variant synthesis (0814)

This directory is isolated from the tutor training code and source dataset. The
pipeline reads the existing Hugging Face DatasetDict and writes code, checkpoints,
raw model responses, summaries, and variant banks only below this directory.

## Contract

For each eligible train/test source:

1. Qwen3-8B returns indexed numeric-token edits, not a rewritten task.
2. The program reconstructs the task and proves the nonnumeric skeleton and
   Asymptote spans are byte-exact.
3. Structural exponent/subscript, Asymptote, and LaTeX thousands-suffix tokens are
   protected. Positive-value edits must stay within 0.5x to 2x.
4. Qwen3-8B performs a source/candidate semantic audit.
5. Qwen3-8B blindly solves the candidate twice in separate calls. Both boxed
   answers must be equivalent.
6. A separate Qwen3-8B answer audit recomputes correctness and well-posedness.
7. Qwen3-1.7B attempts the variant once as a bonus measurement. Student failure is
   preferred but is not an acceptance gate.
8. The accepted bank maps teacher_variant_task to original_retest_task.

Each source keeps its generation, audit, two blind solves, answer audit, and
student attempt in strict order. Independent sources may run concurrently.
Teacher and student requests share one bounded semaphore, so the configured
global maximum is never exceeded.

## Smoke result

The fixed smoke selection contains two train and three test tasks. All 5/5 passed
the mechanical, semantic, two-solve, and answer-audit gates. The student failed on
4/5 accepted variants.

Artifacts:

    runs/smoke_5/summary.json
    runs/smoke_5/variant_bank.jsonl
    runs/smoke_5/records/
    runs/smoke_5/calls.jsonl
    runs/smoke_5/runner.log

## Full inventory

Under the final protection policy:

    train: 759 total, 671 eligible, 88 ineligible
    test:  528 total, 471 eligible, 57 ineligible
    total: 1287 total, 1142 eligible, 145 ineligible

Ineligible means no safe editable numeric literal remains outside protected
structural/diagram positions. These tasks are recorded instead of being forced into
unsafe variants.

## Run and resume

Pure tests:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B \
      examples/tutor/numeric_variant_synthesis_0814/test_numeric_variants.py

Five-task smoke/resume:

    examples/tutor/numeric_variant_synthesis_0814/run_smoke.sh

Full train+test/resume:

    examples/tutor/numeric_variant_synthesis_0814/run_full.sh

Each source has an atomic JSON checkpoint. Accepted and ineligible records are

The full script uses 32 concurrent sources and at most 32 total in-flight model
calls, matching the shared deployment's training-eval call limit. A separate
32-source probe is available as `run_concurrency32_probe.sh`.
skipped on resume. Failed records receive three new attempts per full-script
invocation. A file lock prevents duplicate full processes.

Full artifacts are written below:

    runs/full_train_test/

This pipeline does not change the training config, prompts, workflow, dataset, or
any file outside this directory. Integration into the dialogue/retest training path
is intentionally out of scope until the generated bank is audited and explicitly
approved.
