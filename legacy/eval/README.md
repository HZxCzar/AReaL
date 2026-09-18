# Archived evaluation launchers

Archived on 2026-09-13. Files retain their original bytes and repository-relative
directory layout beneath this directory. This is a historical reference, **not
a supported runnable entrypoint**: internal paths still refer to their original
locations. To reproduce an old run, use its original repository revision and
saved configuration in a separate worktree; do not restore these over live code.

Contents:

- `examples/tutor/scripts/eval_0818*.sh`, `eval_0825*.sh`, `eval_0901*.sh`:
  dated checkpoint, preference-matrix and gate experiment launchers.
- `examples/tutor/scripts/eval_luna_full.*`, `eval_gemini_full.sh`,
  `eval_api_full.py`: superseded provider-specific API launchers.
- `examples/tutor/scripts/eval_suite.py`, `run_eval_suite.sh` and
  `analysis/*`: historical suites, protocol comparisons and their backfill tools.

Supported entrypoints:

- Checkpoint evaluation: [eval_run](../../examples/tutor/scripts/eval_run/README.md).
- API teacher evaluation: [api_run](../../examples/tutor/scripts/api_run/README.md).

The active evaluator engine and its shared helpers are not archived. No model,
checkpoint, output or scientific protocol was moved or changed. `eval_run` still
hashes archived Tutor Python sources using their original logical paths so that
existing experiment manifests remain valid. Do not edit these historical bytes:
doing so must still trigger the existing source-change checks.
