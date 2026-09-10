# Offline training: resume a retained recovery generation

This procedure resumes model/optimizer and trainer state from a retained recovery
generation, not merely a saved Hugging Face adapter. Run on the allocated GPU node.
Stop every previous job using the same run name before changing its pointer or
moving synchronization markers. Do not touch another run's directories.

## 1. Select and inspect the source

Example: restore 975 completed steps and train to 1000 total steps.
`globalstep974` means 975 updates have completed (zero-based indexing).
Replace the example run and generation for other experiments.

```bash
set -euo pipefail
cd /inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL.worktrees/dev-unified

RUN=20260906_133706_0901-preference-v3-reward-v4-step-demonstration-8gpu
ROOT=/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor
GEN=epoch20-epochstep34-globalstep974-1788825257322735365
EXPECTED_GLOBAL_STEP=974
TOTAL_STEPS=1000
RECOVER_DIR="$ROOT/checkpoints/root/tutor-math-baseline/$RUN/recover"

ls "$RECOVER_DIR/generations"
cat "$RECOVER_DIR/current.json"
cat "$RECOVER_DIR/generations/$GEN/recover_info/step_info.json"
ls -lh "$RECOVER_DIR/generations/$GEN/checkpoints/default"
```

Expect `recover_info` and distributed checkpoint shards plus `.metadata`.
The configured `keep_last_n: 3` retains three generations; older ones may already
be gone. A normal `default/epoch...globalstep...` model checkpoint alone is not a
replacement for the full recovery generation.

## 2. Back up and select the pointer

```bash
python - "$RECOVER_DIR" "$GEN" "$EXPECTED_GLOBAL_STEP" <<'PY'
import json
import pathlib
import shutil
import sys
import time

root = pathlib.Path(sys.argv[1])
generation = sys.argv[2]
assert pathlib.Path(generation).name == generation
source = root / "generations" / generation
step = json.loads((source / "recover_info/step_info.json").read_text())
assert step["global_step"] == int(sys.argv[3]), step
assert (source / "checkpoints/default/.metadata").is_file()
assert list((source / "checkpoints/default").glob("*.distcp"))
pointer = root / "current.json"
shutil.copy2(pointer, root / f"current.backup-{time.time_ns()}.json")
temporary = root / "current.resume.tmp"
temporary.write_text(json.dumps({"generation": generation}))
temporary.replace(pointer)
print("Selected completed steps:", step["global_step"] + 1)
PY
```

## 3. Back up stale synchronization markers

Only after the old job has exited:

```bash
SYNC_DIR="$ROOT/name_resolve/root/tutor-math-baseline/$RUN/update_weights_from_disk"
if [ -d "$SYNC_DIR" ]; then
  mv "$SYNC_DIR" "${SYNC_DIR}.backup-$(date +%s%N)"
fi
```

These are readiness signals, not weights. An old `0/ENTRY` can make rollout load
an adapter before its new files are ready, then make actor fail with
`NameEntryExistsError`. Moving the directory preserves it for inspection.
Do not delete checkpoint/recover directories to fix this error.

## 4. Use a compatible copy of the saved config and launch

The saved config preserves the original settings even if repository defaults have
changed. Some saved configs contain `!!python/tuple`, which Hydra rejects. Convert
those tags to ordinary YAML sequences in a separate copy; leave the original intact.

```bash
RESUME_DIR=$(mktemp -d /tmp/tutor-resume.XXXXXX)
sed 's/!!python\/tuple//g' \
  "$ROOT/logs/root/tutor-math-baseline/$RUN/config.yaml" \
  > "$RESUME_DIR/config.yaml"

.venv/bin/python - "$RESUME_DIR/config.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
config = OmegaConf.load(sys.argv[1])
print("Parsed saved config:", config.trial_name)
PY

bash examples/tutor/run_offline.sh 8 \
  "$RESUME_DIR/config.yaml" \
  trial_name="$RUN" \
  recover.mode=on \
  total_train_steps="$TOTAL_STEPS"
```

Explicit `trial_name="$RUN"` is essential: `train.py` adds a timestamp unless this
CLI override is present, even if the YAML already contains a timestamped name.
Saved nested trial names are resolved strings, so omitting it mixes old and new
names. `total_train_steps` is the final total, not the number of additional steps.
For a queued job, create the temporary config inside that job, as above.

## 5. Verify actual recovery, not just service startup

Check the original run's `main.log`, `actor.log`, `rollout.log`, and `metrics.jsonl`.

- Recovery source must be the selected generation.
- The example should report `Recovering from ... global_step=975`.
- Adapter loading must complete and new training metrics must appear.
- Old step 999 rows may still be present. Identify new rows by `wall_time` and file
  order, not the maximum step number.
- Verify LR, entropy, reward, format errors and student call failures on new rows.
- Metrics/W&B history is not automatically rewound. Preserve recovery timestamp
  and generation when selecting the intended branch for plots.

## LR scheduler recovery (fixed for new DCP recoveries)

Observed on 2026-09-08 for the example: resumed step 975 logged LR 0, step 976
logged 1.0638298e-6, and step 982 logged 7.4468085e-6, whereas the original late
training used 5e-5. Warmup restarted.

The single-adapter DCP save/load path now includes LR scheduler state alongside
the optimizer, as the separate-adapter path already did. For recoveries written
with this fix, retain the original schedule configuration: do not set warmup to
zero merely to resume. Scheduler progress and the saved optimizer LR are restored
from the selected generation, including when it is not the latest generation.
LambdaLR's schedule function is constructed from configuration, so changing its
warmup or decay parameters is still a schedule change, not an exact continuation.

Legacy single-adapter recoveries without scheduler state use the selected
generation's `recover_info/step_info.json` to reconstruct completed scheduler
steps. This supports the one-scheduler-step-per-iteration case (`num_iterations=1`)
and verifies the saved optimizer LR against the configured schedule before
restoring scheduler bookkeeping. A mismatch, missing step metadata or ambiguous
multiple-pass setup fails explicitly instead of silently restarting warmup.
Older checkpoints taken during an already-restarted warmup may fail this check;
their exact scheduler progress cannot be recovered from global step alone.

The original incident above remains historical: the first resumed logged LR came
from stale scheduler bookkeeping, while the optimizer LR was loaded separately;
subsequent scheduler steps actually restarted warmup. New code does not rewrite
old checkpoints or update already-running worker processes.
