#!/usr/bin/env python3
"""Fork a full tutor recovery into an isolated trial; never write the source."""

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-recover", type=Path, required=True)
    parser.add_argument("--new-trial", required=True)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate inputs and paths without writing or launching.")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    source = args.source_recover.resolve(strict=True)
    config_path = args.source_config.resolve(strict=True)
    raw = config_path.read_text()
    original = OmegaConf.to_container(
        OmegaConf.create(raw.replace("!!python/tuple", "")), resolve=True
    )
    old = original["trial_name"]
    new = args.new_trial
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", new) or new == old:
        raise ValueError("Use a distinct, simple new trial name (no slashes).")
    step = json.loads((source / "recover_info/step_info.json").read_text())
    completed = int(step["global_step"]) + 1
    if args.total_steps <= completed:
        raise ValueError("Total steps must exceed the recovered completed steps.")
    for name in ("dataloader_info.pkl", "saver_info.json", "evaluator_info.json",
                 "stats_logger_info.json", "checkpoint_info.json"):
        if not (source / "recover_info" / name).is_file():
            raise ValueError(f"Incomplete recovery: missing {name}")
    if not (source / "checkpoints/default/.metadata").is_file():
        raise ValueError("Missing distributed checkpoint metadata.")
    if not list((source / "checkpoints/default").glob("*.distcp")):
        raise ValueError("Missing distributed checkpoint shards.")
    recover = original["recover"]
    if recover.get("no_load_optim") or recover.get("no_save_optim"):
        raise ValueError("Full resume requires saved optimizer state and loading it.")
    optimizer = original["actor"]["optimizer"]
    if optimizer["lr_scheduler_type"] != "constant":
        raise ValueError("This workaround supports only constant-after-warmup LR.")
    # Only valid after the original warmup, with its original LR retained.
    warmup_bound = int(
        float(optimizer["warmup_steps_proportion"])
        * int(original["total_train_epochs"])
        * int(step["steps_per_epoch"])
    )
    if completed < max(1, warmup_bound):
        raise ValueError("Recovery must be after the original warmup period.")

    def rename(value):
        if isinstance(value, dict):
            return {k: rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        return value.replace(old, new) if isinstance(value, str) else value

    config = rename(original)
    config["actor"]["optimizer"]["warmup_steps_proportion"] = 0.0
    config["recover"]["mode"] = "on"
    config["total_train_steps"] = args.total_steps
    # This launcher is deliberately limited to the existing single-actor setup.
    if config.get("critic") is not None:
        raise ValueError("Critic recovery requires a separate review.")
    experiment = config["experiment_name"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment):
        raise ValueError("Unsafe experiment name.")
    root = Path(recover["fileroot"]).resolve()
    user = getpass.getuser()
    destination = root / "checkpoints" / user / experiment / new
    forbidden_existing = [
        destination,
        root / "logs" / user / experiment / new,
        Path(config["cluster"]["name_resolve"]["nfs_record_root"])
        / user / experiment / new,
        Path(config["debug_trace_dir"]),
    ]
    for path in forbidden_existing:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to reuse existing destination: {path}")
    for section, value in config.items():
        if isinstance(value, dict) and "trial_name" in value:
            if value["trial_name"] != new:
                raise ValueError(f"Unaligned trial name in {section}")
        if isinstance(value, dict) and "fileroot" in value:
            if Path(value["fileroot"]).resolve() != root:
                raise ValueError(f"Unexpected output root in {section}")
    if old in json.dumps(config):
        raise ValueError("Old trial references remain in the new config.")
    config_text = OmegaConf.to_yaml(OmegaConf.create(config))
    print(f"Source: {source}", flush=True)
    print(f"New trial: {new}; completed steps: {completed} -> {args.total_steps}", flush=True)
    print(f"LR: constant {optimizer['lr']}; no repeated warmup", flush=True)
    print("New output locations:\n" + "\n".join(map(str, forbidden_existing)), flush=True)
    if args.check_only:
        print("Check passed. No files written; no training launched.")
        return

    # Exclusive creation: never merge into a previous branch, even after failure.
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / "recover/generations" / source.name
    target.parent.mkdir(parents=True)
    shutil.copytree(source, target, symlinks=False)
    source_sizes = {str(p.relative_to(source)): p.stat().st_size
                    for p in source.rglob("*") if p.is_file()}
    target_sizes = {str(p.relative_to(target)): p.stat().st_size
                    for p in target.rglob("*") if p.is_file()}
    if source_sizes != target_sizes:
        raise RuntimeError("Recovery copy verification failed; not launching.")
    with (destination / "recover/current.json").open("x") as f:
        json.dump({"generation": source.name}, f)
    launch = destination / "branch_launch"
    launch.mkdir()
    copied_config = launch / "config.yaml"
    copied_config.write_text(config_text)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_trial": old, "new_trial": new,
        "source_recover": str(source), "source_config": str(config_path),
        "source_config_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "completed_steps": completed, "total_steps": args.total_steps,
        "lr": optimizer["lr"], "warmup_steps_proportion": 0,
        "source_files_bytes": source_sizes,
    }
    (launch / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    env["LOCAL_MODEL_LOG_ROOT"] = str(launch / "student_logs")
    print(f"Verified copied recovery. Provenance: {launch / 'manifest.json'}", flush=True)
    subprocess.run(
        ["bash", str(repo / "examples/tutor/run_offline.sh"), "8",
         str(copied_config), f"trial_name={new}", "recover.mode=on"],
        cwd=repo, env=env, check=True,
    )


if __name__ == "__main__":
    main()
