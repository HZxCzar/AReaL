"""Serial, restartable 8-GPU evaluation suite; no training updates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STUDENT = "qwen3-1.7b-text-original"
TUTOR_SCRIPT = "examples/tutor/scripts/eval_0901_all_reward_v4_step1000_8gpu.sh"
PED_SCRIPT = "examples/pedagogical_rl/run_comparison.sh"
PED_CONFIG = "examples/pedagogical_rl/configs/comparison/eval_8gpu.yaml"
PED_SUMMARY = "examples/pedagogical_rl/scripts/summarize_protocol_pair.py"
EVAL_ON = "examples/tutor/configs/math/0901/pilot/eval-step-demo-step1000.yaml"
EVAL_OFF = "examples/tutor/configs/math/0901/pilot/eval-step1000-no-presolve.yaml"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def lock_suite(root):
    handle = (root / ".suite.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(
            "This suite is already running; do not start another controller"
        ) from None
    return handle


def checkpoints():
    ref = (
        REPO
        / "examples/math_tutor_bench/results/0906-pedagogical-rl-qwen3-8b-lr5e-5-8gpu/epoch21epochstep12globalstep999/run.json"
    )
    ped = Path(json.loads(ref.read_text())["checkpoint"]).resolve()
    output = ped.parents[6]
    ours = (
        output
        / "tutor/checkpoints/root/tutor-math-baseline/20260908_0901-reward-v4-all-id-fork775-8gpu/default/epoch21epochstep12globalstep999"
    )
    paths = {"pedrl": ped, "ours": ours}
    bases = []
    for path in paths.values():
        if path.name != "epoch21epochstep12globalstep999":
            raise ValueError(f"Wrong checkpoint step: {path}")
        if not (path / "adapter_model.safetensors").is_file():
            raise ValueError(f"Missing adapter weights: {path}")
        config = json.loads((path / "adapter_config.json").read_text())
        if config.get("r") != 16 or config.get("peft_type") != "LORA":
            raise ValueError(f"Expected rank-16 LoRA: {path}")
        bases.append(Path(config["base_model_name_or_path"]).resolve())
    if bases[0] != bases[1] or not bases[0].is_dir():
        raise ValueError("Checkpoints need the same available Qwen3-8B base")
    return paths, output


def plan():
    tasks = []
    for stage in ("presolve_on", "ped_protocol", "presolve_off"):
        repeats = range(1, 4) if stage != "ped_protocol" else (1,)
        models = ("pedrl", "ours") if stage != "ped_protocol" else ("ours", "pedrl")
        for repeat in repeats:
            for model in models:
                tasks.append(
                    {
                        "id": f"{len(tasks) + 1:02d}-{stage}-{model}-r{repeat}",
                        "stage": stage,
                        "model": model,
                        "repeat": repeat,
                    }
                )
    return tasks


def clean_env():
    env = dict(os.environ)
    # Prevent inherited single-run overrides from silently changing this suite.
    for key in list(env):
        if key.startswith(("MATRIX_", "TUTOR_EVAL_SHARD_")) or key in {
            "EVAL_RUN_DIR",
            "PAIR_RUN_DIR",
            "CHECKPOINT_ROOT",
            "COMMON_GLOBAL_STEP",
            "TUTOR_FILEROOT",
            "TEACHER_MODEL_PATH",
            "STUDENT_MODEL_PATH",
            "EVAL_STAMP",
            "DRY_RUN",
            "BASE_PORT",
            "STUDENT_PORT",
        }:
            env.pop(key, None)
    env.update(PYTHONHASHSEED="42", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def command(task, root, paths, attempt=1, preflight=False):
    env = clean_env()
    model = task["model"]
    ckpt = paths[model]
    if task["stage"] != "ped_protocol":
        on = task["stage"] == "presolve_on"
        directory = root / task["id"]
        env.update(
            CHECKPOINT_ROOT=str(ckpt.parents[2]),
            COMMON_GLOBAL_STEP="999",
            MATRIX_TEACHER_KEY=model,
            MATRIX_TEACHER_TRIAL=ckpt.parents[1].name,
            MATRIX_TEACHER_KEYS=model,
            MATRIX_INCLUDE_NONE_STUDENT="1",
            MATRIX_ONLY_NONE_STUDENT="1",
            MATRIX_SHARD_NONE="1",
            MATRIX_EXPECT_PRESOLVE=str(int(on)),
            MATRIX_EVAL_CONFIG=str(REPO / (EVAL_ON if on else EVAL_OFF)),
            MATRIX_EVALUATOR=str(
                REPO / "examples/tutor/scripts/evaluate_api_teacher_sharded.py"
            ),
            MATRIX_STRATIFIED_SAMPLES="0",
            MATRIX_EXPECTED_EXPLAIN_RATIO="1.0",
            MATRIX_OUTPUT_TAG=task["id"],
            SAVE_TRACES="all",
            BASE_PORT="37000",
            EVAL_RUN_DIR=str(directory),
            EVAL_CONCURRENCY="16",
            SERVER_MAX_RUNNING_REQUESTS="192",
            CALLER_MAX_CONCURRENT="192",
            EPISODE_ERROR_RETRIES="3",
            EPISODE_TIMEOUT_SECONDS="300",
        )
        return (
            ["bash", TUTOR_SCRIPT, "preflight" if preflight else "run"],
            env,
            directory,
        )
    directory = root / "ped_protocol" / "attempts" / f"{model}-{attempt}"
    env.update(DRY_RUN="1" if preflight else "0", STUDENT_PORT="30001")
    args = [
        "bash",
        PED_SCRIPT,
        PED_CONFIG,
        f"trial_name=suite-{root.name}-{model}-a{attempt}",
        f"actor.init_lora_path={ckpt}",
        "total_train_steps=0",
        "evaluator.eval_before_train=true",
        "evaluator.average_rollouts=1",
        "recover.mode=disabled",
        "max_eval_examples=-1",
        "teacher_pre.enabled=false",
        "teacher_pre.verify=false",
        "evaluation.matrix_enabled=true",
        "evaluation.compute_initial_attempts=true",
        "evaluation.conversation_types=[GUIDED,ATTEMPTED]",
        "evaluation.preference_names=[none]",
        "evaluation.record_turn_leak_diagnostic=false",
        "generation.teacher_output_format=unified_xml",
        "generation.leak_judge_mode=pedagogical_rl",
        "cross_eval.enabled=false",
        f"debug_trace_dir={directory}",
        "debug_trace_every_n_rollouts=1",
    ]
    return args, env, directory


def owned_pids(marker):
    token = f"EVAL_SUITE_OWNER={marker}".encode()
    pids = []

    def namespace_ids(path):
        for line in (path / "status").read_text().splitlines():
            if line.startswith("NSpid:"):
                return [int(value) for value in line.split()[1:]]
        return [int(path.name)]

    # A container may mount host /proc while kill() expects container PIDs.
    self_ids = namespace_ids(Path("/proc/self"))
    level = len(self_ids) - 1
    namespace = Path("/proc/self/ns/pid").readlink()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if token in (path / "environ").read_bytes().split(b"\0"):
                if ") Z " not in (path / "stat").read_text():
                    ids = namespace_ids(path)
                    if (path / "ns/pid").readlink() == namespace and len(ids) > level:
                        pids.append(ids[level])
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return pids


def cleanup_owned(marker):
    # Every process here inherits a random marker from this one launched task.
    # Never kill by model name, port number, or GPU occupancy.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = owned_pids(marker)
        if not pids:
            return
        print(f"[cleanup] signalling owned processes {pids}: {sig.name}", flush=True)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + (30 if sig == signal.SIGTERM else 10)
        while owned_pids(marker) and time.monotonic() < deadline:
            time.sleep(1)
    if owned_pids(marker):
        raise RuntimeError("Owned processes did not exit; refusing next task")


def wait_resources(gpus, timeout=120):
    deadline = time.monotonic() + timeout
    ports = [30001, *[37000 + pair * 10 + side for pair in range(4) for side in (0, 1)]]
    while True:
        busy_ports = []
        for port in ports:
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    busy_ports.append(port)
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpus,
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        busy_gpu = [
            line
            for line in result.stdout.splitlines()
            if line.split(",")[0].strip().isdigit()
        ]
        if not busy_ports and not busy_gpu:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Resources still busy: ports={busy_ports}, GPU processes={busy_gpu}; no next task started"
            )
        print(f"[waiting] ports={busy_ports}, GPU processes={busy_gpu}", flush=True)
        time.sleep(5)


def execute(args, env, log, marker):
    env["EVAL_SUITE_OWNER"] = marker
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        proc = subprocess.Popen(
            args,
            cwd=REPO,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while proc.poll() is None:
                print(f"[running] pid={proc.pid}; log={log}", flush=True)
                time.sleep(20)
            if proc.returncode:
                raise RuntimeError(f"Task exited {proc.returncode}; see {log}")
        finally:
            cleanup_owned(marker)
            proc.wait(timeout=15)


def validate_tutor(task, directory):
    cell = directory / "cells" / task["model"] / STUDENT
    report = json.loads((cell / "summary.json").read_text())
    mode = task["stage"]
    if set(report["modes"]) != {mode}:
        raise ValueError("Presolve mode mismatch")
    data = report["modes"][mode]
    rows = [
        json.loads(line)
        for line in (cell / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if (
        report["dataset_rows"] != 528
        or len(rows) != 528
        or len({r["key"] for r in rows}) != 528
    ):
        raise ValueError("Incomplete/duplicate tutor evaluation")
    if not all(r.get("trace_path") and Path(r["trace_path"]).is_file() for r in rows):
        raise ValueError("Missing tutor trajectory")
    return {
        "pending": report["pending_backfill"]["count"],
        "rows": len(rows),
        "baseline": data["no_teaching_baseline_mean"],
        "retest": data["generalization"]["original"]["accuracy_on_replays"],
        "improvement": data["improvement_over_no_teaching_baseline_mean"],
        "leak": data["leaked_episode_count"] / 528,
        "summary": str(cell / "summary.json"),
    }


def validate_ped(task, directory, root):
    # Use the exact same strict per-model validator as the two-model report.
    sys.path.insert(0, str(REPO))
    from examples.pedagogical_rl.scripts.summarize_protocol_pair import (
        load_arm,
        summarize,
    )

    cells = load_arm(directory.parent, directory.name)
    metrics = {mode: summarize(list(rows.values())) for mode, rows in cells.items()}
    alias = root / "ped_protocol" / task["model"]
    if alias.is_symlink() and alias.resolve() == directory.resolve():
        pass
    elif alias.exists() or alias.is_symlink():
        raise ValueError(f"Refusing conflicting completed-model alias: {alias}")
    else:
        alias.symlink_to(directory.relative_to(alias.parent), target_is_directory=True)
    return {"rows": 1056, "metrics": metrics, "traces": str(directory / "eval")}


def write_progress(root, state):
    write_json(root / "suite_state.json", state)
    lines = [
        "# Evaluation suite",
        "",
        "All tasks run serially. Rates below include recorded diagnostic failures.",
        "",
        "| Task | Status | Baseline | Retest | Improvement (pp) | Leak | Pending |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in state["tasks"]:
        r = task.get("result", {})
        values = [
            f"{100 * r[k]:.2f}" if r.get(k) is not None else "—"
            for k in ("baseline", "retest", "improvement", "leak")
        ]
        lines.append(
            f"| {task['id']} | {task['status']} | {' | '.join(values)} | {r.get('pending', '—')} |"
        )
    (root / "progress.md").write_text("\n".join(lines) + "\n")
    # Sample SD is over three whole-run results, not the 528 individual tasks.
    import statistics

    stats = {}
    for stage in ("presolve_on", "presolve_off"):
        for model in ("pedrl", "ours"):
            done = [
                t["result"]
                for t in state["tasks"]
                if t["stage"] == stage
                and t["model"] == model
                and t["status"].startswith("done")
            ]
            if done:
                stats[f"{stage}/{model}"] = {
                    k: {
                        "n": len(done),
                        "mean": statistics.mean(r[k] for r in done),
                        "sample_std": statistics.stdev(r[k] for r in done)
                        if len(done) > 1
                        else None,
                    }
                    for k in ("baseline", "retest", "improvement", "leak")
                }
    write_json(root / "repeat_statistics.json", stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("plan", "preflight", "run", "resume"),
        nargs="?",
        default="plan",
    )
    parser.add_argument("--output", type=Path, default=os.getenv("SUITE_RUN_DIR"))
    args = parser.parse_args()
    paths, output = checkpoints()
    root = (
        args.output
        or output / "eval_suites" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ).resolve()
    tasks = plan()
    print(f"[suite] {root}", flush=True)
    for task in tasks:
        print(task["id"], flush=True)
    if args.mode == "plan":
        return
    if args.mode == "preflight":
        for i in (0, 1, 6, 7, 8, 9):
            cmd, env, _ = command(tasks[i], root, paths, preflight=True)
            subprocess.run(cmd, env=env, cwd=REPO, check=True)
        print("[preflight] PASS; no GPU processes started", flush=True)
        return
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [s.strip() for s in gpus.split(",")]
    if len(ids) != 8 or len(set(ids)) != 8 or not all(s.isdigit() for s in ids):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly eight distinct GPU ids")
    gpus = ",".join(ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # Refuse silently combining records produced by different code/configs.
    sources = sorted((REPO / "examples/tutor").rglob("*.py")) + sorted(
        (REPO / "examples/pedagogical_rl").rglob("*.py")
    )
    sources += sorted((REPO / "examples/tutor/configs/math/0901").rglob("*.yaml"))
    sources += sorted((REPO / "examples/pedagogical_rl/configs").rglob("*.yaml"))
    sources += [REPO / TUTOR_SCRIPT, REPO / PED_SCRIPT]
    digest = hashlib.sha256(
        b"".join(str(p.relative_to(REPO)).encode() + p.read_bytes() for p in sources)
    ).hexdigest()
    if args.mode == "resume":
        if args.output is None:
            raise ValueError("resume requires --output or SUITE_RUN_DIR")
        _suite_lock = lock_suite(root)
        state = json.loads((root / "suite_state.json").read_text())
        if state["code_sha256"] != digest or state["checkpoints"] != {
            k: str(v) for k, v in paths.items()
        }:
            raise ValueError(
                "Code/config/checkpoints changed; use a fresh suite directory"
            )
        if state["gpus"] != gpus:
            raise ValueError(
                "Resume on the same GPU ids to preserve task configuration"
            )
        if state.get("owner"):
            cleanup_owned(state["owner"])
    else:
        if root.exists():
            raise ValueError("Output already exists; use resume or a new directory")
        root.mkdir(parents=True)
        _suite_lock = lock_suite(root)
        state = {
            "code_sha256": digest,
            "checkpoints": {k: str(v) for k, v in paths.items()},
            "gpus": gpus,
            "tasks": [dict(t, status="pending", attempts=0) for t in tasks],
        }
    write_progress(root, state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for task in state["tasks"]:
            if task["id"].startswith("09-"):
                subprocess.run(
                    [sys.executable, "-B", PED_SUMMARY, str(root / "ped_protocol")],
                    cwd=REPO,
                    check=True,
                )
            if task["status"].startswith("done"):
                continue
            wait_resources(gpus)
            task["attempts"] += 1
            task["status"] = "running"
            marker = uuid.uuid4().hex
            state["owner"] = marker
            write_progress(root, state)
            cmd, env, directory = command(task, root, paths, attempt=task["attempts"])
            print(f"[start] {task['id']}", flush=True)
            execute(cmd, env, root / "logs" / f"{task['id']}.log", marker)
            wait_resources(gpus)
            result = (
                validate_ped(task, directory, root)
                if task["stage"] == "ped_protocol"
                else validate_tutor(task, directory)
            )
            task.update(
                result=result,
                status="done_with_diagnostics" if result.get("pending") else "done",
            )
            state.pop("owner", None)
            write_progress(root, state)
            print(f"[complete] {task['id']} {task['status']}", flush=True)
        print(f"[done] All 14 model evaluations complete: {root}", flush=True)
    except BaseException as exc:
        if state.get("owner"):
            cleanup_owned(state["owner"])
        state["last_error"] = str(exc)
        write_progress(root, state)
        raise


if __name__ == "__main__":
    main()
