"""Own two local TP=4 servers and collect paired trajectories on eight GPUs."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .common import load_dataset, read_json, write_json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def check_ipc_path(output):
    # SGLang uses NamedTemporaryFile(prefix='tmp', random suffix) for ZMQ IPC.
    # Reserve extra filename space rather than only checking the directory.
    if len(os.fsencode(str(Path(output).resolve() / "tmp" / ("tmp" + "x" * 16)))) > 107:
        raise ValueError(
            "Output path is too long for Unix IPC; choose a shorter --output directory"
        )


def gpu_groups(value):
    ids = value.split(",")
    if len(ids) != 8 or len(set(ids)) != 8 or not all(i.isdigit() for i in ids):
        raise ValueError("--gpus requires eight distinct numeric GPU IDs")
    return [",".join(ids[:4]), ",".join(ids[4:])]


def server_command(base, adapter, port, trained):
    command = [
        sys.executable,
        "-B",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(base),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "0901-preference-v3-all-step1500" if trained else "qwen3-8b",
        "--tp-size",
        "4",
        "--context-length",
        "16384",
        "--mem-fraction-static",
        "0.80",
        "--max-running-requests",
        "32",
        "--random-seed",
        "42",
    ]
    if trained:
        command += [
            "--enable-lora",
            "--lora-paths",
            f"step1499={adapter}",
            "--max-loras-per-batch",
            "1",
            "--max-loaded-loras",
            "1",
        ]
    return command


def environment(runtime):
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        INF_API_KEY="EMPTY",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    for name in [
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "FLASHINFER_WORKSPACE_BASE",
    ]:
        directory = runtime / name.lower()
        directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
    temporary = runtime.parent / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(temporary)
    return env


def ready(port):
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as response:
            return response.status == 200
    except Exception:
        return False


def stop_owned(processes):
    # Every group below was created by this invocation with start_new_session=True.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    )
    parser.add_argument("--data", type=Path, default=HERE / "data/ready-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=33100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print plan; no servers, calls, or writes",
    )
    args = parser.parse_args()
    output, data_path = args.output.resolve(), args.data.resolve()
    check_ipc_path(output)
    groups = gpu_groups(args.gpus)
    data = load_dataset(data_path)
    base, adapter = args.base_model.resolve(), args.adapter.resolve()
    if not (base / "config.json").is_file() or not list(base.glob("*.safetensors")):
        parser.error(
            "--base-model must be a complete local snapshot; downloads are disabled"
        )
    if (
        not (adapter / "adapter_config.json").is_file()
        or not (adapter / "adapter_model.safetensors").is_file()
    ):
        parser.error("--adapter must be a complete local LoRA checkpoint")
    if (
        not 1024 <= args.base_port <= 65534
        or args.workers < 1
        or args.startup_timeout < 1
    ):
        parser.error("Invalid port, workers, or startup timeout")
    ports = [args.base_port, args.base_port + 1]
    commands = [server_command(base, adapter, p, i == 1) for i, p in enumerate(ports)]
    plan = {
        "base_model": str(base),
        "adapter": str(adapter),
        "gpu_groups": groups,
        "dataset_fingerprint": data["fingerprint"],
        "count_per_model": len(data["cases"]),
        "servers": commands,
        "output": str(output),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in ports:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))  # refuse to use/kill an existing service
        manifest = output / "deployment.json"
        if manifest.exists() and read_json(manifest) != plan:
            raise ValueError("Deployment changed; use a new output directory")
        write_json(manifest, plan)
        env = environment(output / "runtime")
        env.update(
            HUMAN_STUDY_BASE_URL=f"http://127.0.0.1:{ports[0]}/v1",
            HUMAN_STUDY_TRAINED_URL=f"http://127.0.0.1:{ports[1]}/v1",
        )
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        processes = []

        def launch(command, log, overrides=None):
            with log.open("a", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env={**env, **(overrides or {})},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append(process)
            return process

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        try:
            servers = [
                launch(c, logs / f"server-{i}.log", {"CUDA_VISIBLE_DEVICES": groups[i]})
                for i, c in enumerate(commands)
            ]
            deadline = time.monotonic() + args.startup_timeout
            last_update = 0
            while not all(ready(port) for port in ports):
                if any(p.poll() is not None for p in servers):
                    raise RuntimeError(f"Server exited; inspect {logs}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Server startup timed out; inspect {logs}")
                if time.monotonic() - last_update > 30:
                    print(f"Waiting for two local servers; logs: {logs}", flush=True)
                    last_update = time.monotonic()
                time.sleep(1)
            workers = []
            for name in ["qwen3_8b", "trained_1500"]:
                command = [
                    sys.executable,
                    "-B",
                    "-m",
                    "examples.tutor.human_study.eval",
                    "--data",
                    str(data_path),
                    "--study",
                    str(HERE / "configs/study.yaml"),
                    "--model-config",
                    str(HERE / "configs/models" / f"{name}.yaml"),
                    "--output",
                    str(output / name),
                    "--workers",
                    str(args.workers),
                ]
                if args.retry_incomplete:
                    command.append("--retry-incomplete")
                workers.append(launch(command, logs / f"eval-{name}.log"))
            while any(p.poll() is None for p in workers):
                if any(p.poll() is not None for p in servers):
                    raise RuntimeError("A model server exited during collection")
                if any(p.poll() not in {None, 0} for p in workers):
                    raise RuntimeError(f"Collection failed; inspect {logs}")
                time.sleep(1)
            if any(p.returncode != 0 for p in workers):
                raise RuntimeError(f"Collection incomplete; inspect {logs}")
            print(
                f"Complete: {len(data['cases'])} raw replies per model in {output}",
                flush=True,
            )
        finally:
            stop_owned(processes)


if __name__ == "__main__":
    main()
