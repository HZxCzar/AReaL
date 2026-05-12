from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .inference_engine import resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a vLLM OpenAI-compatible server.")
    parser.add_argument("--model", required=True, help="HF model id or local downloaded model path.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--lora-name", default="student_lora")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = resolve_model_path(args.model)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
    ]
    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.download_dir:
        cmd.extend(["--download-dir", args.download_dir])
    if args.lora_path:
        lora_path = str(Path(args.lora_path).expanduser().resolve())
        cmd.extend(["--enable-lora", "--lora-modules", f"{args.lora_name}={lora_path}"])

    print("Launching vLLM server:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
