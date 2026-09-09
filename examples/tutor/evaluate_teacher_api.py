"""Evaluate an external teacher with the existing tutor evaluation protocol.

Run from the repository root, using the SAME evaluation YAML as the checkpoint
comparison (not a training YAML). No model servers or training jobs are started.
Requires the shared scripts/evaluate_api_teacher.py evaluation library; keep that
library when removing dated experiment launchers.

Example (replace endpoints and model; no vendor endpoints are assumed)::

    .venv/bin/python -m examples.tutor.evaluate_teacher_api \
        --config examples/tutor/configs/math/0901/pilot/eval-step-demo-step1000.yaml \
        --teacher-base-url "$TEACHER_BASE_URL" --teacher-model "$TEACHER_MODEL" \
        --student-base-url "$STUDENT_BASE_URL" --aux-base-url "$AUX_BASE_URL" \
        --output-dir output/api-eval/model-name --dry-run

Export TEACHER_API_KEY, STUDENT_API_KEY, AUX_API_KEY separately. Missing keys
use EMPTY for unauthenticated local servers. Remove --dry-run to evaluate; add
--resume to continue the same output directory. Extra evaluator arguments and
Hydra overrides go after --, e.g. -- --student-name NAME --skip-preflight.
Use --evaluator-help to list those options. --skip-preflight bypasses only the
models-list check, useful for compatible endpoints without GET /models.

Transport: OpenAI-compatible chat completions ONLY. Native Gemini, Anthropic,
and Responses protocols require a future explicit adapter. The complete API
base URL is used verbatim; /v1 is not appended. Provider request options can be
supplied through -- --teacher-request-params-file FILE. Unsupported sampling
options fail visibly; this entrypoint never silently changes prompts or budgets.

Student and auxiliary model identities, prompts, decoding, gates, presolve,
retests, dataset expansion, result aggregation and trace format follow the YAML.
The fixed auxiliary endpoint must serve the same judge model as the checkpoint
comparison, without a teacher LoRA. The local YAML tokenizer remains responsible
for protocol context accounting; provider token counts may differ.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit


def api_url(value: str) -> str:
    """Validate an explicit endpoint without rewriting a provider's API prefix."""
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("Supply a complete http(s) API base URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("Use environment variables for credentials.")
    return value


@contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-base-url", required=True, type=api_url)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-base-url", required=True, type=api_url)
    parser.add_argument("--aux-base-url", required=True, type=api_url)
    parser.add_argument("--teacher-api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--student-api-key-env", default="STUDENT_API_KEY")
    parser.add_argument("--aux-api-key-env", default="AUX_API_KEY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--episode-timeout-seconds", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve roles and workflow locally; no API calls or output writes.",
    )
    parser.add_argument("--evaluator-help", action="store_true")
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    extra = options.evaluator_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    reserved = {
        "--config",
        "--teacher-base-url",
        "--teacher-model",
        "--api-key",
        "--self-aux-via-teacher",
        "--output-dir",
        "--concurrency",
        "--episode-timeout-seconds",
        "--resume",
    }
    if any(arg.split("=", 1)[0] in reserved for arg in extra):
        parser.error(
            "Set role/transport/output options before --; self-aux is unsupported."
        )

    # Import lazily: --help works without loading CUDA/training dependencies.
    from examples.tutor.scripts import evaluate_api_teacher as evaluator

    argv = [
        sys.argv[0],
        "--config",
        options.config,
        "--teacher-base-url",
        options.teacher_base_url,
        "--teacher-model",
        options.teacher_model,
        "--output-dir",
        options.output_dir,
        "--concurrency",
        str(options.concurrency),
        "--episode-timeout-seconds",
        str(options.episode_timeout_seconds),
    ]
    argv += ["--resume"] if options.resume else []
    argv += ["--help"] if options.evaluator_help else extra
    previous_argv = sys.argv
    try:
        sys.argv = argv
        args = evaluator.parse_args()
    finally:
        sys.argv = previous_argv
    args.api_key = os.environ.get(options.teacher_api_key_env) or "EMPTY"

    original_load = evaluator.load_experiment_config
    original_build = evaluator.build_eval_workflow_kwargs
    original_signature = evaluator.build_run_signature

    def load_config(config_path, overrides):
        config, _ = original_load(config_path, overrides)
        if config.auxiliary_model.mode != "api":
            raise ValueError(
                "Use the comparison evaluation YAML with a fixed API auxiliary."
            )
        config.auxiliary_model.base_url = options.aux_base_url
        config.auxiliary_model.api_key = (
            os.environ.get(options.aux_api_key_env) or "EMPTY"
        )
        students = [asdict(student) for student in config.student_models]
        for student in students:
            student["base_url"] = options.student_base_url
            student["api_key"] = os.environ.get(options.student_api_key_env) or "EMPTY"
        return config, students

    def build_workflow(**kwargs):
        config = kwargs["config"]
        effective = original_build(**kwargs)
        # Same compatibility fields used by the checkpoint matrix entrypoint.
        effective.update(
            length_retry_enabled=config.length_retry.enabled,
            length_retry_attempts=config.length_retry.attempts,
            soft_overlong_penalty=asdict(config.reward.soft_overlong),
            student_generalize_gate_pass_credit_only=config.student_generalize.gate_pass_credit_only,
            student_sampling=asdict(config.student_sampling),
        )
        return effective

    def signature(**kwargs):
        result = original_signature(**kwargs)
        result["external_teacher_entrypoint"] = {
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "transport": "openai-compatible-chat-completions",
            "auxiliary_policy": "fixed-configured-model",
        }
        return result

    patches = {
        "load_experiment_config": load_config,
        "build_eval_workflow_kwargs": build_workflow,
        "build_run_signature": signature,
        "deepseek_non_thinking_params": lambda seed: {},
        "normalize_base_url": api_url,
    }
    originals = {key: getattr(evaluator, key) for key in patches}
    # Configuration interpolation only; never source .env or write a YAML copy.
    env = {
        "TUTOR_QWEN3_1_7B_BASE_URL": options.student_base_url,
        "TUTOR_QWEN3_8B_BASE_URL": options.aux_base_url,
        "INF_API_KEY": "EMPTY",
        "DEEPSEEK_API_KEY": "",
    }
    try:
        for key, value in patches.items():
            setattr(evaluator, key, value)
        with environment(env):
            if options.dry_run:
                config, students = load_config(args.config, args.overrides)
                evaluator.tutor_train._apply_eval_average_rollouts(config)
                config.student_generalize.enabled = evaluator.resolve_generalization(
                    args.student_generalization, config.student_generalize.enabled
                )
                evaluator.resolve_teacher_generation_args(args, config)
                evaluator.validate_args(args)
                students = evaluator.select_student_models(students, args.student_name)
                modes = evaluator.resolve_presolve_modes(
                    args.teacher_presolve,
                    evaluator.effective_eval_presolve_enabled(config),
                )
                workflow = build_workflow(
                    config=config,
                    student_models=students,
                    tokenizer=None,
                    args=args,
                    presolve_enabled=modes[0].enabled,
                )
                print(
                    json.dumps(
                        {
                            "teacher": {
                                "model": args.teacher_model,
                                "url": options.teacher_base_url,
                            },
                            "students": [
                                {
                                    "name": s["name"],
                                    "model": s["model"],
                                    "url": s["base_url"],
                                }
                                for s in students
                            ],
                            "auxiliary": {
                                "model": workflow["aux_model"],
                                "url": workflow["aux_base_url"],
                            },
                            "temperature": args.teacher_temperature,
                            "max_tokens": args.teacher_max_tokens,
                            "presolve_enabled": workflow["teacher_pre_enabled"],
                            "presolve_modes": [mode.name for mode in modes],
                            "presolve_verify": workflow["teacher_pre_verify"],
                            "output_dir": options.output_dir,
                            "dry_run": "No API calls or output files; dataset and connectivity not checked.",
                        },
                        indent=2,
                    )
                )
            else:
                asyncio.run(evaluator.main_async(args))
    finally:
        for key, value in originals.items():
            setattr(evaluator, key, value)


if __name__ == "__main__":
    main()
