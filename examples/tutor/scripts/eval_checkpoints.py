#!/usr/bin/env python3
"""Offline evaluation of saved LoRA checkpoints, one eval per checkpoint.

WHAT THIS IS FOR. An arm that trains with evaluation off -- 0818/full-students, say,
where eight students would make a pass cost more than the training -- still needs
numbers. This takes the checkpoints that run wrote, serves each one as a LoRA
adapter, and runs the real free-chat rollout and solo re-test against it.

WHAT IT DOES NOT DO. It does not reimplement the protocol. The evaluation is
evaluate_api_teacher.py, which drives the actual TutorAgentWorkflow; that script
already has the re-test scoring, the leak judge, the pre-solve handling and the
resume logic, and a second copy of any of it would drift. This adds the three things
it lacks: checkpoint discovery, a sweep across them, and -- the reason the file
exists -- proof that the adapter is actually being applied.

THE SILENT FAILURE THIS EXISTS TO PREVENT. Serving a LoRA checkpoint through SGLang
by setting the `model` field DOES NOT WORK: model="step199" quietly returns the base
model. Only extra_body.lora_path applies the adapter. Nothing errors, and a whole
sweep can come back with plausible, internally consistent numbers that are all the
untrained teacher -- a flat line across checkpoints being the only hint, and
"training did nothing" being the natural misreading of it. So before spending hours,
this probes the endpoint and refuses to continue unless the adapter demonstrably
changes the output.

SERVING. This attaches to an OpenAI-compatible endpoint you already have; it does
not start one. The endpoint must serve the BASE model with LoRA enabled, e.g.

    python -m sglang.launch_server --model-path <base> --enable-lora
        --max-loaded-loras 16 --port 30000

and --model is the base model name that endpoint reports.

USAGE

    python examples/tutor/scripts/eval_checkpoints.py
        --trial-dir .../checkpoints/root/tutor-math-baseline/<trial>
        --base-url http://localhost:30000/v1 --model qwen3-8b
        --config examples/tutor/configs/math/0818/base/default.yaml
        --output-root .../offline_eval/<trial>
        --max-samples 128
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "examples/tutor/scripts/evaluate_api_teacher.py"
DEFAULT_CONFIG = "examples/tutor/configs/math/0818/base/default.yaml"

# One short prompt, reused for every liveness probe, answered GREEDILY. A sampled
# probe cannot distinguish "the adapter changed the output" from "the sampler did",
# and this endpoint is measured to ignore `seed`, so pinning it would not help.
PROBE_MESSAGES = [
    {"role": "system", "content": "You are a teacher."},
    {
        "role": "user",
        "content": "In one sentence, how would you open a lesson on the quadratic formula?",
    },
]


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path

    @property
    def label(self) -> str:
        return "step%04d" % self.step


def discover(trial_dir: Path) -> list[Checkpoint]:
    """The step checkpoints a run wrote, in step order.

    They live under <trial>/default/, NOT under <trial>/actor/ -- actor/ holds only
    initial_lora, the adapter shipped to the rollout engine at startup, which is the
    untrained one. The directory names encode epoch, epoch step and global step with
    no separators, and the global step is one BELOW the step you would name, because
    the save fires at the end of it: saver.freq_steps 50 gives globalstep49, 99, 149.
    """
    root = trial_dir / "default"
    if not root.is_dir():
        raise SystemExit(
            "No 'default' directory under %s.\n"
            "That is where step checkpoints go. If only actor/initial_lora exists, the "
            "run never reached saver.freq_steps." % trial_dir
        )
    found: list[Checkpoint] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        match = re.search(r"globalstep(\d+)", child.name)
        if not match:
            continue
        if not (child / "adapter_model.safetensors").is_file():
            print("  skipping %s: no adapter_model.safetensors" % child.name, file=sys.stderr)
            continue
        found.append(Checkpoint(step=int(match.group(1)), path=child.resolve()))
    if not found:
        raise SystemExit("No usable checkpoints under %s." % root)
    return sorted(found, key=lambda c: c.step)


def probe(
    base_url: str, model: str, api_key: str, lora_path: str | None, timeout: float
) -> str:
    """One greedy completion, optionally with an adapter applied."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("the openai package is required for the liveness probe") from exc

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)
    extra_body: dict[str, object] = {}
    if lora_path is not None:
        extra_body["lora_path"] = lora_path
    response = client.chat.completions.create(
        model=model,
        messages=PROBE_MESSAGES,
        temperature=0.0,
        max_tokens=96,
        extra_body=extra_body or None,
    )
    return (response.choices[0].message.content or "").strip()


def verify_adapter_is_live(
    checkpoints: list[Checkpoint],
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
) -> None:
    """Refuse to run a sweep that would silently measure the base model.

    Three checks, cheapest first. Any failure raises rather than warns: a warning in
    a log that a four-hour sweep scrolls past is not a safeguard.
    """
    print("[liveness] probing the endpoint before spending the sweep")

    def must_answer(lora_path: str | None, what: str) -> str:
        """A probe that is expected to succeed. Fails loudly, never with a traceback."""
        try:
            return probe(base_url, model, api_key, lora_path, timeout)
        except Exception as exc:  # noqa: BLE001 - reported, not propagated
            raise SystemExit(
                "The endpoint failed to answer %s: %s: %s\n"
                "Check that %s is serving %r with LoRA enabled."
                % (what, type(exc).__name__, exc, base_url, model)
            ) from exc

    # 0. REACHABILITY FIRST, and this ordering is the point. A connection error looks
    #    exactly like "the endpoint refused a bogus adapter", so probing the bogus
    #    path against a dead endpoint would report check 1 as a PASS and then die
    #    somewhere less obvious. Establish that the endpoint answers at all before
    #    reading anything into a refusal.
    base_text = must_answer(None, "a plain request with no adapter")
    print("  ok    the endpoint answers without an adapter")

    # 1. A bogus adapter path must be REFUSED. If the endpoint answers happily, it is
    #    ignoring lora_path altogether -- the exact silent failure, where every
    #    checkpoint returns the base model and the sweep looks flat.
    try:
        probe(base_url, model, api_key, "/nonexistent/adapter/definitely-not-here", timeout)
    except Exception as exc:  # noqa: BLE001 - the KIND of failure decides the verdict
        if type(exc).__name__ == "APIConnectionError":
            # Reachable a moment ago, unreachable now: that is an endpoint problem,
            # and counting it as a refusal would be exactly the mistake check 0 exists
            # to prevent.
            raise SystemExit(
                "The endpoint answered a plain request but the connection dropped on "
                "the adapter probe (%s). That is an infrastructure failure, not a "
                "refusal, so nothing can be concluded about lora_path." % exc
            ) from exc
        print("  ok    a bogus lora_path is refused (%s)" % type(exc).__name__)
    else:
        raise SystemExit(
            "The endpoint ACCEPTED a nonexistent lora_path and answered anyway, so it "
            "is ignoring the field. Every checkpoint would be served as the base model. "
            "Start the server with --enable-lora, and on builds that require "
            "pre-registration also with --lora-paths naming these adapters."
        )

    # 2. The first checkpoint must differ from the base under greedy decoding. A
    #    rank-16 adapter trained for tens of steps moves the logits; byte-identical
    #    greedy output is the signature of an adapter that was never applied.
    first_text = must_answer(str(checkpoints[0].path), checkpoints[0].label)
    if first_text == base_text:
        raise SystemExit(
            "%s produced output byte-identical to the base model under greedy "
            "decoding, so the adapter is almost certainly not applied.\n"
            "Do NOT pass a checkpoint through the `model` field -- that silently serves "
            "the base. It has to be extra_body.lora_path, which is what this script "
            "does, so the endpoint is the thing to check." % checkpoints[0].label
        )
    print("  ok    %s differs from the base" % checkpoints[0].label)

    # 3. With two or more, earliest and latest must differ from each other. This
    #    catches an endpoint that applies whichever adapter it loaded first and then
    #    reuses it for every later request.
    if len(checkpoints) > 1:
        last_text = must_answer(str(checkpoints[-1].path), checkpoints[-1].label)
        if last_text == first_text:
            raise SystemExit(
                "%s and %s produced identical greedy output. The endpoint is serving "
                "one adapter for every request rather than the one asked for, so "
                "per-checkpoint numbers would all be the same model."
                % (checkpoints[0].label, checkpoints[-1].label)
            )
        print(
            "  ok    %s and %s differ from each other"
            % (checkpoints[0].label, checkpoints[-1].label)
        )
    print("[liveness] the adapter is being applied")


def run_one(checkpoint: Checkpoint, args: argparse.Namespace) -> Path:
    """Delegate to evaluate_api_teacher with this checkpoint as the adapter."""
    out_dir = Path(args.output_root).expanduser().resolve() / checkpoint.label
    out_dir.mkdir(parents=True, exist_ok=True)
    request_params = {"extra_body": {"lora_path": str(checkpoint.path)}}
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--config",
        args.config,
        "--teacher-base-url",
        args.base_url,
        "--teacher-model",
        args.model,
        "--teacher-request-params",
        json.dumps(request_params),
        "--concurrency",
        str(args.concurrency),
        "--output-dir",
        str(out_dir),
    ]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.resume:
        cmd.append("--resume")
    cmd += args.passthrough
    print("\n[%s] %s" % (checkpoint.label, " ".join(cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return out_dir


def summarize(results: dict[str, Path], output_root: Path) -> None:
    """One table across checkpoints, from each run's own summary.json."""
    rows = []
    for label, out_dir in results.items():
        summary_path = out_dir / "summary.json"
        if not summary_path.is_file():
            rows.append((label, None, "no summary.json"))
            continue
        rows.append((label, json.loads(summary_path.read_text(encoding="utf8")), ""))

    print("\n" + "=" * 72)
    print("OFFLINE EVAL ACROSS CHECKPOINTS")
    print("=" * 72)
    for label, summary, note in rows:
        if summary is None:
            print("  %s  --  %s" % (label, note))
            continue
        modes = summary.get("modes") or {}
        print("  %s  %s" % (label, json.dumps(modes, ensure_ascii=False)[:120]))

    combined = output_root / "checkpoint_sweep.json"
    combined.write_text(
        json.dumps(
            {
                label: (summary if summary is not None else {"error": note})
                for label, summary, note in rows
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf8",
    )
    print("\nwrote %s" % combined)
    print(
        "Read the TREND across checkpoints, not one number: a flat line is the "
        "signature of an adapter that never applied, and the liveness probe above is "
        "what makes that reading trustworthy."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--trial-dir", type=Path, help="checkpoints/.../<trial>; sweeps its default/ dir"
    )
    source.add_argument(
        "--checkpoint", type=Path, action="append", help="one checkpoint dir, repeatable"
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible endpoint serving the BASE model with LoRA enabled",
    )
    parser.add_argument("--model", required=True, help="base model name the endpoint reports")
    parser.add_argument("--api-key", default=os.environ.get("INF_API_KEY", "EMPTY"))
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="the arm whose settings the eval runs under"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="problems per checkpoint; None is the whole split",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--steps", default="", help="comma-separated global steps to keep, e.g. 49,199,499"
    )
    parser.add_argument("--probe-timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-liveness",
        action="store_true",
        help="skip the adapter probe; only with independent proof it is applied",
    )
    parser.add_argument("--dry-run", action="store_true", help="discover and probe, then stop")
    parser.add_argument(
        "passthrough", nargs="*", help="extra args forwarded to evaluate_api_teacher"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trial_dir is not None:
        checkpoints = discover(args.trial_dir.expanduser().resolve())
    else:
        checkpoints = []
        for path in args.checkpoint:
            resolved = path.expanduser().resolve()
            match = re.search(r"globalstep(\d+)", resolved.name)
            checkpoints.append(
                Checkpoint(step=int(match.group(1)) if match else 0, path=resolved)
            )
        checkpoints.sort(key=lambda c: c.step)

    if args.steps:
        wanted = {int(s) for s in args.steps.split(",") if s.strip()}
        checkpoints = [c for c in checkpoints if c.step in wanted]
        if not checkpoints:
            raise SystemExit("--steps %s matched none of the discovered checkpoints." % args.steps)

    print("%d checkpoint(s):" % len(checkpoints))
    for c in checkpoints:
        print("  %s  %s" % (c.label, c.path))

    if args.skip_liveness:
        print("[liveness] SKIPPED -- numbers mean nothing unless the adapter is applied")
    else:
        verify_adapter_is_live(
            checkpoints, args.base_url, args.model, args.api_key, args.probe_timeout
        )

    if args.dry_run:
        print("\n--dry-run: stopping before the sweep")
        return 0

    output_root = Path(args.output_root).expanduser().resolve()
    results: dict[str, Path] = {}
    failures: list[str] = []
    for c in checkpoints:
        try:
            results[c.label] = run_one(c, args)
        except subprocess.CalledProcessError as exc:
            # One bad checkpoint must not cost the rest of the sweep.
            print("[%s] FAILED with exit %d" % (c.label, exc.returncode), file=sys.stderr)
            failures.append(c.label)

    summarize(results, output_root)
    if failures:
        print("\n%d checkpoint(s) failed: %s" % (len(failures), ", ".join(failures)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
