"""Shard the original standalone evaluator without changing rollout semantics."""

import hashlib
import os
from pathlib import Path

from examples.tutor.scripts import evaluate_api_teacher as evaluator


def main() -> None:
    count = int(os.environ.get("TUTOR_EVAL_SHARD_COUNT", "1"))
    index = int(os.environ.get("TUTOR_EVAL_SHARD_INDEX", "0"))
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid evaluation shard count/index")
    original_run_all = evaluator.run_all
    original_aggregate = evaluator.aggregate_report
    original_signature = evaluator.build_run_signature

    async def run_all(**kwargs):
        kwargs["specs"] = [
            spec for spec in kwargs["specs"] if spec.dataset_index % count == index
        ]
        return await original_run_all(**kwargs)

    def aggregate_report(results, **kwargs):
        kwargs["dataset_size"] = len(range(index, kwargs["dataset_size"], count))
        if kwargs.get("student_prompt_rows"):
            raise ValueError("Sharded evaluation does not support prompt pools")
        return original_aggregate(results, **kwargs)

    def build_run_signature(**kwargs):
        signature = original_signature(**kwargs)
        signature["shard"] = {"count": count, "index": index}
        signature["shard_evaluator_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        return signature

    evaluator.run_all = run_all
    evaluator.aggregate_report = aggregate_report
    evaluator.build_run_signature = build_run_signature
    evaluator.main()


if __name__ == "__main__":
    main()
