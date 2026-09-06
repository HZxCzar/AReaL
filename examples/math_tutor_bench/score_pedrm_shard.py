#!/usr/bin/env python3
"""Score one deterministic shard of one MathTutorBench pedagogy task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_pedrm import (
    PEDAGOGY_TASKS,
    conversation,
    resolve_local_model,
    response_text,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--task", choices=PEDAGOGY_TASKS, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")

    source = args.tasks_root.resolve() / args.task / "generations.json"
    if not source.is_file():
        raise SystemExit(f"missing generations for Ped-RM: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    indices = list(range(args.shard_index, len(data), args.num_shards))
    model_path = resolve_local_model(args.model)

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    set_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        num_labels=1,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    model.eval()
    device = torch.device("cuda")

    @torch.inference_mode()
    def score(messages: list[dict[str, str]]) -> float:
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to(device)
        return float(model(inputs).logits[0][0].item())

    records = []
    description = f"Ped-RM {args.task} {args.shard_index + 1}/{args.num_shards}"
    for index in tqdm(indices, desc=description):
        item = data[index]
        candidate = score(
            conversation(item, str(item["generated_teacher_utterance"]))
        )
        reference = score(
            conversation(item, response_text(item["ground_truth_response"]))
        )
        records.append(
            {
                "index": index,
                "candidate_score": candidate,
                "reference_score": reference,
            }
        )

    write_json(
        args.output.resolve(),
        {
            "task": args.task,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "total_task_samples": len(data),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
