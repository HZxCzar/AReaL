import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def resolve_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=int, default=1)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("CodeCoach manifest must be a list")

    base_dir = manifest_path.parent
    rows = []
    for item in manifest:
        task_path = resolve_path(base_dir, item["task_markdown_path"])
        code_path = resolve_path(base_dir, item["initial_code_path"])
        evaluator_path = resolve_path(base_dir, item["evaluator_path"])
        rows.append(
            {
                "id": int(item["id"]),
                "task_markdown": task_path.read_text(encoding="utf-8"),
                "initial_code": code_path.read_text(encoding="utf-8"),
                "evaluator_path": str(evaluator_path),
                "target_score": float(item["target_score"]),
                "entry_function": str(item["entry_function"]),
                "eval_timeout_sec": int(item.get("eval_timeout_sec", 120)),
                "metadata": item.get("metadata") or {},
            }
        )

    split_at = max(1, len(rows) - max(1, args.test_size))
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(rows[:split_at]),
            "test": Dataset.from_list(rows[split_at:]),
        }
    )
    dataset.save_to_disk(str(output_path))


if __name__ == "__main__":
    main()
