import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
sys.path.append(str(_REPO_ROOT))

from examples.math_tutor.data_formats.aime import build_aime_splits
from examples.math_tutor.data_formats.math import build_math_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=["aime", "math"],
        default="aime",
        help="Input data format to convert into tutor DatasetDict rows.",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--train-ids", type=Path, default=None)
    parser.add_argument("--test-ids", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--test-jsonl", type=Path, default=None)
    args = parser.parse_args()

    output_path = Path(args.output).resolve()

    if args.format == "aime":
        if not args.manifest:
            raise ValueError("--manifest is required when --format=aime")
        if args.train_jsonl or args.test_jsonl:
            raise ValueError("--train-jsonl/--test-jsonl require --format=math")
        train_rows, test_rows = build_aime_splits(
            manifest_path=Path(args.manifest).resolve(),
            test_size=args.test_size,
            train_ids_path=args.train_ids.resolve() if args.train_ids else None,
            test_ids_path=args.test_ids.resolve() if args.test_ids else None,
        )
    else:
        if args.manifest or args.train_ids or args.test_ids:
            raise ValueError("AIME manifest split args require --format=aime")
        if not args.train_jsonl or not args.test_jsonl:
            raise ValueError("--train-jsonl and --test-jsonl are required for MATH")
        train_rows, test_rows = build_math_splits(
            train_jsonl_path=args.train_jsonl.resolve(),
            test_jsonl_path=args.test_jsonl.resolve(),
        )

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to save tutor datasets."
        ) from exc

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        }
    )
    dataset.save_to_disk(str(output_path))


if __name__ == "__main__":
    main()

