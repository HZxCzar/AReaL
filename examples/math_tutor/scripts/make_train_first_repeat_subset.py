"""
从 HuggingFace `datasets` 磁盘目录导出子集：
- train：取原 train 的一条样本，重复若干次
- test：默认原样拷贝

重复次数：--repeat-count（显式指定）或默认 len(源 train)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train = one row of source train repeated K times "
            "(K = --repeat-count or len(source train)); test copied by default."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="源数据集目录（含 dataset_dict.json、train/、test/）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出目录（save_to_disk）",
    )
    parser.add_argument(
        "--train-index",
        type=int,
        default=0,
        help="用作模板的 train 下标，默认 0",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=None,
        help="重复次数；省略则用 len(源 train)",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="不写 test split",
    )
    args = parser.parse_args()

    src = args.source.resolve()
    out = args.output.resolve()

    ds = load_from_disk(str(src))
    if "train" not in ds:
        raise ValueError(f"源数据集缺少 train split: {src}")

    train = ds["train"]
    if len(train) == 0:
        raise ValueError("源 train 为空")

    k = args.repeat_count if args.repeat_count is not None else len(train)
    if k <= 0:
        raise ValueError(f"重复次数须为正整数，得到 {k}")

    idx = args.train_index
    if idx < 0 or idx >= len(train):
        raise ValueError(f"--train-index={idx} 超出范围 [0, {len(train) - 1}]")

    template = train.select([idx])
    repeated_train = concatenate_datasets([template] * k)

    if args.no_test:
        out_ds = DatasetDict({"train": repeated_train})
    else:
        if "test" not in ds:
            raise ValueError(f"源数据集缺少 test split，请改用 --no-test: {src}")
        out_ds = DatasetDict({"train": repeated_train, "test": ds["test"]})

    out.parent.mkdir(parents=True, exist_ok=True)
    out_ds.save_to_disk(str(out))
    print(f"Wrote {out}: train={len(out_ds['train'])}", end="")
    if "test" in out_ds:
        print(f", test={len(out_ds['test'])}")
    else:
        print()


if __name__ == "__main__":
    main()
