from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from areal.utils.logging import getLogger

logger = getLogger("TutorEmbeddingDownload")

DEFAULT_REPO_ID = "BAAI/bge-m3"
DEFAULT_REVISION = "main"
DEFAULT_IGNORE_PATTERNS = (
    "onnx/*",
    "imgs/*",
    "*.jpg",
    "*.webp",
)
REQUIRED_FILES = (
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
)


def download_bge_m3(
    *,
    cache_dir: Path,
    revision: str = DEFAULT_REVISION,
    max_workers: int = 4,
) -> Path:
    """Download the PyTorch BGE-M3 snapshot into a Hugging Face hub cache."""

    resolved_cache_dir = cache_dir.expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading %s revision %s into %s",
        DEFAULT_REPO_ID,
        revision,
        resolved_cache_dir,
    )
    snapshot_path = Path(
        snapshot_download(
            repo_id=DEFAULT_REPO_ID,
            revision=revision,
            cache_dir=str(resolved_cache_dir),
            ignore_patterns=list(DEFAULT_IGNORE_PATTERNS),
            max_workers=max_workers,
        )
    ).resolve()

    missing_files = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not (snapshot_path / relative_path).is_file()
    ]
    if missing_files:
        raise RuntimeError(
            "BGE-M3 snapshot is incomplete; missing files: " + ", ".join(missing_files)
        )

    logger.info("BGE-M3 snapshot is ready: %s", snapshot_path)
    logger.info("Use this path as embedding_model_path in the Tutor config.")
    return snapshot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download BAAI/bge-m3 into a shared Hugging Face hub cache. "
            "ONNX and image assets are skipped."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Shared Hugging Face hub cache directory containing models--* folders.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face model revision to download (default: main).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel download workers (default: 4).",
    )
    args = parser.parse_args()
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")

    download_bge_m3(
        cache_dir=args.cache_dir,
        revision=args.revision,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
