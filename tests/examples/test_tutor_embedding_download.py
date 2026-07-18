from pathlib import Path

import pytest

from examples.tutor.scripts import download_bge_m3


def _write_required_snapshot_files(snapshot_path: Path) -> None:
    for relative_path in download_bge_m3.REQUIRED_FILES:
        path = snapshot_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_download_bge_m3_uses_shared_cache_and_skips_unused_assets(
    tmp_path, monkeypatch
):
    """Test the downloader targets hub cache layout without fetching ONNX assets."""
    cache_dir = tmp_path / "shared" / "huggingface" / "hub"
    snapshot_path = cache_dir / "models--BAAI--bge-m3" / "snapshots" / "commit"
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        _write_required_snapshot_files(snapshot_path)
        return str(snapshot_path)

    monkeypatch.setattr(download_bge_m3, "snapshot_download", fake_snapshot_download)

    result = download_bge_m3.download_bge_m3(
        cache_dir=cache_dir,
        revision="test-revision",
        max_workers=2,
    )

    assert result == snapshot_path.resolve()
    assert captured == {
        "repo_id": "BAAI/bge-m3",
        "revision": "test-revision",
        "cache_dir": str(cache_dir.resolve()),
        "ignore_patterns": ["onnx/*", "imgs/*", "*.jpg", "*.webp"],
        "max_workers": 2,
    }


def test_download_bge_m3_rejects_incomplete_snapshot(tmp_path, monkeypatch):
    """Test a partial download is reported instead of accepted as usable."""
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()
    (snapshot_path / "config.json").touch()
    monkeypatch.setattr(
        download_bge_m3,
        "snapshot_download",
        lambda **_kwargs: str(snapshot_path),
    )

    with pytest.raises(RuntimeError, match="pytorch_model.bin, tokenizer.json"):
        download_bge_m3.download_bge_m3(cache_dir=tmp_path / "cache")
