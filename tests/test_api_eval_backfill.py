"""Backfill may change retry policy, but not the experimental conditions."""

import json
from copy import deepcopy

import pytest

from examples.tutor.scripts.evaluate_api_teacher import prepare_output_dir


@pytest.mark.parametrize("change", [None, "model", "temperature", "dataset", "retries"])
def test_backfill_allows_only_explicit_retry_policy_change(tmp_path, change):
    """Preserve original signature and reject changes beyond the allowlist."""
    old = {
        "evaluator_sha256": "old",
        "model": "teacher",
        "temperature": 1,
        "dataset": "original",
        "reliability": {"retry_diagnostic_failures": False, "episode_error_retries": 1},
    }
    manifest = tmp_path / "run_config.json"
    manifest.write_text(json.dumps({"signature": old}))
    new = deepcopy(old)
    new["evaluator_sha256"] = "new"
    new["reliability"]["retry_diagnostic_failures"] = True
    if change == "retries":
        new["reliability"]["episode_error_retries"] = 2
    elif change:
        new[change] = "changed"
    kwargs = dict(
        signature=new,
        resume=True,
        allow_evaluator_code_change=True,
        allow_diagnostic_backfill=True,
    )
    if change:
        with pytest.raises(ValueError, match="Resume settings differ"):
            prepare_output_dir(tmp_path, **kwargs)
    else:
        prepare_output_dir(tmp_path, **kwargs)
        assert (tmp_path / "resume_policy_events.jsonl").exists()
    assert json.loads(manifest.read_text())["signature"] == old


def test_backfill_requires_explicit_permission(tmp_path):
    """Code-change permission alone does not authorize a retry policy change."""
    old = {
        "evaluator_sha256": "old",
        "reliability": {"retry_diagnostic_failures": False},
    }
    (tmp_path / "run_config.json").write_text(json.dumps({"signature": old}))
    new = deepcopy(old)
    new["evaluator_sha256"] = "new"
    new["reliability"]["retry_diagnostic_failures"] = True
    with pytest.raises(ValueError):
        prepare_output_dir(
            tmp_path, signature=new, resume=True, allow_evaluator_code_change=True
        )
