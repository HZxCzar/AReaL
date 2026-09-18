import json

import pytest

from examples.tutor.scripts.api_run.benchmark_endpoints import (
    sample_requests,
    summarize,
)


def test_sampling_excludes_paid_teacher(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = []
    for role, system in [
        ("teacher", "Do not replay"),
        ("student:qwen", "Student"),
        ("judge", "Judge one stated student preference"),
        ("judge", "Leak judge"),
        ("answer_judge", "Answer judge"),
    ]:
        rows.append(
            {
                "event": "request",
                "role": role,
                "payload": {"messages": [{"role": "system", "content": system}]},
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows))
    pools = sample_requests(path)
    assert set(pools) == {"preference", "leak", "answer", "student"}
    assert all(len(p) == 1 for p in pools.values())
    assert "Do not replay" not in json.dumps(pools)
    path.write_text("")
    with pytest.raises(ValueError):
        sample_requests(path)


def test_rates_count_only_successful_responses():
    records = [
        {"ok": True, "seconds": 1, "response": {"usage": {"completion_tokens": 100}}},
        {"ok": False, "seconds": 3, "response": {}},
    ]
    result = summarize(records, 4)
    assert result["errors"] == 1
    assert result["requests_per_second"] == 0.25
    assert result["output_tokens_per_second"] == 25
    assert result["p95_seconds"] == 3
