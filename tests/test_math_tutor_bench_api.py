"""Offline-only tests for API MathTutorBench, with no GPU or paid requests."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from examples.math_tutor_bench.api_run import runner as api


@pytest.mark.parametrize("value", ["", "0,", "-1", "0,0", "0,00", "gpu0"])
def test_score_rejects_invalid_gpu_ids(value):
    """Invalid or repeated physical devices must not launch any workers."""
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        api.parse_gpu_ids(value)


def test_score_single_gpu_preserves_existing_entrypoint(tmp_path, monkeypatch):
    """Explicit single GPU selection uses the existing scorer and cache logic."""
    calls = []
    monkeypatch.setattr(api.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    api.score_pedrm(tmp_path, tmp_path / "model", api.parse_gpu_ids(" 3 "))
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert calls[0][0][0][1].endswith("/score_pedrm.py")


@pytest.mark.parametrize("fail", [False, True])
def test_eight_gpu_scoring_resume_and_failure_cleanup(tmp_path, monkeypatch, fail):
    """Eight isolated workers preserve inputs, reuse shards, and clean up on failure."""
    from pathlib import Path

    from examples.math_tutor_bench.score_pedrm import PEDAGOGY_TASKS

    (tmp_path / "score_identity.json").write_text("{}")
    inputs = []
    for task in PEDAGOGY_TASKS:
        folder = tmp_path / "tasks" / task
        folder.mkdir(parents=True)
        source = folder / "generations.json"
        source.write_text('[{"original": "input"}]')
        inputs.append((source, source.read_bytes()))
    audit = tmp_path / "api_requests.jsonl"
    audit.write_text("original request audit\n")
    inputs.append((audit, audit.read_bytes()))
    workers, merges = [], []

    class Worker:
        def __init__(self, command, **kwargs):
            self.command = command
            self.gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            self.returncode = (1 if not workers else None) if fail else 0
            self.terminated = False
            workers.append(self)
            if not fail:
                task = command[command.index("--task") + 1]
                shard = int(command[command.index("--shard-index") + 1])
                Path(command[command.index("--output") + 1]).write_text(
                    json.dumps(
                        {
                            "task": task,
                            "shard_index": shard,
                            "num_shards": 2,
                            "total_task_samples": 1,
                            "records": (
                                [
                                    {
                                        "index": 0,
                                        "candidate_score": 2.0,
                                        "reference_score": 1.0,
                                    }
                                ]
                                if shard == 0
                                else []
                            ),
                        }
                    )
                )

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(api.subprocess, "Popen", Worker)
    monkeypatch.setattr(api.subprocess, "run", lambda *a, **k: merges.append(a))
    devices = api.parse_gpu_ids("0,1,2,3,4,5,6,7")
    if fail:
        with pytest.raises(RuntimeError, match="worker failed"):
            api.score_pedrm(tmp_path, tmp_path / "model", devices)
        assert all(w.terminated for w in workers[1:])
        assert not merges
    else:
        api.score_pedrm(tmp_path, tmp_path / "model", devices)
        assert len(workers) == 8
        assert {w.gpu for w in workers} == set(devices)
        assert all(
            w.command[w.command.index("--num-shards") + 1] == "2" for w in workers
        )
        api.score_pedrm(tmp_path, tmp_path / "model", devices)
        assert len(workers) == 8  # Completed shards are not scored again.
        assert len(merges) == 2
        assert merges[0][0][1].endswith("/merge_pedrm_shards.py")
        # Exercise the real existing merger with deterministic mock GPU scores.
        merge_command = merges[0][0]
        monkeypatch.undo()
        subprocess.run(merge_command, check=True, capture_output=True)
        metrics = json.loads((tmp_path / "pedrm/pedrm_metrics.json").read_text())
        assert set(metrics) == set(PEDAGOGY_TASKS)
        assert all(
            m["win_rate"] == 1.0 and m["total_samples"] == 1 for m in metrics.values()
        )
    assert all(path.read_bytes() == original for path, original in inputs)


def test_gemini_payload_preserves_prompt_and_omits_qwen_options():
    """The provider sees the exact upstream text, not a Qwen-specific suffix."""
    cfg = api.load_config(api.PACKAGE / "gemini-3.8-flash.yaml")
    assert api.request_payload(cfg, "Teacher: original\n") == {
        "model": "gemini-3.8-flash",
        "reasoning_effort": "medium",
        "messages": [{"role": "user", "content": "Teacher: original\n"}],
    }


def test_usage_includes_gemini_hidden_thinking_and_cache():
    """Gemini completion_tokens can omit thoughts included in total_tokens."""
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 180,
        "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
    }
    assert api.usage_cost(usage, [2, 0.2, 2.5, 12]) == pytest.approx(
        (120 + 6 + 25 + 960) / 1e6
    )
    assert api.usage_cost(None, [1, 1, 1, 1]) is None


def test_budget_resume_keeps_unknown_inflight_requests(tmp_path):
    """An interrupted paid request is reserved, never silently counted free."""
    journal = api.Journal(tmp_path, [1, 1, 1, 1], 0.1, 0.1)
    journal.begin("a:0", {"messages": []})
    with pytest.raises(api.BudgetStopped):
        journal.begin("a:1", {})
    resumed = api.Journal(tmp_path, [1, 1, 1, 1], 0.1, 0.1)
    assert resumed.status()["budget_used_usd"] == 0.1


def test_retry_and_resume_capture_full_response_without_auth(tmp_path, monkeypatch):
    """Retry is accounted separately; saved success is reusable without another call."""
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    cfg = api.load_config(api.PACKAGE / "gemini-3.8-flash.yaml")
    requests = []
    response_body = {
        "choices": [
            {
                "message": {
                    "content": "Hello",
                    "reasoning_content": "private thought",
                    "extra_content": {"google": {"thought_signature": "opaque"}},
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 130},
    }

    def respond(request):
        requests.append(json.loads(request.content))
        return (
            httpx.Response(503, json={"error": "secret"})
            if len(requests) == 1
            else httpx.Response(200, json=response_body)
        )

    journal = api.Journal(tmp_path, cfg["prices_per_million"], 2, 0.1)
    with httpx.Client(
        base_url="https://private.test/",
        headers={"Authorization": "Bearer secret"},
        transport=httpx.MockTransport(respond),
    ) as client:
        result = api.fetch(client, journal, cfg, "a:0", "Original prompt", 2)
        assert result == response_body
        resumed = api.Journal(tmp_path, cfg["prices_per_million"], 2, 0.1)
        assert api.fetch(client, resumed, cfg, "a:0", "Original prompt", 2) == result
    assert len(requests) == 2
    records = api.read_jsonl(journal.path)
    assert len(records) == 4 and records[0]["payload"] == requests[0]
    assert records[-1]["payload"] == response_body
    assert "secret" not in journal.path.read_text()
    assert resumed.status()["unknown_requests"] == 1


def test_empty_visible_output_never_exposes_thinking():
    """A hidden thought must not become the scored teacher response."""
    task = SimpleNamespace(
        parse_response=lambda x: x, format_ground_truth=lambda x: "target"
    )
    config = SimpleNamespace(name="scaffolding_generation", stop=None)
    body = {
        "choices": [
            {"message": {"content": None, "reasoning_content": "secret answer"}}
        ]
    }
    r = api.build_record(config.name, config, task, {}, 0, "prompt", body)
    assert r["visible_response"] == ""
    assert r["generation"]["generated_teacher_utterance"] == ""


def test_manifest_rejects_different_model_or_prompts(tmp_path):
    """Resuming must not mix two evaluation protocols."""
    path = tmp_path / "run.json"
    api.check_manifest(path, {"model": "a", "prompts": "hash"})
    with pytest.raises(ValueError):
        api.check_manifest(path, {"model": "b", "prompts": "hash"})


def test_jsonl_repairs_only_partial_final_record(tmp_path):
    """A process crash may truncate the tail, not justify dropping interior data."""
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"a":1}\n{"a":')
    assert api.read_jsonl(path) == [{"a": 1}]
    assert path.read_bytes() == b'{"a":1}\n'
    path.write_bytes(b'{bad}\n{"a":1}\n')
    with pytest.raises(ValueError):
        api.read_jsonl(path)


def test_generation_export_and_resume_use_no_additional_calls(tmp_path, monkeypatch):
    """A full local export creates Ped-RM inputs and resumes solely from the journal."""
    cfg = api.load_config(api.PACKAGE / "gemini-3.8-flash.yaml")
    monkeypatch.setenv(cfg["endpoint_env"], "https://private.test/v1")
    monkeypatch.setenv(cfg["key_env"], "secret")
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hint"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )

    client_class = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(respond)),
    )
    task = SimpleNamespace(
        parse_response=lambda s: s,
        format_ground_truth=lambda e: "reference",
        compute_metrics=lambda p, t: {"count": len(p)},
    )
    tc = SimpleNamespace(name="scaffolding_generation", stop=None)
    loaded = [
        (
            tc.name,
            tc,
            task,
            [{"question": "Problem", "dialog_history": "Student: hi"}],
            ["Official prompt"],
        )
    ]
    args = SimpleNamespace(
        limit=1,
        budget_usd=2,
        unknown_reserve_usd=0.1,
        concurrency=1,
        retries=0,
        timeout=1,
    )
    api.generate(args, cfg, loaded, tmp_path)
    api.generate(args, cfg, loaded, tmp_path)
    assert len(calls) == 1
    assert (tmp_path / "generation_complete.json").exists()
    generations = json.loads(
        (tmp_path / "tasks/scaffolding_generation/generations.json").read_text()
    )
    assert generations[0]["generated_teacher_utterance"] == "Hint"
    assert generations[0]["ground_truth_response"] == ""
    assert json.loads((tmp_path / "pending.json").read_text()) == []


def test_auth_failure_stops_following_requests(tmp_path):
    """Invalid credentials must not generate one failed paid attempt per example."""
    cfg = api.load_config(api.PACKAGE / "gemini-3.8-flash.yaml")
    j = api.Journal(tmp_path, cfg["prices_per_million"], 2, 0.1)
    with httpx.Client(
        base_url="https://example.test/",
        transport=httpx.MockTransport(lambda _: httpx.Response(401)),
    ) as client:
        with pytest.raises(RuntimeError):
            api.fetch(client, j, cfg, "a:0", "prompt", 0)
        with pytest.raises(api.BudgetStopped):
            api.fetch(client, j, cfg, "a:1", "prompt", 0)
    assert len(j.requests) == 1


def test_report_uses_official_f1_metrics(tmp_path):
    """Legacy summary aliases must not turn F1 into accuracy or macro-F1."""
    api.run_task.write_json(tmp_path / "run.json", {"evaluation_mode": "api-chat"})
    api.run_task.write_json(
        tmp_path / "summary.json",
        {
            "leaderboard": {},
            "official_task_metrics": {
                "solution_correctness": {"metrics": {"f1": 0.8, "accuracy": 0.7}},
                "mistake_location": {"metrics": {"f1_micro": 0.6, "f1_macro": 0.5}},
            },
        },
    )
    api.finalize_report(tmp_path)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["leaderboard"] == {
        "solution_correctness": 0.8,
        "mistake_location": 0.6,
    }


@pytest.mark.slow
@pytest.mark.skipif(
    not (api.BENCH / ".runtime/upstream/registry.py").exists(),
    reason="Pinned runtime assets not staged",
)
def test_nine_staged_tasks_export_with_offline_http_transport(tmp_path):
    """Exercise real official datasets/parsers/metrics, with zero external calls."""
    code = """
import sys, httpx
from examples.math_tutor_bench.api_run import runner as api
original = httpx.Client
calls = []
def respond(request):
    calls.append(request)
    return httpx.Response(200, json={"choices": [{"message": {"content": "1"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})
httpx.Client = lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(respond))
sys.argv = ["runner", "generate", "--limit", "1", "--concurrency", "1", "--output-dir", sys.argv[1]]
api.main()
assert len(calls) == 9
"""
    repo = api.BENCH.parents[1]
    env = {
        **os.environ,
        "GEMINI_BASE_URL": "https://offline.test/v1",
        "GEMINI_API_KEY": "fake",
    }
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert len(list((tmp_path / "tasks").glob("*/metrics.json"))) == 9
    assert len(list((tmp_path / "tasks").glob("*/generations.json"))) == 4
    assert len(api.read_jsonl(tmp_path / "api_requests.jsonl")) == 18
