from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples.tutor.scripts import eval_suite as suite


def test_plan_is_serial_and_has_exact_repetitions():
    tasks = suite.plan()
    assert len(tasks) == 14
    assert [t["stage"] for t in tasks] == ["presolve_on"] * 6 + ["ped_protocol"] * 2 + [
        "presolve_off"
    ] * 6
    for stage in ("presolve_on", "presolve_off"):
        for model in ("ours", "pedrl"):
            assert [
                t["repeat"]
                for t in tasks
                if t["stage"] == stage and t["model"] == model
            ] == [1, 2, 3]


def test_tutor_modes_checkpoints_and_directories_are_isolated(monkeypatch, tmp_path):
    paths = {
        key: Path(
            f"/output/{key}/checkpoints/root/exp/trial/default/epoch21epochstep12globalstep999"
        )
        for key in ("ours", "pedrl")
    }
    monkeypatch.setenv("MATRIX_TEACHER_KEYS", "wrong")
    monkeypatch.setenv("EVAL_RUN_DIR", "/old")
    outputs = []
    for task in suite.plan():
        cmd, env, directory = suite.command(task, tmp_path, paths)
        outputs.append(directory)
        if task["stage"] != "ped_protocol":
            assert env["MATRIX_EXPECT_PRESOLVE"] == str(
                int(task["stage"] == "presolve_on")
            )
            assert env["MATRIX_TEACHER_KEYS"] == task["model"]
            assert env["SAVE_TRACES"] == "all"
            assert env["MATRIX_SHARD_NONE"] == "1"
            assert env["EVAL_RUN_DIR"] != "/old"
        else:
            assert "total_train_steps=0" in cmd
            assert "evaluation.preference_names=[none]" in cmd
            assert "EVAL_RUN_DIR" not in env
    assert len(set(outputs)) == 14


def test_cross_protocol_retry_preserves_previous_attempt(tmp_path):
    paths = {"ours": Path("/checkpoints/trial/default/step999")}
    a = suite.command(suite.plan()[6], tmp_path, paths, attempt=1)[2]
    b = suite.command(suite.plan()[6], tmp_path, paths, attempt=2)[2]
    assert a != b


def test_lock_refuses_two_controllers(tmp_path):
    first = suite.lock_suite(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            suite.lock_suite(tmp_path)
    finally:
        first.close()


def test_busy_gpu_refuses_next_task_without_killing(monkeypatch):
    monkeypatch.setattr(suite.socket, "socket", MagicMock())
    monkeypatch.setattr(
        suite.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="123, GPU-ours\n"),
    )
    monkeypatch.setattr(
        suite.os, "kill", lambda *a: pytest.fail("Must not kill a GPU occupant")
    )
    with pytest.raises(RuntimeError, match="Resources still busy"):
        suite.wait_resources("0,1,2,3,4,5,6,7", timeout=0)


def test_progress_keeps_completed_results(tmp_path):
    state = {
        "tasks": [
            dict(
                suite.plan()[0],
                status="done",
                result={
                    "baseline": 0.3,
                    "retest": 0.7,
                    "improvement": 0.4,
                    "leak": 0.1,
                    "pending": 0,
                },
            )
        ]
    }
    suite.write_progress(tmp_path, state)
    assert "40.00" in (tmp_path / "progress.md").read_text()
    assert (tmp_path / "repeat_statistics.json").is_file()
