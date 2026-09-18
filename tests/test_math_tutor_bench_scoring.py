"""Regression tests for local StepVerify scoring; no generation or GPU."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCH = Path(__file__).resolve().parents[1] / "examples/math_tutor_bench"
sys.path.insert(0, str(BENCH))
import rescore_stepverify  # noqa: E402
import run_task  # noqa: E402


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A: Yes\nExplanation\nA: No", False),
        ("A: No\nActually wrong.\nFinal Answer: Yes", True),
        ("**Answer: No**\nDo not answer yes without checking.", False),
        ("Yes, initially.\nNo, the solution is correct.", False),
        ("Yesterday nobody checked", True),
        ("The answer is no.", False),
        ("", True),
    ],
)
def test_final_judgment(text, expected):
    assert run_task.parse_correctness(text) is expected


def test_correction_keeps_headings_and_hides_native_thinking():
    raw = "<think>Final Answer: 99</think>### Problem:\nQuestion\nQuoted Student: mistaken\nFinal Answer: 27"
    visible = run_task.response_for_task(
        "mistake_correction", raw, ["Problem:", "Student:"]
    )
    assert visible == "### Problem:\nQuestion\nQuoted Student: mistaken\nFinal Answer: 27"
    assert (
        rescore_stepverify.correction_parser(
            BENCH / ".runtime/upstream"
        ).parse_response(visible)
        == 27
    )


@pytest.mark.parametrize(
    "raw",
    [
        "<reasoning>analysis</reasoning><output>Answer</output>",
        "<reasoning>unfinished",
        "<output>unfinished",
        "<end>",
        "<end></end>",
    ],
)
def test_training_xml_is_not_interpreted(raw):
    assert run_task.extract_visible_teacher_output(raw) == raw
    assert (
        run_task.response_for_task("mistake_correction", raw, ["Problem:", "Student:"])
        == raw
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("</think>\n\nA: 2", "A: 2"),
        ("<think>unfinished", ""),
        ("  Ordinary reply.  ", "Ordinary reply."),
    ],
)
def test_native_thinking_behavior_retained(raw, expected):
    assert run_task.extract_visible_teacher_output(raw) == expected


def test_paragraphs_preserved_and_answer_parser_unchanged():
    raw = "Teacher: Hint.\n\nNext paragraph"
    stops = ["Teacher:", "Student:", "\n\n"]
    assert run_task.response_for_task("scaffolding_generation", raw, stops) == "Hint.\n\nNext paragraph"
    parser = SimpleNamespace(parse_response=lambda s: 3)
    assert run_task.parse_task_response("mistake_location", "A: 3", parser) == 3


@pytest.mark.parametrize("task", [
    "problem_solving", "socratic_questioning", "student_solution_correctness",
    "mistake_location", "mistake_correction", "scaffolding_generation",
    "pedagogy_following", "scaffolding_generation_hard", "pedagogy_following_hard",
])
def test_all_tasks_use_same_role_boundary_rule(task):
    """Only the initial teacher label and subsequent line-start roles are removed."""
    raw = "Teacher: Hint.\n\nProblem: example\nQ: why?\nExplanation: detail\nStudent: invented\nTeacher: next"
    expected = "Hint.\n\nProblem: example\nQ: why?\nExplanation: detail"
    assert run_task.response_for_task(task, raw, ["\n", "Problem:", "Q:", "Explanation:"]) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Teacher： First\n\n  student： next", "First"),
    ("Teacher: First\nTeacher: second", "First"),
    ("Mention Student: inline\nTutor: retained", "Mention Student: inline\nTutor: retained"),
    ("Student: invented", ""),
])
def test_role_boundaries_match_reviewed_diagnostic(raw, expected):
    assert run_task.response_for_task("pedagogy_following", raw, None) == expected


def test_rescore_non_destructive(tmp_path):
    source = tmp_path / "run"
    metric_payloads = {}
    for task, raw, target, pred in [
        ("student_solution_correctness", "A: Yes\nA: No", "No", True),
        ("mistake_correction", "### Problem:\nQuestion\nFinal Answer: 27", "27", None),
    ]:
        folder = source / "tasks" / task
        folder.mkdir(parents=True)
        row = {
            "index": 0,
            "raw_response": raw,
            "visible_response": "",
            "prediction": pred,
            "target": target,
        }
        (folder / "predictions.jsonl").write_text(json.dumps(row) + "\n")
        name = (
            "solution_correctness" if task == "student_solution_correctness" else task
        )
        payload = {"task_name": name, "metrics": {"accuracy": 0}, "num_examples": 1}
        (folder / "metrics.json").write_text(json.dumps(payload))
        metric_payloads[name] = payload
    metric_payloads["mistake_location"] = {"metrics": {"f1_micro": 0.4}}
    (source / "summary.json").write_text(
        json.dumps({"official_task_metrics": metric_payloads, "leaderboard": {}})
    )
    before = {str(f): f.read_bytes() for f in source.rglob("*") if f.is_file()}
    result = rescore_stepverify.rescore(source, BENCH / ".runtime/upstream")
    assert result["changes"]["mistake_correction"]["after"]["accuracy"] == 1
    assert all(Path(f).read_bytes() == content for f, content in before.items())
    with pytest.raises(FileExistsError):
        rescore_stepverify.rescore(source, BENCH / ".runtime/upstream")
