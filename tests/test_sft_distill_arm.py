"""The SFT arm: the no-dialogue scorers and the distillation data build.

The arm answers "what is plain distillation worth", so the two things that can
quietly break it are (a) its numbers not coming from the same scorers the taught
arms use, and (b) the SFT sequence training on the wrong tokens.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
from types import SimpleNamespace

import pytest

from examples.pedagogical_rl import cross_eval as ce

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "tutor"
    / "scripts"
    / "build_sft_distill_dataset.py"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_sft_distill", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Recorder:
    def __init__(self, student_text: str = "the answer is \\boxed{7}") -> None:
        self.student_text = student_text
        self.calls: list[tuple[list[dict[str, str]], int]] = []

    async def student(
        self, messages, *, n=1, max_tokens=None, rid_prefix="", timeout=None
    ):
        self.calls.append((messages, n))
        return [self.student_text] * n


async def always_correct(*, task, ground_truth, answer):
    return True


# --------------------------------------------------------------------------
# The no-dialogue row
# --------------------------------------------------------------------------


def test_no_dialogue_runs_both_scorers_on_an_empty_transcript():
    rec = Recorder()
    results = asyncio.run(
        ce.score_no_dialogue(
            task="2+2?",
            ground_truth="7",
            retest=ce.RetestSpec(replays=4),
            interview=ce.InterviewSpec(attempts=8),
            student_call=rec.student,
            answer_judge=always_correct,
        )
    )
    assert set(results) == {"retest", "interview"}
    assert results["retest"].complete and results["retest"].attempts == 4
    assert results["interview"].complete and results["interview"].attempts == 8
    # Four independent requests for ours, one n-choice request for theirs --
    # the same asymmetry the taught arms are measured under.
    assert sum(1 for _m, n in rec.calls if n == 1) == 4
    assert sum(1 for _m, n in rec.calls if n == 8) == 1


def test_no_dialogue_still_shows_the_student_the_problem():
    """An empty transcript is a complete prompt for both scorers, not a
    degenerate one: ours puts the task in the final turn, theirs in the system
    prompt. If either dropped it, this arm would be measuring nothing."""

    rec = Recorder()
    asyncio.run(
        ce.score_no_dialogue(
            task="UNIQUE_TASK_TOKEN",
            ground_truth="7",
            retest=ce.RetestSpec(replays=1),
            interview=ce.InterviewSpec(attempts=1),
            student_call=rec.student,
            answer_judge=always_correct,
        )
    )
    for messages, _n in rec.calls:
        assert "UNIQUE_TASK_TOKEN" in "".join(m["content"] for m in messages)
        # And no dialogue was invented to fill the gap.
        assert len(messages) == 2


def test_no_dialogue_metrics_use_the_shared_naming():
    rec = Recorder()
    results = asyncio.run(
        ce.score_no_dialogue(
            task="t",
            ground_truth="7",
            retest=ce.RetestSpec(replays=2),
            interview=ce.InterviewSpec(attempts=2),
            student_call=rec.student,
            answer_judge=always_correct,
        )
    )
    metrics = ce.no_dialogue_metrics(results)
    assert metrics["xeval/no_dialogue/retest/success"] == 1.0
    assert metrics["xeval/no_dialogue/interview/success"] == 1.0
    assert "xeval/no_dialogue/interview/success_math" in metrics
    for key in metrics:
        assert key.startswith("xeval/no_dialogue/")


# --------------------------------------------------------------------------
# The distillation data build
# --------------------------------------------------------------------------


def test_the_training_prompt_is_byte_identical_to_the_eval_input():
    """The premise of this arm. It is fine-tuned on the prompt it is tested
    under, so if these two ever differ it stops being the best case it is
    supposed to be -- and nothing else would notice."""

    builder = _load_builder()
    trained_on = builder.retest_prompt("What is 2+2?")
    tested_under = ce.retest_messages(transcript=[], task="What is 2+2?")
    assert trained_on == tested_under
    # The two-message shape the scorer uses, not a bare user turn.
    assert [m["role"] for m in trained_on] == ["system", "user"]
    assert "student talking with a teacher" in trained_on[0]["content"]
    assert "What is 2+2?" in trained_on[1]["content"]


def test_the_retest_prompt_is_what_the_no_dialogue_scorer_actually_sends():
    """Tighter than equality with retest_messages: equal to what
    score_no_dialogue puts on the wire."""

    builder = _load_builder()
    rec = Recorder()
    asyncio.run(
        ce.score_no_dialogue(
            task="What is 2+2?",
            ground_truth="4",
            retest=ce.RetestSpec(replays=1),
            # 2, not 1: the interview issues one request for n attempts, so at
            # attempts=1 both scorers send n=1 and the two are indistinguishable
            # in the recorder.
            interview=ce.InterviewSpec(attempts=2),
            student_call=rec.student,
            answer_judge=always_correct,
        )
    )
    sent = [messages for messages, n in rec.calls if n == 1]
    assert len(sent) == 1, "expected exactly the one re-test call"
    assert sent[0] == builder.retest_prompt("What is 2+2?")
    # And their scorer really does prompt the student a different way, which is
    # why the no_dialogue/interview cell is not aligned to this arm's training.
    interview_sent = [messages for messages, n in rec.calls if n == 2]
    assert interview_sent and interview_sent[0] != sent[0]


def test_the_math_template_is_still_available_as_the_neutral_alternative():
    builder = _load_builder()
    messages = builder.math_prompt("What is 2+2?")
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("What is 2+2?")
    assert messages[0]["content"].endswith(
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    assert set(builder.PROMPTS) == {"retest", "math"}


class StubTokenizer:
    """Enough of a tokenizer to check the mask, with no model on disk."""

    eos_token_id = 99

    def apply_chat_template(
        self, messages, *, add_generation_prompt, tokenize, enable_thinking
    ):
        assert add_generation_prompt and tokenize and not enable_thinking
        # One token per word across every turn, so the two-message re-test prompt
        # and the single-turn math prompt are both handled.
        return [1] * sum(len(m["content"].split()) for m in messages)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [2] * len(text.split())


def test_the_prompt_is_masked_out_and_the_solution_is_trained_on():
    builder = _load_builder()
    records = [
        {"task": "a b c", "solution": "x y z w", "correct": True},
    ]
    dataset, dropped = builder._tokenize(
        records, tokenizer=StubTokenizer(), max_length=100, prompt="math"
    )
    assert dropped == 0 and len(dataset) == 1
    row = dataset[0]
    ids, mask = row["input_ids"], row["loss_mask"]
    assert len(ids) == len(mask)
    # "a b c" plus the instruction words -> prompt; "x y z w" plus eos -> target.
    prompt_len = mask.index(1)
    assert set(mask[:prompt_len]) == {0}
    assert set(mask[prompt_len:]) == {1}
    assert sum(mask) == 5  # four solution tokens plus the eos
    assert ids[-1] == StubTokenizer.eos_token_id


def test_an_empty_solution_is_dropped_rather_than_trained_on():
    builder = _load_builder()
    dataset, _dropped = builder._tokenize(
        [{"task": "a", "solution": "   ", "correct": True}],
        tokenizer=StubTokenizer(),
        max_length=100,
        prompt="math",
    )
    assert len(dataset) == 0


def test_an_overlong_example_is_dropped_and_counted():
    builder = _load_builder()
    records = [{"task": "a b", "solution": " ".join(["w"] * 50), "correct": True}]
    dataset, dropped = builder._tokenize(
        records, tokenizer=StubTokenizer(), max_length=20, prompt="math"
    )
    assert len(dataset) == 0 and dropped == 1


def test_the_retest_prompt_tokenizes_the_same_way():
    """Same masking guarantee under the default prompt, which is a two-message
    chat rather than the single user turn the math template produces."""

    builder = _load_builder()
    dataset, dropped = builder._tokenize(
        [{"task": "a b c", "solution": "x y z", "correct": True}],
        tokenizer=StubTokenizer(),
        max_length=200,
        prompt="retest",
    )
    assert dropped == 0 and len(dataset) == 1
    ids, mask = dataset[0]["input_ids"], dataset[0]["loss_mask"]
    prompt_len = mask.index(1)
    assert set(mask[:prompt_len]) == {0}
    assert sum(mask) == 4  # three solution tokens plus the eos
    assert ids[-1] == StubTokenizer.eos_token_id


# --------------------------------------------------------------------------
# The sampling loop: try until correct, skip if it never is
# --------------------------------------------------------------------------


class _Args2:
    """Just the fields _sample_rows reads."""

    def __init__(self, attempts=3, per_task=1, generate_prompt="retest"):
        self.attempts = attempts
        self.per_task = per_task
        self.generate_prompt = generate_prompt
        self.max_tokens = 2048
        self.temperature = 0.7
        self.top_p = 0.8


class ScriptedTeacher:
    """Returns a fixed sequence of solutions, cycling per problem."""

    def __init__(self, per_problem: dict[str, list[str]]) -> None:
        self.per_problem = per_problem
        self.calls: dict[str, int] = {}

    async def generate(self, messages, *, n, max_tokens, temperature, top_p):
        # The problem text is in the last user turn under either prompt.
        body = messages[-1]["content"]
        key = next(k for k in self.per_problem if k in body)
        index = self.calls.get(key, 0)
        self.calls[key] = index + 1
        seq = self.per_problem[key]
        return [seq[min(index, len(seq) - 1)]]


def _verdict(correct: bool):
    return SimpleNamespace(correct=correct)


async def _score_by_marker(*, task, ground_truth, student_answer, judge):
    """Accept a solution iff it contains "GOOD"."""

    return _verdict("GOOD" in student_answer)


def _rows(*names):
    return [
        {"id": name, "task": f"solve {name}", "ground_truth": "7"} for name in names
    ]


def test_sampling_stops_at_the_first_accepted_solution():
    builder = _load_builder()
    teacher = ScriptedTeacher({"alpha": ["GOOD first"]})
    records = asyncio.run(
        builder._generate(
            _Args2(),
            _rows("alpha"),
            teacher=teacher,
            judge=object(),
            score=_score_by_marker,
        )
    )
    assert len(records) == 1
    assert records[0]["kept"] and records[0]["correct"]
    assert records[0]["attempt"] == 1
    # One try, not three: the extra calls would be pure waste.
    assert teacher.calls["alpha"] == 1


def test_sampling_retries_up_to_the_limit_and_keeps_the_first_success():
    builder = _load_builder()
    teacher = ScriptedTeacher({"alpha": ["bad", "bad", "GOOD third"]})
    records = asyncio.run(
        builder._generate(
            _Args2(attempts=3),
            _rows("alpha"),
            teacher=teacher,
            judge=object(),
            score=_score_by_marker,
        )
    )
    assert teacher.calls["alpha"] == 3
    kept = [r for r in records if r["kept"]]
    assert len(kept) == 1
    assert kept[0]["attempt"] == 3 and "GOOD" in kept[0]["solution"]
    # The failures are recorded for inspection but never kept.
    assert [r["kept"] for r in records] == [False, False, True]


def test_a_problem_never_solved_is_skipped_rather_than_trained_on_wrong():
    """The invariant the whole arm rests on: no wrong target ever reaches the
    student. A floor built by distilling the teacher's mistakes would measure
    something else entirely."""

    builder = _load_builder()
    teacher = ScriptedTeacher({"alpha": ["bad", "bad", "bad"]})
    records = asyncio.run(
        builder._generate(
            _Args2(attempts=3),
            _rows("alpha"),
            teacher=teacher,
            judge=object(),
            score=_score_by_marker,
        )
    )
    assert teacher.calls["alpha"] == 3
    assert not any(r["kept"] for r in records)
    assert all(not r["correct"] for r in records)


def test_one_row_per_problem_regardless_of_how_easy_it_is():
    """Balance is the point of --per-task 1. An easy problem must not outweigh a
    hard one, or the student trains mostly on what was already easy."""

    builder = _load_builder()
    teacher = ScriptedTeacher(
        {"alpha": ["GOOD a"], "beta": ["bad", "GOOD b"], "gamma": ["bad", "bad", "bad"]}
    )
    records = asyncio.run(
        builder._generate(
            _Args2(attempts=3, per_task=1),
            _rows("alpha", "beta", "gamma"),
            teacher=teacher,
            judge=object(),
            score=_score_by_marker,
        )
    )
    kept = [r for r in records if r["kept"]]
    assert sorted(r["id"] for r in kept) == ["alpha", "beta"]
    assert len(kept) == 2  # exactly one each, and nothing for gamma


def test_per_task_zero_keeps_every_accepted_solution():
    builder = _load_builder()
    teacher = ScriptedTeacher({"alpha": ["GOOD 1", "GOOD 2", "GOOD 3"]})
    records = asyncio.run(
        builder._generate(
            _Args2(attempts=3, per_task=0),
            _rows("alpha"),
            teacher=teacher,
            judge=object(),
            score=_score_by_marker,
        )
    )
    assert teacher.calls["alpha"] == 3
    assert sum(1 for r in records if r["kept"]) == 3


def test_a_teacher_error_does_not_end_the_retries():
    builder = _load_builder()

    class FlakyTeacher(ScriptedTeacher):
        async def generate(self, messages, **kwargs):
            index = self.calls.get("alpha", 0)
            self.calls["alpha"] = index + 1
            if index == 0:
                raise RuntimeError("endpoint hiccup")
            return ["GOOD second"]

    teacher = FlakyTeacher({"alpha": []})
    records = asyncio.run(
        builder._generate(
            _Args2(attempts=3),
            _rows("alpha"),
            teacher=teacher,
            judge=object(),
            score=_score_by_marker,
        )
    )
    assert records[0]["error"] and not records[0]["kept"]
    assert records[1]["kept"]


# --------------------------------------------------------------------------
# The local serving adapter
#
# The engine itself needs a GPU, so what is covered here is the layer between
# cross_eval's student callable and SGLang: the chat template, the sample count,
# and what happens when the engine returns the wrong number of them.
# --------------------------------------------------------------------------

_EVAL_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "tutor"
    / "scripts"
    / "eval_sft_student.py"
)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("eval_sft_student", _EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEngine:
    def __init__(self, texts=("\\boxed{7}",)) -> None:
        self.texts = list(texts)
        self.calls: list[tuple[str, dict]] = []

    async def async_generate(self, *, prompt, sampling_params):
        self.calls.append((prompt, dict(sampling_params)))
        return [{"text": text} for text in self.texts[: sampling_params["n"]]]


class TemplateTokenizer:
    def apply_chat_template(
        self, messages, *, add_generation_prompt, tokenize, enable_thinking
    ):
        # The student is served with thinking off in the RL arms; if this flipped,
        # the SFT arm would be measuring a different model's behaviour.
        assert add_generation_prompt and not tokenize and not enable_thinking
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


class _Args:
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    max_tokens = 2048


def test_the_adapter_flattens_messages_through_the_chat_template():
    evaluator = _load_evaluator()
    engine = FakeEngine(texts=["a", "b"])
    student_call = evaluator._make_student_call(engine, TemplateTokenizer(), _Args())
    out = asyncio.run(
        student_call([{"role": "user", "content": "hi"}], n=2, max_tokens=64)
    )
    assert out == ["a", "b"]
    prompt, params = engine.calls[0]
    assert prompt == "user:hi"
    assert params["n"] == 2
    assert params["max_new_tokens"] == 64
    # Sampling matches student_models[0] in the RL arms.
    assert params["temperature"] == 0.7 and params["top_p"] == 0.8
    assert params["top_k"] == 20


def test_the_adapter_refuses_a_short_engine_response():
    """A silent short return would average over fewer samples than asked for and
    read as a real score instead of as missing data."""

    evaluator = _load_evaluator()
    engine = FakeEngine(texts=["only one"])
    student_call = evaluator._make_student_call(engine, TemplateTokenizer(), _Args())
    with pytest.raises(RuntimeError, match="expected 4"):
        asyncio.run(student_call([{"role": "user", "content": "hi"}], n=4))


def test_a_short_n_choice_return_fails_only_the_cell_that_asked_for_n():
    """An engine that can only produce one sample per request breaks their
    scorer and not ours: the interview is one n=8 request, the re-test is four
    n=1 requests. The interview cell must come back incomplete rather than
    averaging over fewer samples than it asked for."""

    evaluator = _load_evaluator()
    engine = FakeEngine(texts=["\\boxed{7}"])
    student_call = evaluator._make_student_call(engine, TemplateTokenizer(), _Args())
    results = asyncio.run(
        ce.score_no_dialogue(
            task="t",
            ground_truth="7",
            retest=ce.RetestSpec(replays=4),
            interview=ce.InterviewSpec(attempts=8),
            student_call=student_call,
            answer_judge=always_correct,
        )
    )
    assert results["retest"].complete and results["retest"].attempts == 4
    assert not results["interview"].complete
    assert results["interview"].score == 0.0
    metrics = ce.no_dialogue_metrics(results)
    assert metrics["xeval/no_dialogue/interview/incomplete"] == 1.0
    assert metrics["xeval/no_dialogue/retest/incomplete"] == 0.0


def test_a_dead_engine_marks_both_cells_incomplete():
    evaluator = _load_evaluator()

    class DeadEngine:
        async def async_generate(self, *, prompt, sampling_params):
            raise RuntimeError("engine crashed")

    student_call = evaluator._make_student_call(
        DeadEngine(), TemplateTokenizer(), _Args()
    )
    results = asyncio.run(
        ce.score_no_dialogue(
            task="t",
            ground_truth="7",
            retest=ce.RetestSpec(replays=2),
            interview=ce.InterviewSpec(attempts=2),
            student_call=student_call,
            answer_judge=always_correct,
        )
    )
    assert not results["retest"].complete and results["retest"].score == 0.0
    assert not results["interview"].complete and results["interview"].score == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
