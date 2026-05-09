from __future__ import annotations

import importlib
import asyncio
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight local test env
    torch = ModuleType("torch")

    class _FakeTensor:
        def __init__(self, data):
            self.data = data

        @property
        def shape(self):
            if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            if isinstance(self.data, list):
                return (len(self.data),)
            return ()

        @property
        def ndim(self):
            return len(self.shape)

        def dim(self):
            return self.ndim

        def unsqueeze(self, dim):
            assert dim == 0
            return _FakeTensor([self.data])

        def bool(self):
            if self.ndim == 2:
                return _FakeTensor([[bool(x) for x in row] for row in self.data])
            return _FakeTensor([bool(x) for x in self.data])

        def tolist(self):
            return self.data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, key):
            if isinstance(key, _FakeTensor):
                return _FakeTensor([x for x, keep in zip(self.data, key.data) if keep])
            value = self.data[key]
            return _FakeTensor(value) if isinstance(value, list) else value

    def _tensor(data, dtype=None):
        del dtype
        return _FakeTensor(list(data) if isinstance(data, tuple) else data)

    def _ones(length, dtype=None):
        return _FakeTensor([True if dtype is torch.bool else 1 for _ in range(length)])

    def _cat(values, dim=0):
        assert dim == 0
        if values[0].ndim == 2:
            data = []
            for value in values:
                data.extend(value.data)
            return _FakeTensor(data)
        data = []
        for value in values:
            data.extend(value.data)
        return _FakeTensor(data)

    def _pad(tensor_value, pad, value_fill=0.0, **kwargs):
        fill = kwargs.get("value", value_fill)
        right = pad[1] if pad else 0
        if tensor_value.ndim == 2:
            return _FakeTensor([row + [fill] * right for row in tensor_value.data])
        return _FakeTensor(tensor_value.data + [fill] * right)

    torch.Tensor = _FakeTensor
    torch.tensor = _tensor
    torch.ones = _ones
    torch.cat = _cat
    torch.long = "long"
    torch.float32 = "float32"
    torch.bool = "bool"
    torch.nn = SimpleNamespace(functional=SimpleNamespace(pad=_pad))
    sys.modules["torch"] = torch


def _install_areal_stubs_if_needed() -> None:
    try:
        importlib.import_module("areal.utils.hf_utils")
        return
    except Exception:
        for name in list(sys.modules):
            if name == "areal" or name.startswith("areal."):
                del sys.modules[name]

    areal = ModuleType("areal")
    workflow_context = SimpleNamespace(
        stat_scope=lambda: "examples",
        get=lambda: SimpleNamespace(task_id=None, is_eval=False),
    )
    api = ModuleType("areal.api")

    @dataclass
    class _ModelRequest:
        rid: str = ""
        input_ids: list[int] = field(default_factory=list)
        gconfig: Any | None = None
        metadata: dict[str, Any] = field(default_factory=dict)
        tokenizer: Any | None = None

    @dataclass
    class _ModelResponse:
        input_tokens: list[int] = field(default_factory=list)
        output_tokens: list[int] = field(default_factory=list)
        output_logprobs: list[float] = field(default_factory=list)
        output_versions: list[int] = field(default_factory=list)
        tokenizer: Any | None = None

        @property
        def input_len(self) -> int:
            return len(self.input_tokens)

        @property
        def output_len(self) -> int:
            return len(self.output_tokens)

    api.ModelRequest = _ModelRequest
    api.ModelResponse = _ModelResponse
    api.RolloutWorkflow = object
    utils = ModuleType("areal.utils")
    logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
    utils.logging = SimpleNamespace(getLogger=lambda *args, **kwargs: logger)
    utils.stats_tracker = SimpleNamespace(
        get=lambda _scope: SimpleNamespace(scalar=lambda **kwargs: None)
    )
    data_mod = ModuleType("areal.utils.data")

    def _concat_padded_tensors(tensor_dicts: list[dict[str, Any]], pad_value=0.0):
        keys = set(tensor_dicts[0])
        for item in tensor_dicts:
            assert set(item) == keys
        result = {}
        for key in tensor_dicts[0]:
            values = [item[key] for item in tensor_dicts]
            if isinstance(values[0], torch.Tensor):
                max_len = max(value.shape[-1] if value.ndim >= 2 else 1 for value in values)
                padded = []
                for value in values:
                    if value.ndim >= 2 and value.shape[-1] < max_len:
                        fill = 0.0 if key == "attention_mask" else pad_value
                        value = torch.nn.functional.pad(
                            value, (0, max_len - value.shape[-1]), value=fill
                        )
                    padded.append(value)
                result[key] = torch.cat(padded, dim=0)
            else:
                result[key] = values[0]
        return result

    data_mod.concat_padded_tensors = _concat_padded_tensors
    hf_utils = ModuleType("areal.utils.hf_utils")
    hf_utils.load_hf_tokenizer = lambda path: path

    areal.workflow_context = workflow_context
    sys.modules.update(
        {
            "areal": areal,
            "areal.api": api,
            "areal.utils": utils,
            "areal.utils.data": data_mod,
            "areal.utils.hf_utils": hf_utils,
        }
    )


_install_areal_stubs_if_needed()

tutor_workflow = importlib.import_module("examples.tutor.workflow")
LeakCheckResult = tutor_workflow.LeakCheckResult
ProgressJudgment = tutor_workflow.ProgressJudgment
PublicHistoryState = tutor_workflow.PublicHistoryState
StudentTurnState = tutor_workflow.StudentTurnState
TutorAgentWorkflow = tutor_workflow.TutorAgentWorkflow


class _FakeTokenizer:
    eos_token_id = None
    pad_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(int(token)) for token in tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> list[int]:
        del tokenize, add_generation_prompt, enable_thinking
        text = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        return self.encode(text)


class _FakeGConfig:
    temperature = 1.0
    top_p = 1.0
    max_new_tokens = 128

    def new(self, **kwargs):
        cfg = _FakeGConfig()
        cfg.__dict__.update(kwargs)
        return cfg


class _FakeEngine:
    def __init__(self, tokenizer: _FakeTokenizer, outputs: list[str]):
        self.tokenizer = tokenizer
        self.outputs = list(outputs)
        self.requests = []

    async def agenerate(self, req):
        self.requests.append(req)
        text = self.outputs.pop(0)
        output_tokens = self.tokenizer.encode(text)
        return tutor_workflow.ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=output_tokens,
            output_logprobs=[-0.1] * len(output_tokens),
            output_versions=[1] * len(output_tokens),
            tokenizer=self.tokenizer,
        )


class _ScriptedTutorWorkflow(TutorAgentWorkflow):
    def __init__(
        self,
        *,
        student_outputs: list[str],
        leak_results: list[LeakCheckResult] | None = None,
        progress_results: list[ProgressJudgment] | None = None,
    ):
        super().__init__(
            gconfig=_FakeGConfig(),
            tokenizer=_FakeTokenizer(),
            max_turns=3,
            debug_trace_dir="",
            success_reward=1.0,
            leak_penalty=-1.0,
            progress_improved_reward=0.3,
            progress_same_reward=0.0,
            progress_regressed_reward=-0.3,
        )
        self.student_outputs = list(student_outputs)
        self.leak_results = list(leak_results or [])
        self.progress_results = list(progress_results or [])
        self.student_prompts: list[str] = []
        self.summary_inputs: list[dict[str, str]] = []
        self.progress_inputs: list[dict[str, str]] = []
        self.leak_inputs: list[str] = []

    async def _call_student_prompt(self, prompt: str) -> str:
        self.student_prompts.append(prompt)
        return self.student_outputs.pop(0)

    async def _run_leak_check(
        self, task: str, ground_truth: str, teacher_action: str
    ) -> LeakCheckResult:
        del task, ground_truth
        self.leak_inputs.append(teacher_action)
        if self.leak_results:
            return self.leak_results.pop(0)
        return LeakCheckResult("", False, "ok", None, {})

    async def _run_progress_judge(
        self,
        *,
        task: str,
        ground_truth: str,
        previous_student_answer: str,
        current_student_answer: str,
        tutor_visible_output: str,
    ) -> ProgressJudgment:
        self.progress_inputs.append(
            {
                "task": task,
                "ground_truth": ground_truth,
                "previous_student_answer": previous_student_answer,
                "current_student_answer": current_student_answer,
                "tutor_visible_output": tutor_visible_output,
            }
        )
        if self.progress_results:
            return self.progress_results.pop(0)
        return ProgressJudgment("", "same", "low", "same")

    async def _run_public_summary_update(
        self,
        *,
        old_public_history: PublicHistoryState,
        previous_student_answer: str,
        tutor_visible_output: str,
        current_student_answer: str,
    ) -> PublicHistoryState:
        self.summary_inputs.append(
            {
                "old_public_history": old_public_history.summary,
                "previous_student_answer": previous_student_answer,
                "tutor_visible_output": tutor_visible_output,
                "current_student_answer": current_student_answer,
            }
        )
        return PublicHistoryState(
            summary=(
                f"{old_public_history.summary}\nTutor: {tutor_visible_output}\n"
                f"Student: {current_student_answer}"
            ).strip(),
            turn_count=old_public_history.turn_count + 1,
        )


def _progress(label: str) -> ProgressJudgment:
    return ProgressJudgment("", label, "high", f"{label} feedback")


def test_pre_solved_returns_none_and_skips_tutor_generation():
    workflow = _ScriptedTutorWorkflow(student_outputs=["The answer is 7."])
    engine = _FakeEngine(workflow.tokenizer, outputs=["should not be used"])

    result = asyncio.run(
        workflow.arun_episode(engine, {"task": "original task", "ground_truth": "7"})
    )

    assert result is None
    assert engine.requests == []
    assert workflow.last_history == []


def test_two_turn_episode_returns_multi_sample_tensor_batch():
    workflow = _ScriptedTutorWorkflow(
        student_outputs=[
            "The answer is 1.",
            "Still working, maybe 2.",
            "The answer is 7.",
        ],
        progress_results=[_progress("improved")],
    )
    engine = _FakeEngine(
        workflow.tokenizer,
        outputs=["First concrete hint", "Second concrete hint"],
    )

    result = asyncio.run(
        workflow.arun_episode(engine, {"task": "original task", "ground_truth": "7"})
    )

    assert result is not None
    assert set(result) == {
        "input_ids",
        "logprobs",
        "loss_mask",
        "versions",
        "attention_mask",
        "rewards",
    }
    assert result["input_ids"].shape[0] == 2
    assert result["attention_mask"].shape == result["input_ids"].shape
    assert result["rewards"].tolist() == pytest.approx([0.3, 1.0])
    assert result["input_ids"].shape[-1] >= max(
        len(request.input_ids) for request in engine.requests
    )


def test_leak_turn_is_negative_sample_and_not_public_history():
    workflow = _ScriptedTutorWorkflow(
        student_outputs=[
            "The answer is 1.",
            "The answer is 7.",
        ],
        leak_results=[
            LeakCheckResult("", True, "leaked answer", None, {}),
            LeakCheckResult("", False, "ok", None, {}),
        ],
    )
    engine = _FakeEngine(
        workflow.tokenizer,
        outputs=["Leaked final answer", "Safe hint"],
    )

    result = asyncio.run(
        workflow.arun_episode(engine, {"task": "original task", "ground_truth": "7"})
    )

    assert result is not None
    assert result["rewards"].tolist() == pytest.approx([-1.0, 1.0])
    assert len(workflow.student_prompts) == 2  # initial attempt + non-leaked turn
    assert workflow.summary_inputs
    assert "Leaked final answer" not in workflow.summary_inputs[0]["old_public_history"]
    assert "Leaked final answer" not in workflow.summary_inputs[0]["tutor_visible_output"]


def test_reasoning_kept_in_training_but_stripped_from_contexts():
    workflow = _ScriptedTutorWorkflow(
        student_outputs=[
            "<think>student private</think>The answer is 1.",
            "The answer is 7.",
        ],
    )
    raw_tutor = "<think>private tutor reasoning</think>Visible hint"
    engine = _FakeEngine(workflow.tokenizer, outputs=[raw_tutor])

    result = asyncio.run(
        workflow.arun_episode(engine, {"task": "original task", "ground_truth": "7"})
    )

    assert result is not None
    decoded_training_text = workflow.tokenizer.decode(
        result["input_ids"][0][result["attention_mask"][0].bool()].tolist()
    )
    assert "private tutor reasoning" in decoded_training_text
    assert all("private tutor reasoning" not in prompt for prompt in workflow.student_prompts)
    assert all("private tutor reasoning" not in text for text in workflow.leak_inputs)
    assert all(
        "private tutor reasoning" not in item["tutor_visible_output"]
        for item in workflow.summary_inputs
    )


def test_summary_prompt_excludes_private_fields():
    workflow = TutorAgentWorkflow(
        tokenizer=_FakeTokenizer(),
        gconfig=_FakeGConfig(),
        debug_trace_dir="",
    )

    prompt = workflow._build_summary_prompt(
        old_public_history=PublicHistoryState("visible history"),
        previous_student_answer="student previous",
        tutor_visible_output="visible hint",
        current_student_answer="student current",
    )

    assert "visible history" in prompt
    assert "visible hint" in prompt
    assert "ground truth" not in prompt.lower().replace("do not include ground truth", "")
    assert "judge feedback" not in prompt.lower().replace("judge feedback", "")
    assert "leak feedback" not in prompt.lower().replace("leak feedback", "")


def test_student_prompt_excludes_private_fields():
    workflow = TutorAgentWorkflow(
        tokenizer=_FakeTokenizer(),
        gconfig=_FakeGConfig(),
        debug_trace_dir="",
    )
    prompt = workflow._build_student_prompt_from_state(
        StudentTurnState(
            task="task",
            public_history=PublicHistoryState("public summary"),
            previous_student_output="previous student",
            latest_tutor_visible_output="visible hint",
        )
    )

    assert "public summary" in prompt
    assert "visible hint" in prompt
    assert "ground truth" not in prompt.lower()
    assert "judge" not in prompt.lower()
    assert "leak" not in prompt.lower()


def test_progress_parser_falls_back_to_unknown():
    workflow = TutorAgentWorkflow(
        tokenizer=_FakeTokenizer(),
        gconfig=_FakeGConfig(),
        debug_trace_dir="",
    )

    parsed = workflow._parse_progress_judgment('{"label": "nonsense"}')

    assert parsed.label == "unknown"
    assert parsed.parse_error is not None


def test_response_to_tensordict_core_keys_and_shapes():
    workflow = TutorAgentWorkflow(
        tokenizer=_FakeTokenizer(),
        gconfig=_FakeGConfig(),
        debug_trace_dir="",
    )
    response = tutor_workflow.ModelResponse(
        input_tokens=[1, 2, 3],
        output_tokens=[4, 5],
        output_logprobs=[-0.1, -0.2],
        output_versions=[1, 1],
        tokenizer=workflow.tokenizer,
    )

    sample = workflow._response_to_tensordict(response, reward=0.3)

    assert set(sample) == {
        "input_ids",
        "logprobs",
        "loss_mask",
        "versions",
        "attention_mask",
        "rewards",
    }
    assert sample["input_ids"].shape == (1, 5)
    assert sample["loss_mask"].tolist() == [[0, 0, 0, 1, 1]]
    assert sample["rewards"].tolist() == pytest.approx([0.3])
