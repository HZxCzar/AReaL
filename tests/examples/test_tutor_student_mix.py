from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from examples.tutor import workflow as tutor_workflow
from examples.tutor.configs import (
    TUTOR_EVAL_STUDENT_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD,
    TutorConfig,
    TutorEvaluatorConfig,
    TutorStudentModelConfig,
)
from examples.tutor.train import (
    EvalStudentPrompt,
    _expand_eval_dataset_for_student_prompts,
    _expand_eval_dataset_for_students,
    _resolve_eval_student_names,
)


@pytest.fixture(autouse=True)
def _disable_real_openai_clients(monkeypatch):
    """Keep student-mix unit tests independent of network and proxy settings."""

    def build_caller(config):
        return SimpleNamespace(config=config, request_config={})

    monkeypatch.setattr(tutor_workflow, "AsyncLLMCaller", build_caller)


def _student(name: str, *, weight: float = 1.0) -> TutorStudentModelConfig:
    return TutorStudentModelConfig(
        name=name,
        base_url="http://student.invalid/v1",
        model=name,
        weight=weight,
    )


def _student_dict(name: str, *, weight: float = 1.0) -> dict:
    return vars(_student(name, weight=weight))


def test_tutor_config_accepts_arbitrary_student_model_count():
    """Test TutorConfig accepts an extensible pool rather than two fixed models."""
    # Arrange / Act
    config = TutorConfig(
        dataset_type="math",
        student_models=[
            _student("qwen3-4b", weight=0.25),
            _student("qwen3-8b", weight=0.5),
            _student("future-student", weight=0.25),
        ],
    )

    # Assert
    assert [student.name for student in config.student_models] == [
        "qwen3-4b",
        "qwen3-8b",
        "future-student",
    ]


def test_mixed_student_example_yaml_loads_typed_student_configs(monkeypatch):
    """Test the checked-in mixed experiment resolves into typed student configs."""
    # Arrange
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path(
        "examples/tutor/configs/math/july/"
        "baseline-overfit-1-generalize-001020-lora-batch128-rebn-nomean-5-"
        "mixed-students.yaml"
    )

    # Act
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )

    # Assert
    assert isinstance(config, TutorConfig)
    assert [student.name for student in config.student_models] == [
        "qwen3-4b",
        "qwen3-8b",
    ]
    assert [student.max_tokens for student in config.student_models] == [2048, 1024]
    expected_base_url = (
        "https://9jgodagb8ogpceemkhc58boma8oddqoa.openapi-qb-ai.sii.edu.cn/v1"
    )
    assert config.auxiliary_model.base_url == expected_base_url
    assert all(
        student.base_url == expected_base_url for student in config.student_models
    )
    assert config.evaluator.average_rollouts == 3


def test_mixed_17b_7b_yaml_trains_both_and_evaluates_only_17b(monkeypatch):
    """Test the mixed experiment samples both students but pins 1.7B for eval."""
    # Arrange
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path(
        "examples/tutor/configs/math/july/pass@2/"
        "qwen8b-qwen1.7b-math-baseline-aleak-mix1.7_7.yaml"
    )

    # Act
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )

    # Assert
    assert [student.name for student in config.student_models] == [
        "qwen3-1.7b",
        "qwen2.5-7b-instruct",
    ]
    assert [student.weight for student in config.student_models] == [0.5, 0.5]
    qwen25_student = config.student_models[1]
    assert qwen25_student.temperature == 0.7
    assert qwen25_student.top_p == 0.8
    assert qwen25_student.request_params["extra_body"] == {
        "top_k": 20,
        "repetition_penalty": 1.05,
    }
    assert _resolve_eval_student_names(config) == ["qwen3-1.7b"]


def test_eval7b_yaml_trains_17b_and_evaluates_17b_and_7b(monkeypatch):
    """Test zero-weight 7B is excluded from training but included in evaluation."""
    # Arrange
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path(
        "examples/tutor/configs/math/july/pass@2/"
        "qwen8b-qwen1.7b-math-baseline-aleak-eval7b-aleak-pre.yaml"
    )

    # Act
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )

    # Assert
    assert [student.name for student in config.student_models] == [
        "qwen3-1.7b",
        "qwen2.5-7b-instruct",
    ]
    assert [student.weight for student in config.student_models] == [1.0, 0.0]
    assert _resolve_eval_student_names(config) == [
        "qwen3-1.7b",
        "qwen2.5-7b-instruct",
    ]


def test_train7b_yaml_trains_7b_and_evaluates_17b_and_7b(monkeypatch):
    """Test zero-weight 1.7B is excluded from training but included in evaluation."""
    # Arrange
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path(
        "examples/tutor/configs/math/july/pass@2/"
        "qwen8b-qwen2.5-7b-math-baseline-aleak-train7b-eval1.7_7-pre.yaml"
    )

    # Act
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )

    # Assert
    assert [student.name for student in config.student_models] == [
        "qwen3-1.7b",
        "qwen2.5-7b-instruct",
    ]
    assert [student.weight for student in config.student_models] == [0.0, 1.0]
    assert _resolve_eval_student_names(config) == [
        "qwen3-1.7b",
        "qwen2.5-7b-instruct",
    ]


def test_eval_student_subset_defaults_to_all_configured_students():
    """Test existing mixed configs retain exhaustive per-student evaluation."""
    # Arrange
    config = TutorConfig(
        dataset_type="math",
        student_models=[_student("student-a"), _student("student-b")],
    )

    # Act / Assert
    assert _resolve_eval_student_names(config) == ["student-a", "student-b"]


def test_tutor_config_rejects_unknown_eval_student_name():
    """Test evaluation subsets only reference configured student runtimes."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="unknown names"):
        TutorConfig(
            dataset_type="math",
            student_models=[_student("student-a")],
            evaluator=TutorEvaluatorConfig(student_model_names=["student-b"]),
        )


@pytest.mark.parametrize("name", ["contains/slash", "contains space", "_leading"])
def test_student_model_config_rejects_metric_unsafe_name(name):
    """Test student names cannot corrupt dynamic metric namespaces."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="student_models.name"):
        _student(name)


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_url": ""},
        {"model": ""},
        {"base_url": None},
    ],
)
def test_student_model_config_rejects_missing_api_identity(overrides):
    """Test each student supplies a usable API endpoint and routed model name."""
    # Arrange
    kwargs = {
        "name": "qwen3-4b",
        "base_url": "http://student.invalid/v1",
        "model": "qwen3-4b",
    }
    kwargs.update(overrides)

    # Act / Assert
    with pytest.raises(ValueError, match="require non-empty"):
        TutorStudentModelConfig(**kwargs)


def test_tutor_config_rejects_duplicate_student_names():
    """Test each configured student has a unique selection and metric identity."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="names must be unique"):
        TutorConfig(
            dataset_type="math",
            student_models=[_student("qwen3-4b"), _student("qwen3-4b")],
        )


def test_tutor_config_rejects_student_pool_with_zero_total_weight():
    """Test training sampling always has at least one selectable student."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="positive weight"):
        TutorConfig(
            dataset_type="math",
            student_models=[
                _student("qwen3-4b", weight=0.0),
                _student("qwen3-8b", weight=0.0),
            ],
        )


def test_expand_eval_dataset_covers_every_row_for_every_student():
    """Test each configured student receives the complete validation dataset."""
    # Arrange
    dataset = Dataset.from_dict({"id": [1, 2], "task": ["a", "b"]})

    # Act
    expanded = _expand_eval_dataset_for_students(
        dataset, ["qwen3-4b", "qwen3-8b", "future-student"]
    )

    # Assert
    assert len(expanded) == 6
    assert expanded[0:2]["id"] == [1, 2]
    assert expanded[2:4]["id"] == [1, 2]
    assert expanded[4:6]["id"] == [1, 2]
    assert expanded[TUTOR_EVAL_STUDENT_FIELD] == [
        "qwen3-4b",
        "qwen3-4b",
        "qwen3-8b",
        "qwen3-8b",
        "future-student",
        "future-student",
    ]


def test_expand_eval_dataset_covers_every_row_for_every_student_prompt():
    """Test base-prompt rows remain alongside every persona prompt index."""
    dataset = Dataset.from_dict({"id": [1, 2], "task": ["a", "b"]})

    prompts = tuple(
        EvalStudentPrompt(pool="seen", index=index, suffix=f"prompt-{index}")
        for index in range(3)
    )
    expanded = _expand_eval_dataset_for_student_prompts(dataset, prompts)

    assert len(expanded) == 8
    assert expanded["id"] == [1, 2, 1, 2, 1, 2, 1, 2]
    assert expanded[TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD] == [
        None,
        None,
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert expanded[TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD] == [
        None,
        None,
        "seen",
        "seen",
        "seen",
        "seen",
        "seen",
        "seen",
    ]


def test_expand_eval_dataset_covers_student_model_prompt_cross_product():
    """Test model and prompt dimensions form a complete evaluation product."""
    dataset = Dataset.from_dict({"id": [1, 2], "task": ["a", "b"]})
    expanded = _expand_eval_dataset_for_students(dataset, ["student-a", "student-b"])

    prompts = tuple(
        EvalStudentPrompt(pool="seen", index=index, suffix=f"prompt-{index}")
        for index in range(3)
    )
    expanded = _expand_eval_dataset_for_student_prompts(expanded, prompts)

    combinations = set(
        zip(
            expanded["id"],
            expanded[TUTOR_EVAL_STUDENT_FIELD],
            expanded[TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD],
            strict=True,
        )
    )
    expected = {
        (item_id, student_name, prompt_index)
        for item_id in (1, 2)
        for student_name in ("student-a", "student-b")
        for prompt_index in (None, 0, 1, 2)
    }
    assert len(expanded) == 16
    assert combinations == expected


def test_expand_eval_dataset_with_zero_student_prompts_returns_input():
    """Test disabled all-prompt evaluation leaves the dataset untouched."""
    dataset = Dataset.from_dict({"id": [1], "task": ["a"]})

    assert _expand_eval_dataset_for_student_prompts(dataset, ()) is dataset


def test_expand_eval_dataset_rejects_reserved_student_prompt_column():
    """Test user data cannot collide with the forced prompt-index column."""
    dataset = Dataset.from_dict(
        {
            "id": [1],
            TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD: [0],
        }
    )

    with pytest.raises(ValueError, match="reserved column"):
        _expand_eval_dataset_for_student_prompts(
            dataset,
            (EvalStudentPrompt(pool="seen", index=0, suffix="prompt"),),
        )


def test_expand_eval_dataset_separates_seen_and_heldout_prompt_indices():
    """Test equal local indices remain distinct across persona pools."""
    dataset = Dataset.from_dict({"id": [1, 2]})
    prompts = (
        EvalStudentPrompt(pool="seen", index=0, suffix="seen"),
        EvalStudentPrompt(pool="heldout", index=0, suffix="heldout"),
    )

    expanded = _expand_eval_dataset_for_student_prompts(dataset, prompts)

    assert expanded[TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD] == [
        None,
        None,
        "seen",
        "seen",
        "heldout",
        "heldout",
    ]
    assert expanded[TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD] == [
        None,
        None,
        0,
        0,
        0,
        0,
    ]


def test_select_student_uses_weighted_training_sample(monkeypatch):
    """Test training selection uses configured weights once an episode starts."""
    # Arrange
    workflow = tutor_workflow.TutorAgentWorkflow(
        student_models=[
            _student_dict("qwen3-4b", weight=0.25),
            _student_dict("qwen3-8b", weight=0.75),
        ]
    )
    captured = {}

    def choose(population, *, weights, k):
        captured["weights"] = weights
        captured["k"] = k
        return [population[1]]

    monkeypatch.setattr(tutor_workflow.random, "choices", choose)
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: SimpleNamespace(is_eval=False),
    )

    # Act
    selected = workflow._select_student({}, aux_caller=object())

    # Assert
    assert selected.name == "qwen3-8b"
    assert captured == {"weights": [0.25, 0.75], "k": 1}


def test_select_student_honors_forced_eval_student(monkeypatch):
    """Test expanded validation rows bypass random weights for complete coverage."""
    # Arrange
    workflow = tutor_workflow.TutorAgentWorkflow(
        student_models=[
            _student_dict("qwen3-4b", weight=0.0),
            _student_dict("qwen3-8b", weight=1.0),
        ]
    )
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: SimpleNamespace(is_eval=True),
    )

    # Act
    selected = workflow._select_student(
        {TUTOR_EVAL_STUDENT_FIELD: "qwen3-4b"}, aux_caller=object()
    )

    # Assert
    assert selected.name == "qwen3-4b"


def test_student_pool_confidence_caller_follows_each_student_api():
    """Test confidence probes share the selected student's API configuration."""
    # Arrange / Act
    workflow = tutor_workflow.TutorAgentWorkflow(
        aux_mode="self",
        student_generalize_enabled=True,
        student_generalize_confidence_enabled=True,
        student_models=[
            _student_dict("qwen3-4b"),
            _student_dict("qwen3-8b"),
        ],
    )

    # Assert
    for runtime in workflow.student_model_runtimes.values():
        assert runtime.confidence_caller is not None
        assert runtime.confidence_caller.caller is runtime.caller.caller
        assert runtime.confidence_caller.request_config["logprobs"] is True


def test_select_student_without_pool_preserves_legacy_auxiliary_caller(monkeypatch):
    """Test empty student_models keeps the current single-student behavior."""
    # Arrange
    workflow = tutor_workflow.TutorAgentWorkflow(aux_model="legacy-student")
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: SimpleNamespace(is_eval=False),
    )

    # Act
    selected = workflow._select_student({}, aux_caller=workflow.aux_caller)

    # Assert
    assert selected.name == "legacy-student"
    assert selected.caller is workflow.aux_caller


def test_rollout_metrics_include_overall_and_dynamic_per_student_values(monkeypatch):
    """Test adding a student automatically creates conditional success metrics."""
    # Arrange
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_model_runtimes = {
        "qwen3-4b": object(),
        "qwen3-8b": object(),
        "future-student": object(),
    }
    workflow.student_generalize_enabled = False
    captured = {}
    monkeypatch.setattr(
        tutor_workflow, "_safe_scalar", lambda **metrics: captured.update(metrics)
    )
    trace = SimpleNamespace(
        turn_idx=2,
        judge_correct=True,
        invalid_due_to_leak=False,
        reward_components={},
    )

    # Act
    workflow._log_rollout_stats(
        total_reward=1.5,
        traces=[trace],
        termination_reason="success",
        pre_success=False,
        leak_count=0,
        student_name="qwen3-8b",
        student_call_failed=False,
    )

    # Assert
    assert captured["solved"] == 1.0
    assert captured["reward"] == 1.5
    assert captured["student/qwen3-4b/selected"] == 0.0
    assert captured["student/qwen3-8b/selected"] == 1.0
    assert captured["student/future-student/selected"] == 0.0
    assert captured["student/qwen3-8b/solved"] == 1.0
    assert captured["student/qwen3-8b/reward"] == 1.5
    assert captured["student/qwen3-8b/call_failed"] == 0.0
