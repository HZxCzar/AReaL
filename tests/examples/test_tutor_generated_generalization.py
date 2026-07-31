import asyncio
import json

import pytest
from datasets import Dataset

from examples.tutor import train as tutor_train
from examples.tutor.configs import TutorStudentGeneralizeConfig
from examples.tutor.workflow import TutorAgentWorkflow
from areal.infra.data_service import RDataset


def test_generated_source_config_requires_path_when_enabled():
    with pytest.raises(ValueError, match="requires student_generalize.path"):
        TutorStudentGeneralizeConfig(
            enabled=True,
            source="generated",
        )

    config = TutorStudentGeneralizeConfig(
        enabled=True,
        source="generated",
        path="verified_triples.json",
    )
    assert config.source == "generated"


def test_workflow_loads_generated_triples(tmp_path):
    path = tmp_path / "verified_triples.json"
    path.write_text(
        json.dumps(
            {
                "triples": [
                    {
                        "source_id": "train-1",
                        "variant1": {"task": "v1", "ground_truth": "11"},
                        "variant2": {"task": "v2", "ground_truth": "12"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    bank = TutorAgentWorkflow._load_student_generalize_bank(
        str(path),
        source="generated",
    )

    assert bank["train-1"]["level1"]["task"] == "v1"
    assert bank["train-1"]["level2"]["task"] == "v2"


def test_generated_source_filters_unpaired_rows_before_rollout():
    dataset = Dataset.from_list(
        [
            {"id": "train-1", "task": "one", "ground_truth": "1"},
            {"id": "train-2", "task": "two", "ground_truth": "2"},
            {"id": "train-3", "task": "three", "ground_truth": "3"},
        ]
    )
    bank = {
        "train-1": {
            "level1": {"task": "one-v1", "ground_truth": "11"},
            "level2": {"task": "one-v2", "ground_truth": "12"},
        },
        "train-3": {
            "level1": {"task": "three-v1", "ground_truth": "31"},
            "level2": {"task": "three-v2", "ground_truth": "32"},
        },
    }

    filtered = tutor_train._filter_generated_generalization_dataset(
        dataset,
        bank=bank,
        split_name="train",
    )

    assert filtered["id"] == ["train-1", "train-3"]


def test_generated_source_defers_filter_for_unconnected_remote_dataset():
    dataset = RDataset(path="unused", split="train")

    result = tutor_train._filter_generated_generalization_dataset(
        dataset,
        bank={},
        split_name="train",
    )

    assert result is dataset
    assert dataset.connected is False


def test_workflow_skips_missing_generated_sample_before_model_calls():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.student_generalize_enabled = True
    workflow.student_generalize_source = "generated"
    workflow.student_generalize_bank = {}

    result = asyncio.run(
        workflow._run_episode(
            {
                "id": "train-missing",
                "task": "original",
                "ground_truth": "1",
            },
            external_client=object(),
        )
    )

    assert result is None
    assert workflow.last_history == []
    assert workflow.last_student_generalization_results == []
    assert workflow.last_total_reward == 0.0


def test_generated_source_rejects_split_without_any_matching_variants():
    dataset = Dataset.from_list([{"id": "test-1", "task": "one", "ground_truth": "1"}])

    with pytest.raises(ValueError, match="found no complete generated variants"):
        tutor_train._filter_generated_generalization_dataset(
            dataset,
            bank={},
            split_name="test",
        )
