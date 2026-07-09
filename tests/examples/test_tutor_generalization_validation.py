import json

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from examples.tutor.core.generalization import validate_student_generalize_dataset
from examples.tutor.core.polaris_generalization import (
    prepare_polaris_generalization_dataset,
)


def test_validate_student_generalize_dataset_accepts_sidecar_and_metadata():
    dataset = [
        {"id": "a", "metadata": {}},
        {
            "id": "b",
            "metadata": {
                "student_generalize": {
                    "level1": {"task": "m1", "ground_truth": "1"},
                    "level2": {"task": "m2", "ground_truth": "2"},
                }
            },
        },
    ]
    bank = {
        "a": {
            "level1": {"task": "s1", "ground_truth": "1"},
            "level2": {"task": "s2", "ground_truth": "2"},
        }
    }

    validate_student_generalize_dataset(dataset, split_name="train", bank=bank)


def test_validate_student_generalize_dataset_rejects_missing_cases():
    dataset = [
        {
            "id": "a",
            "metadata": {
                "student_generalize": {
                    "level1": {"task": "m1", "ground_truth": "1"},
                }
            },
        },
        {"id": "b", "metadata": {}},
    ]

    with pytest.raises(ValueError, match="train split has 2/2 samples"):
        validate_student_generalize_dataset(dataset, split_name="train", bank={})


def test_prepare_polaris_generalization_dataset_matches_by_difficulty(tmp_path):
    rows = []
    for difficulty in range(4):
        for index in range(6):
            rows.append(
                {
                    "id": f"d{difficulty}-{index}",
                    "task": f"task {difficulty}-{index}",
                    "ground_truth": str(index),
                    "metadata": {
                        "source": "polaris",
                        "difficulty": f"{difficulty}/8",
                    },
                }
            )
    source = tmp_path / "source"
    output = tmp_path / "derived"
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(source))

    result = prepare_polaris_generalization_dataset(
        source_path=source,
        output_path=output,
        seed=7,
        train_ratio=0.75,
        generalize_ratio=0.5,
        reuse_generalize_tasks=True,
        overwrite=False,
        show_progress=False,
    )

    derived = load_from_disk(str(result.dataset_path))
    bank = result.sidecar_path.read_text(encoding="utf-8")
    assert set(derived.keys()) == {
        "train",
        "test",
        "train_generalize_pool",
        "test_generalize_pool",
    }
    assert len(bank) > 0

    sidecar = json.loads(bank)
    all_main_rows = list(derived["train"]) + list(derived["test"])
    rows_by_id = {row["id"]: row for row in all_main_rows}
    assert set(sidecar) == set(rows_by_id)
    validate_student_generalize_dataset(
        derived["train"],
        split_name="train",
        bank=sidecar,
    )
    validate_student_generalize_dataset(
        derived["test"],
        split_name="test",
        bank=sidecar,
    )

    for sample_id, cases in sidecar.items():
        difficulty = int(rows_by_id[sample_id]["metadata"]["difficulty"].split("/")[0])
        level1_difficulty = int(cases["level1"]["difficulty"])
        level2_difficulty = int(cases["level2"]["difficulty"])
        if difficulty == 0:
            assert level1_difficulty == difficulty
        else:
            assert difficulty - 2 <= level1_difficulty <= difficulty - 1
        if difficulty == 3:
            assert level2_difficulty == difficulty
        else:
            assert difficulty + 1 <= level2_difficulty <= difficulty + 2
