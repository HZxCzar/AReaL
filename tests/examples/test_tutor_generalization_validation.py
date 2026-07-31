import json

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from examples.tutor.core.generalization import (
    load_student_generalize_bank,
    validate_student_generalize_dataset,
)
from examples.tutor.core.math_generalization import (
    prepare_math_generalization_sidecar,
)
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


def test_load_generated_generalize_bank_maps_variants_to_levels(tmp_path):
    path = tmp_path / "verified_triples.json"
    path.write_text(
        json.dumps(
            {
                "triples": [
                    {
                        "source_id": "train-7",
                        "original": {
                            "task": "original",
                            "ground_truth": "1",
                        },
                        "variant1": {
                            "task": "same graph",
                            "ground_truth": "2",
                        },
                        "variant2": {
                            "task": "one graph edit",
                            "ground_truth": "3",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    bank = load_student_generalize_bank(str(path), source="generated")

    assert bank == {
        "train-7": {
            "level1": {"task": "same graph", "ground_truth": "2"},
            "level2": {"task": "one graph edit", "ground_truth": "3"},
        }
    }


def test_load_generated_generalize_bank_rejects_incomplete_triple(tmp_path):
    path = tmp_path / "verified_triples.json"
    path.write_text(
        json.dumps(
            {
                "triples": [
                    {
                        "source_id": "train-7",
                        "variant1": {
                            "task": "same graph",
                            "ground_truth": "2",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing variant2"):
        load_student_generalize_bank(str(path), source="generated")


def test_prepare_math_generalization_sidecar_pairs_fixed_train_samples(tmp_path):
    """Test seeded preprocessing fixes two other train questions per sample."""
    train_rows = [
        {
            "id": f"train-{index}",
            "task": f"train task {index}",
            "ground_truth": str(index),
            "metadata": {},
        }
        for index in range(5)
    ]
    test_rows = [
        {
            "id": "test-0",
            "task": "test task",
            "ground_truth": "5",
            "metadata": {},
        }
    ]
    source = tmp_path / "source"
    DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        }
    ).save_to_disk(str(source))

    first = prepare_math_generalization_sidecar(
        train_dataset_path=source,
        valid_dataset_path=source,
        output_path=tmp_path / "first",
        seed=42,
        sample_count=2,
        show_progress=False,
    )
    second = prepare_math_generalization_sidecar(
        train_dataset_path=source,
        valid_dataset_path=source,
        output_path=tmp_path / "second",
        seed=42,
        sample_count=2,
        show_progress=False,
    )

    first_bank = json.loads(first.sidecar_path.read_text(encoding="utf-8"))
    second_bank = json.loads(second.sidecar_path.read_text(encoding="utf-8"))
    assert first_bank == second_bank
    assert set(first_bank) == {row["id"] for row in [*train_rows, *test_rows]}
    train_ids = {row["id"] for row in train_rows}
    for sample_id, payload in first_bank.items():
        paired_ids = [sample["id"] for sample in payload["samples"]]
        assert len(paired_ids) == 2
        assert len(set(paired_ids)) == 2
        assert set(paired_ids) <= train_ids
        assert sample_id not in paired_ids

    validate_student_generalize_dataset(
        Dataset.from_list(train_rows),
        split_name="train",
        bank=first_bank,
        sample_count=2,
    )


def test_prepare_math_generalization_sidecar_reuses_matching_manifest(tmp_path):
    """Test startup preprocessing reuses fixed pairs with matching settings."""
    rows = [
        {
            "id": f"train-{index}",
            "task": f"task {index}",
            "ground_truth": str(index),
        }
        for index in range(3)
    ]
    source = tmp_path / "source"
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(source))
    output = tmp_path / "output"

    prepare_math_generalization_sidecar(
        train_dataset_path=source,
        valid_dataset_path=None,
        output_path=output,
        seed=7,
        sample_count=2,
        show_progress=False,
    )
    reused = prepare_math_generalization_sidecar(
        train_dataset_path=source,
        valid_dataset_path=None,
        output_path=output,
        seed=7,
        sample_count=2,
        show_progress=False,
    )

    assert reused.reused is True
    assert reused.main_size == 3
    assert reused.train_pool_size == 3


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
