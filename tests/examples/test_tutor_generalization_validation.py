import pytest

from examples.tutor.core.generalization import validate_student_generalize_dataset


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
