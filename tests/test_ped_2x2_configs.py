"""The two head-to-head configs, checked against each other.

The comparison is only paired if both arms cross the same protocols with the
same numbers, and nothing at runtime can notice when they stop doing so: each
arm reads its own YAML and neither can see the other's. These checks are that
missing link.

Needs the endpoint environment, same as the other config tests:

    set -a && . ./.env && set +a
    PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_ped_2x2_configs.py -q
"""

from __future__ import annotations

import os

import pytest
from omegaconf import OmegaConf

TUTOR_CONFIG_DIR = "examples/tutor/configs/math/0810/4gpu"
PED_CONFIG_DIR = "examples/pedagogical_rl/configs"
TUTOR_ARM = "leak-local-stable-xeval"
TUTOR_CONTROL = "leak-local-stable"
PED_ARM = "qwen3_8b_qwen3_1_7b_math_pass2_baseline_xeval"
PED_CONTROL = "qwen3_8b_qwen3_1_7b_math_pass2_baseline"

pytestmark = pytest.mark.skipif(
    not os.environ.get("TUTOR_QWEN3_8B_BASE_URL"),
    reason="endpoint env not loaded; source .env first",
)


# trial_name is interpolated into rollout, actor, saver, recover, stats_logger
# and debug_trace_dir, so two configs that differ only in their trial name
# differ in six more blocks as well. The comparison fixtures below pin it to one
# value so that cascade cancels and a real difference is the only thing left.
PINNED_TRIAL = "pinned-for-comparison"


def _compose(config_dir: str, config_name: str, *, trial_name: str | None = None):
    from hydra import compose, initialize_config_dir

    overrides = [] if trial_name is None else [f"trial_name={trial_name}"]
    with initialize_config_dir(
        config_dir=os.path.abspath(config_dir), version_base=None
    ):
        return compose(config_name=config_name, overrides=overrides)


@pytest.fixture(scope="module")
def tutor_arm():
    return _compose(TUTOR_CONFIG_DIR, TUTOR_ARM)


@pytest.fixture(scope="module")
def tutor_pinned():
    return _compose(TUTOR_CONFIG_DIR, TUTOR_ARM, trial_name=PINNED_TRIAL)


@pytest.fixture(scope="module")
def tutor_control():
    return _compose(TUTOR_CONFIG_DIR, TUTOR_CONTROL, trial_name=PINNED_TRIAL)


@pytest.fixture(scope="module")
def ped_arm():
    return _compose(PED_CONFIG_DIR, PED_ARM)


@pytest.fixture(scope="module")
def ped_pinned():
    return _compose(PED_CONFIG_DIR, PED_ARM, trial_name=PINNED_TRIAL)


@pytest.fixture(scope="module")
def ped_control():
    return _compose(PED_CONFIG_DIR, PED_CONTROL, trial_name=PINNED_TRIAL)


def test_the_two_cross_eval_blocks_are_identical(tutor_arm, ped_arm):
    """The single invariant the whole table rests on."""

    ours = OmegaConf.to_container(tutor_arm.cross_eval, resolve=True)
    theirs = OmegaConf.to_container(ped_arm.cross_eval, resolve=True)
    assert ours == theirs
    assert ours["enabled"] is True
    assert ours["run_other_protocol"] is True


def test_our_arm_changes_nothing_about_its_own_training(tutor_pinned, tutor_control):
    """Everything except the trial name and the cross must be inherited.

    If the xeval arm trains differently from leak-local-stable then it is not
    the arm being compared, and its own headline series cannot be read against
    the runs already on disk.
    """

    ours = OmegaConf.to_container(tutor_pinned, resolve=True)
    control = OmegaConf.to_container(tutor_control, resolve=True)
    ours.pop("cross_eval", None)
    control.pop("cross_eval", None)
    differing = {
        key
        for key in set(ours) | set(control)
        if ours.get(key) != control.get(key)
    }
    assert differing == set(), differing


def test_their_arm_changes_only_the_eval_cadence(ped_pinned, ped_control):
    """Their training must be their published baseline.

    freq_steps and average_rollouts are allowed: they line the eval sets up with
    ours and touch nothing about the method. Anything else means this stopped
    being PedagogicalRL's baseline.
    """

    ours = OmegaConf.to_container(ped_pinned, resolve=True)
    control = OmegaConf.to_container(ped_control, resolve=True)
    ours.pop("cross_eval", None)
    control.pop("cross_eval", None)
    evaluator_ours = ours.pop("evaluator")
    evaluator_control = control.pop("evaluator")
    differing = {
        key
        for key in set(ours) | set(control)
        if ours.get(key) != control.get(key)
    }
    assert differing == set(), differing
    evaluator_differing = {
        key
        for key in set(evaluator_ours) | set(evaluator_control)
        if evaluator_ours.get(key) != evaluator_control.get(key)
    }
    assert evaluator_differing == {"freq_steps", "average_rollouts"}
    assert evaluator_ours["average_rollouts"] == 1


def test_the_free_chat_budget_matches_our_own_rollout(tutor_arm):
    """The workflow also raises on this, but a config test says so before a
    4-GPU job is submitted."""

    assert tutor_arm.cross_eval.free_chat.budget == tutor_arm.free_chat.budget
    assert tutor_arm.cross_eval.free_chat.budget == tutor_arm.max_turns


def test_the_history_tags_match_our_own_rollout(tutor_arm):
    """The tutor workflow raises on this too; here it is caught before a job is
    submitted. A mismatch would run the ped arm's teacher through our protocol
    under a setting our own teacher never rolls out with."""

    assert (
        tutor_arm.cross_eval.free_chat.teacher_history_tags
        == tutor_arm.teacher_history_tags
    )
    assert tutor_arm.teacher_history_tags == "masked"


def test_the_retest_cell_measures_what_our_arm_is_rewarded_on(tutor_arm):
    assert tutor_arm.cross_eval.retest.replays == tutor_arm.student_generalize.replays


def test_the_interview_cell_measures_what_their_arm_reports(ped_arm):
    assert (
        ped_arm.cross_eval.interview.attempts
        == ped_arm.generation.number_student_attempts
    )


def test_the_classroom_protocol_matches_the_one_they_train_with(ped_arm):
    """The crossed classroom dialogue has to be their dialogue, not a variant of
    it, or the classroom column is not measuring their protocol."""

    classroom = ped_arm.cross_eval.classroom
    generation = ped_arm.generation
    assert classroom.max_teacher_turns == generation.max_teacher_turns
    assert classroom.max_tokens_in_conversation == generation.max_tokens_in_conversation
    assert (
        classroom.max_tokens_per_student_turn == generation.max_tokens_per_student_turn
    )
    assert (
        classroom.max_tokens_per_student_attempt
        == generation.max_tokens_per_student_attempt
    )
    assert classroom.include_thinking == generation.use_thinking


def test_both_arms_evaluate_on_the_same_steps_and_the_same_data(tutor_arm, ped_arm):
    assert tutor_arm.evaluator.freq_steps == ped_arm.evaluator.freq_steps
    assert tutor_arm.evaluator.average_rollouts == ped_arm.evaluator.average_rollouts
    # One outer step is batch_size problems x n_samples rollouts on both arms, so
    # the W&B x-axis is comparable without rescaling.
    assert tutor_arm.train_dataset.batch_size == ped_arm.train_dataset.batch_size
    assert tutor_arm.gconfig.n_samples == ped_arm.gconfig.n_samples
    # And the same problems.
    assert (
        tutor_arm.valid_dataset.path.split("examples/")[-1].split("/", 1)[-1]
        == ped_arm.valid_dataset.path.split("examples/")[-1].split("/", 1)[-1]
    )


def test_the_two_arms_fit_on_one_node_together(tutor_arm, ped_arm):
    assert tutor_arm.cluster.n_gpus_per_node + ped_arm.cluster.n_gpus_per_node <= 8


def test_each_arm_trains_its_own_way(tutor_arm, ped_arm):
    """The premise of the comparison, asserted so a later edit cannot quietly
    converge the two training setups and make the result meaningless."""

    # Ours: leak stays on the turn that leaked, a malformed turn ends the
    # episode, free chat scored by the delayed solo re-test.
    assert tutor_arm.free_chat.enabled is True
    assert tutor_arm.leak_handling_mode == "terminate"
    assert list(tutor_arm.reward.turn_local_components) == ["leak"]
    assert tutor_arm.student_generalize.retest_reward > 0

    # Theirs: their whole-dialogue leak gate, no teacher pre-solve.
    assert ped_arm.generation.leak_judge_mode == "pedagogical_rl"
    assert ped_arm.teacher_pre.enabled is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
