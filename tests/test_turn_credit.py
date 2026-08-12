"""Per-turn re-test credit: the split must not change the episode total.

The whole point of paying turns individually is credit assignment, not a new
objective. So the invariant that matters is that the marginals telescope back to
retest_reward * (S(T) - S(0)) -- the number the trainer already used -- including
when an intermediate prefix fails to score.
"""

from types import SimpleNamespace

import pytest

from examples.tutor.configs import TutorStudentGeneralizeConfig
from examples.tutor.workflow import TutorAgentWorkflow


def _turns(n):
    return [SimpleNamespace(turn_idx=i + 1) for i in range(n)]


def _credits(prefix_scores, final, baseline, scale=1.0, n=5):
    return TutorAgentWorkflow._turn_credits_from_prefixes(
        SimpleNamespace(),
        turn_artifacts=_turns(n),
        prefix_scores=prefix_scores,
        final_fraction=final,
        baseline=baseline,
        scale=scale,
    )


def test_marginals_telescope_to_the_episode_reward():
    credits = _credits({1: 0.25, 2: 0.25, 3: 0.5, 4: 0.75}, final=1.0, baseline=0.0)
    assert credits == {1: 0.25, 2: 0.0, 3: 0.25, 4: 0.25, 5: 0.25}
    assert sum(credits.values()) == pytest.approx(1.0 - 0.0)


def test_total_matches_even_when_a_prefix_fails_to_score():
    # Turn 3's prefix re-test came back empty. Its share is carried into turn 4
    # rather than dropped, so the episode total is still S(T) - S(0).
    credits = _credits({1: 0.25, 2: 0.5, 4: 0.75}, final=1.0, baseline=0.0)
    assert 3 not in credits
    assert sum(credits.values()) == pytest.approx(1.0)


def test_a_turn_that_sets_the_student_back_is_charged():
    credits = _credits({1: 0.75, 2: 0.25, 3: 0.25, 4: 0.5}, final=0.5, baseline=0.5)
    assert credits[1] == pytest.approx(0.25)
    assert credits[2] == pytest.approx(-0.5)
    assert sum(credits.values()) == pytest.approx(0.0)


def test_baseline_is_the_zeroth_prefix():
    # With no teaching effect at all every marginal is zero, not just the total.
    credits = _credits({1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}, final=0.4, baseline=0.4)
    assert all(v == pytest.approx(0.0) for v in credits.values())


def test_scale_is_retest_reward():
    scaled = _credits({1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}, final=1.0, baseline=0.0, scale=2.0)
    assert sum(scaled.values()) == pytest.approx(2.0)


def test_single_turn_episode_pays_the_whole_gain_to_that_turn():
    credits = _credits({}, final=0.75, baseline=0.25, n=1)
    assert credits == {1: pytest.approx(0.5)}


def test_config_defaults_off():
    cfg = TutorStudentGeneralizeConfig()
    assert cfg.turn_credit is False
    assert cfg.turn_credit_replays == 0
