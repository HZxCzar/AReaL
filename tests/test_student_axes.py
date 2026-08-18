"""The student pool declared as its two axes instead of as their product.

A student is a (behavior, information) pair. Both fields already live on a
student_models entry, so the pool can always be written out by hand; what these
cover is that declaring the axes produces exactly the same pool, because omegaconf
replaces a list on merge and six near-identical entries is how an endpoint change
ends up applied to four of them.
"""

from __future__ import annotations

import pytest

from examples.tutor.configs import (
    TutorStudentAxesConfig,
    TutorStudentMaskConfig,
    TutorStudentModelConfig,
)


def _template(**kw) -> TutorStudentModelConfig:
    base = dict(
        name="qwen3-1.7b",
        base_url="http://endpoint",
        model="qwen3-1.7b",
        weight=1.0,
    )
    base.update(kw)
    return TutorStudentModelConfig(**base)


def test_the_product_is_the_pool():
    axes = TutorStudentAxesConfig(
        behaviors=["text", "code"],
        informations={
            "original": TutorStudentMaskConfig(),
            "teacher_fade": TutorStudentMaskConfig(mode="teacher_fade"),
            "long_drop": TutorStudentMaskConfig(mode="long_drop", long_drop_words=40),
        },
        template=_template(),
    )
    pool = axes.expand()
    assert len(pool) == 6
    assert [s.name for s in pool] == [
        "qwen3-1.7b-text-original",
        "qwen3-1.7b-text-teacher_fade",
        "qwen3-1.7b-text-long_drop",
        "qwen3-1.7b-code-original",
        "qwen3-1.7b-code-teacher_fade",
        "qwen3-1.7b-code-long_drop",
    ]
    # The axes set these two and nothing else varies.
    assert [s.mode for s in pool] == ["text"] * 3 + ["code"] * 3
    assert [s.mask.mode for s in pool] == ["full", "teacher_fade", "long_drop"] * 2
    assert {s.base_url for s in pool} == {"http://endpoint"}
    assert pool[2].mask.long_drop_words == 40


def test_the_template_weight_is_the_pools_total():
    """One block at weight 1.0 draws as often as one student at weight 1.0,
    whatever it expands to -- otherwise adding an axis would quietly change how
    often this endpoint is picked against the others."""

    axes = TutorStudentAxesConfig(
        behaviors=["text", "code"],
        informations={"original": TutorStudentMaskConfig()},
        template=_template(weight=1.0),
    )
    pool = axes.expand()
    assert sum(s.weight for s in pool) == pytest.approx(1.0)
    assert {s.weight for s in pool} == {0.5}


def test_a_single_axis_degrades_to_plain_expansion():
    axes = TutorStudentAxesConfig(template=_template())
    pool = axes.expand()
    assert len(pool) == 1
    assert pool[0].name == "qwen3-1.7b-text-original"
    assert pool[0].mode == "text"
    assert pool[0].mask.mode == "full"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"behaviors": []}, "behaviors must not be empty"),
        ({"behaviors": ["text", "text"]}, "duplicates"),
        ({"informations": {}}, "informations must not be empty"),
    ],
)
def test_an_empty_or_repeated_axis_is_refused(kwargs, message):
    axes = TutorStudentAxesConfig(template=_template(), **kwargs)
    with pytest.raises(ValueError, match=message):
        axes.expand()


def test_the_stem_has_to_be_named():
    """Names are what the evaluator keys its per-student expansion on, so an
    unnamed stem would produce a pool it cannot report separately.

    Refused at TEMPLATE construction, which is earlier than expand() and applies
    to a written-out entry too; expand() keeps its own check only for the path
    where a template arrives already built."""

    with pytest.raises(ValueError, match="non-empty values for: name"):
        _template(name="")


def test_a_mask_given_as_a_plain_dict_is_accepted():
    """omegaconf hands nested blocks over as mappings, not as the dataclass."""

    axes = TutorStudentAxesConfig(
        behaviors=["text"],
        informations={"fade": {"mode": "student_fade", "keep_recent": 1}},
        template=_template(),
    )
    pool = axes.expand()
    assert pool[0].mask.mode == "student_fade"
    assert pool[0].mask.keep_recent == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_expansion_is_idempotent_across_a_round_trip():
    """The config is rebuilt on every worker, so expansion must not re-run.

    THE REGRESSION THIS PINS. The trainer serializes the config and rebuilds it on
    each RPC worker, which runs __post_init__ again. Expansion used to leave the
    axes blocks in place, so the second pass appended the same cells on top of an
    already-expanded student_models and raised "student_models names must be
    unique" -- inside the RPC deserializer, which caught it, fell back to handing
    the worker a plain dict, and died four frames later on
    `'dict' object has no attribute 'seed'`. Nothing in that message points at
    student_axes, and a single in-process load cannot reproduce it, which is why
    this rebuilds the config the way the transport does.

    Consuming the blocks is what makes the second pass a no-op.
    """

    from omegaconf import OmegaConf

    from areal.api.cli_args import load_expr_config, to_structured_cfg
    from examples.tutor.configs import TutorConfig

    arm = "examples/tutor/configs/math/0818/4gpu/full-students.yaml"
    config, _ = load_expr_config(["--config", arm], TutorConfig)
    assert len(config.student_models) == 8
    # The blocks are consumed, so a second __post_init__ has nothing to expand.
    assert list(config.student_axes) == []

    def rebuild(cfg):
        container = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
        return OmegaConf.to_object(
            to_structured_cfg(OmegaConf.create(container), config_cls=TutorConfig)
        )

    names = [s.name for s in config.student_models]
    once = rebuild(config)
    twice = rebuild(once)
    assert [s.name for s in once.student_models] == names
    assert [s.name for s in twice.student_models] == names
    assert len(names) == len(set(names))
    # And the pool still means what it did: eight cells over the two axes.
    assert {s.mode for s in twice.student_models} == {"text", "code"}
    assert {s.mask.mode for s in twice.student_models} == {
        "full",
        "student_fade",
        "teacher_fade",
        "long_drop",
    }
