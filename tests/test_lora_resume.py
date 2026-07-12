import pytest
from torch import nn

from areal.api.cli_args import TrainEngineConfig


class DummyParam:
    def __init__(self, *, requires_grad: bool):
        self.requires_grad = requires_grad


class DummyModel:
    def __init__(self):
        self.enable_input_require_grads_calls = 0
        self.print_trainable_parameters_calls = 0
        self._named_parameters = []
        self._named_modules = [
            ("", self),
            ("model.layers.0.q_proj", nn.Linear(2, 2)),
            ("model.layers.0.v_proj", nn.Linear(2, 2)),
        ]

    def enable_input_require_grads(self):
        self.enable_input_require_grads_calls += 1

    def named_parameters(self):
        return iter(self._named_parameters)

    def named_modules(self):
        return iter(self._named_modules)

    def print_trainable_parameters(self):
        self.print_trainable_parameters_calls += 1


def make_fsdp_engine(fsdp_module, config: TrainEngineConfig):
    engine = fsdp_module.FSDPEngine.__new__(fsdp_module.FSDPEngine)
    engine.config = config
    engine.model = DummyModel()
    engine.rank = 0
    return engine


def make_adapter_config(fsdp_module, **overrides):
    kwargs = {
        "task_type": fsdp_module.TaskType.CAUSAL_LM,
        "r": 32,
        "lora_alpha": 16,
        "target_modules": "all-linear",
        "bias": "none",
    }
    kwargs.update(overrides)
    return fsdp_module.LoraConfig(**kwargs)


def test_train_engine_config_rejects_init_lora_path_without_lora():
    with pytest.raises(ValueError, match="init_lora_path requires use_lora=True"):
        TrainEngineConfig(
            backend="fsdp:d1",
            experiment_name="test-experiment",
            trial_name="trial0",
            path="test-model",
            init_lora_path="/tmp/adapter",
            use_lora=False,
        )


def test_train_engine_config_rejects_init_lora_path_with_megatron():
    """Megatron must fail instead of silently ignoring an initial adapter."""
    with pytest.raises(
        ValueError,
        match="init_lora_path is not supported by the Megatron backend",
    ):
        TrainEngineConfig(
            backend="megatron:d1",
            experiment_name="test-experiment",
            trial_name="trial0",
            path="test-model",
            init_lora_path="/tmp/adapter",
            use_lora=True,
        )


def test_fsdp_apply_peft_wrapper_loads_initial_lora_adapter(monkeypatch):
    import areal.engine.fsdp_engine as fsdp_module

    calls = []
    adapter_config = make_adapter_config(fsdp_module)

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, **kwargs):
            calls.append((model, adapter_path, kwargs))
            model._named_parameters = [
                (
                    "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
                    DummyParam(requires_grad=True),
                ),
                (
                    "base_model.model.layers.0.self_attn.q_proj.weight",
                    DummyParam(requires_grad=False),
                ),
            ]
            return model

    monkeypatch.setattr(fsdp_module, "PeftModel", FakePeftModel)
    monkeypatch.setattr(
        fsdp_module.PeftConfig,
        "from_pretrained",
        staticmethod(lambda _path: adapter_config),
    )

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
        init_lora_path="/tmp/adapter",
    )
    engine = make_fsdp_engine(fsdp_module, config)

    engine._apply_peft_wrapper()

    assert calls == [
        (
            engine.model,
            "/tmp/adapter",
            {
                "config": adapter_config,
                "is_trainable": True,
                "autocast_adapter_dtype": False,
            },
        )
    ]
    assert engine.model.enable_input_require_grads_calls == 1
    assert engine.model.print_trainable_parameters_calls == 1


def test_fsdp_apply_peft_wrapper_requires_trainable_lora_params(monkeypatch):
    import areal.engine.fsdp_engine as fsdp_module

    adapter_config = make_adapter_config(fsdp_module)

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, *_args, **_kwargs):
            model._named_parameters = [
                (
                    "base_model.model.layers.0.self_attn.q_proj.weight",
                    DummyParam(requires_grad=False),
                ),
            ]
            return model

    monkeypatch.setattr(fsdp_module, "PeftModel", FakePeftModel)
    monkeypatch.setattr(
        fsdp_module.PeftConfig,
        "from_pretrained",
        staticmethod(lambda _path: adapter_config),
    )

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
        init_lora_path="/tmp/adapter",
    )
    engine = make_fsdp_engine(fsdp_module, config)

    with pytest.raises(RuntimeError, match="No trainable LoRA parameters"):
        engine._apply_peft_wrapper()


@pytest.mark.parametrize(
    ("adapter_overrides", "field_name"),
    [
        ({"r": 8}, "r"),
        ({"lora_alpha": 32}, "lora_alpha"),
        ({"lora_dropout": 0.1}, "lora_dropout"),
        ({"use_rslora": True}, "use_rslora"),
    ],
)
def test_fsdp_rejects_initial_lora_config_mismatch_before_model_mutation(
    monkeypatch, adapter_overrides, field_name
):
    """Adapter settings must match actor settings before PEFT mutates the model."""
    import areal.engine.fsdp_engine as fsdp_module

    adapter_config = make_adapter_config(fsdp_module, **adapter_overrides)
    monkeypatch.setattr(
        fsdp_module.PeftConfig,
        "from_pretrained",
        staticmethod(lambda _path: adapter_config),
    )

    class FailPeftModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise AssertionError("mismatched adapter must not be loaded")

    monkeypatch.setattr(fsdp_module, "PeftModel", FailPeftModel)
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
        init_lora_path="/tmp/adapter",
    )
    engine = make_fsdp_engine(fsdp_module, config)

    with pytest.raises(ValueError, match=field_name):
        engine._apply_peft_wrapper()

    assert engine.model.enable_input_require_grads_calls == 0


def test_fsdp_initial_lora_all_linear_matches_expanded_checkpoint_targets(monkeypatch):
    """PEFT's expanded target set is equivalent to actor all-linear shorthand."""
    import areal.engine.fsdp_engine as fsdp_module

    adapter_config = make_adapter_config(
        fsdp_module,
        target_modules={"v_proj", "q_proj"},
    )
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
        init_lora_path="/tmp/adapter",
    )
    engine = make_fsdp_engine(fsdp_module, config)

    engine._validate_init_lora_config(adapter_config)


def test_fsdp_rejects_initial_lora_target_module_mismatch():
    """Checkpoint and actor must select the same base-model modules."""
    import areal.engine.fsdp_engine as fsdp_module

    adapter_config = make_adapter_config(
        fsdp_module,
        target_modules={"q_proj"},
    )
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
        init_lora_path="/tmp/adapter",
    )
    engine = make_fsdp_engine(fsdp_module, config)

    with pytest.raises(ValueError, match="target_modules"):
        engine._validate_init_lora_config(adapter_config)


def test_fsdp_lora_dcp_metadata_round_trip(tmp_path):
    import areal.engine.fsdp_engine as fsdp_module

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
    )
    engine = make_fsdp_engine(fsdp_module, config)
    engine._lora_config_signature = {
        "config": {"r": 32, "target_parameters": []},
        "resolved_target_modules": ["model.layers.0.q_proj"],
    }
    metadata = engine._current_lora_dcp_metadata()

    engine._write_lora_dcp_metadata(str(tmp_path), metadata)

    assert engine._read_lora_dcp_metadata(str(tmp_path)) == metadata


def test_fsdp_lora_config_normalization_preserves_ordered_lists():
    import areal.engine.fsdp_engine as fsdp_module

    normalize = fsdp_module.FSDPEngine._normalize_lora_config_value

    assert normalize([1, 2]) == [1, 2]
    assert normalize([2, 1]) == [2, 1]
    assert normalize({1, 2}) == [1, 2]


def test_fsdp_lora_resume_rejects_config_mismatch_before_loading_weights(
    monkeypatch,
):
    import areal.engine.fsdp_engine as fsdp_module

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
    )
    engine = make_fsdp_engine(fsdp_module, config)
    engine._lora_config_signature = {
        "config": {"r": 32},
        "resolved_target_modules": ["model.layers.0.q_proj"],
    }
    checkpoint_metadata = {
        "schema_version": fsdp_module._LORA_DCP_METADATA_SCHEMA_VERSION,
        "lora_signature": {
            "config": {"r": 8},
            "resolved_target_modules": ["model.layers.0.q_proj"],
        },
    }
    monkeypatch.setattr(
        engine,
        "_read_lora_dcp_metadata_distributed",
        lambda _path: checkpoint_metadata,
    )
    load_called = False

    def fail_if_load_called(*_args, **_kwargs):
        nonlocal load_called
        load_called = True

    monkeypatch.setattr(fsdp_module.dcp, "load", fail_if_load_called)

    with pytest.raises(ValueError, match="LoRA config mismatch"):
        engine._load_from_dcp("/tmp/checkpoint", with_optim=False)

    assert load_called is False


def test_fsdp_lora_resume_rejects_checkpoint_without_config_metadata(tmp_path):
    import areal.engine.fsdp_engine as fsdp_module

    with pytest.raises(ValueError, match="missing areal_lora_config.json"):
        fsdp_module.FSDPEngine._read_lora_dcp_metadata(str(tmp_path))
