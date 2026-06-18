import pytest

from areal.api.cli_args import TrainEngineConfig


class DummyParam:
    def __init__(self, *, requires_grad: bool):
        self.requires_grad = requires_grad


class DummyModel:
    def __init__(self):
        self.enable_input_require_grads_calls = 0
        self.print_trainable_parameters_calls = 0
        self._named_parameters = []

    def enable_input_require_grads(self):
        self.enable_input_require_grads_calls += 1

    def named_parameters(self):
        return iter(self._named_parameters)

    def print_trainable_parameters(self):
        self.print_trainable_parameters_calls += 1


def make_fsdp_engine(fsdp_module, config: TrainEngineConfig):
    engine = fsdp_module.FSDPEngine.__new__(fsdp_module.FSDPEngine)
    engine.config = config
    engine.model = DummyModel()
    engine.rank = 0
    return engine


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


def test_fsdp_apply_peft_wrapper_loads_initial_lora_adapter(monkeypatch):
    import areal.engine.fsdp_engine as fsdp_module

    calls = []

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, **kwargs):
            calls.append((model, adapter_path, kwargs))
            model._named_parameters = [
                (
                    "base_model.model.layers.0.self_attn.q_proj."
                    "lora_A.default.weight",
                    DummyParam(requires_grad=True),
                ),
                (
                    "base_model.model.layers.0.self_attn.q_proj.weight",
                    DummyParam(requires_grad=False),
                ),
            ]
            return model

    monkeypatch.setattr(fsdp_module, "PeftModel", FakePeftModel)

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
            {"is_trainable": True, "autocast_adapter_dtype": False},
        )
    ]
    assert engine.model.enable_input_require_grads_calls == 1
    assert engine.model.print_trainable_parameters_calls == 1


def test_fsdp_apply_peft_wrapper_requires_trainable_lora_params(monkeypatch):
    import areal.engine.fsdp_engine as fsdp_module

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
