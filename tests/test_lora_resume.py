from types import SimpleNamespace

import pytest
import torch
from torch import nn

from areal.api.cli_args import FSDPEngineConfig, OptimizerConfig, TrainEngineConfig
from areal.api.io_struct import FinetuneSpec
from areal.engine.fsdp_utils.checkpoint import DCPState


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
    engine._separate_lora_enabled = False
    engine._active_lora_adapter = "default"
    engine._adapter_optimizers = {}
    engine._adapter_lr_schedulers = {}
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
    ("config_overrides", "message"),
    [
        ({"use_lora": False}, "requires FSDP LoRA training"),
        ({"use_lora": True, "optimizer": None}, "requires an optimizer"),
        (
            {
                "use_lora": True,
                "optimizer": OptimizerConfig(),
                "fsdp": FSDPEngineConfig(per_layer_optim_step=True),
            },
            "does not support per-layer optimizer stepping",
        ),
        (
            {
                "use_lora": True,
                "optimizer": OptimizerConfig(),
                "enable_tree_training": True,
            },
            "does not support tree training",
        ),
    ],
)
def test_fsdp_separate_lora_validation_rejects_incompatible_config(
    config_overrides, message
):
    """Dual-LoRA requirements are enforced once at the FSDP engine boundary."""

    import areal.engine.fsdp_engine as fsdp_module

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        **config_overrides,
    )
    engine = make_fsdp_engine(fsdp_module, config)
    engine._separate_lora_enabled = True

    with pytest.raises(ValueError, match=message):
        engine._validate_separate_lora_config()


def test_fsdp_apply_peft_wrapper_adds_fresh_world_model_adapter(monkeypatch):
    """Dual mode creates both adapters before FSDP and leaves the policy active."""

    import areal.engine.fsdp_engine as fsdp_module

    class DualPeftModel(DummyModel):
        def __init__(self):
            super().__init__()
            self.added_adapters = []
            self.active_adapter = None
            self.policy = nn.Parameter(torch.zeros(1))
            self.world_model = nn.Parameter(torch.zeros(1))
            self._named_parameters = [
                ("layer.lora_A.default.weight", self.policy),
                ("layer.lora_A.world_model.weight", self.world_model),
            ]

        def add_adapter(self, adapter_name, adapter_config):
            self.added_adapters.append((adapter_name, adapter_config))

        def set_adapter(self, adapter_name):
            self.active_adapter = adapter_name
            self.policy.requires_grad_(adapter_name == "default")
            self.world_model.requires_grad_(adapter_name == "world_model")

    model = DualPeftModel()
    monkeypatch.setattr(
        fsdp_module,
        "get_peft_model",
        lambda *_args, **_kwargs: model,
    )
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
    )
    engine = make_fsdp_engine(fsdp_module, config)
    engine._separate_lora_enabled = True

    engine._apply_peft_wrapper()

    assert [name for name, _ in model.added_adapters] == ["world_model"]
    assert model.active_adapter == "default"
    assert model.policy.requires_grad
    assert model.world_model.requires_grad


def test_merge_saved_lora_adapter_state_restores_named_adapter():
    import areal.engine.fsdp_engine as fsdp_module

    policy = torch.zeros(2, 2)
    world_model = torch.zeros(2, 2)
    full_state = {
        "base_model.layer.lora_A.default.weight": policy,
        "base_model.layer.lora_A.world_model.weight": world_model,
        "base_model.layer.weight": torch.ones(2, 2),
    }
    saved_world_model = {
        "base_model.layer.lora_A.weight": torch.full((2, 2), 3.0),
    }

    fsdp_module._merge_saved_lora_adapter_state(
        full_state,
        saved_world_model,
        "world_model",
    )

    torch.testing.assert_close(
        full_state["base_model.layer.lora_A.world_model.weight"],
        torch.full((2, 2), 3.0),
    )
    torch.testing.assert_close(
        full_state["base_model.layer.lora_A.default.weight"],
        policy,
    )


def test_merge_saved_lora_adapter_state_rejects_mismatch():
    import areal.engine.fsdp_engine as fsdp_module

    with pytest.raises(ValueError, match="does not match the current model"):
        fsdp_module._merge_saved_lora_adapter_state(
            {"base_model.layer.lora_A.world_model.weight": torch.zeros(1)},
            {"base_model.other.lora_A.weight": torch.zeros(1)},
            "world_model",
        )


def test_fsdp_dual_lora_hf_load_resolves_repo_and_restores_both_adapters(
    monkeypatch, tmp_path
):
    """A Hub repo is downloaded once before loading its WM subdirectory."""

    import huggingface_hub

    import areal.engine.fsdp_engine as fsdp_module

    engine = make_fsdp_engine(
        fsdp_module,
        TrainEngineConfig(
            backend="fsdp:d1",
            experiment_name="test-experiment",
            trial_name="trial0",
            path="test-model",
            use_lora=True,
        ),
    )
    engine._separate_lora_enabled = True
    engine._initialized = True
    engine._cpu_group = object()
    engine.cpu_offload = None
    engine.model_config = SimpleNamespace(tie_word_embeddings=False)
    full_state = {
        "layer.lora_A.default.weight": torch.zeros(1),
        "layer.lora_A.world_model.weight": torch.zeros(1),
    }
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / "world_model").mkdir()
    monkeypatch.setattr(fsdp_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        fsdp_module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        fsdp_module, "get_model_state_dict", lambda *_args, **_kwargs: full_state
    )
    snapshot_calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id: snapshot_calls.append(repo_id) or str(downloaded),
    )

    def load_adapter(path):
        value = 2.0 if str(path).endswith("world_model") else 1.0
        return {"layer.lora_A.weight": torch.tensor([value])}

    monkeypatch.setattr(
        fsdp_module, "get_state_dict_from_repo_id_or_path", load_adapter
    )
    loaded = []
    monkeypatch.setattr(
        fsdp_module,
        "fsdp2_load_full_state_dict",
        lambda _model, state, *_args, **_kwargs: loaded.append(state.copy()),
    )

    engine._load_model_from_hf("org/repo")

    assert snapshot_calls == ["org/repo"]
    torch.testing.assert_close(
        loaded[0]["layer.lora_A.default.weight"],
        torch.tensor([1.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        loaded[0]["layer.lora_A.world_model.weight"],
        torch.tensor([2.0]),
        rtol=0,
        atol=0,
    )


def test_fsdp_dual_lora_hf_load_broadcasts_rank_zero_error(monkeypatch, tmp_path):
    """Every rank exits before FSDP loading when rank zero cannot read an adapter."""

    import areal.engine.fsdp_engine as fsdp_module

    engine = make_fsdp_engine(
        fsdp_module,
        TrainEngineConfig(
            backend="fsdp:d1",
            experiment_name="test-experiment",
            trial_name="trial0",
            path="test-model",
            use_lora=True,
        ),
    )
    engine._separate_lora_enabled = True
    engine._initialized = True
    engine._cpu_group = object()
    engine.cpu_offload = None
    engine.model_config = SimpleNamespace(tie_word_embeddings=False)
    monkeypatch.setattr(fsdp_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        fsdp_module, "get_model_state_dict", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        fsdp_module,
        "get_state_dict_from_repo_id_or_path",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing adapter")),
    )
    broadcast_payloads = []
    monkeypatch.setattr(
        fsdp_module.dist,
        "broadcast_object_list",
        lambda payload, **_kwargs: broadcast_payloads.append(payload.copy()),
    )
    monkeypatch.setattr(
        fsdp_module,
        "fsdp2_load_full_state_dict",
        lambda *_args, **_kwargs: pytest.fail("FSDP load must not start after error"),
    )

    with pytest.raises(RuntimeError, match="missing adapter"):
        engine._load_model_from_hf(str(tmp_path))

    assert broadcast_payloads == [["FileNotFoundError: missing adapter"]]


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


def test_fsdp_separate_lora_optimizers_are_disjoint_and_switch_together():
    """Each adapter owns only its parameters and its optimizer/scheduler aliases."""

    import areal.engine.fsdp_engine as fsdp_module

    class DualAdapterModel:
        def __init__(self):
            self.policy = nn.Parameter(torch.tensor([1.0]))
            self.world_model = nn.Parameter(torch.tensor([2.0]))
            self.active = "default"

        def named_parameters(self):
            return iter(
                [
                    ("layer.lora_A.default.weight", self.policy),
                    ("layer.lora_A.world_model.weight", self.world_model),
                ]
            )

        def set_adapter(self, adapter_name):
            self.active = adapter_name
            self.policy.requires_grad_(adapter_name == "default")
            self.world_model.requires_grad_(adapter_name == "world_model")

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
        optimizer=OptimizerConfig(type="sgd", lr=0.1, weight_decay=0.0),
    )
    engine = make_fsdp_engine(fsdp_module, config)
    engine.model = DualAdapterModel()
    engine.optimizer_config = config.optimizer
    engine._separate_lora_enabled = True
    engine.is_vision_model = False
    engine.logger = type(
        "FakeLogger",
        (),
        {"info": lambda *_: None, "warning": lambda *_: None},
    )()

    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1)
    )

    policy_optimizer = engine._adapter_optimizers["default"]
    world_model_optimizer = engine._adapter_optimizers["world_model"]
    assert policy_optimizer.param_groups[0]["params"][0] is engine.model.policy
    assert (
        world_model_optimizer.param_groups[0]["params"][0] is engine.model.world_model
    )
    engine.activate_world_model_adapter()
    assert engine.model.active == "world_model"
    assert engine.optimizer is world_model_optimizer
    assert engine.lr_scheduler is engine._adapter_lr_schedulers["world_model"]
    engine.activate_policy_adapter()
    assert engine.model.active == "default"
    assert engine.optimizer is policy_optimizer


def test_dcp_state_round_trips_two_optimizers_and_schedulers():
    """Dual-LoRA recovery preserves both optimizer and scheduler states."""

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    policy_optimizer = torch.optim.SGD(model[0].parameters(), lr=0.1)
    world_model_optimizer = torch.optim.SGD(model[1].parameters(), lr=0.2)
    schedulers = (
        torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=1, gamma=0.5),
        torch.optim.lr_scheduler.StepLR(world_model_optimizer, step_size=1, gamma=0.25),
    )
    schedulers[0].step()
    schedulers[1].step()
    state = DCPState(
        model,
        (optimizer for optimizer in (policy_optimizer, world_model_optimizer)),
        lr_schedulers=schedulers,
    )
    checkpoint = state.state_dict()

    policy_optimizer.param_groups[0]["lr"] = 9.0
    world_model_optimizer.param_groups[0]["lr"] = 8.0
    schedulers[0].last_epoch = 99
    schedulers[1].last_epoch = 98
    state.load_state_dict(checkpoint)

    assert policy_optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert world_model_optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert schedulers[0].last_epoch == 1
    assert schedulers[1].last_epoch == 1


def test_fsdp_lora_config_normalization_preserves_ordered_lists():
    import areal.engine.fsdp_engine as fsdp_module

    normalize = fsdp_module.FSDPEngine._normalize_lora_config_value

    assert normalize([1, 2]) == [1, 2]
    assert normalize([2, 1]) == [2, 1]
    assert normalize({1, 2}) == [1, 2]


def test_fsdp_dual_lora_rejects_single_adapter_checkpoint_metadata():
    """A legacy single-LoRA checkpoint is not silently converted to dual mode."""

    import areal.engine.fsdp_engine as fsdp_module

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        use_lora=True,
    )
    engine = make_fsdp_engine(fsdp_module, config)
    engine._separate_lora_enabled = True
    engine._lora_config_signature = {
        "config": {"r": 32},
        "resolved_target_modules": ["model.layers.0.q_proj"],
    }
    legacy_metadata = {
        "schema_version": fsdp_module._LORA_DCP_METADATA_SCHEMA_VERSION,
        "lora_signature": engine._lora_config_signature,
    }

    with pytest.raises(ValueError, match="adapter layout mismatch"):
        engine._validate_lora_dcp_metadata(legacy_metadata, "/tmp/checkpoint")


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
