"""Tests for the recovery configuration and functionality."""

import json
import os
import shutil
import tempfile
from unittest.mock import Mock

import pytest

from areal.api.cli_args import RecoverConfig
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.utils.recover import (
    InValidRecoverInfo,
    RecoverHandler,
    check_if_auto_recover,
    check_if_recover,
)


class TestRecoverConfig:
    """Tests for RecoverConfig dataclass validation."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
        )
        assert config.mode == "disabled"
        assert config.retries == 3

    @pytest.mark.parametrize("mode", ["on", "off", "auto", "disabled"])
    def test_valid_modes(self, mode):
        """Test that all valid modes are accepted."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
            mode=mode,
        )
        assert config.mode == mode

    @pytest.mark.parametrize("mode", ["fault", "resume", "invalid", "ON", "OFF", ""])
    def test_invalid_modes(self, mode):
        """Test that invalid modes raise ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot="/tmp",
                mode=mode,
            )
        error_msg = str(exc_info.value)
        assert f"Invalid recover mode '{mode}'" in error_msg
        assert "fault" in error_msg and "resume" in error_msg  # Migration hint


class TestCheckIfRecover:
    """Tests for the check_if_recover function."""

    @pytest.mark.parametrize("mode", ["disabled", "off"])
    def test_disabled_modes_return_false(self, mode):
        """Test that disabled modes always return False."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
            mode=mode,
        )
        # Should return False regardless of run_id
        assert check_if_recover(config, 0) is False
        assert check_if_recover(config, 1) is False
        assert check_if_recover(config, 10) is False

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_enabled_modes_check_for_checkpoint(self, mode):
        """Test that enabled modes check for existing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode=mode,
            )
            # No checkpoint exists, should return False
            assert check_if_recover(config, 0) is False

    @pytest.mark.parametrize("run_id", [0, 1, 5, 100])
    def test_run_id_parameter_unused(self, run_id):
        """Test that run_id parameter doesn't affect the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with disabled mode
            config_disabled = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="disabled",
            )
            assert check_if_recover(config_disabled, run_id) is False

            # Test with enabled mode (no checkpoint)
            config_enabled = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            # Result should be the same regardless of run_id
            result = check_if_recover(config_enabled, run_id)
            assert result == check_if_recover(config_enabled, 0)


class TestCheckIfAutoRecover:
    """Tests for the check_if_auto_recover function."""

    def test_no_checkpoint_returns_false(self):
        """Test that missing checkpoint returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            assert check_if_auto_recover(config) is False

    def test_empty_directory_returns_false(self):
        """Test that empty directory (no checkpoint) returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            assert check_if_auto_recover(config) is False


class TestModeEquivalence:
    """Tests to verify mode equivalences (on=auto, off=disabled)."""

    def test_on_equals_auto(self):
        """Test that 'on' and 'auto' modes behave identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_on = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            config_auto = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="auto",
            )
            # Both should return the same result
            assert check_if_recover(config_on, 0) == check_if_recover(config_auto, 0)

    def test_off_equals_disabled(self):
        """Test that 'off' and 'disabled' modes behave identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_off = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="off",
            )
            config_disabled = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="disabled",
            )
            # Both should return False
            assert check_if_recover(config_off, 0) is False
            assert check_if_recover(config_disabled, 0) is False


class TestRecoverHandler:
    @staticmethod
    def _make_handler(tmpdir: str, mode: str) -> RecoverHandler:
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot=tmpdir,
            mode=mode,
            freq_steps=1,
        )
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=8,
            train_batch_size=2,
        )
        return RecoverHandler(config, ft_spec)

    @staticmethod
    def _make_gateway_controller() -> GatewayTrainController:
        return GatewayTrainController.__new__(GatewayTrainController)

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_load_rejects_gateway_train_controller(self, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, mode)

            with pytest.raises(NotImplementedError) as exc_info:
                handler.load(
                    self._make_gateway_controller(),
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                )

            assert "GatewayTrainController" in str(exc_info.value)
            assert '`_version="v2"`' in str(exc_info.value)

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_dump_rejects_gateway_train_controller(self, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, mode)
            step_info = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )

            with pytest.raises(NotImplementedError) as exc_info:
                handler.dump(
                    self._make_gateway_controller(),
                    step_info,
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                )

            assert "GatewayTrainController" in str(exc_info.value)
            assert "recover.mode" in str(exc_info.value)

    def test_failed_save_does_not_replace_last_complete_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "on")

            class Stateful:
                @staticmethod
                def state_dict():
                    return {}

            class Engine:
                fail = False

                def save(self, meta):
                    if self.fail:
                        raise RuntimeError("save failed")
                    with open(os.path.join(meta.path, "complete"), "w") as f:
                        f.write("ok")

            engine = Engine()
            stateful = Stateful()
            first_step = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            handler.dump(
                engine,
                first_step,
                stateful,
                stateful,
                stateful,
                stateful,
            )
            pointer_path = os.path.join(
                handler._recover_root("test_exp", "test_trial", tmpdir),
                handler._CURRENT_FILE,
            )
            with open(pointer_path) as f:
                first_generation = json.load(f)["generation"]

            engine.fail = True
            second_step = StepInfo(
                epoch=0,
                epoch_step=1,
                global_step=1,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            with pytest.raises(RuntimeError, match="save failed"):
                handler.dump(
                    engine,
                    second_step,
                    stateful,
                    stateful,
                    stateful,
                    stateful,
                )

            with open(pointer_path) as f:
                assert json.load(f)["generation"] == first_generation
            assert check_if_auto_recover(handler.config) is True

    def test_failed_state_collection_removes_uncommitted_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "on")

            class Engine:
                @staticmethod
                def save(meta):
                    with open(os.path.join(meta.path, "complete"), "w") as f:
                        f.write("ok")

            class Stateful:
                @staticmethod
                def state_dict():
                    return {}

            class FailingStateful:
                @staticmethod
                def state_dict():
                    raise RuntimeError("state collection failed")

            step = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            with pytest.raises(RuntimeError, match="state collection failed"):
                handler.dump(
                    Engine(),
                    step,
                    Stateful(),
                    Stateful(),
                    Stateful(),
                    FailingStateful(),
                )

            recover_root = handler._recover_root("test_exp", "test_trial", tmpdir)
            generations_root = os.path.join(recover_root, handler._GENERATIONS_DIR)
            assert not os.path.exists(os.path.join(recover_root, handler._CURRENT_FILE))
            assert not os.path.exists(generations_root) or not os.listdir(
                generations_root
            )

    def test_complete_save_replaces_invalid_current_pointer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "on")
            recover_root = handler._recover_root("test_exp", "test_trial", tmpdir)
            os.makedirs(recover_root, exist_ok=True)
            pointer_path = os.path.join(recover_root, handler._CURRENT_FILE)
            with open(pointer_path, "w") as f:
                json.dump({"generation": ".."}, f)
            orphan_path = os.path.join(recover_root, handler._GENERATIONS_DIR, "orphan")
            os.makedirs(orphan_path)
            with open(os.path.join(orphan_path, "partial"), "w") as f:
                f.write("incomplete")

            class Engine:
                @staticmethod
                def save(meta):
                    with open(os.path.join(meta.path, "complete"), "w") as f:
                        f.write("ok")

            class Stateful:
                @staticmethod
                def state_dict():
                    return {}

            step = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            handler.dump(
                Engine(),
                step,
                Stateful(),
                Stateful(),
                Stateful(),
                Stateful(),
            )

            generation = handler._read_current_generation(
                "test_exp", "test_trial", tmpdir
            )
            assert generation not in {".", ".."}
            assert os.path.isdir(
                handler._generation_path("test_exp", "test_trial", tmpdir, generation)
            )
            assert not os.path.exists(orphan_path)

    def test_dump_rejects_unsafe_engine_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "on")
            step = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )

            with pytest.raises(InValidRecoverInfo, match="recovery engine name"):
                handler.dump(
                    {"..": Mock()},
                    step,
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                )

    def test_load_restores_local_state_before_entering_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "on")

            class Engine:
                load_called = False

                @staticmethod
                def save(meta):
                    with open(os.path.join(meta.path, "complete"), "w") as f:
                        f.write("ok")

                def load(self, _meta):
                    self.load_called = True

            class Stateful:
                @staticmethod
                def state_dict():
                    return {}

                @staticmethod
                def load_state_dict(_state):
                    return None

            class FailingDataloader(Stateful):
                @staticmethod
                def load_state_dict(_state):
                    raise RuntimeError("local restore failed")

            engine = Engine()
            stateful = Stateful()
            step = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            handler.dump(
                engine,
                step,
                stateful,
                stateful,
                stateful,
                stateful,
            )

            with pytest.raises(RuntimeError, match="local restore failed"):
                handler.load(
                    engine,
                    stateful,
                    stateful,
                    stateful,
                    FailingDataloader(),
                )

            assert engine.load_called is False

    def test_missing_checkpoint_does_not_mutate_local_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "on")

            class Engine:
                @staticmethod
                def save(meta):
                    with open(os.path.join(meta.path, "complete"), "w") as f:
                        f.write("ok")

            class Stateful:
                load_calls = 0

                @staticmethod
                def state_dict():
                    return {}

                def load_state_dict(self, _state):
                    self.load_calls += 1

            engine = Engine()
            stateful = Stateful()
            step = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            handler.dump(
                engine,
                step,
                stateful,
                stateful,
                stateful,
                stateful,
            )
            generation = handler._read_current_generation(
                "test_exp", "test_trial", tmpdir
            )
            checkpoint_path = os.path.join(
                handler._generation_path("test_exp", "test_trial", tmpdir, generation),
                "checkpoints",
                "default",
            )
            shutil.rmtree(checkpoint_path)

            assert handler.load(engine, stateful, stateful, stateful, stateful) is None
            assert stateful.load_calls == 0

    def test_directory_sync_failure_keeps_previous_generation(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "on")

            class Engine:
                @staticmethod
                def save(meta):
                    with open(os.path.join(meta.path, "complete"), "w") as f:
                        f.write("ok")

            class Stateful:
                @staticmethod
                def state_dict():
                    return {}

            stateful = Stateful()
            first_step = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            handler.dump(
                Engine(),
                first_step,
                stateful,
                stateful,
                stateful,
                stateful,
            )

            from areal.utils import recover as recover_module

            original_os_open = recover_module.os.open

            def fail_directory_open(*_args, **_kwargs):
                raise OSError("directory fsync unsupported")

            monkeypatch.setattr(recover_module.os, "open", fail_directory_open)
            second_step = StepInfo(
                epoch=0,
                epoch_step=1,
                global_step=1,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            handler.dump(
                Engine(),
                second_step,
                stateful,
                stateful,
                stateful,
                stateful,
            )
            monkeypatch.setattr(recover_module.os, "open", original_os_open)

            generations_root = os.path.join(
                handler._recover_root("test_exp", "test_trial", tmpdir),
                handler._GENERATIONS_DIR,
            )
            assert len(os.listdir(generations_root)) == 2
