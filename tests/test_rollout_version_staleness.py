import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from areal.api import (
    Job,
    LocalInfServerInfo,
    ModelRequest,
    RolloutWorkflow,
    Worker,
)
from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig
from areal.api.io_struct import HttpGenerationResult, HttpRequest, WeightUpdateMeta
from areal.infra import remote_inf_engine as remote_inf_engine_module
from areal.infra import workflow_context
from areal.infra import workflow_executor as workflow_executor_module
from areal.infra.async_task_runner import TimedResult
from areal.infra.controller.rollout_controller import (
    RolloutController,
    _RemoteRolloutResult,
    _RemoteRolloutTaskInput,
)
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_context import WorkflowContext
from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    RolloutStaleError,
    WorkflowExecutor,
    _RolloutResult,
    _RolloutTaskInput,
)


class FakeInferenceEngine:
    def __init__(self, version: int = 0):
        self.version = version
        self.leased_versions = []

    def get_version(self) -> int:
        return self.version

    def set_version(self, version: int) -> None:
        self.version = version

    @contextmanager
    def lora_version_lease(self, version: int | None, *, workflow: bool = False):
        leased_version = self.version if version is None else version
        self.leased_versions.append((leased_version, workflow))
        yield leased_version


class StaticWorkflow(RolloutWorkflow):
    def __init__(self, on_run=None):
        self.on_run = on_run
        self.seen_lora_version = None

    async def arun_episode(
        self, engine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        self.seen_lora_version = workflow_context.get().lora_version
        if self.on_run is not None:
            self.on_run()
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 2), dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1]], dtype=torch.int32),
            "versions": torch.tensor([[-1, self.seen_lora_version]], dtype=torch.int32),
            "rewards": torch.tensor([1.0], dtype=torch.float32),
        }


class FakeBackend:
    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ):
        return HttpRequest(endpoint="/generate", payload={"version": version})

    def parse_generation_response(self, response: dict[str, Any]):
        raise AssertionError("stale rollout should not issue a backend request")

    def get_pause_request(self):
        return HttpRequest(endpoint="/pause", payload={})

    def get_resume_request(self):
        return HttpRequest(endpoint="/resume", payload={})

    def get_health_check_request(self):
        return HttpRequest(endpoint="/health", payload={}, method="GET")


class SuccessfulFakeBackend(FakeBackend):
    def parse_generation_response(self, response: dict[str, Any]):
        return HttpGenerationResult(
            output_tokens=[7], output_logprobs=[0.0], stop_reason="stop"
        )


class FakeSubmissionExecutor:
    def __init__(self):
        self.paused = False
        self.min_allowed_rollout_version = None

    def set_min_allowed_rollout_version(self, version: int) -> None:
        self.min_allowed_rollout_version = version

    def is_rollout_version_stale(self, version: int) -> bool:
        return False

    def pause_submission(self) -> None:
        self.paused = True

    def resume_submission(self) -> None:
        self.paused = False


def make_executor(version: int = 0) -> tuple[WorkflowExecutor, FakeInferenceEngine]:
    config = InferenceEngineConfig(
        backend="sglang:d1",
        max_concurrent_rollouts=4,
        consumer_batch_size=2,
        max_head_offpolicyness=1,
        queue_size=8,
        use_lora=True,
    )
    engine = FakeInferenceEngine(version=version)
    manager = StalenessManager(
        version_provider=engine,
        max_concurrent_rollouts=4,
        consumer_batch_size=2,
        max_staleness=1,
    )
    executor = WorkflowExecutor(
        config=config, inference_engine=engine, staleness_manager=manager
    )
    executor.logger = None
    return executor, engine


def test_workflow_task_uses_start_lora_version_in_context():
    """A queued task should bind the latest LoRA version when execution starts."""
    executor, engine = make_executor(version=3)
    workflow = StaticWorkflow()
    task_fn = executor._create_workflow_task(
        _RolloutTaskInput(task_id=1, data={}, workflow=workflow)
    )

    engine.set_version(5)
    result = asyncio.run(task_fn())

    assert result is not None
    assert workflow.seen_lora_version == 5
    assert result.rollout_version == 5
    assert result.trajectory["versions"][0, 1].item() == 5
    assert engine.leased_versions == [(5, True)]


def test_workflow_task_rejects_stale_before_running_workflow():
    """Already-stale tasks should reject without invoking the workflow or filter."""
    executor, _ = make_executor(version=0)
    workflow = StaticWorkflow()
    should_accept_called = False

    def should_accept(_traj):
        nonlocal should_accept_called
        should_accept_called = True
        return True

    task_fn = executor._create_workflow_task(
        _RolloutTaskInput(
            task_id=2,
            data={},
            workflow=workflow,
            should_accept_fn=should_accept,
        )
    )
    executor.set_min_allowed_rollout_version(1)

    result = asyncio.run(task_fn())

    assert result is None
    assert workflow.seen_lora_version is None
    assert should_accept_called is False
    stats = executor.staleness_manager.get_stats()
    assert stats.rejected == 1


def test_workflow_task_rejects_stale_after_workflow_returns():
    """Tasks that become stale during workflow execution should be rejected."""
    executor, _ = make_executor(version=0)
    workflow = StaticWorkflow(
        on_run=lambda: executor.set_min_allowed_rollout_version(1)
    )
    should_accept_called = False

    def should_accept(_traj):
        nonlocal should_accept_called
        should_accept_called = True
        return True

    task_fn = executor._create_workflow_task(
        _RolloutTaskInput(
            task_id=3,
            data={},
            workflow=workflow,
            should_accept_fn=should_accept,
        )
    )

    result = asyncio.run(task_fn())

    assert result is None
    assert workflow.seen_lora_version == 0
    assert should_accept_called is False
    stats = executor.staleness_manager.get_stats()
    assert stats.rejected == 1


def test_workflow_task_logs_stale_zero_for_accepted_and_one_for_stale(monkeypatch):
    """The stale metric should average over finalized rollout tasks."""

    class CapturingStatsTracker:
        def __init__(self):
            self.records = []

        def get(self, name: str):
            assert name == "rollout"
            return self

        def scalar(self, **metrics):
            self.records.append(metrics)

    tracker = CapturingStatsTracker()
    monkeypatch.setattr(workflow_executor_module, "stats_tracker", tracker)

    accepted_executor, _ = make_executor(version=0)
    accepted_task = accepted_executor._create_workflow_task(
        _RolloutTaskInput(task_id=20, data={}, workflow=StaticWorkflow())
    )
    accepted_result = asyncio.run(accepted_task())
    assert accepted_result is not None
    assert accepted_executor._transform_dequeued_result(accepted_result) is not None

    stale_executor, _ = make_executor(version=0)
    stale_task = stale_executor._create_workflow_task(
        _RolloutTaskInput(task_id=21, data={}, workflow=StaticWorkflow())
    )
    stale_executor.set_min_allowed_rollout_version(1)
    assert asyncio.run(stale_task()) is None

    assert tracker.records == [
        {"accepted": 1, "stale": 0},
        {"rejected": 1, "stale": 1},
    ]


def _buffer_rollout_result(
    executor: WorkflowExecutor,
    result: _RolloutResult,
) -> None:
    manager = executor.staleness_manager
    dispatcher = BatchTaskDispatcher(
        max_queue_size=4,
        task_factory=lambda _input: None,
        staleness_manager=manager,
        result_transform=executor._transform_dequeued_result,
    )
    executor._dispatcher = dispatcher
    manager.on_rollout_enqueued()
    manager.on_rollout_submitted()
    manager.on_rollout_accepted()
    dispatcher._active_task_ids.add(result.task_id)
    dispatcher._pending_results[result.task_id] = TimedResult(
        create_time=1,
        data=result,
        task_id=result.task_id,
    )


def test_discard_before_worker_submit_prevents_task_from_starting():
    executor, _ = make_executor(version=0)
    dispatcher = BatchTaskDispatcher(
        max_queue_size=4,
        task_factory=lambda _input: None,
        staleness_manager=executor.staleness_manager,
    )
    task_input = _RolloutTaskInput(task_id=29, data={}, workflow=StaticWorkflow())

    assert dispatcher.discard_task(29) is True
    dispatcher.submit_task_input(task_input)

    assert list(dispatcher._pending_inputs) == []
    assert dispatcher._active_task_ids == set()
    assert dispatcher._discarded_task_ids == set()
    assert dispatcher._pre_discarded_task_deadlines == {}


def test_pre_discard_tombstone_expires():
    executor, _ = make_executor(version=0)
    dispatcher = BatchTaskDispatcher(
        max_queue_size=4,
        task_factory=lambda _input: None,
        staleness_manager=executor.staleness_manager,
    )
    task_input = _RolloutTaskInput(task_id=27, data={}, workflow=StaticWorkflow())

    assert dispatcher.discard_task(27, tombstone_ttl_seconds=0) is True
    dispatcher.submit_task_input(task_input)

    assert list(dispatcher._pending_inputs) == [task_input]
    assert dispatcher._active_task_ids == {27}
    assert dispatcher._pre_discarded_task_deadlines == {}


def test_discarded_pending_result_uses_discard_callback_not_result_transform():
    executor, _ = make_executor(version=0)
    discarded = []

    def fail_transform(_result):
        raise AssertionError("discarded results must not be accepted")

    dispatcher = BatchTaskDispatcher(
        max_queue_size=4,
        task_factory=lambda _input: None,
        staleness_manager=executor.staleness_manager,
        result_transform=fail_transform,
        discarded_result_callback=discarded.append,
    )
    result = _RolloutResult(
        task_id=28,
        trajectory={"input_ids": torch.tensor([[1]])},
        rollout_version=0,
    )
    dispatcher._active_task_ids.add(28)
    dispatcher._pending_results[28] = TimedResult(
        create_time=1,
        data=result,
        task_id=28,
    )

    assert dispatcher.discard_task(28) is True

    assert discarded == [result]
    assert dispatcher._pending_results == {}
    assert dispatcher._active_task_ids == set()


def test_buffered_rollout_is_rejected_if_stale_at_dequeue():
    """A result that aged in the worker buffer must not reach the trainer."""
    executor, _ = make_executor(version=0)
    result = _RolloutResult(
        task_id=30,
        trajectory={"input_ids": torch.tensor([[1]])},
        rollout_version=0,
    )
    _buffer_rollout_result(executor, result)
    executor.set_min_allowed_rollout_version(1)

    assert executor.wait(count=1, timeout=0.1) == [None]
    stats = executor.staleness_manager.get_stats()
    assert stats.accepted == 0
    assert stats.rejected == 1
    assert stats.running == 0


def test_buffered_rollout_at_minimum_version_remains_accepted():
    """The minimum allowed rollout version is inclusive."""
    executor, _ = make_executor(version=1)
    trajectory = {"input_ids": torch.tensor([[1]])}
    result = _RolloutResult(
        task_id=31,
        trajectory=trajectory,
        rollout_version=1,
    )
    _buffer_rollout_result(executor, result)
    executor.set_min_allowed_rollout_version(1)

    dequeued = executor.wait(count=1, timeout=0.1)

    assert len(dequeued) == 1
    assert dequeued[0] is trajectory
    stats = executor.staleness_manager.get_stats()
    assert stats.accepted == 1
    assert stats.rejected == 0


def test_controller_buffered_rollout_is_rejected_if_stale_at_dequeue():
    """The controller's second result buffer must also enforce freshness."""
    config = InferenceEngineConfig(
        backend="sglang:d1",
        consumer_batch_size=2,
        max_concurrent_rollouts=4,
        max_head_offpolicyness=1,
    )
    controller = RolloutController(
        inf_engine=RemoteInfEngine,
        config=config,
        scheduler=object(),
    )
    controller._staleness_manager = StalenessManager(
        version_provider=controller,
        max_concurrent_rollouts=4,
        consumer_batch_size=2,
        max_staleness=1,
    )
    controller._version = 2
    manager = controller.staleness_manager
    manager.on_rollout_enqueued()
    manager.on_rollout_submitted()
    manager.on_rollout_accepted()
    result = _RemoteRolloutResult(
        task_id=32,
        trajectory={"input_ids": torch.tensor([[1]])},
        rollout_version=0,
    )

    assert controller._transform_dequeued_result(result) is None
    stats = manager.get_stats()
    assert stats.accepted == 0
    assert stats.rejected == 1
    assert stats.running == 0


def test_remote_agenerate_raises_before_backend_request_when_rollout_stale():
    """Remote generation should stop cooperatively before issuing stale requests."""
    config = InferenceEngineConfig(
        backend="sglang:d1",
        max_head_offpolicyness=1,
        tokenizer_path="dummy-tokenizer",
        use_lora=True,
    )
    engine = RemoteInfEngine(config=config, backend=FakeBackend())
    executor, _ = make_executor(version=0)
    engine.workflow_executor = executor
    engine.set_version(2)
    engine.addresses = ["127.0.0.1:1"]
    workflow_context.set(WorkflowContext(task_id=10, lora_version=0))
    req = ModelRequest(
        rid="stale",
        input_ids=[1, 2],
        gconfig=GenerationHyperparameters(max_new_tokens=1, max_tokens=8),
    )

    with pytest.raises(RolloutStaleError):
        asyncio.run(engine.agenerate(req))

    workflow_context.set(WorkflowContext())


def test_current_lora_lease_protects_version_from_unload_selection():
    config = InferenceEngineConfig(
        backend="sglang:d1",
        use_lora=True,
        request_timeout=1,
    )
    engine = RemoteInfEngine(config=config, backend=FakeBackend())
    engine.addresses = ["127.0.0.1:1"]
    engine._version = 3
    engine._loaded_lora_versions_by_addr = {"127.0.0.1:1": {3}}
    engine._unloading_lora_versions_by_addr = {"127.0.0.1:1": set()}
    engine._max_loaded_loras_by_addr = {"127.0.0.1:1": 1}

    with engine.lora_version_lease(None, workflow=True) as leased_version:
        assert leased_version == 3
        with pytest.raises(TimeoutError, match="inactive LoRA adapter slots"):
            engine._select_inactive_loras_to_unload(4, time.monotonic())

    assert engine._select_inactive_loras_to_unload(4, time.monotonic()) == {
        3: ["127.0.0.1:1"]
    }


def test_vllm_server_args_record_loaded_lora_and_capacity():
    config = InferenceEngineConfig(backend="vllm:d1", use_lora=True)
    engine = RemoteInfEngine(config=config, backend=FakeBackend())
    addr = "127.0.0.1:1"
    engine.addresses = [addr]

    engine.record_lora_server_args(
        {
            "lora_modules": ["actor-v0=/tmp/adapter"],
            "max_loras": 4,
        }
    )

    assert engine._loaded_lora_versions_by_addr == {addr: {0}}
    assert engine._max_loaded_loras_by_addr == {addr: 4}
    with engine.lora_version_lease(0) as leased_version:
        assert leased_version == 0


@pytest.mark.asyncio
async def test_eval_controller_uses_updated_lora_on_shared_server():
    """Eval workers should use LoRAs loaded by the owning rollout controller."""

    class FakeScheduler:
        def __init__(self, config):
            self.worker = Worker(id="eval-rollout/0", ip="127.0.0.1")
            self.engine = RemoteInfEngine(config=config, backend=FakeBackend())

        def create_workers(self, job):
            return [self.worker.id]

        def get_workers(self, role):
            return [self.worker]

        async def create_engine(self, **_kwargs):
            return None

        async def async_call_engine(self, *, method, **kwargs):
            if method == "initialize":
                self.engine.addresses = [kwargs["addr"]]
            elif method == "record_lora_server_args":
                self.engine.record_lora_server_args(kwargs["server_args"])
            return None

    config = InferenceEngineConfig(backend="sglang:d1", use_lora=True)
    scheduler = FakeScheduler(config)
    controller = RolloutController(
        inf_engine=RemoteInfEngine,
        config=config,
        scheduler=scheduler,
    )
    controller._worker_role = "eval-rollout"

    await controller._async_initialize(
        Job(role="eval-rollout"),
        server_args={
            "lora_paths": ["default_lora-v0=/tmp/initial_lora"],
            "max_loaded_loras": 16,
        },
        server_infos=[LocalInfServerInfo(host="127.0.0.1", port=30000, process=None)],
    )

    scheduler.engine.set_version(10)
    with scheduler.engine.lora_version_lease(None, workflow=True) as version:
        assert version == 10


def test_concurrent_lora_disk_update_fails_instead_of_waiting(monkeypatch):
    config = InferenceEngineConfig(backend="sglang:d1", use_lora=True)
    engine = RemoteInfEngine(config=config, backend=FakeBackend())
    update_started = threading.Event()
    release_update = threading.Event()

    def fake_serial_update(*_args):
        update_started.set()
        assert release_update.wait(timeout=1)

    monkeypatch.setattr(
        engine, "_update_lora_weights_from_disk_serial", fake_serial_update
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            engine._update_lora_weights_from_disk,
            "experiment",
            "trial",
            1,
            None,
        )
        assert update_started.wait(timeout=1)
        second = executor.submit(
            engine._update_lora_weights_from_disk,
            "experiment",
            "trial",
            2,
            None,
        )
        with pytest.raises(RuntimeError, match="already running"):
            second.result()
        release_update.set()
        first.result()


def test_lora_lease_rejects_version_already_selected_for_unload():
    config = InferenceEngineConfig(backend="sglang:d1", use_lora=True)
    engine = RemoteInfEngine(config=config, backend=FakeBackend())
    addr = "127.0.0.1:1"
    engine.addresses = [addr]
    engine._loaded_lora_versions_by_addr = {addr: {3}}
    engine._unloading_lora_versions_by_addr = {addr: {3}}

    with pytest.raises(RuntimeError, match="not available for a new lease"):
        with engine.lora_version_lease(3):
            pass


@pytest.mark.asyncio
async def test_controller_rejects_concurrent_lora_update_before_worker_rpc(
    monkeypatch,
):
    config = InferenceEngineConfig(backend="sglang:d1", use_lora=True)
    controller = RolloutController(
        inf_engine=RemoteInfEngine,
        config=config,
        scheduler=object(),
    )
    first_rpc_started = asyncio.Event()
    release_first_rpc = asyncio.Event()
    rpc_calls = 0

    async def fake_collective_rpc(*_args, **_kwargs):
        nonlocal rpc_calls
        rpc_calls += 1
        first_rpc_started.set()
        await release_first_rpc.wait()

    monkeypatch.setattr(controller, "_collective_rpc_async", fake_collective_rpc)
    meta = WeightUpdateMeta(type="disk", path="/tmp/missing", use_lora=True)

    first_update = asyncio.create_task(controller.update_weights_from_disk(meta))
    await first_rpc_started.wait()
    with pytest.raises(RuntimeError, match="already running"):
        await controller.update_weights_from_disk(meta)
    release_first_rpc.set()
    await first_update

    assert rpc_calls == 1


@pytest.mark.asyncio
async def test_controller_discards_worker_result_that_arrives_after_timeout():
    class Scheduler:
        def __init__(self):
            self.calls = []

        async def async_call_engine(self, worker_id, method, **kwargs):
            self.calls.append((worker_id, method, kwargs))
            return None

    scheduler = Scheduler()
    controller = RolloutController(
        inf_engine=RemoteInfEngine,
        config=InferenceEngineConfig(backend="sglang:d1"),
        scheduler=scheduler,
    )
    await controller._discard_worker_task(
        worker_id="worker-0",
        engine_name="rollout-0",
        engine_task_id=17,
    )

    assert len(scheduler.calls) == 1
    worker_id, method, kwargs = scheduler.calls[0]
    assert worker_id == "worker-0"
    assert method == "_discard_task"
    assert kwargs["task_id"] == 17


def test_controller_no_worker_failure_releases_running_capacity():
    config = InferenceEngineConfig(
        backend="sglang:d1",
        consumer_batch_size=2,
        max_concurrent_rollouts=2,
    )
    controller = RolloutController(
        inf_engine=RemoteInfEngine,
        config=config,
        scheduler=object(),
    )
    controller._staleness_manager = StalenessManager(
        version_provider=controller,
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
        max_staleness=1,
    )
    manager = controller.staleness_manager
    manager.on_rollout_enqueued()
    manager.on_rollout_submitted()
    task_fn = controller._create_submit_callback(
        _RemoteRolloutTaskInput(
            task_id=19,
            data={},
            workflow=None,
            workflow_kwargs={},
            should_accept_fn=None,
        )
    )

    assert asyncio.run(task_fn()) is None
    stats = manager.get_stats()
    assert stats.running == 0
    assert stats.rejected == 1


@pytest.mark.asyncio
async def test_controller_callback_does_not_resolve_cancelled_future():
    controller = RolloutController(
        inf_engine=RemoteInfEngine,
        config=InferenceEngineConfig(backend="sglang:d1"),
        scheduler=object(),
    )
    loop = asyncio.get_running_loop()
    errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        future = loop.create_future()
        controller._pending_futures[23] = future
        controller._resolve_task_future(23)
        future.cancel()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert future.cancelled()
    assert errors == []


def make_remote_engine_for_pause_tests() -> tuple[
    RemoteInfEngine, FakeSubmissionExecutor
]:
    config = InferenceEngineConfig(
        backend="sglang:d1",
        max_head_offpolicyness=1,
        tokenizer_path="dummy-tokenizer",
        request_timeout=1,
        pause_grace_period=0,
        use_lora=False,
    )
    engine = RemoteInfEngine(config=config, backend=SuccessfulFakeBackend())
    executor = FakeSubmissionExecutor()
    engine.workflow_executor = executor
    engine.addresses = ["127.0.0.1:1"]
    return engine, executor


def test_agenerate_supports_a_private_per_request_attempt_limit(monkeypatch):
    engine, _ = make_remote_engine_for_pause_tests()
    attempt_limits = []

    async def fake_request(**kwargs):
        attempt_limits.append(kwargs["max_retries"])
        return {"ok": True}

    monkeypatch.setattr(remote_inf_engine_module, "arequest_with_retry", fake_request)

    ordinary = ModelRequest(
        rid="ordinary-retries",
        input_ids=[1, 2],
        gconfig=GenerationHyperparameters(max_new_tokens=1, max_tokens=8),
    )
    one_attempt = ModelRequest(
        rid="one-attempt",
        input_ids=[1, 2],
        gconfig=GenerationHyperparameters(max_new_tokens=1, max_tokens=8),
        metadata={"_request_max_attempts": 1},
    )

    asyncio.run(engine.agenerate(ordinary))
    asyncio.run(engine.agenerate(one_attempt))

    assert attempt_limits == [engine.config.request_retries, 1]
    assert one_attempt.metadata == {"_request_max_attempts": 1}


@pytest.mark.parametrize("invalid_limit", [True, 0, -1, "1"])
def test_agenerate_rejects_invalid_per_request_attempt_limits(
    monkeypatch, invalid_limit
):
    engine, _ = make_remote_engine_for_pause_tests()
    requests = []

    async def fake_request(**kwargs):
        requests.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(remote_inf_engine_module, "arequest_with_retry", fake_request)
    req = ModelRequest(
        rid="invalid-attempt-limit",
        input_ids=[1, 2],
        gconfig=GenerationHyperparameters(max_new_tokens=1, max_tokens=8),
        metadata={"_request_max_attempts": invalid_limit},
    )

    with pytest.raises(ValueError, match="must be a positive integer"):
        asyncio.run(engine.agenerate(req))

    assert requests == []


def test_rollout_submission_pause_aliases_do_not_hard_pause_generation(monkeypatch):
    engine, executor = make_remote_engine_for_pause_tests()
    requests = []

    async def fake_request(**kwargs):
        requests.append(kwargs["payload"])
        return {"ok": True}

    monkeypatch.setattr(remote_inf_engine_module, "arequest_with_retry", fake_request)

    engine.pause_rollout_submission()
    assert executor.paused is True

    req = ModelRequest(
        rid="soft-pause",
        input_ids=[1, 2],
        gconfig=GenerationHyperparameters(max_new_tokens=1, max_tokens=8),
    )
    resp = asyncio.run(engine.agenerate(req))

    assert resp.output_tokens == [7]
    assert requests == [{"version": 0}]

    engine.resume()
    assert executor.paused is False

    engine.pause()
    assert executor.paused is True
    engine.resume_rollout_submission()
    assert executor.paused is False


def test_generation_pause_blocks_agenerate_until_resumed(monkeypatch):
    engine, _ = make_remote_engine_for_pause_tests()
    requests = []

    async def fake_request(**kwargs):
        requests.append(kwargs["payload"])
        return {"ok": True}

    monkeypatch.setattr(remote_inf_engine_module, "arequest_with_retry", fake_request)
    monkeypatch.setattr(engine, "_run_request_on_all_servers", lambda requests: None)

    async def run_test():
        engine.pause_generation()
        req = ModelRequest(
            rid="hard-pause",
            input_ids=[1, 2],
            gconfig=GenerationHyperparameters(max_new_tokens=1, max_tokens=8),
        )
        task = asyncio.create_task(engine.agenerate(req))
        await asyncio.sleep(0.05)
        assert requests == []

        engine.resume_generation()
        resp = await asyncio.wait_for(task, timeout=2)
        assert resp.output_tokens == [7]
        assert requests == [{"version": 0}]

        engine.pause_generation()
        engine.continue_generation()
        assert engine.is_generation_paused() is False

    asyncio.run(run_test())
    workflow_context.set(WorkflowContext())
