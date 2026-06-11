import asyncio
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from areal.api import ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig
from areal.api.io_struct import HttpRequest
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_context import WorkflowContext
from areal.infra.workflow_executor import (
    RolloutStaleError,
    WorkflowExecutor,
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
    def lora_version_lease(self, version: int, *, workflow: bool = False):
        self.leased_versions.append((version, workflow))
        yield


class StaticWorkflow(RolloutWorkflow):
    def __init__(self, on_run=None):
        self.on_run = on_run
        self.seen_lora_version = None

    async def arun_episode(self, engine, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.seen_lora_version = workflow_context.get().lora_version
        if self.on_run is not None:
            self.on_run()
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 2), dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1]], dtype=torch.int32),
            "versions": torch.tensor(
                [[-1, self.seen_lora_version]], dtype=torch.int32
            ),
            "rewards": torch.tensor([1.0], dtype=torch.float32),
        }


class FakeBackend:
    def build_generation_request(self, req: ModelRequest, with_lora: bool, version: int):
        return HttpRequest(endpoint="/generate", payload={"version": version})

    def parse_generation_response(self, response: dict[str, Any]):
        raise AssertionError("stale rollout should not issue a backend request")

    def get_pause_request(self):
        return HttpRequest(endpoint="/pause", payload={})

    def get_resume_request(self):
        return HttpRequest(endpoint="/resume", payload={})

    def get_health_check_request(self):
        return HttpRequest(endpoint="/health", payload={}, method="GET")


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
    """Task execution should expose the LoRA version captured at task creation."""
    executor, engine = make_executor(version=3)
    workflow = StaticWorkflow()
    task_fn = executor._create_workflow_task(
        _RolloutTaskInput(task_id=1, data={}, workflow=workflow)
    )

    engine.set_version(5)
    result = asyncio.run(task_fn())

    assert result is not None
    assert workflow.seen_lora_version == 3
    assert result.trajectory["versions"][0, 1].item() == 3


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
    workflow = StaticWorkflow(on_run=lambda: executor.set_min_allowed_rollout_version(1))
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
