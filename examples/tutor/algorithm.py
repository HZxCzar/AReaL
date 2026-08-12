"""Tutor-local training engine.

This lives in its own module and not in train.py on purpose. The scheduler ships
an engine to its workers by ``module.name`` and re-imports it there, and a worker
runs ``python3 -m areal.infra.rpc.rpc_server``, so its ``__main__`` is not train.py.
A class defined inside ``train.main`` resolves to ``__main__.TutorFSDPPPOActor``
and the worker fails with EngineImportError before the first step. This is why
examples/pedagogical_rl keeps its actor in algorithm.py.
"""

from __future__ import annotations

from typing import Any

from areal.engine import FSDPPPOActor


class TutorFSDPPPOActor(FSDPPPOActor):
    """FSDP actor that runs ``actor.num_iterations`` passes over each batch.

    ``ppo_n_minibatches`` splits one collected batch into chunks and takes a step
    on each: several optimizer steps, but a single pass over the data. This
    repeats the whole update, so 2 means two passes. The extra pass is off-policy
    against the parameters the first one produced, and the PPO clip is what bounds
    it -- which is the difference between this and simply raising the learning
    rate.

    The outer trainer steps the scheduler once after this returns, so the
    scheduler is stepped between internal passes too and every pass corresponds to
    one scheduler step, as it would if they were separate updates.
    """

    def ppo_update(
        self, data: Any, world_model_batch: Any | None = None
    ) -> None:
        iterations = max(1, int(getattr(self.config, "num_iterations", 1)))
        for iteration in range(iterations):
            if world_model_batch is None:
                super().ppo_update(data)
            else:
                super().ppo_update(data, world_model_batch)
            if iteration + 1 < iterations:
                self.lr_scheduler_step()
