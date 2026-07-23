# SPDX-License-Identifier: Apache-2.0

"""FSDP checkpointing utilities for DCP (Distributed Checkpoint) integration."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


class DCPState(Stateful):
    """Wrapper for checkpointing the State using DCP.

    This class implements the Stateful protocol, so DCP will automatically call
    state_dict/load_state_dict as needed in the dcp.save/load APIs.

    It handles calling distributed state dict methods on the model and optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
        | Iterable[torch.optim.Optimizer]
        | None = None,
        lr_schedulers: Iterable[Any] | None = None,
    ):
        self.model = model
        self.optimizer = (
            tuple(optimizer)
            if optimizer is not None
            and not isinstance(optimizer, torch.optim.Optimizer)
            else optimizer
        )
        self.lr_schedulers = tuple(lr_schedulers) if lr_schedulers is not None else None

    def state_dict(self) -> dict[str, Any]:
        """
        Get state dict for model and optimizer using DCP utilities.
        This automatically manages FSDP FQN's and
        sets default state dict type to FSDP.SHARDED_STATE_DICT
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
            state_dict = {"model": model_state_dict, "optim": optimizer_state_dict}
        elif self.optimizer is not None:
            optimizers = tuple(self.optimizer)
            if not optimizers:
                raise ValueError("Optimizer collection must not be empty.")
            model_state_dict, first_optimizer_state = get_state_dict(
                self.model, optimizers[0]
            )
            optimizer_states = [first_optimizer_state]
            for optimizer in optimizers[1:]:
                _, optimizer_state = get_state_dict(self.model, optimizer)
                optimizer_states.append(optimizer_state)
            state_dict = {
                "model": model_state_dict,
                "optimizers": optimizer_states,
            }
        else:
            state_dict = {"model": get_model_state_dict(self.model)}
        if self.lr_schedulers is not None:
            state_dict["lr_schedulers"] = [
                scheduler.state_dict() for scheduler in self.lr_schedulers
            ]
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load state dicts onto model and optimizer.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"],
            )
        elif self.optimizer is not None:
            optimizers = tuple(self.optimizer)
            optimizer_states = state_dict["optimizers"]
            if len(optimizer_states) != len(optimizers):
                raise ValueError(
                    "Checkpoint optimizer count does not match the current engine."
                )
            for optimizer, optimizer_state in zip(
                optimizers, optimizer_states, strict=True
            ):
                set_state_dict(
                    self.model,
                    optimizer,
                    model_state_dict=state_dict["model"],
                    optim_state_dict=optimizer_state,
                )
        else:
            set_model_state_dict(
                self.model,
                model_state_dict=state_dict["model"],
            )
        if self.lr_schedulers is not None:
            scheduler_states = state_dict["lr_schedulers"]
            if len(scheduler_states) != len(self.lr_schedulers):
                raise ValueError(
                    "Checkpoint scheduler count does not match the current engine."
                )
            for scheduler, scheduler_state in zip(
                self.lr_schedulers, scheduler_states, strict=True
            ):
                scheduler.load_state_dict(scheduler_state)
