# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
from typing import Any


class EnvironmentService(abc.ABC):
    async def step(self, action: Any):
        raise NotImplementedError

    async def reset(self, seed=None, options=None):
        raise NotImplementedError


ALL_ENV_CLASSES = {}


def register_environment(name, env_cls):
    if name in ALL_ENV_CLASSES:
        return
    if "/" in name:
        raise ValueError(f"Environment name must not contain '/': {name}")
    ALL_ENV_CLASSES[name] = env_cls


class NullEnvironment:
    async def step(self, action):
        await asyncio.sleep(1)
        return None, 0.0, True, False, {}

    async def reset(self, seed=None, options=None):
        await asyncio.sleep(0.1)
        return None, {}


register_environment("null", NullEnvironment)


def make_env(cfg):
    env_type = getattr(cfg, "type_", getattr(cfg, "type", None))
    args = getattr(cfg, "args", {})
    if env_type not in ALL_ENV_CLASSES:
        raise ValueError(f"Unknown environment type: {env_type}")
    return ALL_ENV_CLASSES[env_type](**args)
