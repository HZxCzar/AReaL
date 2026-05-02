# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "RLVRWorkflow",
    "MultiTurnWorkflow",
    "VisionRLVRWorkflow",
    "HanabiWorkflow",
    "HanabiOnlineSFTWorkflow",
    "KuhnPokerWorkflow",
    "WerewolfWorkflow",
]

_LAZY_IMPORTS = {
    "RLVRWorkflow": "areal.workflow.rlvr",
    "MultiTurnWorkflow": "areal.workflow.multi_turn",
    "VisionRLVRWorkflow": "areal.workflow.vision_rlvr",
    "HanabiWorkflow": "areal.workflow.hanabi",
    "HanabiOnlineSFTWorkflow": "areal.workflow.hanabi_online_sft",
    "KuhnPokerWorkflow": "areal.workflow.kuhn_poker",
    "WerewolfWorkflow": "areal.workflow.werewolf",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
