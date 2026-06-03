from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import GRPOConfig


@dataclass
class CodeCoachAuxiliaryModelConfig:
    mode: str = field(
        default="api",
        metadata={
            "help": (
                "Student backend: 'api' calls an external OpenAI-compatible "
                "student model, 'self' uses the actor base model with LoRA disabled."
            ),
            "choices": ["api", "self"],
        },
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to enable thinking for mode='self' student calls."},
    )
    base_url: str = field(default="http://127.0.0.1:30000/v1")
    model: str = field(default="qwen-student")
    api_key: str = field(default="EMPTY")
    timeout: int = field(default=120)
    max_tokens: int = field(default=4096)
    temperature: float = field(default=0.7)
    top_p: float | None = field(default=None)
    max_concurrent_calls: int = field(default=8)
    request_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Additional OpenAI chat.completions.create keyword arguments for "
                "mode='api'. Use extra_body for backend-specific parameters."
            )
        },
    )


@dataclass
class CodeCoachPairwiseRewardConfig:
    enabled: bool = field(default=False)
    reference_lag_steps: int = field(default=5)
    scale: float = field(default=0.05)
    compare_all_turns: bool = field(default=True)


@dataclass
class CodeCoachRewardConfig:
    token_budget_penalty: float = field(default=-0.2)
    pairwise: CodeCoachPairwiseRewardConfig = field(
        default_factory=CodeCoachPairwiseRewardConfig
    )


@dataclass
class CodeCoachConfig(GRPOConfig):
    workflow: str = field(
        default="examples.codecoach.workflow.CodeCoachAgentWorkflow",
        metadata={"help": "Training workflow import path."},
    )
    eval_workflow: str = field(
        default="examples.codecoach.workflow.CodeCoachAgentWorkflow",
        metadata={"help": "Evaluation workflow import path."},
    )
    max_turns: int = field(default=6)
    enable_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable thinking mode for the teacher rollout model."
        },
    )
    auxiliary_model: CodeCoachAuxiliaryModelConfig = field(
        default_factory=CodeCoachAuxiliaryModelConfig
    )
    reward: CodeCoachRewardConfig = field(default_factory=CodeCoachRewardConfig)
    work_dir_root: str = field(default="examples/codecoach/artifacts")
    debug_trace_dir: str = field(default="")
    debug_trace_every_n_rollouts: int = field(default=10)
