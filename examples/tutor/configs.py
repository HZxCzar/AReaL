from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig
from examples.tutor.prompts import (
    DEFAULT_LEAK_CHECK_SYSTEM_PROMPT,
    DEFAULT_STUDENT_SYSTEM_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_TEACHER_SYSTEM_PROMPT,
)


@dataclass
class TutorAuxiliaryModelConfig:
    mode: str = field(
        default="api",
        metadata={
            "help": (
                "Auxiliary caller backend: 'api' uses base_url, "
                "'self' uses the actor base model without LoRA."
            ),
            "choices": ["api", "self"],
        },
    )
    enable_thinking: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable thinking mode for self auxiliary calls. "
                "Only affects mode='self'."
            )
        },
    )
    base_url: str = field(default="http://127.0.0.1:30000/v1")
    model: str = field(default="qwen-aux")
    api_key: str = field(default="EMPTY")
    timeout: int = field(default=120)
    max_tokens: int = field(default=2048)
    temperature: float = field(default=0.7)
    top_p: float | None = field(default=None)
    max_concurrent_calls: int = field(default=8)
    api_params_config_path: str = field(default="")
    api_params_key: str = field(default="")


@dataclass
class TutorPairwiseRewardConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable post-hoc pairwise tutor reward against lagged LoRA."},
    )
    reference_lag_steps: int = field(default=5)
    scale: float = field(default=0.05)
    compare_all_turns: bool = field(default=True)


@dataclass
class TutorRewardConfig:
    success: float = field(default=1.0)
    leak_penalty: float = field(default=-1.0)
    outcome_prior_turn_weight: float = field(default=0.1)
    outcome_credit_gamma: float = field(default=0.9)
    early_success_bonus: float = field(default=0.3)
    turn_penalty: float = field(default=-0.01)
    length_penalty_threshold_chars: int = field(default=1200)
    length_penalty_per_100_chars: float = field(default=-0.005)
    length_penalty_min: float = field(default=-0.1)
    pairwise: TutorPairwiseRewardConfig = field(
        default_factory=TutorPairwiseRewardConfig
    )


@dataclass
class TutorConfig(GRPOConfig):
    workflow: str = field(
        default="examples.tutor.workflow.TutorAgentWorkflow",
        metadata={"help": "Training workflow import path."},
    )
    eval_workflow: str = field(
        default="examples.tutor.workflow.TutorAgentWorkflow",
        metadata={"help": "Evaluation workflow import path."},
    )
    max_turns: int = field(default=6, metadata={"help": "Maximum teacher turns."})
    enable_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable thinking mode for the tutor rollout model."
        },
    )
    auxiliary_model: TutorAuxiliaryModelConfig = field(
        default_factory=TutorAuxiliaryModelConfig
    )
    reward: TutorRewardConfig = field(default_factory=TutorRewardConfig)
    teacher_system_prompt: str = field(default=DEFAULT_TEACHER_SYSTEM_PROMPT)
    student_system_prompt: str = field(default=DEFAULT_STUDENT_SYSTEM_PROMPT)
    leak_check_system_prompt: str = field(default=DEFAULT_LEAK_CHECK_SYSTEM_PROMPT)
    summary_system_prompt: str = field(default=DEFAULT_SUMMARY_SYSTEM_PROMPT)
    debug_trace_dir: str = field(
        default="",
        metadata={"help": "Optional directory to dump readable per-rollout tutor traces."},
    )
    debug_trace_every_n_rollouts: int = field(
        default=10,
        metadata={"help": "Dump one readable tutor trace every N rollout episodes."},
    )
