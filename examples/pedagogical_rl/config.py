from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from examples.pedagogical_rl.cross_eval_config import CrossEvalConfig

from areal.api.cli_args import (
    EvaluatorConfig,
    GRPOConfig,
    PPOActorConfig,
)


@dataclass
class PedagogicalAPIModelConfig:
    """Frozen OpenAI-compatible model used by the classroom workflow."""

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout: float = 120.0
    max_retries: int = 2
    max_concurrent_calls: int = 4
    seed: int = 42
    top_k: int = 20
    min_p: float = 0.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False}
    )

    def __post_init__(self) -> None:
        if self.max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class PedagogicalGenerationConfig:
    max_teacher_turns: int = 10
    max_tokens_in_conversation: int = 24576
    max_tokens_per_teacher_turn: int = 4096
    max_tokens_per_student_turn: int = 2048
    max_tokens_per_student_attempt: int = 2048
    max_tokens_per_judge_attempt: int = 1024
    number_student_attempts: int = 8
    number_judge_attempts: int = 2
    student_temperature: float = 0.7
    student_top_p: float = 0.8
    judge_temperature: float = 0.0
    judge_top_p: float = 1.0
    extra_penalty_for_rejected_judges: float = 1.0
    use_thinking: bool = False
    # Keep this as ``str`` rather than ``Literal`` because the OmegaConf
    # version used by AReaL cannot construct structured configs containing
    # Literal annotations. ``__post_init__`` still enforces the enum values.
    leak_judge_mode: str = "pedagogical_rl"

    def __post_init__(self) -> None:
        positive_fields = {
            "max_teacher_turns": self.max_teacher_turns,
            "max_tokens_in_conversation": self.max_tokens_in_conversation,
            "max_tokens_per_teacher_turn": self.max_tokens_per_teacher_turn,
            "max_tokens_per_student_turn": self.max_tokens_per_student_turn,
            "max_tokens_per_student_attempt": self.max_tokens_per_student_attempt,
            "max_tokens_per_judge_attempt": self.max_tokens_per_judge_attempt,
            "number_student_attempts": self.number_student_attempts,
            "number_judge_attempts": self.number_judge_attempts,
        }
        invalid = [name for name, value in positive_fields.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"generation values must be positive: {invalid}")
        if self.leak_judge_mode not in {"pedagogical_rl", "turn"}:
            raise ValueError(
                "generation.leak_judge_mode must be 'pedagogical_rl' or 'turn'"
            )


@dataclass
class PedagogicalTeacherPreConfig:
    """Private teacher draft generated before the classroom dialogue."""

    enabled: bool = False
    verify: bool = True
    attempts: int = 3
    max_tokens: int = 0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("teacher_pre.attempts must be positive")


@dataclass
class PedagogicalEvaluatorConfig(EvaluatorConfig):
    average_rollouts: int = 3

    def __post_init__(self) -> None:
        if self.average_rollouts < 1:
            raise ValueError("evaluator.average_rollouts must be positive")


@dataclass
class PedagogicalActorConfig(PPOActorConfig):
    """PPO actor with PedagogicalRL's rollout reuse count."""

    behave_imp_weight_cap: float | None = None
    behave_imp_weight_mode: str = "disabled"
    num_iterations: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_iterations < 1:
            raise ValueError("actor.num_iterations must be positive")


@dataclass
class PedagogicalRLConfig(GRPOConfig):
    workflow: str = "examples.pedagogical_rl.workflow.PedagogicalRLWorkflow"
    eval_workflow: str = "examples.pedagogical_rl.workflow.PedagogicalRLWorkflow"
    actor: PedagogicalActorConfig = field(default_factory=PedagogicalActorConfig)
    evaluator: PedagogicalEvaluatorConfig = field(
        default_factory=PedagogicalEvaluatorConfig
    )
    student_model: PedagogicalAPIModelConfig = field(
        default_factory=PedagogicalAPIModelConfig
    )
    judge_model: PedagogicalAPIModelConfig = field(
        default_factory=PedagogicalAPIModelConfig
    )
    generation: PedagogicalGenerationConfig = field(
        default_factory=PedagogicalGenerationConfig
    )
    teacher_pre: PedagogicalTeacherPreConfig = field(
        default_factory=PedagogicalTeacherPreConfig
    )
    # The head-to-head cross. Same dataclass the tutor arm uses, so the two YAML
    # blocks are the same shape and a difference between them is visible by
    # reading them side by side.
    cross_eval: CrossEvalConfig = field(default_factory=CrossEvalConfig)
    debug_trace_dir: str = ""
    debug_trace_every_n_rollouts: int = 10
    max_train_examples: int = 640
    max_eval_examples: int = -1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.debug_trace_every_n_rollouts < 1:
            raise ValueError("debug_trace_every_n_rollouts must be positive")
        if self.max_train_examples == 0 or self.max_train_examples < -1:
            raise ValueError("max_train_examples must be -1 or positive")
        if self.max_eval_examples == 0 or self.max_eval_examples < -1:
            raise ValueError("max_eval_examples must be -1 or positive")
        if self.actor.kl_ctl != 0.0:
            raise ValueError(
                "the aligned PedagogicalRL baseline requires actor.kl_ctl=0"
            )
        if self.critic is not None or self.ref is not None:
            raise ValueError(
                "the aligned PedagogicalRL baseline does not use critic/ref"
            )
        if self.teacher_pre.enabled and self.dynamic_bs:
            raise ValueError(
                "teacher presolve requires dynamic_bs=false to preserve complete "
                "GRPO groups when a verified draft is rejected"
            )
