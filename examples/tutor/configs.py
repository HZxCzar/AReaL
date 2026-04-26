from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig
from examples.tutor.prompts import (
    DEFAULT_GENERATOR_SYSTEM_PROMPT,
    DEFAULT_JUDGE_SYSTEM_PROMPT,
    DEFAULT_LEAK_CHECK_SYSTEM_PROMPT,
    DEFAULT_STUDENT_SYSTEM_PROMPT,
    DEFAULT_TEACHER_SYSTEM_PROMPT,
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
    aux_base_url: str = field(default="http://127.0.0.1:30000/v1")
    aux_model: str = field(default="qwen-aux")
    aux_api_key: str = field(default="EMPTY")
    aux_timeout: int = field(default=120)
    aux_max_tokens: int = field(default=4096)
    aux_temperature: float = field(default=0.7)
    aux_top_p: float | None = field(default=None)
    max_concurrent_aux_calls: int = field(default=8)
    api_params_config_path: str = field(default="")
    api_params_key: str = field(default="")
    primary_success_reward: float = field(default=1.0)
    transfer_bonus_reward: float = field(default=0.5)
    token_budget_penalty: float = field(default=-1.0)
    transfer_success_reward: float = field(
        default=1.2,
        metadata={"help": "Deprecated. Use primary_success_reward and transfer_bonus_reward."},
    )
    transfer_fail_reward: float = field(
        default=0.6,
        metadata={"help": "Deprecated. Transfer failure now gives no positive reward."},
    )
    teacher_system_prompt: str = field(default=DEFAULT_TEACHER_SYSTEM_PROMPT)
    student_system_prompt: str = field(default=DEFAULT_STUDENT_SYSTEM_PROMPT)
    judge_system_prompt: str = field(default=DEFAULT_JUDGE_SYSTEM_PROMPT)
    leak_check_system_prompt: str = field(default=DEFAULT_LEAK_CHECK_SYSTEM_PROMPT)
    generator_system_prompt: str = field(default=DEFAULT_GENERATOR_SYSTEM_PROMPT)
    debug_trace_dir: str = field(
        default="",
        metadata={"help": "Optional directory to dump readable per-rollout tutor traces."},
    )
    debug_trace_every_n_rollouts: int = field(
        default=10,
        metadata={"help": "Dump one readable tutor trace every N rollout episodes."},
    )
