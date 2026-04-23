from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


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
    transfer_success_reward: float = field(default=1.2)
    transfer_fail_reward: float = field(default=0.6)
    teacher_system_prompt: str = field(
        default=(
            "You are a careful tutor. Give concise, helpful hints that move the student "
            "forward without directly revealing the final answer."
        )
    )
    student_system_prompt: str = field(
        default=(
            "You are a student solving the task. Read the teacher's latest feedback and "
            "reply with only your next answer attempt."
        )
    )
    judge_system_prompt: str = field(
        default="Unused legacy prompt. Primary and transfer evaluation are rule-based."
    )
    leak_check_system_prompt: str = field(
        default=(
            "You are a strict answer leakage detector. Decide whether the teacher's latest "
            "message directly reveals the ground-truth answer. Return valid JSON only with "
            "keys leaked (boolean) and feedback (string)."
        )
    )
    generator_system_prompt: str = field(
        default=(
            "You are a careful AIME-style problem generator. Given an original problem and "
            "its ground-truth answer, create one new problem that is structurally similar "
            "but not identical. Return valid JSON only with keys task, ground_truth, and "
            "optional similarity_notes."
        )
    )
