from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig


@dataclass
class CodeCoachConfig(PPOConfig):
    workflow: str = field(
        default="examples.codecoach.workflow.CodeCoachAgentWorkflow",
        metadata={"help": "Training workflow import path."},
    )
    eval_workflow: str = field(
        default="examples.codecoach.workflow.CodeCoachAgentWorkflow",
        metadata={"help": "Evaluation workflow import path."},
    )
    max_turns: int = field(default=6)
    student_base_url: str = field(default="http://127.0.0.1:30000/v1")
    student_model: str = field(default="qwen-student")
    student_api_key: str = field(default="EMPTY")
    student_timeout: int = field(default=120)
    student_max_tokens: int = field(default=4096)
    student_temperature: float = field(default=0.7)
    student_top_p: float = field(default=1.0)
    max_concurrent_students: int = field(default=8)
    api_params_config_path: str = field(default="")
    api_params_key: str = field(default="")
    work_dir_root: str = field(default="examples/codecoach/artifacts")
