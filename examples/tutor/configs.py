import re
from dataclasses import dataclass, field
from typing import Any

from examples.tutor.prompts import (
    DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT,
    DEFAULT_LEAK_CHECK_SYSTEM_PROMPT,
    DEFAULT_STUDENT_SYSTEM_PROMPT,
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    DEFAULT_WORLD_MODEL_SYSTEM_PROMPT,
    TEACHER_STATE_USER_TEMPLATE,
)

from areal.api.cli_args import MISSING, EvaluatorConfig, GRPOConfig

_LEAK_HANDLING_MODES = {"disabled", "reward_only", "terminate", "feedback"}
_DATASET_TYPES = {"aime", "math", "polaris"}
_ANSWER_SCORERS = {"auto", "aime", "math", "polaris"}
_STUDENT_GENERALIZE_MODES = {"only_success", "always"}
_STUDENT_GENERALIZE_SOURCES = {"sidecar", "train"}
_STUDENT_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

TUTOR_EVAL_STUDENT_FIELD = "__tutor_student_name"
TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD = "__tutor_student_prompt_group"
TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD = "__tutor_student_prompt_index"


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
    base_url: str = field(
        default="https://choab9kmmqm8cbcbmqjbeg5jdej8ahaj.openapi-qb-ai.sii.edu.cn/v1"
    )
    model: str = field(default="qwen3-4b")
    api_key: str = field(default="${oc.env:INF_API_KEY}")
    timeout: int = field(default=120)
    max_tokens: int = field(default=2048)
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
    answer_judge_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Use this auxiliary model as an LLM fallback judge when exact "
                "answer matching fails."
            )
        },
    )
    answer_judge_max_tokens: int = field(
        default=256,
        metadata={"help": "Maximum completion tokens for answer judge JSON output."},
    )


@dataclass
class TutorTeacherWarmupPromptConfig:
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to condition teacher rollouts on a full warm-up system "
                "prompt with linearly annealed probability."
            )
        },
    )
    prompt_path: str = field(
        default="",
        metadata={"help": "Path to the full teacher warm-up system prompt."},
    )
    steps: int = field(
        default=0,
        metadata={
            "help": (
                "Policy-version step at which the warm-up prompt probability "
                "reaches zero."
            )
        },
    )

    def __post_init__(self) -> None:
        self.prompt_path = str(self.prompt_path or "").strip()
        self.steps = int(self.steps)
        if self.enabled and not self.prompt_path:
            raise ValueError(
                "prompt_pool.teacher_warmup.prompt_path is required when enabled."
            )
        if self.enabled and self.steps <= 0:
            raise ValueError(
                "prompt_pool.teacher_warmup.steps must be positive when enabled."
            )


@dataclass
class TutorPromptPoolConfig:
    teacher_path: str = field(
        default="",
        metadata={
            "help": (
                "Optional JSON string-array of teacher strategy suffixes sampled "
                "once per training episode."
            )
        },
    )
    student_seen_path: str = field(
        default="",
        metadata={
            "help": (
                "JSON string-array of seen student persona suffixes sampled during "
                "training and, when test_persona is enabled, covered exhaustively "
                "during evaluation."
            )
        },
    )
    student_heldout_path: str = field(
        default="",
        metadata={
            "help": (
                "Optional JSON string-array of held-out student persona suffixes. "
                "These are never sampled during training and, when test_persona is "
                "enabled, are evaluated exhaustively alongside the seen personas."
            )
        },
    )
    include_base: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether training samples the clean base student prompt as an "
                "equal-probability option alongside configured student suffixes."
            )
        },
    )
    test_persona: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether evaluation covers every configured student persona in "
                "addition to the clean base prompt. When disabled, evaluation "
                "uses only the base prompt."
            )
        },
    )
    teacher_warmup: TutorTeacherWarmupPromptConfig = field(
        default_factory=TutorTeacherWarmupPromptConfig
    )

    def __post_init__(self) -> None:
        self.student_seen_path = str(self.student_seen_path or "").strip()
        self.student_heldout_path = str(self.student_heldout_path or "").strip()
        if self.student_heldout_path and not self.student_seen_path:
            raise ValueError(
                "prompt_pool.student_seen_path is required when "
                "prompt_pool.student_heldout_path is set."
            )

    @property
    def student_train_path(self) -> str:
        """Return the only student persona pool eligible for training."""

        return self.student_seen_path

    @property
    def student_eval_paths(self) -> dict[str, str]:
        """Return named persona pools covered exhaustively during evaluation."""

        if not self.test_persona or not self.student_seen_path:
            return {}
        paths = {"seen": self.student_seen_path}
        if self.student_heldout_path:
            paths["heldout"] = self.student_heldout_path
        return paths


@dataclass
class TutorStudentModelConfig:
    name: str = field(
        default=MISSING,
        metadata={
            "help": (
                "Unique student identifier used for rollout selection, traces, and "
                "per-student metrics. Use letters, digits, dots, underscores, or dashes."
            )
        },
    )
    base_url: str = field(
        default=MISSING,
        metadata={"help": "OpenAI-compatible API base URL for this student."},
    )
    model: str = field(
        default=MISSING,
        metadata={"help": "Model name sent to chat.completions.create."},
    )
    weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Non-negative training rollout sampling weight. Weights are "
                "normalized across all configured students."
            )
        },
    )
    api_key: str = field(default="EMPTY")
    timeout: int = field(default=120)
    max_tokens: int = field(default=2048)
    temperature: float = field(default=0.7)
    top_p: float | None = field(default=None)
    max_concurrent_calls: int = field(default=8)
    request_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Additional OpenAI chat.completions.create keyword arguments. "
                "Use extra_body for backend-specific parameters."
            )
        },
    )

    def __post_init__(self) -> None:
        missing = [
            field_name
            for field_name in ("name", "base_url", "model")
            if getattr(self, field_name) is None
            or getattr(self, field_name) is MISSING
            or str(getattr(self, field_name)).strip() in {"", "???"}
        ]
        if missing:
            raise ValueError(
                "student_models entries require non-empty values for: "
                f"{', '.join(missing)}."
            )

        self.name = str(self.name).strip()
        self.base_url = str(self.base_url).strip()
        self.model = str(self.model).strip()
        if not _STUDENT_MODEL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                "student_models.name must start with a letter or digit and contain "
                "only letters, digits, dots, underscores, or dashes."
            )
        if not self.base_url or not self.model:
            raise ValueError("student_models base_url and model must be non-empty.")
        self.weight = float(self.weight)
        if self.weight < 0.0:
            raise ValueError("student_models.weight must be non-negative.")
        self.timeout = int(self.timeout)
        if self.timeout <= 0:
            raise ValueError("student_models.timeout must be positive.")
        self.max_tokens = int(self.max_tokens)
        if self.max_tokens <= 0:
            raise ValueError("student_models.max_tokens must be positive.")
        self.max_concurrent_calls = int(self.max_concurrent_calls)
        if self.max_concurrent_calls <= 0:
            raise ValueError("student_models.max_concurrent_calls must be positive.")


@dataclass
class TutorStudentGeneralizeConfidenceConfig:
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Give a confidence bonus to a correct generalized answer. "
                "The auxiliary API must return token logprobs for the generated "
                "student answer."
            )
        },
    )
    reward_scale: float = field(
        default=0.25,
        metadata={
            "help": (
                "Confidence reward as a fraction of the corresponding level "
                "correctness reward. Must be in (0, 1)."
            )
        },
    )

    def __post_init__(self) -> None:
        if self.enabled and not 0.0 < float(self.reward_scale) < 1.0:
            raise ValueError(
                "student_generalize.confidence.reward_scale must be in (0, 1)."
            )


@dataclass
class TutorStudentGeneralizeConfig:
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Run post-success student generalization tests and add their "
                "correctness as reward components."
            )
        },
    )
    mode: str = field(
        default="only_success",
        metadata={
            "help": (
                "When to run student generalization tests: 'only_success' keeps "
                "the current post-success behavior, while 'always' runs after "
                "every completed tutor rollout except pre-solved and "
                "teacher-pre-solve skipped samples."
            ),
            "choices": ["only_success", "always"],
        },
    )
    source: str = field(
        default="sidecar",
        metadata={
            "help": (
                "Where generalization questions come from: 'sidecar' uses the "
                "configured JSON/metadata level1 and level2 cases; 'train' "
                "builds fixed seeded pairs from the training split at startup."
            ),
            "choices": ["sidecar", "train"],
        },
    )
    path: str = field(
        default="",
        metadata={
            "help": (
                "Optional JSON sidecar keyed by sample id with level1/level2 "
                "generalization tasks."
            )
        },
    )
    level1_reward: float = field(default=0.2)
    level2_reward: float = field(default=0.5)
    confidence: TutorStudentGeneralizeConfidenceConfig = field(
        default_factory=TutorStudentGeneralizeConfidenceConfig
    )

    def __post_init__(self) -> None:
        if self.mode not in _STUDENT_GENERALIZE_MODES:
            raise ValueError(
                "student_generalize.mode must be one of: 'only_success', 'always'."
            )
        if self.source not in _STUDENT_GENERALIZE_SOURCES:
            raise ValueError(
                "student_generalize.source must be one of: 'sidecar', 'train'."
            )
        if self.confidence.enabled and not self.enabled:
            raise ValueError(
                "student_generalize.confidence.enabled=true requires "
                "student_generalize.enabled=true."
            )
        if self.confidence.enabled and (
            float(self.level1_reward) <= 0.0 or float(self.level2_reward) <= 0.0
        ):
            raise ValueError(
                "student_generalize rewards must be positive when confidence reward "
                "is enabled."
            )


@dataclass
class TutorPolarisProcessingConfig:
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "For dataset_type='polaris', derive deterministic train/test "
                "splits plus per-split generalization pools before training."
            )
        },
    )
    output_path: str = field(
        default="",
        metadata={
            "help": (
                "Where to save the derived Polaris dataset. If empty, a path "
                "next to train_dataset.path is generated from seed and ratios."
            )
        },
    )
    train_ratio: float = field(
        default=0.98,
        metadata={"help": "Fraction of source Polaris rows assigned to train."},
    )
    generalize_ratio: float = field(
        default=0.5,
        metadata={
            "help": (
                "Fraction of each train/test split reserved as the corresponding "
                "generalization pool."
            )
        },
    )
    reuse_generalize_tasks: bool = field(
        default=True,
        metadata={
            "help": (
                "Allow multiple main samples to share the same generalization "
                "probe. Recommended when keeping a larger main training set."
            )
        },
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Overwrite an existing derived Polaris dataset."},
    )
    sidecar_filename: str = field(
        default="student_generalize.json",
        metadata={"help": "JSON sidecar filename under the derived dataset path."},
    )

    def __post_init__(self) -> None:
        if not 0.0 < float(self.train_ratio) < 1.0:
            raise ValueError("polaris_processing.train_ratio must be in (0, 1).")
        if not 0.0 < float(self.generalize_ratio) < 1.0:
            raise ValueError("polaris_processing.generalize_ratio must be in (0, 1).")
        if not self.sidecar_filename.strip():
            raise ValueError("polaris_processing.sidecar_filename must be non-empty.")


@dataclass
class TutorTeacherPreConfig:
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Privately ask the teacher to solve the task before tutoring. "
                "Accepted solutions are hidden from the student and appended "
                "to later teacher prompts as a private reference."
            )
        },
    )
    mode: str = field(
        default="filter_solver",
        metadata={
            "help": (
                "Teacher pre-solve prompt mode. 'filter_solver' uses the same "
                "clean solver context as tutor dataset filtering."
            ),
            "choices": ["filter_solver"],
        },
    )
    verify: bool = field(
        default=True,
        metadata={
            "help": (
                "Judge teacher pre-solve drafts and skip the rollout unless one "
                "is correct. When disabled, generate exactly one unverified draft "
                "and continue without calling the answer judge."
            )
        },
    )
    attempts: int = field(
        default=3,
        metadata={
            "help": (
                "Maximum teacher pre-solve attempts when verification is enabled. "
                "If no attempt is correct, the sample is skipped for this rollout."
            )
        },
    )
    max_tokens: int = field(
        default=0,
        metadata={
            "help": (
                "Maximum completion tokens for teacher pre-solve. Non-positive "
                "values reuse the tutor rollout max_new_tokens."
            )
        },
    )

    def __post_init__(self) -> None:
        if self.mode != "filter_solver":
            raise ValueError("teacher_pre.mode must be 'filter_solver'.")
        if int(self.attempts) < 1:
            raise ValueError("teacher_pre.attempts must be >= 1.")


@dataclass
class TutorEvaluatorConfig(EvaluatorConfig):
    max_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of validation samples to evaluate. "
                "None or non-positive values evaluate the full validation set."
            )
        },
    )
    average_rollouts: int = field(
        default=3,
        metadata={
            "help": (
                "Number of independent tutor rollout episodes to run for each "
                "validation sample. Metrics are averaged over rollout attempts, and "
                "per-task correctness stability is reported under eval-rollout/repeat."
            )
        },
    )
    student_prompt_average_rollouts: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of rollout episodes for each additional seen or held-out "
                "student instruction prompt during evaluation. None reuses "
                "average_rollouts."
            )
        },
    )
    student_model_names: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional subset of student_models to evaluate. None evaluates every "
                "configured student model."
            )
        },
    )

    def __post_init__(self) -> None:
        self.average_rollouts = int(self.average_rollouts)
        if self.average_rollouts < 1:
            raise ValueError("evaluator.average_rollouts must be >= 1.")
        if self.student_prompt_average_rollouts is not None:
            self.student_prompt_average_rollouts = int(
                self.student_prompt_average_rollouts
            )
            if self.student_prompt_average_rollouts < 1:
                raise ValueError(
                    "evaluator.student_prompt_average_rollouts must be >= 1."
                )
        if self.student_model_names is None:
            return

        self.student_model_names = [
            str(name).strip() for name in self.student_model_names
        ]
        if not self.student_model_names or any(
            not name for name in self.student_model_names
        ):
            raise ValueError(
                "evaluator.student_model_names must contain non-empty names."
            )
        if len(self.student_model_names) != len(set(self.student_model_names)):
            raise ValueError("evaluator.student_model_names must be unique.")


@dataclass
class TutorTeacherDiversityRewardConfig:
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Penalize semantic similarity between consecutive visible teacher "
                "outputs relative to the current training-batch mean. The local "
                "auxiliary advantage is attached only to the later teacher turn."
            )
        },
    )
    weight: float = field(
        default=0.1,
        metadata={
            "help": (
                "Coefficient for the local auxiliary advantage applied when a "
                "teacher pair is more similar than the current training-batch mean."
            )
        },
    )
    embedding_model_path: str = field(
        default="",
        metadata={
            "help": (
                "Local Hugging Face model directory used to embed teacher outputs. "
                "The model is loaded offline and shared within each rollout process."
            )
        },
    )
    embedding_device: str = field(
        default="cuda",
        metadata={
            "help": (
                "Device for the local embedding model. Use cuda on rollout nodes "
                "with sufficient GPU memory."
            )
        },
    )
    embedding_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Floating-point dtype for the local embedding model."},
    )
    embedding_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum token length for each teacher output embedding."},
    )
    embedding_batch_wait_ms: float = field(
        default=2.0,
        metadata={
            "help": (
                "Time window used to merge concurrent trajectory embedding requests."
            )
        },
    )
    embedding_max_batch_texts: int = field(
        default=64,
        metadata={"help": "Maximum number of texts in one embedding microbatch."},
    )
    embedding_max_batch_tokens: int = field(
        default=32768,
        metadata={
            "help": (
                "Maximum padded token count in one embedding microbatch. Long texts "
                "are automatically placed in smaller microbatches."
            )
        },
    )

    def __post_init__(self) -> None:
        self.weight = float(self.weight)
        self.embedding_model_path = str(self.embedding_model_path or "").strip()
        self.embedding_device = str(self.embedding_device or "").strip()
        self.embedding_dtype = str(self.embedding_dtype or "").strip().lower()
        self.embedding_max_length = int(self.embedding_max_length)
        self.embedding_batch_wait_ms = float(self.embedding_batch_wait_ms)
        self.embedding_max_batch_texts = int(self.embedding_max_batch_texts)
        self.embedding_max_batch_tokens = int(self.embedding_max_batch_tokens)
        if self.enabled and self.weight <= 0.0:
            raise ValueError("reward.teacher_diversity.weight must be positive.")
        if self.enabled and not self.embedding_model_path:
            raise ValueError(
                "reward.teacher_diversity.embedding_model_path is required when enabled."
            )
        if self.enabled and not self.embedding_device:
            raise ValueError(
                "reward.teacher_diversity.embedding_device is required when enabled."
            )
        if self.embedding_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "reward.teacher_diversity.embedding_dtype must be float32, "
                "float16, or bfloat16."
            )
        if self.embedding_max_length <= 0:
            raise ValueError(
                "reward.teacher_diversity.embedding_max_length must be positive."
            )
        if self.embedding_batch_wait_ms < 0.0:
            raise ValueError(
                "reward.teacher_diversity.embedding_batch_wait_ms must be non-negative."
            )
        if self.embedding_max_batch_texts <= 0:
            raise ValueError(
                "reward.teacher_diversity.embedding_max_batch_texts must be positive."
            )
        if self.embedding_max_batch_tokens <= 0:
            raise ValueError(
                "reward.teacher_diversity.embedding_max_batch_tokens must be positive."
            )


@dataclass
class TutorTeacherContextRewardConfig:
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Reward a later teacher turn when its sampled output is more likely "
                "at its real position than when moved to the preceding teacher "
                "turn's position. The length-normalized, batch-centered signal can "
                "be logged alone or attached to the later turn's local advantage."
            )
        },
    )
    apply_to_advantage: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to add the computed context signal to the training "
                "advantage. Disable this to compute and log the signal only."
            )
        },
    )
    weight: float = field(
        default=0.1,
        metadata={
            "help": (
                "Coefficient for the local context-awareness advantage after "
                "subtracting the valid training-batch mean information gain."
            )
        },
    )
    score_clip: float = field(
        default=5.0,
        metadata={
            "help": (
                "Symmetric clip, in nats per output token, applied to the real-minus-"
                "moved log-probability gap before batch centering."
            )
        },
    )

    def __post_init__(self) -> None:
        self.weight = float(self.weight)
        self.score_clip = float(self.score_clip)
        if self.enabled and self.weight <= 0.0:
            raise ValueError("reward.teacher_context.weight must be positive.")
        if self.enabled and self.score_clip <= 0.0:
            raise ValueError("reward.teacher_context.score_clip must be positive.")


@dataclass
class TutorTeacherProgressJudgeConfig:
    """Give a teacher turn local advantage for observed student improvement."""

    enabled: bool = field(default=False)
    weight: float = field(
        default=0.5,
        metadata={
            "help": (
                "Magnitude of the local turn advantage. Judge scores 0/1/2 map "
                "to -weight/0/+weight."
            )
        },
    )

    def __post_init__(self) -> None:
        self.weight = float(self.weight)
        if self.enabled and self.weight <= 0.0:
            raise ValueError("reward.teacher_progress_judge.weight must be positive.")


@dataclass
class TutorSuccessTurnShapingConfig:
    """Shape clean success reward linearly by the turn where success occurs."""

    enabled: bool = field(default=False)
    min_reward: float = field(
        default=1.0,
        metadata={"help": "Success reward at the final allowed teacher turn."},
    )
    max_reward: float = field(
        default=1.0,
        metadata={"help": "Success reward at the first teacher turn."},
    )

    def __post_init__(self) -> None:
        self.min_reward = float(self.min_reward)
        self.max_reward = float(self.max_reward)
        if self.enabled and self.min_reward < 0.0:
            raise ValueError("reward.success_turn_shaping.min_reward must be >= 0.")
        if self.enabled and self.max_reward < self.min_reward:
            raise ValueError(
                "reward.success_turn_shaping.max_reward must be >= min_reward."
            )


@dataclass
class TutorRewardConfig:
    success: float = field(default=1.0)
    leaked_success_reward_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiplier applied to success reward when any tutor turn in "
                "the episode leaked. Use 0.0 to give no success reward to "
                "leaked-success episodes."
            )
        },
    )
    leak_penalty_mode: str = field(
        default="binary",
        metadata={
            "help": "Leak penalty mode: 'binary' uses leaked=true/false, "
            "'staged' uses leak levels 1-4, and 'rawbase' asks the "
            "auxiliary judge whether the public tutor output contains the "
            "final answer or an equivalent numeric expression.",
            "choices": ["binary", "staged", "rawbase"],
        },
    )
    leak_penalty: float | None = field(default=-1.0)
    leak_penalty_final_answer: float | None = field(default=None)
    leak_penalty_compute: float | None = field(default=None)
    leak_penalty_formula: float | None = field(default=None)
    leak_penalty_aggregation: str = field(
        default="turn",
        metadata={
            "help": (
                "Leak penalty aggregation: 'turn' applies the configured "
                "penalty to every leaked turn; 'episode' applies one "
                "saturated penalty per episode."
            ),
            "choices": ["turn", "episode"],
        },
    )
    format_error_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Per-turn penalty for malformed tagged teacher output in "
                "non-thinking mode. Ignored when teacher thinking is enabled."
            )
        },
    )
    assign_success_reward: bool = field(default=False)
    outcome_prior_turn_weight: float = field(default=0.1)
    outcome_credit_gamma: float = field(default=0.9)
    early_success_bonus: float = field(default=0.3)
    success_turn_shaping: TutorSuccessTurnShapingConfig = field(
        default_factory=TutorSuccessTurnShapingConfig
    )
    max_turn_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Terminal reward applied once to the final teacher turn when an "
                "episode ends at max_turns without success."
            )
        },
    )
    enable_turn_penalty: bool = field(default=False)
    turn_penalty: float = field(default=-0.01)
    length_penalty_threshold_chars: int = field(default=1200)
    length_penalty_per_100_chars: float = field(default=-0.005)
    length_penalty_min: float = field(default=-0.1)
    zero_reward_on_length_stop: bool = field(
        default=False,
        metadata={
            "help": (
                "Set a tutor turn's training reward to zero when inference "
                "stops because it reached the generation length limit."
            )
        },
    )
    teacher_diversity: TutorTeacherDiversityRewardConfig = field(
        default_factory=TutorTeacherDiversityRewardConfig
    )
    teacher_context: TutorTeacherContextRewardConfig = field(
        default_factory=TutorTeacherContextRewardConfig
    )
    teacher_progress_judge: TutorTeacherProgressJudgeConfig = field(
        default_factory=TutorTeacherProgressJudgeConfig
    )

    def __post_init__(self) -> None:
        if self.format_error_penalty > 0.0:
            raise ValueError("reward.format_error_penalty must be <= 0.")
        if self.leak_penalty_mode not in {"binary", "staged", "rawbase"}:
            raise ValueError(
                "reward.leak_penalty_mode must be one of: "
                "'binary', 'staged', 'rawbase'."
            )
        if self.leaked_success_reward_scale < 0.0:
            raise ValueError("reward.leaked_success_reward_scale must be >= 0.")
        if self.leak_penalty_aggregation not in {"turn", "episode"}:
            raise ValueError(
                "reward.leak_penalty_aggregation must be one of: 'turn', 'episode'."
            )
        if self.leak_penalty_mode in {"binary", "rawbase"}:
            if self.leak_penalty is None:
                raise ValueError(
                    "reward.leak_penalty must be set when "
                    f"reward.leak_penalty_mode={self.leak_penalty_mode!r}."
                )
            return

        missing = [
            name
            for name, value in {
                "reward.leak_penalty_final_answer": self.leak_penalty_final_answer,
                "reward.leak_penalty_compute": self.leak_penalty_compute,
                "reward.leak_penalty_formula": self.leak_penalty_formula,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "staged leak penalty mode requires explicit values for "
                f"{', '.join(missing)}."
            )


@dataclass
class TutorPaWConfig:
    """Optional PaW-style filtering and robust World Model objective."""

    enabled: bool = field(default=False)
    entropy_filter_enabled: bool = field(default=True)
    entropy_keep_ratio: float = field(default=0.75)
    cmae_enabled: bool = field(default=True)
    confidence_threshold: float = field(default=0.2)
    reward_adaptive_enabled: bool = field(default=False)
    max_episode_return: float = field(default=1.0)

    def __post_init__(self) -> None:
        self.entropy_keep_ratio = float(self.entropy_keep_ratio)
        self.confidence_threshold = float(self.confidence_threshold)
        self.max_episode_return = float(self.max_episode_return)
        if not 0.0 < self.entropy_keep_ratio <= 1.0:
            raise ValueError("world_model.paw.entropy_keep_ratio must be in (0, 1].")
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError("world_model.paw.confidence_threshold must be in (0, 1).")
        if self.max_episode_return <= 0.0:
            raise ValueError("world_model.paw.max_episode_return must be positive.")


@dataclass
class TutorWorldModelRLReweightConfig:
    """Optional sign-aware PPO advantage scaling from Student prediction NLL."""

    enabled: bool = field(default=False)
    mode: str = field(
        default="continuous",
        metadata={"choices": ["continuous", "threshold"]},
    )
    positive_strength: float = field(default=1.0)
    negative_strength: float = field(default=1.0)
    min_weight: float = field(default=0.5)
    max_weight: float = field(default=2.0)
    continuous_temperature: float = field(default=1.0)
    low_threshold: float = field(default=-0.5)
    high_threshold: float = field(default=0.5)
    min_turn_std: float = field(default=1e-6)

    def __post_init__(self) -> None:
        self.positive_strength = float(self.positive_strength)
        self.negative_strength = float(self.negative_strength)
        self.min_weight = float(self.min_weight)
        self.max_weight = float(self.max_weight)
        self.continuous_temperature = float(self.continuous_temperature)
        self.low_threshold = float(self.low_threshold)
        self.high_threshold = float(self.high_threshold)
        self.min_turn_std = float(self.min_turn_std)
        if self.mode not in {"continuous", "threshold"}:
            raise ValueError(
                "world_model.rl_reweight.mode must be 'continuous' or 'threshold'."
            )
        if self.positive_strength < 0.0 or self.negative_strength < 0.0:
            raise ValueError("world_model.rl_reweight strengths must be non-negative.")
        if not 0.0 < self.min_weight <= 1.0:
            raise ValueError("world_model.rl_reweight.min_weight must be in (0, 1].")
        if self.max_weight < 1.0 or self.max_weight < self.min_weight:
            raise ValueError(
                "world_model.rl_reweight.max_weight must be >= 1 and min_weight."
            )
        if self.continuous_temperature <= 0.0:
            raise ValueError(
                "world_model.rl_reweight.continuous_temperature must be positive."
            )
        if self.low_threshold >= self.high_threshold:
            raise ValueError(
                "world_model.rl_reweight.low_threshold must be smaller than "
                "high_threshold."
            )
        if self.min_turn_std <= 0.0:
            raise ValueError("world_model.rl_reweight.min_turn_std must be positive.")


@dataclass
class TutorWorldModelConfig:
    """Auxiliary supervised objective for predicting the next Student reply."""

    enabled: bool = field(default=False)
    separate_lora_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Train the World Model objective with a separate LoRA adapter. "
                "Disabled by default to preserve joint PPO/World Model training."
            )
        },
    )
    loss_weight: float = field(
        default=0.05,
        metadata={
            "help": (
                "Weight of the response-balanced Student prediction auxiliary "
                "loss added to the PPO actor loss."
            )
        },
    )
    system_prompt: str = field(default=DEFAULT_WORLD_MODEL_SYSTEM_PROMPT)
    log_per_turn_nll: bool = field(
        default=False,
        metadata={
            "help": (
                "Log the current actor's mean token NLL for every valid World "
                "Model turn before the PPO update. This adds one extra WM forward "
                "pass on logging steps."
            )
        },
    )
    per_turn_nll_log_every_n_steps: int = field(
        default=1,
        metadata={"help": "Log per-turn World Model NLL every N training steps."},
    )
    paw: TutorPaWConfig = field(default_factory=TutorPaWConfig)
    rl_reweight: TutorWorldModelRLReweightConfig = field(
        default_factory=TutorWorldModelRLReweightConfig
    )

    def __post_init__(self) -> None:
        self.loss_weight = float(self.loss_weight)
        self.system_prompt = str(self.system_prompt or "").strip()
        self.per_turn_nll_log_every_n_steps = int(self.per_turn_nll_log_every_n_steps)
        if self.enabled and self.loss_weight <= 0.0:
            raise ValueError("world_model.loss_weight must be positive when enabled.")
        if self.enabled and not self.system_prompt:
            raise ValueError("world_model.system_prompt is required when enabled.")
        if self.separate_lora_enabled and not self.enabled:
            raise ValueError(
                "world_model.enabled must be true when "
                "world_model.separate_lora_enabled."
            )
        if self.per_turn_nll_log_every_n_steps < 1:
            raise ValueError(
                "world_model.per_turn_nll_log_every_n_steps must be at least 1."
            )
        if self.paw.enabled and not self.enabled:
            raise ValueError(
                "world_model.enabled must be true when world_model.paw.enabled."
            )
        if self.rl_reweight.enabled and not self.enabled:
            raise ValueError(
                "world_model.enabled must be true when world_model.rl_reweight.enabled."
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
    dataset_type: str = field(
        default=MISSING,
        metadata={
            "help": "Tutor dataset source type.",
            "choices": ["aime", "math", "polaris"],
        },
    )
    answer_scorer: str = field(
        default="auto",
        metadata={
            "help": "Answer scorer used by tutor workflow. 'auto' follows dataset_type.",
            "choices": ["auto", "aime", "math", "polaris"],
        },
    )
    max_turns: int = field(default=6, metadata={"help": "Maximum teacher turns."})
    enable_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable thinking mode for the tutor rollout model."
        },
    )
    leak_handling_mode: str = field(
        default="reward_only",
        metadata={
            "help": (
                "Leak handling mode: 'disabled' skips leak checks; "
                "'reward_only' checks after rollout and applies reward penalties; "
                "'terminate' stops before the student sees leaked tutor output; "
                "'feedback' calls the student, invalidates leaked turns, and "
                "feeds persistent private leak feedback to later tutor turns."
            ),
            "choices": ["disabled", "reward_only", "terminate", "feedback"],
        },
    )
    teacher_show_ground_truth: bool = field(
        default=False,
        metadata={"help": "Whether teacher prompts include the ground-truth answer."},
    )
    prompt_pool: TutorPromptPoolConfig = field(default_factory=TutorPromptPoolConfig)
    teacher_pre: TutorTeacherPreConfig = field(default_factory=TutorTeacherPreConfig)
    auxiliary_model: TutorAuxiliaryModelConfig = field(
        default_factory=TutorAuxiliaryModelConfig
    )
    student_models: list[TutorStudentModelConfig] = field(
        default_factory=list,
        metadata={
            "help": (
                "Optional pool of API students sampled once per training episode. "
                "An empty list preserves the legacy auxiliary_model student behavior."
            )
        },
    )
    student_generalize: TutorStudentGeneralizeConfig = field(
        default_factory=TutorStudentGeneralizeConfig
    )
    polaris_processing: TutorPolarisProcessingConfig = field(
        default_factory=TutorPolarisProcessingConfig
    )
    evaluator: TutorEvaluatorConfig = field(default_factory=TutorEvaluatorConfig)
    reward: TutorRewardConfig = field(default_factory=TutorRewardConfig)
    world_model: TutorWorldModelConfig = field(default_factory=TutorWorldModelConfig)
    teacher_system_prompt: str = field(default=DEFAULT_TEACHER_SYSTEM_PROMPT)
    teacher_anti_leak_instruction_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to append an instruction forbidding the teacher from "
                "revealing the answer to the student."
            )
        },
    )
    teacher_adaptive_instruction_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to tell the teacher to adapt to the student's latest "
                "responses instead of following or repeating a fixed approach."
            )
        },
    )
    teacher_user_prompt_template: str = field(default=TEACHER_STATE_USER_TEMPLATE)
    student_system_prompt: str = field(default=DEFAULT_STUDENT_SYSTEM_PROMPT)
    leak_check_system_prompt: str = field(default=DEFAULT_LEAK_CHECK_SYSTEM_PROMPT)
    answer_judge_system_prompt: str = field(default=DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT)
    debug_trace_dir: str = field(
        default="",
        metadata={
            "help": "Optional directory to dump readable per-rollout tutor traces."
        },
    )
    debug_trace_every_n_rollouts: int = field(
        default=10,
        metadata={"help": "Dump one readable tutor trace every N rollout episodes."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.world_model.separate_lora_enabled:
            if not self.actor.backend.startswith("fsdp:"):
                raise ValueError(
                    "world_model.separate_lora_enabled currently requires an "
                    "FSDP actor backend."
                )
            if self.actor._version != "v1":
                raise ValueError(
                    "world_model.separate_lora_enabled currently requires the v1 "
                    "training controller."
                )
        if self.actor.mask_no_eos_with_zero:
            raise ValueError(
                "Tutor does not support actor.mask_no_eos_with_zero because its "
                "turn-level tensors are dynamically padded. Use "
                "reward.zero_reward_on_length_stop instead."
            )
        student_names = [student.name for student in self.student_models]
        if len(student_names) != len(set(student_names)):
            raise ValueError("student_models names must be unique.")
        eval_student_names = self.evaluator.student_model_names
        if eval_student_names is not None:
            unknown_eval_students = sorted(set(eval_student_names) - set(student_names))
            if unknown_eval_students:
                raise ValueError(
                    "evaluator.student_model_names must reference configured "
                    f"student_models; unknown names: {unknown_eval_students}."
                )
        if self.student_models and not any(
            student.weight > 0.0 for student in self.student_models
        ):
            raise ValueError(
                "student_models must contain at least one student with positive weight."
            )
        if self.dataset_type is MISSING or str(self.dataset_type) == "???":
            raise ValueError("dataset_type must be one of: 'aime', 'math', 'polaris'.")
        self.dataset_type = str(self.dataset_type).strip().lower()
        if self.dataset_type not in _DATASET_TYPES:
            raise ValueError("dataset_type must be one of: 'aime', 'math', 'polaris'.")
        self.answer_scorer = str(self.answer_scorer).strip().lower()
        if self.answer_scorer not in _ANSWER_SCORERS:
            raise ValueError(
                "answer_scorer must be one of: 'auto', 'aime', 'math', 'polaris'."
            )
        if self.answer_scorer == "auto":
            self.answer_scorer = self.dataset_type
        elif self.answer_scorer != self.dataset_type:
            raise ValueError(
                "answer_scorer must be 'auto' or match dataset_type; "
                f"got dataset_type={self.dataset_type!r}, "
                f"answer_scorer={self.answer_scorer!r}."
            )
        if self.leak_handling_mode not in _LEAK_HANDLING_MODES:
            raise ValueError(
                "leak_handling_mode must be one of: 'disabled', "
                "'reward_only', 'terminate', or 'feedback'."
            )
        if self.dataset_type == "polaris":
            if self.leak_handling_mode != "disabled":
                raise ValueError(
                    "dataset_type='polaris' is incompatible with leak checks; "
                    "set leak_handling_mode='disabled'."
                )
            if self.auxiliary_model.answer_judge_enabled:
                raise ValueError(
                    "dataset_type='polaris' uses the Polaris rule judge and is "
                    "incompatible with auxiliary_model.answer_judge_enabled=true."
                )
        elif self.polaris_processing.enabled:
            raise ValueError(
                "polaris_processing.enabled=true requires dataset_type='polaris'."
            )
        if self.student_generalize.source == "train" and self.dataset_type != "math":
            raise ValueError(
                "student_generalize.source='train' currently requires "
                "dataset_type='math'."
            )
