import re
from dataclasses import dataclass, field, replace
from typing import Any

# The head-to-head cross is configured from one place for both arms; see the
# module docstring for why it is not defined here.
from examples.pedagogical_rl.cross_eval_config import CrossEvalConfig
from examples.tutor.prompts import (
    DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT,
    DEFAULT_LEAK_CHECK_SYSTEM_PROMPT,
    DEFAULT_STUDENT_SYSTEM_PROMPT,
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    DEFAULT_WORLD_MODEL_SYSTEM_PROMPT,
    TEACHER_STATE_USER_TEMPLATE,
)

from areal.api.cli_args import (
    MISSING,
    EvaluatorConfig,
    GRPOConfig,
    PPOActorConfig,
)

_LEAK_HANDLING_MODES = {"disabled", "reward_only", "terminate"}
_FORMAT_HANDLING_MODES = {"continue", "terminate"}
_DATASET_TYPES = {"aime", "math", "polaris"}
_ANSWER_SCORERS = {"auto", "aime", "math", "polaris"}
_STUDENT_GENERALIZE_MODES = {"only_success", "always"}
_STUDENT_GENERALIZE_SOURCES = {"generated", "sidecar", "train"}
_STUDENT_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

STUDENT_MODE_TEXT = "text"
STUDENT_MODE_CODE = "code"
# The action space a student is allowed. See TutorStudentModelConfig.mode.
STUDENT_MODES = frozenset({STUDENT_MODE_TEXT, STUDENT_MODE_CODE})

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
class TutorStudentTurnBehaviorConfig:
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether training samples a fresh weighted behavior instruction "
                "before every student response. Evaluation always disables this "
                "feature."
            )
        },
    )
    path: str = field(
        default="",
        metadata={
            "help": (
                "JSON array of weighted turn-local student behaviors. Entries "
                "contain name, probability, and instruction; an empty instruction "
                "represents the clean base behavior."
            )
        },
    )
    separate_call_behavior_names: list[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Behavior names generated by a separate student-model call after "
                "an unsuccessful normal response. All other behavior instructions "
                "remain inline suffixes on the normal student prompt."
            )
        },
    )

    def __post_init__(self) -> None:
        self.path = str(self.path or "").strip()
        self.separate_call_behavior_names = [
            str(name).strip()
            for name in self.separate_call_behavior_names
            if str(name).strip()
        ]
        if self.enabled and not self.path:
            raise ValueError(
                "prompt_pool.student_turn_behavior.path is required when enabled."
            )
        if self.separate_call_behavior_names and not self.enabled:
            raise ValueError(
                "prompt_pool.student_turn_behavior.separate_call_behavior_names "
                "requires student turn behaviors to be enabled."
            )
        if len(self.separate_call_behavior_names) != len(
            set(self.separate_call_behavior_names)
        ):
            raise ValueError(
                "prompt_pool.student_turn_behavior.separate_call_behavior_names "
                "must be unique."
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
    student_turn_behavior: TutorStudentTurnBehaviorConfig = field(
        default_factory=TutorStudentTurnBehaviorConfig
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
class TutorStudentMaskConfig:
    """What this student is allowed to see of the dialogue it is having.

    The mechanism behind heterogeneous students that need no fine-tuning: the
    same frozen model behaves like a different learner because a different part
    of the transcript reaches it. See core/attention_mask.py for why this is
    preferred over a persona (which this student ignores) and over an SFT
    profile (which has to be defended as a modelling choice).

    Applied to the HISTORY only. The turn being answered is always visible, so
    the mask is a memory limit rather than deafness.
    """

    mode: str = field(
        default="full",
        metadata={
            "help": (
                "full: nothing hidden. teacher_fade: the teacher's earlier turns "
                "are dropped, so only what the student said itself survives and "
                "teaching persists only if the student was made to produce it. "
                "student_fade: the student's own earlier turns are dropped, so "
                "each teacher message must stand alone. long_drop: teacher text "
                "past long_drop_words is never read, in history AND in the turn "
                "being answered, so only short messages land at all. "
                "The two fades are MEMORY limits and touch history only; "
                "long_drop is an ATTENTION limit and also truncates the live turn."
            )
        },
    )
    keep_recent: int = field(
        default=0,
        metadata={
            "help": (
                "For the fade modes: how many of the faded role's most recent "
                "turns survive. 0 forgets all of them; 1 remembers only the last. "
                "A value at or above the turn count makes the mask a no-op. Note "
                "that with the live turn always visible, teacher_fade at 1 lets "
                "the student effectively retain two teacher turns."
            )
        },
    )
    long_drop_words: int = field(
        default=75,
        metadata={
            "help": (
                "long_drop only: the word count past which the student stops "
                "reading a teacher message. 75 is set so a short teaching turn "
                "(under ~70 words) arrives intact while a 150-250 word "
                "explanation loses roughly two thirds -- the cut is exactly "
                "'longer than a brief teacher writes'."
            )
        },
    )
    placeholder: bool = field(
        default=True,
        metadata={
            "help": (
                "Replace masked content with a short marker instead of deleting "
                "it. A faded turn keeps its role and says its content is no "
                "longer remembered; a truncated message ends in an ellipsis. "
                "This preserves strict user/assistant alternation -- deletion "
                "leaves runs of same-role messages that some chat templates "
                "merge or reject -- and it keeps the next turn's references "
                "coherent, so a reply reads as forgetful rather than confused. "
                "Set false for hard deletion; the mask label records which, "
                "because two runs differing only in this are not comparable."
            )
        },
    )


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
    mode: str = field(
        default="text",
        metadata={
            "help": (
                "How this student is allowed to act. 'text' replies in prose and "
                "algebra, as before. 'code' is a CodeAct student: every reply is "
                "one Python program, it is run in a persistent notebook-style "
                "session, and what it produced is the only thing the student can "
                "say -- the teacher sees the program and the result. The re-test "
                "is a program whose output is judged as the answer. Two entries "
                "on the SAME underlying model differing only in mode give a "
                "student pair whose demand differs by action space rather than by "
                "capability, which is the point: prompting cannot hold the code "
                "channel across a conversation (compliance collapses after the "
                "first turn), so it is enforced structurally instead."
            )
        },
    )
    api_key: str = field(default="EMPTY")
    timeout: int = field(default=120)
    max_tokens: int = field(default=2048)
    temperature: float = field(default=0.7)
    top_p: float | None = field(default=None)
    max_concurrent_calls: int = field(default=8)
    # Two entries may share base_url and model and differ only here: that is the
    # intended way to define several students over one served endpoint, and it is
    # why `name` rather than `model` keys the per-student metrics.
    mask: TutorStudentMaskConfig = field(default_factory=TutorStudentMaskConfig)
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
        self.mode = str(self.mode).strip().lower()
        if self.mode not in STUDENT_MODES:
            raise ValueError(
                "student_models.mode must be one of "
                f"{sorted(STUDENT_MODES)}, got {self.mode!r}."
            )
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
class TutorStudentAxesConfig:
    """One endpoint, expanded into the (behavior, information) students over it.

    A student in this rollout is a PAIR: behavior is its action space (text or
    code) and information is what it may see of the dialogue (a mask). The two are
    independent, and both already live on a student_models entry, so the pool can
    always be written out by hand -- 2 behaviors x 3 informations is six entries.

    WHY THIS EXISTS ANYWAY. Those six entries differ only in `name`, `mode` and
    `mask`; the endpoint, model, key, timeout and sampling parameters are identical
    and repeated six times. omegaconf REPLACES a list on merge rather than merging
    element-wise, so an arm that overrides student_models restates every entry, and
    a changed endpoint has to be edited in all six or the pool silently splits
    across two services. Declaring the axes instead keeps one copy of the endpoint
    and makes the product a property of the config rather than of the typing.

    Expansion happens in TutorConfig.__post_init__, before every student_models
    check runs, so an expanded pool is validated exactly like a written-out one and
    nothing downstream can tell the difference.

    Names are f"{template.name}-{behavior}-{information}". They have to be distinct
    because the evaluator duplicates the validation set once per student NAME, which
    is also how eval reports the axes separately.
    """

    behaviors: list[str] = field(
        default_factory=lambda: ["text"],
        metadata={
            "help": (
                "The behavior axis: which action spaces to build over this "
                "endpoint. 'text' replies in prose, 'code' is a CodeAct student "
                "whose every reply is one program run in a persistent session."
            )
        },
    )
    informations: dict[str, TutorStudentMaskConfig] = field(
        default_factory=lambda: {"original": TutorStudentMaskConfig()},
        metadata={
            "help": (
                "The information axis, as name -> mask. The name goes into the "
                "student name, so call the unmasked one 'original' rather than "
                "'full'. Default is a single unmasked entry, which makes this "
                "block degrade to plain behavior expansion."
            )
        },
    )
    # MISSING, not a default_factory: omegaconf builds the schema for a
    # list[TutorStudentAxesConfig] by CALLING the factory, and
    # TutorStudentModelConfig refuses to construct without a name, base_url and
    # model -- so a factory here makes the whole config class unloadable.
    template: TutorStudentModelConfig = field(
        default=MISSING,
        metadata={
            "help": (
                "The endpoint every combination shares. Its `weight` is the TOTAL "
                "mass for this pool and is divided evenly across the combinations, "
                "so one axes block with weight 1.0 behaves like a single student "
                "with weight 1.0 however many cells it expands to. Its `name` is "
                "the stem; its `mode` and `mask` are ignored, because the axes set "
                "them."
            )
        },
    )

    def expand(self) -> list[TutorStudentModelConfig]:
        """The student_models entries this block stands for."""
        behaviors = [str(b).strip() for b in (self.behaviors or []) if str(b).strip()]
        if not behaviors:
            raise ValueError("student_axes.behaviors must not be empty.")
        if len(behaviors) != len(set(behaviors)):
            raise ValueError(f"student_axes.behaviors has duplicates: {behaviors}.")
        informations = dict(self.informations or {})
        if not informations:
            raise ValueError("student_axes.informations must not be empty.")
        if self.template is MISSING or self.template is None:
            raise ValueError("student_axes.template is required.")
        template = self.template
        if not isinstance(template, TutorStudentModelConfig):
            template = TutorStudentModelConfig(**dict(template))
        stem = str(template.name or "").strip()
        if not stem:
            raise ValueError("student_axes.template.name must be non-empty.")
        cells = len(behaviors) * len(informations)
        share = float(template.weight) / cells
        expanded: list[TutorStudentModelConfig] = []
        for behavior in behaviors:
            for label, mask in informations.items():
                label = str(label).strip()
                if not label:
                    raise ValueError("student_axes.informations keys must be non-empty.")
                entry = replace(
                    template,
                    name=f"{stem}-{behavior}-{label}",
                    mode=behavior,
                    weight=share,
                    mask=mask if isinstance(mask, TutorStudentMaskConfig)
                    else TutorStudentMaskConfig(**dict(mask)),
                )
                expanded.append(entry)
        return expanded


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
                "builds fixed seeded pairs from the training split at startup; "
                "'generated' reads verified V1/V2 triples and filters train/test "
                "to source ids with complete generated variants."
            ),
            "choices": ["sidecar", "train", "generated"],
        },
    )
    path: str = field(
        default="",
        metadata={
            "help": (
                "JSON input for generalization tasks. Sidecar inputs are keyed by "
                "sample id with level1/level2 cases; generated inputs contain a "
                "top-level triples list with source_id, variant1, and variant2."
            )
        },
    )
    level1_reward: float = field(default=0.2)
    level2_reward: float = field(default=0.5)
    turn_credit: bool = field(
        default=False,
        metadata={
            "help": (
                "Pay each turn for the re-test gain it produced instead of "
                "putting the whole episode reward on the last turn. After the "
                "dialogue we cut the history after the student's reply to turn "
                "t and re-run the solo re-test, giving S(t); turn t earns "
                "retest_reward * (S(t) - S(t-1)), with S(0) the no-teaching "
                "baseline. The marginals telescope to the old episode reward, "
                "so the total is unchanged and only credit assignment differs. "
                "Costs (turns - 1) * turn_credit_replays extra student calls "
                "per episode. Requires free_chat.no_teaching_baseline."
            )
        },
    )
    turn_credit_replays: int = field(
        default=0,
        metadata={
            "help": (
                "Replays per intermediate prefix re-test. 0 means use "
                "student_generalize.replays, which keeps S(t) and S(T) measured "
                "the same way. Lower values are cheaper but noisier, and the "
                "noise lands on the per-turn split rather than the total."
            )
        },
    )
    retest_reward: float = field(
        default=0.0,
        metadata={
            "help": (
                "Reward for the ORIGINAL re-test. 0.0 is the historical "
                "behaviour, where the re-test is scored and logged but never "
                "rewarded. The free-chat rollout sets this and makes it the "
                "entire episode reward: with replays=4 an episode is worth the "
                "fraction of 4 independent solo attempts the student gets "
                "right, so the signal has five levels instead of two."
            )
        },
    )
    retest_original: bool = field(
        default=False,
        metadata={
            "help": (
                "Also re-test the ORIGINAL task after tutoring: the student is "
                "asked to write a complete standalone solution. Scored and logged "
                "only, never rewarded. An independent branch of the same chat, so "
                "it cannot see the transfer probes."
            )
        },
    )
    level1_enabled: bool = field(
        default=True,
        metadata={
            "help": (
                "Run the level1 transfer probe (variant1 in a generated bank). "
                "Turn it off to keep the original re-test without paying for a "
                "variant that is not being measured: the probe is skipped, its "
                "student calls are not made, and the startup validator stops "
                "requiring a level1 case on every row."
            )
        },
    )
    level2_enabled: bool = field(
        default=True,
        metadata={
            "help": (
                "Run the level2 transfer probe (variant2 in a generated bank). "
                "Same semantics as level1_enabled. With both off and "
                "retest_original on, student_generalize needs no bank at all."
            )
        },
    )
    replays: int = field(
        default=1,
        metadata={
            "help": (
                "Number of independent student attempts per original/transfer probe. "
                "The reward and the logged success become the fraction correct "
                "over these attempts instead of a single binary outcome, which "
                "removes most of the student-resampling noise from the signal. "
                "Costs (replays - 1) extra student calls per variant."
            )
        },
    )
    confidence: TutorStudentGeneralizeConfidenceConfig = field(
        default_factory=TutorStudentGeneralizeConfidenceConfig
    )

    def transfer_levels(self) -> tuple[str, ...]:
        """The transfer probes that are switched on, in their fixed order.

        Order matters: a 'train' sidecar stores cases as a positional list and
        the probe builder zips it against ("level1", "level2").
        """
        return tuple(
            level
            for level, on in (
                ("level1", self.level1_enabled),
                ("level2", self.level2_enabled),
            )
            if on
        )

    def probe_levels(self) -> tuple[str, ...]:
        """Every probe that will run, re-test included."""
        levels = self.transfer_levels()
        return ("original", *levels) if self.retest_original else levels

    def __post_init__(self) -> None:
        self.replays = int(self.replays)
        if self.mode not in _STUDENT_GENERALIZE_MODES:
            raise ValueError(
                "student_generalize.mode must be one of: 'only_success', 'always'."
            )
        if self.source not in _STUDENT_GENERALIZE_SOURCES:
            raise ValueError(
                "student_generalize.source must be one of: "
                "'sidecar', 'train', 'generated'."
            )
        if self.replays < 1:
            raise ValueError(
                f"student_generalize.replays must be >= 1, got {self.replays}."
            )
        if (
            self.enabled
            and self.source == "generated"
            and self.transfer_levels()
            and not self.path.strip()
        ):
            raise ValueError(
                "student_generalize.source='generated' requires "
                "student_generalize.path."
            )
        if self.enabled and not self.probe_levels():
            raise ValueError(
                "student_generalize.enabled=true with every probe switched off. "
                "Set at least one of retest_original, level1_enabled or "
                "level2_enabled, or set student_generalize.enabled=false."
            )
        if self.confidence.enabled and not self.enabled:
            raise ValueError(
                "student_generalize.confidence.enabled=true requires "
                "student_generalize.enabled=true."
            )
        if self.confidence.enabled and not self.transfer_levels():
            raise ValueError(
                "student_generalize.confidence.enabled=true scores the transfer "
                "probes, so it requires level1_enabled or level2_enabled."
            )
        if self.confidence.enabled and any(
            reward <= 0.0
            for level, reward in (
                ("level1", float(self.level1_reward)),
                ("level2", float(self.level2_reward)),
            )
            if level in self.transfer_levels()
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
                "If no attempt is correct, teacher_pre.on_reject decides what "
                "happens to the group."
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
    visibility: str = field(
        default="rollout",
        metadata={
            "help": (
                "Where an accepted draft is visible. 'rollout' is the historical "
                "behaviour: the draft rides in the teacher's context for every "
                "turn it then generates, so the policy is a with-draft teacher. "
                "'opd_only' keeps the draft OUT of the rollout -- the conversation "
                "is generated by the no-draft teacher, byte-identical to "
                "teacher_pre.enabled=false -- and hands it only to the privileged "
                "teacher that opd.context='presolve' scores the policy against. "
                "That is the whole switch: the draft stops being a prompt the "
                "deployed model needs and becomes a supervision signal."
            ),
            "choices": ["rollout", "opd_only"],
        },
    )
    share_per_group: bool = field(
        default=True,
        metadata={
            "help": (
                "Sample one draft per problem per weight version and reuse it "
                "across that problem's whole rollout group, instead of one per "
                "rollout. ON BY DEFAULT, AND THERE IS NO REASON TO TURN IT OFF: "
                "the draft rides in the teacher's prompt, so one draft per rollout "
                "gives a group gconfig.n_samples DIFFERENT prompts, and GRPO's "
                "leave-one-out baseline then averages partly over which draft was "
                "drawn rather than over how well the teacher taught. That variance "
                "lands in every advantage in the group. It also costs n_samples "
                "generations per problem instead of one, and denies the group a "
                "shared prompt prefix the generation cache could reuse. "
                "This defaulted to false until 20260815, so every arm that did not "
                "set it explicitly ran with per-rollout drafts -- measured on "
                "20260814_223024, 12 of 19 problems had all 8 rollouts on 8 "
                "different drafts. The field survives only so that a config.yaml "
                "saved by one of those runs still loads for --resume; new arms "
                "should leave it alone. The cache is keyed on the weight version "
                "and never survives an update, because the teacher is the model "
                "being trained. Under verify=true the filter becomes "
                "all-or-nothing per problem: the expected fraction of episodes "
                "lost is unchanged, since it is still `attempts` tries, but it "
                "concentrates into whole groups instead of scattering across them "
                "-- which is what on_reject is for."
            )
        },
    )
    on_reject: str = field(
        default="skip",
        metadata={
            "help": (
                "What happens when verification runs and no attempt is judged "
                "correct. Because share_per_group makes the draft one decision per "
                "problem per step, this is now a choice about the whole GROUP and "
                "not about one rollout: all n_samples rollouts see the same "
                "rejected result and take the same branch. "
                "'skip' drops the entire group for this step, which is what makes "
                "teacher_pre a dataset filter as well as a context; the problem is "
                "simply absent from the batch and contributes no gradient, rather "
                "than contributing a short group whose leave-one-out baseline is "
                "computed over fewer episodes. 'continue' keeps the group and "
                "leaves the result unaccepted, so the draft is absent. "
                "WHICH ONE DEPENDS ON WHAT THE DRAFT IS FOR. With "
                "visibility='rollout' the draft is part of the prompt this arm is "
                "defined by, so an episode without it is a different arm -- 'skip' "
                "is right, and the default. With visibility='opd_only' the draft "
                "is supervision, and an episode without it is the same rollout "
                "carrying no OPD term (_opd_skip_reason returns 'no_presolve'), "
                "so 'continue' is right: it keeps the episode set identical to the "
                "no-draft arm's, and otherwise the two arms differ by the filter "
                "as well as by the distillation. "
                "stop/teacher_pre_skipped reports the rate either way, and under "
                "sharing that fraction of episodes is also the fraction of groups."
            ),
            "choices": ["skip", "continue"],
        },
    )

    def __post_init__(self) -> None:
        if self.mode != "filter_solver":
            raise ValueError("teacher_pre.mode must be 'filter_solver'.")
        if int(self.attempts) < 1:
            raise ValueError("teacher_pre.attempts must be >= 1.")
        self.visibility = str(self.visibility or "rollout").strip()
        if self.visibility not in {"rollout", "opd_only"}:
            raise ValueError(
                "teacher_pre.visibility must be 'rollout' or 'opd_only', got "
                f"{self.visibility!r}."
            )
        self.on_reject = str(self.on_reject or "skip").strip()
        if self.on_reject not in {"skip", "continue"}:
            raise ValueError(
                "teacher_pre.on_reject must be 'skip' or 'continue', got "
                f"{self.on_reject!r}."
            )


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
    leak_terminate: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Whether a leak ends the conversation during EVALUATION. None "
                "follows leak_handling_mode, which is the historical behaviour. "
                "False evaluates in the wild: the conversation runs the full "
                "budget however much the teacher gives away, the leak judge still "
                "runs so the leak rate is known, and the re-test sees the whole "
                "transcript. Terminating is a training policy -- nothing "
                "truncates a real conversation -- so it should not decide what "
                "gets measured. With False the train-consistent number is still "
                "reported, as student_original_preleak_success, computed on the "
                "same rollout."
            )
        },
    )
    format_terminate: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Whether a malformed teacher turn ends the conversation during "
                "EVALUATION. None follows format_handling_mode, which is the "
                "historical behaviour. False evaluates in the wild: the "
                "conversation runs the full budget, a malformed turn hands the "
                "student an empty message and the episode continues, and "
                "format_errors still reports the rate so nothing is hidden. "
                "Exactly the argument leak_terminate makes -- terminating is a "
                "training policy, nothing truncates a real conversation, so it "
                "should not decide what gets measured. Measured on the step-0 "
                "eval of 20260814_053136, under terminate 83.5% of eval episodes "
                "ended on a format error and only 16.5% reached the budget, which "
                "is the baseline every later point is read against."
            )
        },
    )
    teacher_pre_verify: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Override teacher_pre.verify during evaluation only. None keeps "
                "eval identical to training. False evaluates under the deployment "
                "condition: the teacher pre-solve produces one unverified draft, "
                "the answer judge is not called, and no sample is skipped for "
                "having failed it. Training can still verify, which keeps the "
                "training support clean, while the reported number stops "
                "depending on a checker that does not exist at deployment."
            )
        },
    )
    teacher_pre_enabled: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Override teacher_pre.enabled during evaluation only. None keeps eval "
                "identical to training, which with teacher_pre_verify False is the "
                "usual setting: the teacher drafts a solution but nothing checks it. "
                "False evaluates a teacher that never drafts at all -- the deployment "
                "condition for an arm whose claim is that the trained teacher no "
                "longer needs one. True forces the draft on where training had it off. "
                "Together with teacher_pre_verify this spans the three eval regimes: "
                "draft-unverified, draft-verified, and no draft. "
                "NOTE under teacher_pre.visibility opd_only the draft has no reader "
                "at eval at all -- the rollout hides it and OPD does not run there -- "
                "so _presolve_unused_at_eval skips the generation whatever this is set "
                "to. There is nothing to measure either way, only a call to save."
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
class TutorStudentRequestJudgeConfig:
    """Reward a teacher for satisfying a sampled student turn request."""

    enabled: bool = field(default=False)
    weight: float = field(
        default=0.5,
        metadata={
            "help": (
                "Magnitude of the target teacher-turn reward. Judge scores "
                "-1/1 map to -weight/+weight."
            )
        },
    )
    behavior_names: list[str] = field(
        default_factory=lambda: ["ask_question"],
        metadata={
            "help": (
                "Names from prompt_pool.student_turn_behavior.path whose student "
                "reply should trigger the request-fulfillment judge."
            )
        },
    )

    def __post_init__(self) -> None:
        self.weight = float(self.weight)
        self.behavior_names = [
            str(name).strip() for name in self.behavior_names if str(name).strip()
        ]
        if self.enabled and self.weight <= 0.0:
            raise ValueError("reward.student_request_judge.weight must be positive.")
        if self.enabled and not self.behavior_names:
            raise ValueError(
                "reward.student_request_judge.behavior_names must not be empty."
            )
        if len(self.behavior_names) != len(set(self.behavior_names)):
            raise ValueError(
                "reward.student_request_judge.behavior_names must be unique."
            )


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
    turn_local_components: list[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Reward component names that stay on the turn that produced "
                "them instead of being accumulated backward onto earlier turns "
                "by advantage_estimator='rebn'. Empty keeps the previous "
                "behaviour, where a leak on the last turn discounts the "
                "returns of every good turn before it. Names match the "
                "reward_component/* metrics, e.g. 'leak', 'format_error', and "
                "in staged leak mode 'leak_final_answer', 'leak_compute', "
                "'leak_formula'. Requires advantage_estimator='rebn' and is "
                "incompatible with actor.reward_norm."
            )
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
    student_request_judge: TutorStudentRequestJudgeConfig = field(
        default_factory=TutorStudentRequestJudgeConfig
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
class TutorGuidedSlotsConfig:
    """Reserve rollout slots in each group for move-prescribed teacher turns.

    The trained policy plays one move (PINPOINT 86% of turns), so a group of eight
    free rollouts contains the state's best move only 37.7% of the time and GRPO
    has nothing to compare against. Prescribing a move in a few slots lifts that
    to 78%. The instruction is stripped from the prompt the sample is trained on,
    so what the policy learns is "this move suited this task", not "obey
    instructions".
    """

    enabled: bool = field(default=False)
    slots: int = field(
        default=3,
        metadata={
            "help": (
                "Rollouts per group that receive a prescribed move. Must be less "
                "than gconfig.n_samples so free rollouts remain the majority."
            )
        },
    )
    moves: tuple[str, ...] = field(
        default=("DECOMPOSE", "REFRAME", "PROBE"),
        metadata={
            "help": (
                "Moves eligible for the reserved slots, drawn from "
                "prompts.TEACHER_MOVE_INSTRUCTIONS. Assigned to slots in order, "
                "cycling if there are more slots than moves."
            )
        },
    )
    turns: tuple[int, ...] = field(
        default=(1,),
        metadata={
            "help": (
                "1-based turn indices that receive a prescribed move. Defaults to "
                "the first turn only, which is where the group still shares a "
                "state and where the measured task-level effect lives."
            )
        },
    )
    rotate_by_task: bool = field(
        default=True,
        metadata={
            "help": (
                "Offset the move-to-slot assignment by a hash of the task so that "
                "with fewer slots than moves, different tasks still cover "
                "different moves across the dataset."
            )
        },
    )

    def __post_init__(self) -> None:
        from examples.tutor.prompts import TEACHER_MOVE_INSTRUCTIONS

        self.slots = int(self.slots)
        self.moves = tuple(str(m).strip().upper() for m in self.moves if str(m).strip())
        self.turns = tuple(sorted({int(t) for t in self.turns}))
        if not self.enabled:
            return
        if self.slots < 1:
            raise ValueError("guided_slots.slots must be at least 1 when enabled.")
        if not self.moves:
            raise ValueError("guided_slots.moves must be non-empty when enabled.")
        unknown = [m for m in self.moves if m not in TEACHER_MOVE_INSTRUCTIONS]
        if unknown:
            raise ValueError(
                "guided_slots.moves contains unknown moves: "
                f"{unknown}; known moves are "
                f"{sorted(TEACHER_MOVE_INSTRUCTIONS)}."
            )
        if any(t < 1 for t in self.turns):
            raise ValueError("guided_slots.turns must be 1-based positive integers.")


@dataclass
class TutorInstructionPromptConfig:
    """Control arm: leave the instruction in the prompt instead of distilling it.

    This exists so that "OPD worked" is a defensible claim. Appending a sentence
    to the prompt is free, so OPD has to beat that, not beat nothing. The arm is
    identical to OPD in what the teacher is told and when -- same wording, same
    turn gate -- and differs only in where it ends up: here it stays in the
    prompt at training and at evaluation, nothing is stripped, nothing is
    distilled, and no part of the turn is off-policy.

    Reading the three arms together:
      opd ~ baseline                     the distillation did nothing
      prompt_instruction ~ baseline      the sentence stops paying once the
                                         policy is trained, i.e. it only ever
                                         raised the starting point
      opd ~ prompt_instruction > baseline the effect of the prompt is now in the
                                         weights, which is the whole point
      prompt_instruction > opd > baseline only partly transferred
    """

    enabled: bool = field(default=False)
    instruction: str = field(
        default="",
        metadata={
            "help": (
                "Empty uses prompts.TEACHER_REPAIR_INSTRUCTION, so this arm and "
                "opd are told the same thing."
            )
        },
    )
    min_prior_failed_turns: int = field(
        default=2,
        metadata={
            "help": (
                "Same gate as opd.min_prior_failed_turns, so the two arms differ "
                "only in where the instruction ends up."
            )
        },
    )

    def __post_init__(self) -> None:
        self.instruction = str(self.instruction or "").strip()
        self.min_prior_failed_turns = int(self.min_prior_failed_turns)
        if self.enabled and self.min_prior_failed_turns < 0:
            raise ValueError(
                "prompt_instruction.min_prior_failed_turns must be non-negative."
            )
        self.resolved_instruction  # raises on an unknown "@name"

    @property
    def resolved_instruction(self) -> str:
        from examples.tutor.prompts import resolve_teacher_instruction

        return resolve_teacher_instruction(self.instruction)[0]

    @property
    def instruction_name(self) -> str:
        """Short label for logging, so a run says which sentence it was given."""
        from examples.tutor.prompts import resolve_teacher_instruction

        return resolve_teacher_instruction(self.instruction)[1]


@dataclass
class TutorOpdConfig:
    """On-policy distillation from a teacher given a privileged instruction.

    The policy rolls out normally -- no instruction, fully on-policy. The same
    weights are then run teacher-forced over the policy's own tokens with the
    repair instruction appended, and a token-level KL pulls the policy toward
    that instructed distribution. This is context distillation: the effect of
    the instruction is moved into the weights, so the deployed model behaves as
    if it had been told, without being told.

    Nothing is sampled from the teacher and no prompt mismatch enters the policy
    gradient, so there is no importance weight and no off-policy correction.
    Cost is one extra forward pass over the supervised turns.
    """

    enabled: bool = field(default=False)
    teacher_source: str = field(
        default="policy",
        metadata={
            "help": (
                "Which weights score the privileged prompt. 'policy' is the "
                "historical behaviour and true context distillation: the SAME "
                "live weights, differing from the rollout only by the prompt. "
                "'checkpoint' scores it under the frozen model held in the `ref` "
                "engine, which is the reference on-policy-distillation setting.  "
                "WHY IT EXISTS. With 'policy' the target moves with every update, "
                "and the reverse KL has no fixed point to converge to. Measured on "
                "20260815_054008 at loss_weight 1.0, the divergence GREW 0.083 -> "
                "0.509 in 7 steps while entropy went 0.216 -> 0.441 and "
                "update/clip_ratio went 0.0002 -> 0.274 against a 0.004-0.007 "
                "reference; retest fell 0.435 -> 0.271. At loss_weight 0.05 the "
                "same drift was present but too weak to matter (0.074 -> 0.126 "
                "over 70 steps), so there is no weight at which 'policy' both "
                "converges and does anything.  "
                "A per-token KL penalty applied through the PPO surrogate can only "
                "push DOWN the probability of tokens that were sampled; it never "
                "pushes up the tokens the teacher preferred, because those carry "
                "no gradient. Against a fixed, genuinely better teacher that still "
                "works -- suppressing its dispreferred tokens redistributes mass "
                "toward it. Against a teacher that moves with the policy, the only "
                "response available is to flatten, which is the entropy climb.  "
                "'checkpoint' requires a `ref` block configured with use_lora and "
                "init_lora_path pointing at the teacher adapter. The ref engine is "
                "reused because it is the one frozen, optimizer-less engine that "
                "already has offload wiring and a compute_logp; its own ref_logp "
                "pass is skipped while actor.kl_ctl is 0."
            ),
            "choices": ["policy", "checkpoint"],
        },
    )
    context: str = field(
        default="instruction",
        metadata={
            "help": (
                "What privilege the teacher gets over the policy. 'instruction' is "
                "the historical behaviour: `instruction` is appended to the "
                "teacher's system prompt. 'presolve' instead inserts the episode's "
                "private solution draft into the teacher's context as the two "
                "messages teacher_pre already generates -- the solve request and "
                "the accepted draft -- so the teacher is a with-draft teacher and "
                "the policy is the no-draft one that produced the conversation. "
                "Requires free_chat.enabled and teacher_pre.visibility='opd_only'; "
                "`instruction` is then unused. Turns whose draft was rejected are "
                "skipped as 'no_presolve' rather than supervised against an "
                "identical prompt."
            ),
            "choices": ["instruction", "presolve"],
        },
    )
    loss_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Coefficient on the per-token reverse KL, subtracted from the "
                "advantage. This is the reference implementation's "
                "kl_penalty_coef, whose default is also 1.0. Note the scale is "
                "set by the KL being in nats, not by the task reward, so it is "
                "not comparable to a loss weight. NOTE the KL is added after the "
                "group baseline and is not baselined, so it competes with the task "
                "advantage on absolute scale: the instruction arms ran "
                "opd_reverse_kl/avg at 0.016-0.021 nats/token against an "
                "advantages/avg of -0.04 to -0.08. A whole solution draft moves "
                "the context far more than one appended sentence, so measure "
                "opd_reverse_kl for a step at a small weight before choosing one."
            )
        },
    )
    instruction: str = field(
        default="",
        metadata={
            "help": (
                "Instruction the teacher is conditioned on. Empty uses "
                "prompts.TEACHER_REPAIR_INSTRUCTION. A leading @ names one of "
                "prompts.TEACHER_NAMED_INSTRUCTIONS, e.g. \"@handback\"; "
                "anything else is used verbatim."
            )
        },
    )
    reward_clip: float = field(
        default=0.0,
        metadata={
            "help": (
                "Clamp on the per-token reverse KL in nats, 0 to disable. Not part "
                "of the reference implementation, which does not clip; 0 is the "
                "default so the standard behaviour is what runs unless asked for. "
                "Raise it above 0 only if a single outlier token is observed "
                "dominating opd_advantage."
            )
        },
    )
    min_prior_failed_turns: int = field(
        default=2,
        metadata={
            "help": (
                "Supervise a turn only after the tutor has already failed this "
                "many times in this episode. Episodes terminate on success, so "
                "this is equivalent to turn_idx > min_prior_failed_turns. Default "
                "2 matches the repair instruction's premise that the previous "
                "message did not get through."
            )
        },
    )
    max_turns_per_episode: int = field(
        default=0,
        metadata={
            "help": (
                "Cap on supervised turns per episode, counted from the earliest "
                "eligible turn. 0 means no cap."
            )
        },
    )
    skip_guided_rows: bool = field(
        default=True,
        metadata={
            "help": (
                "Skip turns that were generated under a guided_slots move "
                "instruction. Those rows are already trained on a rewritten "
                "prompt; stacking a second perturbation on them muddies both."
            )
        },
    )
    skip_leaked_rows: bool = field(
        default=True,
        metadata={
            "help": "Skip turns whose teacher output leaked the answer.",
        },
    )

    def __post_init__(self) -> None:
        self.loss_weight = float(self.loss_weight)
        self.reward_clip = float(self.reward_clip)
        self.instruction = str(self.instruction or "").strip()
        self.min_prior_failed_turns = int(self.min_prior_failed_turns)
        self.max_turns_per_episode = int(self.max_turns_per_episode)
        self.context = str(self.context or "instruction").strip()
        if self.context not in {"instruction", "presolve"}:
            raise ValueError(
                "opd.context must be 'instruction' or 'presolve', got "
                f"{self.context!r}."
            )
        self.teacher_source = str(self.teacher_source or "policy").strip()
        if self.teacher_source not in {"policy", "checkpoint"}:
            raise ValueError(
                "opd.teacher_source must be 'policy' or 'checkpoint', got "
                f"{self.teacher_source!r}."
            )
        if not self.enabled:
            return
        if self.loss_weight <= 0.0:
            raise ValueError("opd.loss_weight must be positive when enabled.")
        if self.reward_clip < 0.0:
            raise ValueError("opd.reward_clip must be non-negative (0 disables).")
        if self.min_prior_failed_turns < 0:
            raise ValueError("opd.min_prior_failed_turns must be non-negative.")
        if self.max_turns_per_episode < 0:
            raise ValueError("opd.max_turns_per_episode must be non-negative.")
        if self.context == "instruction":
            # Only meaningful in the instruction arm; under 'presolve' the
            # privilege is the draft and a stale instruction string is dead
            # config rather than a typo worth failing on.
            self.resolved_instruction  # raises on an unknown "@name"

    @property
    def resolved_instruction(self) -> str:
        from examples.tutor.prompts import resolve_teacher_instruction

        return resolve_teacher_instruction(self.instruction)[0]

    @property
    def instruction_name(self) -> str:
        """Short label for logging, so a run says which sentence it was given."""
        from examples.tutor.prompts import resolve_teacher_instruction

        return resolve_teacher_instruction(self.instruction)[1]


@dataclass
class TutorActorConfig(PPOActorConfig):
    """PPO actor with a configurable number of passes over each rollout batch."""

    num_iterations: int = field(
        default=1,
        metadata={
            "help": (
                "How many times to run the whole PPO update over one collected "
                "batch. 1 is the historical behaviour: ppo_n_minibatches splits "
                "the batch and takes one step per chunk, which is several "
                "optimizer steps but a single pass over the data. 2 takes two "
                "passes. Rollout dominates the cost here, so this buys "
                "optimization without buying generation. The extra passes are "
                "off-policy against the updated parameters and the PPO clip is "
                "what bounds them, which is why this is safer than the "
                "equivalent increase in learning rate."
            )
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_iterations < 1:
            raise ValueError("actor.num_iterations must be at least 1.")


@dataclass
class TutorFreeChatConfig:
    """Teacher-first free conversation scored only by a delayed solo re-test.

    The answer-attempt loop tells the student to solve the task on every turn,
    judges every reply, and stops the moment one is correct. That makes every
    student turn an answer attempt and every episode as short as the student's
    luck allows. Here the teacher opens with no student input, the pair talks for
    exactly ``budget`` rounds with nothing judged in between, and the reward is
    the student's solo re-test afterwards.
    """

    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Run the free-chat rollout instead of the answer-attempt loop: "
                "no student pre-attempt, the teacher speaks first, no per-turn "
                "answer judge, no early termination on a correct answer, and "
                "the reward comes from student_generalize's original re-test. "
                "Requires student_generalize.enabled, retest_original and "
                "retest_reward > 0; reward.max_turn_penalty must be 0 because "
                "reaching the budget is now the normal ending."
            )
        },
    )
    student_has_not_seen_problem: bool = field(
        default=False,
        metadata={
            "help": (
                "State in the teacher system prompt that the student has not "
                "seen the math problem yet. This adds only that fact before "
                "the problem statement; it does not instruct the teacher how "
                "to respond."
            )
        },
    )
    transfer_prompts: bool = field(
        default=False,
        metadata={
            "help": (
                "Use the transfer wording for the teacher system prompt and the "
                "student re-test prompt. For datasets where the dialogue task "
                "and the re-test task are DIFFERENT problems -- rows carrying "
                "retest_task/retest_ground_truth, as the numeric-variant build "
                "does. The default wording tells the teacher the student will be "
                "asked to solve 'a problem from scratch ... on their own' and "
                "tells the student to solve 'the problem from scratch', and "
                "neither is true when the test problem is a variant: it aims the "
                "teacher at the instance rather than the method, and it points "
                "the student at a problem it is assumed to have already seen. "
                "With this on the teacher is asked to make the student "
                "'understand the underlying concepts' for 'some related "
                "questions', and the re-test says 'Now try to solve this "
                "problem', which presupposes no prior reference. Off leaves both "
                "prompts byte-for-byte unchanged, so it cannot affect any "
                "non-transfer arm. Set it only alongside a dataset whose "
                "retest_task differs from task -- nothing validates that pairing."
            )
        },
    )
    no_teaching_baseline: bool = field(
        default=False,
        metadata={
            "help": (
                "Score the re-test against no teaching at all. Once per problem "
                "the student is given the re-test input with an empty transcript "
                "and asked to solve it student_generalize.replays times; every "
                "episode on that problem then reports and is rewarded on its "
                "re-test fraction MINUS that baseline. The extra calls are "
                "outside the GRPO group. Note that actor.group_baseline already "
                "removes a per-problem constant, so while that is on this changes "
                "the logged improvement and not the advantage."
            )
        },
    )
    budget: int = field(
        default=0,
        metadata={
            "help": (
                "Number of (teacher, student) rounds. 0 means use max_turns. "
                "Whatever it resolves to also overwrites max_turns, so every "
                "downstream turn count and the budget the teacher is told in "
                "its system prompt cannot disagree."
            )
        },
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
                "'terminate' stops before the student sees leaked tutor output."
            ),
            "choices": ["disabled", "reward_only", "terminate"],
        },
    )
    format_handling_mode: str = field(
        default="continue",
        metadata={
            "help": (
                "What to do when a teacher turn cannot be parsed. 'continue' "
                "penalises the turn, hands the student an empty message and keeps "
                "going -- the historical behaviour. 'terminate' ends the episode "
                "at that turn, exactly like leak_handling_mode='terminate': the "
                "malformed turn carries reward.format_error_penalty, the episode "
                "is TRAINED, and the re-test scores the prefix through the last "
                "completed round.  "
                "Terminating is worth it because a malformed turn writes a blank "
                "assistant message into the tutor's own history and 88-98% of the "
                "following turns are malformed too, so the rest of the episode "
                "teaches nothing (the student solves 0.3% of the time after one) "
                "while still consuming turns and OPD supervision.  "
                "reward.format_error_penalty must be at least as negative as the "
                "worst honest outcome, or terminating becomes the cheap way out of "
                "a losing episode. With free_chat.no_teaching_baseline on, an "
                "honest episode that teaches nothing is worth about -baseline, and "
                "the measured baseline runs 0.33-0.53, so -0.5 clears it. "
                "reward.turn_local_components=['leak'] helps too, by keeping a "
                "later leak from discounting the good turns before it. An older "
                "note here recommended 0.0 on the grounds that the episode was "
                "discarded anyway; that discarding was a bug, now fixed, and 0.0 "
                "would leave malformed output entirely uncharged.  "
                "Track stop/format_error together with rollout/turns, and watch "
                "ppo_actor/update/clip_ratio: actor.episode_loss_weighting scales "
                "an episode's advantage by mean_tokens/episode_tokens with no cap, "
                "so a one-turn episode can carry a ~5x per-token weight. That "
                "amplifies the penalty, which is the wanted direction, but it "
                "concentrates a large advantage on few tokens."
            ),
            "choices": ["continue", "terminate"],
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
    student_axes: list[TutorStudentAxesConfig] = field(
        default_factory=list,
        metadata={
            "help": (
                "Declare the student pool as its two axes instead of writing the "
                "product out. Each block is one endpoint times behaviors times "
                "informations, expanded into student_models before any of its "
                "checks run. Expanded entries are APPENDED, so a config may mix "
                "the two forms; leave this empty and nothing changes."
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
    guided_slots: TutorGuidedSlotsConfig = field(
        default_factory=TutorGuidedSlotsConfig
    )
    opd: TutorOpdConfig = field(default_factory=TutorOpdConfig)
    prompt_instruction: TutorInstructionPromptConfig = field(
        default_factory=TutorInstructionPromptConfig
    )
    free_chat: TutorFreeChatConfig = field(default_factory=TutorFreeChatConfig)
    cross_eval: CrossEvalConfig = field(default_factory=CrossEvalConfig)
    actor: TutorActorConfig = field(default_factory=TutorActorConfig)
    teacher_history_tags: str = field(
        default="unmasked",
        metadata={
            "help": (
                "How the teacher sees its own earlier turns. 'unmasked' is the "
                "default: each valid prior reply is replayed verbatim, private "
                "reasoning included, in the teacher view only. 'stripped' hands "
                "back the visible text with the output tags removed, which the "
                "model imitates -- malformed turns run 2.1% at depth 1 and 73.4% "
                "at depth 10 on an untrained teacher, so it is the worst mode and "
                "is kept only to reproduce a run that used it. 'masked' restores "
                "the tag skeleton with the reasoning replaced by a placeholder, "
                "and is what a malformed turn falls back to under 'unmasked'. "
                "Observation side only: reward and sampling are untouched."
            )
        },
    )
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
        if (
            self.reward.student_request_judge.enabled
            and not self.prompt_pool.student_turn_behavior.enabled
        ):
            raise ValueError(
                "reward.student_request_judge.enabled requires "
                "prompt_pool.student_turn_behavior.enabled."
            )
        if self.reward.student_request_judge.enabled:
            missing_separate_call_names = sorted(
                set(self.reward.student_request_judge.behavior_names)
                - set(
                    self.prompt_pool.student_turn_behavior.separate_call_behavior_names
                )
            )
            if missing_separate_call_names:
                raise ValueError(
                    "reward.student_request_judge.behavior_names must also appear "
                    "in prompt_pool.student_turn_behavior."
                    "separate_call_behavior_names: "
                    f"{missing_separate_call_names}."
                )
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
        if self.guided_slots.enabled:
            if self.gconfig.n_samples < 2:
                raise ValueError(
                    "guided_slots.enabled requires gconfig.n_samples >= 2 so a "
                    "group contains both guided and free rollouts."
                )
            if self.guided_slots.slots >= self.gconfig.n_samples:
                raise ValueError(
                    "guided_slots.slots must be smaller than gconfig.n_samples "
                    f"({self.guided_slots.slots} >= {self.gconfig.n_samples}); "
                    "otherwise no free rollout is left to compare against."
                )
            if not self.actor.use_decoupled_loss:
                raise ValueError(
                    "guided_slots.enabled requires actor.use_decoupled_loss=true. "
                    "Guided turns are trained on a prompt they were not generated "
                    "from, and the decoupled loss is what recomputes the proximal "
                    "log-probabilities on the training prompt and applies "
                    "actor.behave_imp_weight_cap to the resulting weight."
                )
        if self.prompt_instruction.enabled and self.opd.enabled:
            raise ValueError(
                "prompt_instruction and opd are the two arms of the same "
                "comparison -- the first keeps the instruction in the prompt, the "
                "second distils it into the weights. Enabling both tells the "
                "teacher the same thing twice and makes neither attributable."
            )
        if self.prompt_instruction.enabled and self.guided_slots.enabled:
            raise ValueError(
                "prompt_instruction is a control arm and must differ from the "
                "baseline in exactly one way; guided_slots adds a second change."
            )
        # A frozen OPD teacher lives in the `ref` engine, so that engine has to
        # exist AND be pointed at the teacher's adapter. Without init_lora_path it
        # would silently load the bare base model, and the run would look fine
        # while distilling from an untrained teacher.
        if self.opd.enabled and self.opd.teacher_source == "checkpoint":
            if self.ref is None:
                raise ValueError(
                    "opd.teacher_source='checkpoint' requires a `ref` block to "
                    "hold the frozen teacher."
                )
            if not getattr(self.ref, "use_lora", False):
                raise ValueError(
                    "opd.teacher_source='checkpoint' requires ref.use_lora=true; "
                    "the teacher checkpoints in this tree are LoRA adapters over "
                    "the shared base."
                )
            if not getattr(self.ref, "init_lora_path", ""):
                raise ValueError(
                    "opd.teacher_source='checkpoint' requires ref.init_lora_path "
                    "pointing at the teacher adapter directory. Without it the ref "
                    "engine loads the bare base model and the run would distil "
                    "from an untrained teacher without complaining."
                )
            if self.actor.kl_ctl > 0:
                raise ValueError(
                    "opd.teacher_source='checkpoint' reuses the `ref` engine, "
                    "which actor.kl_ctl also claims as its KL reference. They "
                    "would need different weights; set kl_ctl to 0."
                )
        if self.opd.enabled and self.actor.backend.startswith("megatron"):
            raise ValueError(
                "opd.enabled currently requires an FSDP actor backend; the "
                "instructed-teacher forward pass reuses actor.compute_logp."
            )
        # teacher_pre.visibility='opd_only' and opd.context='presolve' are two
        # halves of one switch and neither does anything alone: the first hides a
        # draft that then supervises nothing, the second asks for a privileged
        # context the policy already had. Requiring both makes the useless halves
        # unrepresentable instead of silently paying for a generation.
        presolve_opd = self.opd.enabled and self.opd.context == "presolve"
        opd_only = self.teacher_pre.visibility == "opd_only"
        if presolve_opd and not opd_only:
            raise ValueError(
                "opd.context='presolve' requires teacher_pre.visibility="
                "'opd_only'. With 'rollout' the draft is already in the policy's "
                "own prompt, so the teacher and the policy would differ by "
                "nothing and every reverse KL would be zero."
            )
        if opd_only and not presolve_opd:
            raise ValueError(
                "teacher_pre.visibility='opd_only' requires opd.enabled=true with "
                "opd.context='presolve'. Otherwise the pre-solve is generated, "
                "hidden from the rollout, and read by nobody."
            )
        if presolve_opd and not self.teacher_pre.enabled:
            raise ValueError(
                "opd.context='presolve' requires teacher_pre.enabled=true; there "
                "is no draft to condition the teacher on otherwise."
            )
        if presolve_opd and not self.free_chat.enabled:
            raise ValueError(
                "opd.context='presolve' requires free_chat.enabled=true. The draft "
                "enters the teacher's context through the free-chat preamble; "
                "outside free chat it is appended to the system prompt by a "
                "different path that this switch does not gate."
            )
        # The axes blocks become student_models entries HERE, before any check
        # below runs, so an expanded pool is validated exactly like a written-out
        # one and nothing downstream can tell which form produced it. Appended
        # rather than replacing, so the two forms can be mixed.
        for axes in list(self.student_axes or []):
            if not isinstance(axes, TutorStudentAxesConfig):
                axes = TutorStudentAxesConfig(**dict(axes))
            self.student_models = list(self.student_models) + axes.expand()
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
        if self.teacher_user_prompt_template != TEACHER_STATE_USER_TEMPLATE:
            raise ValueError(
                "teacher_user_prompt_template is no longer used: the teacher "
                "prompt is a real multi-turn message list and the task rides in "
                "the system prompt. Customise the system prompt instead."
            )
        if self.format_handling_mode not in _FORMAT_HANDLING_MODES:
            raise ValueError(
                "format_handling_mode must be one of: 'continue', 'terminate'."
            )
        if self.leak_handling_mode not in _LEAK_HANDLING_MODES:
            raise ValueError(
                "leak_handling_mode must be one of: 'disabled', "
                "'reward_only', or 'terminate'."
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
