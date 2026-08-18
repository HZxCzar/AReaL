"""Configuration for the tutor / PedagogicalRL cross, shared by both arms.

Separate from cross_eval.py, and importing nothing from either arm, so that
examples/tutor/configs.py and examples/pedagogical_rl/config.py can both point
at it without an import cycle.

ONE DEFINITION ON PURPOSE. The comparison is read off one set of series, and it
is only paired if both arms ran the same protocols with the same numbers. If
each arm owned its own copy of these fields, a change to one side would be
invisible from the other and would show up as an unexplained gap in the table.
The two YAML blocks are therefore the same shape, key for key, and the tutor arm
validates the one field it can check against its own rollout
(`free_chat.budget`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrossEvalFreeChatConfig:
    """The tutor protocol, as the cross runs it.

    Written out on both arms even though the tutor arm's own eval rollout
    already produced this transcript: the ped arm has to run the protocol from
    these numbers, and the tutor arm checks them against its own so a mismatch
    is a startup error rather than a silent difference.
    """

    budget: int = field(
        default=5,
        metadata={
            "help": (
                "Rounds of (teacher, student). Must equal the tutor arm's "
                "free_chat.budget."
            )
        },
    )
    max_student_tokens: int = field(default=2048)
    teacher_history_tags: str = field(
        default="unmasked",
        metadata={
            "help": (
                "How the teacher sees its own earlier turns in this protocol: "
                "'stripped', 'masked' or 'unmasked'. MUST EQUAL the tutor arm "
                "teacher_history_tags, which math/0810 sets to 'unmasked'. Under "
                "'stripped' the teacher imitates its own untagged replies and "
                "malformed turns run 6.3% at depth 1 rising to 41.7% at depth 5, so "
                "a mismatch here would compare a teacher that keeps its format "
                "contract against one that loses it. 'unmasked' replays the exact "
                "prior reply, reasoning included, to the teacher only -- the student "
                "and the re-test still see the public visible text, so the "
                "comparison stays on the same transcript."
            )
        },
    )


@dataclass
class CrossEvalClassroomConfig:
    """PedagogicalRL's GUIDED/ATTEMPTED dialogue, as the cross runs it.

    Defaults are their baseline's `generation` block. The ped arm should leave
    these at the values it trains with; the tutor arm has to be given the same
    ones.
    """

    max_teacher_turns: int = field(default=10)
    max_tokens_in_conversation: int = field(default=24576)
    max_tokens_per_student_turn: int = field(default=2048)
    max_tokens_per_student_attempt: int = field(default=2048)
    include_thinking: bool = field(default=False)


@dataclass
class CrossEvalRetestConfig:
    """Our measurement: solo re-test on a replayed transcript, LLM answer judge.

    `replays` should match the tutor arm's student_generalize.replays, so the
    cross reports the same quantity that arm is rewarded on.
    """

    replays: int = field(default=4)
    max_tokens: int = field(default=2048)


@dataclass
class CrossEvalInterviewConfig:
    """Their measurement: n solutions from one request, exact boxed match.

    `attempts` should match their generation.number_student_attempts.
    """

    attempts: int = field(default=8)
    max_tokens: int = field(default=2048)
    student_name: str = field(
        default="",
        metadata={
            "help": (
                "Their student persona name at interview time. Empty selects "
                "their unnamed variant, which is right for a free-chat "
                "transcript because that dialogue never used one."
            )
        },
    )
    timeout: float | None = field(
        default=None,
        metadata={
            "help": (
                "Timeout for the single n-choice interview request. Null uses "
                "the student's own timeout times attempts, since that request "
                "generates that many completions and would otherwise time out "
                "on every problem and score as a teaching failure."
            )
        },
    )


@dataclass
class CrossEvalLeakJudgeConfig:
    """Both leak judges on both transcripts, as metrics only.

    Each arm optimises one of them -- the tutor arm AReaL's turn-level rawbase
    judge, the ped arm PedagogicalRL's whole-dialogue judge -- so a leak rate
    read under the arm's own judge flatters it. Neither judge terminates a
    rollout and neither touches a reward.
    """

    enabled: bool = field(default=True)
    turn_enabled: bool = field(default=True)
    native_enabled: bool = field(default=True)
    native_attempts: int = field(default=2)
    native_max_retries: int = field(default=5)


@dataclass
class CrossEvalConfig:
    """Two dialogue protocols x two scorers, at evaluation, on both arms.

    Each arm keeps the transcript its own eval rollout produced, runs the OTHER
    protocol's dialogue with the same teacher, and scores both transcripts with
    both scorers. Four cells per arm under

        xeval/<free_chat|classroom>/<retest|interview>/success

    and because both arms emit those same names, the eight-cell table is one set
    of series rather than two that have to be aligned by hand.

    Training is untouched: every switch here applies only when is_eval is set,
    and every series it adds is namespaced under xeval/.

    COST. One extra dialogue plus three extra scorings per crossed problem,
    against a tutor eval pass that already runs 1500-1800 s. `sample_rate` is
    the lever: it selects on a hash of the problem text, so both arms cross
    exactly the same subset without agreeing on a seed or an ordering.
    """

    enabled: bool = field(
        default=False,
        metadata={"help": "Run the two-protocol, two-scorer cross at evaluation."},
    )
    sample_rate: float = field(
        default=1.0,
        metadata={
            "help": (
                "Fraction of evaluated problems to cross, chosen by a stable "
                "hash of the problem text so both arms pick the same ones. "
                "1.0 crosses every problem."
            )
        },
    )
    run_other_protocol: bool = field(
        default=True,
        metadata={
            "help": (
                "Run the other arm's dialogue. False keeps only the two scorers "
                "on this arm's own transcript: that is the cheap half of the "
                "cross and it answers the scoring question but not the protocol "
                "one."
            )
        },
    )
    free_chat: CrossEvalFreeChatConfig = field(
        default_factory=CrossEvalFreeChatConfig
    )
    classroom: CrossEvalClassroomConfig = field(
        default_factory=CrossEvalClassroomConfig
    )
    retest: CrossEvalRetestConfig = field(default_factory=CrossEvalRetestConfig)
    interview: CrossEvalInterviewConfig = field(
        default_factory=CrossEvalInterviewConfig
    )
    leak_judges: CrossEvalLeakJudgeConfig = field(
        default_factory=CrossEvalLeakJudgeConfig
    )

    def __post_init__(self) -> None:
        # The ped arm reconstructs this from asdict(), which flattens the nested
        # dataclasses to plain dicts, and a workflow built from workflow_kwargs
        # would otherwise get dicts where it expects configs -- and only find
        # out at the first attribute access, inside an eval rollout.
        for name, config_cls in (
            ("free_chat", CrossEvalFreeChatConfig),
            ("classroom", CrossEvalClassroomConfig),
            ("retest", CrossEvalRetestConfig),
            ("interview", CrossEvalInterviewConfig),
            ("leak_judges", CrossEvalLeakJudgeConfig),
        ):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, config_cls(**value))
        self.sample_rate = float(self.sample_rate)
        if not 0.0 < self.sample_rate <= 1.0:
            raise ValueError("cross_eval.sample_rate must be in (0, 1].")
        if self.free_chat.budget < 1:
            raise ValueError("cross_eval.free_chat.budget must be at least 1.")
        if self.classroom.max_teacher_turns < 1:
            raise ValueError("cross_eval.classroom.max_teacher_turns must be >= 1.")
        if self.retest.replays < 1:
            raise ValueError("cross_eval.retest.replays must be at least 1.")
        if self.interview.attempts < 1:
            raise ValueError("cross_eval.interview.attempts must be at least 1.")
