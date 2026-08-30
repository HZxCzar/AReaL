"""Depth of a tutoring episode, and whether the tutor is still saying anything new.

The thing being measured is not repetition for its own sake. Episodes that run
long are episodes where the student is stuck, and the objective is for those to
end in a solve. This module supplies the two halves of that picture:

  - how often a tutor turn at a given depth actually gets the student unstuck
    (the hazard -- the objective), and
  - how much of that turn the tutor had already said (the mechanism -- one
    reason the hazard falls).

DEPTH

A turn's depth is how many times the student had already answered incorrectly
when the tutor produced it. Depth 1 is the reply to the student's first wrong
attempt, depth k the reply to the k-th. It equals the 1-based turn index,
because an episode only continues while the student is wrong.

Measured on 1520 baseline episodes, the hazard and the overlap move in opposite
directions and do so monotonically:

    depth      1      2      3      4      5      6      7      8      9     10
    resolve  38.2%  28.0%  22.7%  12.2%  15.4%  10.5%   0.8%   6.0%   0.0%   2.0%
    overlap   --    0.218  0.304  0.348  0.391  0.467  0.576  0.606  0.643  0.649

OVERLAP

Fraction of THIS turn's word bigrams that already appeared in the previous turn.
Containment rather than a symmetric measure: the question is how much of what
the tutor just said is new, and a short restatement of a long earlier turn
should score high.

Text is lowercased and split into words and LaTeX command names; a small
stopword list is removed so the score reflects content rather than the phrasing
that wraps every tutor message. Turns with fewer than MIN_BIGRAMS content
bigrams score None -- too short to say anything about.

What the score does and does not catch, checked against real pairs:

    0.06  different error corrected, different content            not a repeat
    0.22  same problem, a genuinely different route offered       not a repeat
    0.47  restates the previous turn and adds one concrete step   the boundary
    0.63  same claim reworded, nothing added                      a repeat
    0.83  same sentence, one word changed                         a repeat

It is a surface measure. A tutor that says the same thing in new words scores
low, so the rate UNDER-counts. Shared boilerplate and the problem's own symbols
push short turns up, so it can over-count there -- hence MIN_BIGRAMS.

REPEAT_CONTAINMENT_THRESHOLD sits at 0.5, which the table above places right at
"restates and adds one step". That is a real ambiguity and it is why the mean
containment is the more robust of the two statistics: it needs no threshold and
it moved 0.218 -> 0.649 across depth where the thresholded rate moved
8.4% -> 63.6%. Read the mean first and the rate as a summary.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

REPEAT_CONTAINMENT_THRESHOLD = 0.5
MIN_BIGRAMS = 5

_TOKEN = re.compile(r"[a-z0-9]+|\\[a-zA-Z]+")
_STOPWORDS = frozenset(
    """the a an is are was were be been to of in on for and or not you your i it
    this that these those we us he she they them so if then than as at by with
    from can could should would will shall do does did have has had but there
    here what which who when where how why now next step let s t re ve ll m d""".split()
)


def normalize_exact_teacher_output(text: str | None) -> str:
    """Canonical text for exact-repeat detection.

    Only whitespace is normalized. Case, punctuation, words, numbers, and math
    remain significant so a stable teaching scaffold with new content is not
    mistaken for a repeat.
    """
    return " ".join((text or "").split())


# Depth buckets. Kept coarse so the metric namespace stays readable, and cut
# where the measured behaviour changes: 1-2 is the healthy end, 3-4 is where
# the overlap crosses 20%, 5-7 is where the hazard reaches single digits, 8-10
# is the tail that almost never resolves.
DEPTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("d1", 1, 1),
    ("d2", 2, 2),
    ("d3_4", 3, 4),
    ("d5_7", 5, 7),
    ("d8_10", 8, 10),
)
# Depth at which an episode counts as stuck: the tutor has explained twice and
# the student is still wrong.
STUCK_DEPTH = 3


def content_tokens(text: str | None) -> list[str]:
    """Lowercased words and LaTeX command names, stopwords dropped."""
    return [word for word in _TOKEN.findall((text or "").lower())
            if word not in _STOPWORDS]


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def bigram_containment(current: str | None, previous: str | None) -> float | None:
    """How much of `current` was already in `previous`, in [0, 1].

    None when either side has fewer than MIN_BIGRAMS content bigrams, which
    means the pair is too short for the score to mean anything.
    """
    current_bigrams = _bigrams(content_tokens(current))
    previous_bigrams = _bigrams(content_tokens(previous))
    if len(current_bigrams) < MIN_BIGRAMS or len(previous_bigrams) < MIN_BIGRAMS:
        return None
    return len(current_bigrams & previous_bigrams) / len(current_bigrams)


def is_repeat(containment: float | None) -> bool:
    return containment is not None and containment >= REPEAT_CONTAINMENT_THRESHOLD


def bucket_for_depth(depth: int) -> str | None:
    for name, low, high in DEPTH_BUCKETS:
        if low <= depth <= high:
            return name
    return None


def depth_metrics(
    messages: Iterable[str | None],
    resolved: Iterable[bool],
) -> dict[str, float]:
    """Per-episode depth counters, given the tutor's turns in order.

    COUNTS, not rates. A batch mean of a per-episode rate would weight an
    episode with one deep turn the same as an episode with eight, and would
    have to invent a value for episodes with none. Divide the means instead:

        resolve rate at depth 3-4  =  mean(depth/d3_4/resolved)
                                      / mean(depth/d3_4/turns)
        repeat rate at depth 3-4   =  mean(depth/d3_4/repeats)
                                      / mean(depth/d3_4/repeat_scored)
        mean overlap at depth 3-4  =  mean(depth/d3_4/containment_sum)
                                      / mean(depth/d3_4/repeat_scored)

    The two episode-level gauges are the objective itself:

        depth/stuck_episode         reached STUCK_DEPTH
        depth/stuck_episode_solved  reached it and still ended solved

    and their ratio of means is "of the episodes where the student was still
    wrong after two explanations, how many did the tutor eventually get there".
    Baseline is 46.3%; with the hazard not decaying it would be 74.7%.
    """
    messages = list(messages)
    resolved = list(resolved)
    metrics: dict[str, float] = {}
    names = [name for name, _, _ in DEPTH_BUCKETS] + ["deep"]
    for name in names:
        for suffix in ("turns", "resolved", "repeats", "repeat_scored",
                       "containment_sum"):
            metrics[f"depth/{name}/{suffix}"] = 0.0

    previous: str | None = None
    for index, message in enumerate(messages):
        depth = index + 1
        buckets = [bucket_for_depth(depth)]
        if depth >= STUCK_DEPTH:
            buckets.append("deep")
        containment = bigram_containment(message, previous) if previous else None
        for bucket in buckets:
            if bucket is None:
                continue
            metrics[f"depth/{bucket}/turns"] += 1.0
            if index < len(resolved) and resolved[index]:
                metrics[f"depth/{bucket}/resolved"] += 1.0
            if containment is not None:
                metrics[f"depth/{bucket}/repeat_scored"] += 1.0
                metrics[f"depth/{bucket}/containment_sum"] += containment
                if is_repeat(containment):
                    metrics[f"depth/{bucket}/repeats"] += 1.0
        previous = message

    stuck = len(messages) >= STUCK_DEPTH
    metrics["depth/stuck_episode"] = float(stuck)
    metrics["depth/stuck_episode_solved"] = float(stuck and any(resolved))
    metrics["depth/turns"] = float(len(messages))
    return metrics
