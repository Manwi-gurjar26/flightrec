"""Diffing two runs, which is a sequence alignment problem and not a ``zip()``.

Two runs of the same task have different numbers of steps. One retried a tool
twice; the other did not. Pairing them index-by-index means every step after the
first insertion is compared against the wrong partner, and the tool then reports
a wall of divergence starting at the retry -- which is both useless and actively
misleading, because the retry is usually not the interesting difference.

So the pipeline is: align first, compare second.

1. A **similarity function** over steps. Deliberately *not* string equality of
   the whole span, and deliberately **not a function of the step's output**:
   two runs that called the same tool with the same arguments and got different
   results are the *same step with a divergent result*, which is precisely the
   thing worth reporting. Folding output into similarity would push those two
   steps apart and align each against something else entirely.
2. **Needleman-Wunsch with affine gap penalties** (the Gotoh variant). A retry
   is one event that produced a run of extra steps, not several independent
   insertions. Affine gaps say so -- though see ``GAP_OPEN`` for how much less
   this buys than it first appears, and for the way an over-tuned version of it
   made the alignment worse rather than better.
3. Only then, a **first genuine divergence**, where "genuine" excludes cosmetic
   differences: same tool, same result, differently worded arguments. That case
   is not hypothetical, it is what the demo model does above temperature 0, and
   reporting it as the cause of a failure would be the tool's most likely way of
   wasting somebody's afternoon.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Any

from flightrec.spans import (
    FR_INPUT,
    FR_OUTPUT,
    GEN_AI_TOOL_NAME,
    Run,
    Span,
    stable_key,
    step_signature,
)

#: Gap penalties, in the same units as the substitution score (which spans
#: [-1, +1]).
#:
#: **The gap-open penalty is a tie-breaker, not a force, and tuning it as a
#: force is a bug.** It exists because two alignments frequently score exactly
#: the same -- three extra steps can be gapped as one block or as a block of two
#: plus a stray, and a linear penalty charges both identically, so which one you
#: get is down to the order the DP happens to evaluate its options in. Opening a
#: gap costing something makes "one event produced one block" the principled
#: answer instead of the lucky one.
#:
#: But it has a ceiling, and the first version of this file was over it. The
#: smallest meaningful difference in similarity is 0.2 (one component of the
#: similarity function), which is 0.4 in score terms. A gap-open bigger than
#: that outbids real evidence: at -0.6 the aligner paired a tool step against a
#: *chat* step, similarity 0.0, purely to avoid opening a second gap. So
#: ``|GAP_OPEN|`` stays below 0.4 and the total cost of a length-1 gap stays
#: where it was.
GAP_OPEN = -0.25
GAP_EXTEND = -0.6

#: Long inputs get truncated before the character-level ratio, which is O(n^2).
#: Agent steps are short; a runaway prompt should not make the diff quadratic in
#: its length.
_RATIO_CAP = 400

_NEG = float("-inf")


class Op(str, Enum):
    """What the alignment says about one pair of steps."""

    MATCH = "match"  # identical, to the byte
    COSMETIC = "cosmetic"  # differently phrased, same result
    CHANGED = "changed"  # aligned, but the outcome differs
    REMOVED = "removed"  # in the left run only
    INSERTED = "inserted"  # in the right run only


@dataclass
class AlignedStep:
    """One column of the alignment."""

    op: Op
    left: Span | None = None
    right: Span | None = None
    left_index: int | None = None
    right_index: int | None = None
    similarity: float = 0.0

    @property
    def genuine(self) -> bool:
        """Is this a difference worth pointing a human at?

        Cosmetic rewording is not. It is a real difference between the two runs
        and it is reported as one -- it just is not a candidate for "where did
        this go wrong".
        """
        return self.op not in (Op.MATCH, Op.COSMETIC)

    @property
    def index(self) -> int:
        """Where this column sits, preferring the left run's numbering."""
        if self.left_index is not None:
            return self.left_index
        return self.right_index if self.right_index is not None else -1


@dataclass
class RunDiff:
    left: Run
    right: Run
    columns: list[AlignedStep]

    @property
    def first_divergence(self) -> AlignedStep | None:
        return next((c for c in self.columns if c.genuine), None)

    @property
    def divergence_index(self) -> int | None:
        column = self.first_divergence
        return None if column is None else column.index

    @property
    def identical(self) -> bool:
        return all(c.op is Op.MATCH for c in self.columns)

    def count(self, op: Op) -> int:
        return sum(1 for c in self.columns if c.op is op)

    @property
    def counts(self) -> dict[Op, int]:
        return {op: self.count(op) for op in Op}


# --- similarity ---------------------------------------------------------------


def similarity(left: Span, right: Span) -> float:
    """How likely two steps are to be *the same step*, in [0, 1].

    Weights: what the step is (0.5) dominates what it was given (0.3), which
    dominates how it turned out (0.2). A step keeps its identity when its
    arguments are reworded; it loses it when it becomes a different tool.
    """
    if left.kind is not right.kind:
        return 0.0

    score = 0.5 if _label(left) == _label(right) else 0.0
    score += 0.3 * _input_similarity(left.attr(FR_INPUT), right.attr(FR_INPUT))
    score += 0.2 if left.status is right.status else 0.0
    return score


def _input_similarity(left: Any, right: Any) -> float:
    if left is None and right is None:
        return 1.0
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        if not keys:
            return 1.0
        return sum(
            _input_similarity(left.get(k), right.get(k)) for k in keys
        ) / len(keys)
    if isinstance(left, str) and isinstance(right, str):
        return _ratio(left, right)
    return 1.0 if left == right else 0.0


def _ratio(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SequenceMatcher(None, left[:_RATIO_CAP], right[:_RATIO_CAP]).ratio()


def _label(span: Span) -> str:
    return str(span.attr(GEN_AI_TOOL_NAME) or span.name)


# --- alignment ----------------------------------------------------------------


def align(
    left: list[Span],
    right: list[Span],
    *,
    gap_open: float = GAP_OPEN,
    gap_extend: float = GAP_EXTEND,
) -> list[AlignedStep]:
    """Needleman-Wunsch with affine gaps (Gotoh), over three score matrices.

    ``M`` ends in a substitution, ``X`` in a gap on the right (a step only the
    left run has), ``Y`` in a gap on the left. Each keeps its own predecessor so
    the traceback knows whether it is continuing a gap or opening one, which a
    single matrix cannot express and is exactly what makes the penalty affine.
    """
    n, m = len(left), len(right)
    if n == 0 or m == 0:
        return [_gap(Op.REMOVED, s, i) for i, s in enumerate(left)] + [
            _gap(Op.INSERTED, s, j) for j, s in enumerate(right)
        ]

    M = [[_NEG] * (m + 1) for _ in range(n + 1)]
    X = [[_NEG] * (m + 1) for _ in range(n + 1)]
    Y = [[_NEG] * (m + 1) for _ in range(n + 1)]
    from_M = [[""] * (m + 1) for _ in range(n + 1)]
    from_X = [[""] * (m + 1) for _ in range(n + 1)]
    from_Y = [[""] * (m + 1) for _ in range(n + 1)]

    M[0][0] = 0.0
    for i in range(1, n + 1):
        X[i][0] = gap_open + gap_extend * i
        from_X[i][0] = "X"
    for j in range(1, m + 1):
        Y[0][j] = gap_open + gap_extend * j
        from_Y[0][j] = "Y"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Substitution score maps similarity onto [-1, +1], so an unrelated
            # pair is a genuine cost rather than a free zero.
            sub = 2.0 * similarity(left[i - 1], right[j - 1]) - 1.0
            M[i][j], from_M[i][j] = _best(
                sub, {"M": M[i - 1][j - 1], "X": X[i - 1][j - 1], "Y": Y[i - 1][j - 1]}
            )
            X[i][j], from_X[i][j] = _best(
                0.0,
                {
                    "M": M[i - 1][j] + gap_open + gap_extend,
                    "X": X[i - 1][j] + gap_extend,
                },
            )
            Y[i][j], from_Y[i][j] = _best(
                0.0,
                {
                    "M": M[i][j - 1] + gap_open + gap_extend,
                    "Y": Y[i][j - 1] + gap_extend,
                },
            )

    state = max(("M", "X", "Y"), key=lambda s: {"M": M, "X": X, "Y": Y}[s][n][m])
    pointers = {"M": from_M, "X": from_X, "Y": from_Y}

    columns: list[AlignedStep] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            state = "Y"
        elif j == 0:
            state = "X"

        previous = pointers[state][i][j] or ("X" if j == 0 else "Y")
        if state == "M":
            columns.append(_pair(left[i - 1], right[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif state == "X":
            columns.append(_gap(Op.REMOVED, left[i - 1], i - 1))
            i -= 1
        else:
            columns.append(_gap(Op.INSERTED, right[j - 1], j - 1))
            j -= 1
        state = previous

    columns.reverse()
    return columns


def _best(bonus: float, options: dict[str, float]) -> tuple[float, str]:
    state = max(options, key=lambda k: options[k])
    value = options[state]
    return (_NEG if value == _NEG else value + bonus), state


def _gap(op: Op, span: Span, index: int) -> AlignedStep:
    if op is Op.REMOVED:
        return AlignedStep(op=op, left=span, left_index=index)
    return AlignedStep(op=op, right=span, right_index=index)


def _pair(left: Span, right: Span, i: int, j: int) -> AlignedStep:
    return AlignedStep(
        op=_classify(left, right),
        left=left,
        right=right,
        left_index=i,
        right_index=j,
        similarity=similarity(left, right),
    )


def _classify(left: Span, right: Span) -> Op:
    if step_signature(left) == step_signature(right):
        return Op.MATCH
    same_outcome = left.status is right.status and stable_key(
        left.attr(FR_OUTPUT)
    ) == stable_key(right.attr(FR_OUTPUT))
    if _label(left) == _label(right) and same_outcome:
        return Op.COSMETIC
    return Op.CHANGED


# --- the two pairings ---------------------------------------------------------


def diff_runs(left: Run, right: Run, **kwargs: Any) -> RunDiff:
    """Align two runs and classify every column. The real one."""
    return RunDiff(
        left=left, right=right, columns=align(left.steps(), right.steps(), **kwargs)
    )


def diff_runs_by_index(left: Run, right: Run) -> RunDiff:
    """Pair step *i* with step *i*. The baseline the alignment is measured against.

    Committed rather than described, because "alignment beats zip" is a claim in
    the README and a claim in a README needs something to run against.
    """
    left_steps, right_steps = left.steps(), right.steps()
    columns: list[AlignedStep] = []
    for index in range(max(len(left_steps), len(right_steps))):
        if index >= len(left_steps):
            columns.append(_gap(Op.INSERTED, right_steps[index], index))
        elif index >= len(right_steps):
            columns.append(_gap(Op.REMOVED, left_steps[index], index))
        else:
            columns.append(_pair(left_steps[index], right_steps[index], index, index))
    return RunDiff(left=left, right=right, columns=columns)
