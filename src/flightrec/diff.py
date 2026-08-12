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


#: How certain a leftover pairing has to look before the move pass will claim
#: it. Identical steps are rescued regardless; this governs the second pass,
#: which is matching steps whose *content* differs -- the moved step that also
#: changed. 0.9 keeps it to same-kind, same-tool, same-outcome candidates.
MOVE_CONFIDENCE = 0.9

#: How certain an existing "changed" pairing has to be before the move pass
#: leaves it alone.
#:
#: A separate constant because it is a separate decision, and sharing one number
#: between them was a bug: at 0.9 a step paired with its *positional* neighbour
#: at 0.955 was protected, and the genuine counterpart it had been moved away
#: from never got a look. The principled value is total certainty. Similarity
#: ignores output, so 1.0 means same kind, same tool, same arguments, same
#: outcome class -- "the same step with a different result", which is the diff's
#: most valuable finding and the one thing here worth defending from a
#: reshuffle. Anything less is a guess and can be re-examined.
PROTECT_CONFIDENCE = 1.0


class Op(str, Enum):
    """What the alignment says about the *content* of one pair of steps.

    Whether a pair also moved is tracked separately, on ``AlignedStep.moved``.
    The two are independent: a step can be reordered and unchanged, reordered
    and changed, or changed where it stands, and collapsing that into one
    enumeration would force the diff to pick which half of the truth to report.
    """

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
    moved: bool = False
    """This pair breaks the run's step order: the two runs did it at different points."""

    @property
    def genuine(self) -> bool:
        """Is this a difference worth pointing a human at?

        Cosmetic rewording is not. It is a real difference between the two runs
        and it is reported as one -- it just is not a candidate for "where did
        this go wrong". A move is: doing the same steps in a different order is
        exactly the kind of thing that changes an outcome.
        """
        return self.moved or self.op not in (Op.MATCH, Op.COSMETIC)

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
        """Same steps, same content, same order.

        The order clause is not pedantry: every column of a pure reordering is a
        ``MATCH``, so checking content alone reports two runs that did the same
        things in a different sequence as the same run.
        """
        return all(c.op is Op.MATCH and not c.moved for c in self.columns)

    @property
    def moved_count(self) -> int:
        return sum(1 for c in self.columns if c.moved)

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
    detect_moves: bool = True,
) -> list[AlignedStep]:
    """Align two step sequences, then recover any reorderings.

    Two passes, because one algorithm cannot do both jobs. Needleman-Wunsch
    produces a *monotonic* correspondence -- step order preserved on both sides
    -- which is the right model for insertions, deletions and edits and is
    structurally incapable of expressing a transposition. Set ``detect_moves``
    to ``False`` to see that first pass on its own.
    """
    columns = _align_monotonic(left, right, gap_open, gap_extend)
    return _recover_moves(left, right, columns) if detect_moves else columns


def _align_monotonic(
    left: list[Span],
    right: list[Span],
    gap_open: float,
    gap_extend: float,
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


# --- recovering reorderings ---------------------------------------------------


def _recover_moves(
    left: list[Span], right: list[Span], columns: list[AlignedStep]
) -> list[AlignedStep]:
    """Re-pair steps the monotonic pass could not place, and mark what moved.

    Only steps it was *not* confident about are touched: anything it matched
    exactly or called cosmetic keeps its partner. What is left over -- gaps and
    weak substitutions -- gets a second look, because a transposition surfaces
    in the first pass as either shape and neither is recoverable from the other.

    A reordering preserves length, which is what makes it nastier than it looks:
    with the same number of steps on both sides the aligner can pair everything
    positionally for the price of a few substitutions, cheaper than opening the
    gaps that would tell the truth. So this pass cannot just examine gap blocks
    the way a line differ does -- in the common case there are no gaps at all.
    """
    confident: dict[int, int] = {}
    weak_left: list[int] = []
    weak_right: list[int] = []

    for column in columns:
        # A changed pair is not automatically in doubt. "Same tool, same
        # arguments, different result" is the diff's most valuable finding, and
        # it scores near 1.0 because similarity ignores output -- so re-opening
        # those lets the exact-match pass hand the counterpart to an unrelated
        # step that happens to be identical to something. Only weak pairings go
        # back in the pool.
        settled = column.op in (Op.MATCH, Op.COSMETIC) or (
            column.op is Op.CHANGED and column.similarity >= PROTECT_CONFIDENCE
        )
        if settled and column.left_index is not None and column.right_index is not None:
            confident[column.left_index] = column.right_index
            continue
        if column.left_index is not None:
            weak_left.append(column.left_index)
        if column.right_index is not None:
            weak_right.append(column.right_index)

    if not weak_left or not weak_right:
        return columns

    rescued = _rescue(left, right, weak_left, weak_right)
    if not rescued:
        return columns

    pairs = {**confident, **rescued}
    return _rebuild(left, right, pairs, _out_of_order(pairs))


def _rescue(
    left: list[Span],
    right: list[Span],
    weak_left: list[int],
    weak_right: list[int],
) -> dict[int, int]:
    """Pair up leftover steps, certainty first, best candidate first.

    Both passes go through the candidate list globally rather than walking the
    left steps in order, and that is not a detail. Taking them in left order
    lets an early step claim a partner that a later one matches better -- in a
    run with two steps deleted, a deleted step would reach the changed step's
    counterpart first and pair with it, which loses the deletion *and* the edit
    in one move. Best-first gives every candidate its strongest partner.
    """
    available = set(weak_right)
    rescued: dict[int, int] = {}

    def take(order: list[tuple[Any, ...]]) -> None:
        for *_, i, j in order:
            if i in rescued or j not in available:
                continue
            rescued[i] = j
            available.discard(j)

    # Pass one: identical content. A step that appears verbatim on both sides is
    # the same step whatever the alignment decided, so this needs no threshold --
    # only a tie-break, by distance, because when a run contains the same step
    # twice, assuming the far one moved is a worse guess than the near one.
    take(
        sorted(
            (abs(j - i), i, j)
            for i in weak_left
            for j in weak_right
            if step_signature(left[i]) == step_signature(right[j])
        )
    )

    # Pass two: what is left, by similarity. This is what catches the step that
    # was moved *and* edited -- its content differs, so pass one cannot see it,
    # and similarity ignores output precisely so that it still matches here.
    # Anything below the threshold is left unpaired: an unexplained step is a
    # better answer than a confident wrong pairing.
    scored = [
        (-score, abs(j - i), i, j)
        for i in weak_left
        if i not in rescued
        for j in available
        if (score := similarity(left[i], right[j])) >= MOVE_CONFIDENCE
    ]
    take(sorted(scored))

    return rescued


def _out_of_order(pairs: dict[int, int]) -> set[int]:
    """Which pairs break the run's order, given that most of it did not move.

    The backbone is the longest run of pairs whose order is consistent on both
    sides -- the longest increasing subsequence of right indices, read in left
    order. Everything off it is what moved. Calling the *shorter* side the move
    is the whole judgement here: two runs that share nine steps in order and
    disagree about two did not reorder nine of them.
    """
    ordered = sorted(pairs)
    values = [pairs[i] for i in ordered]
    if not values:
        return set()

    # O(n^2) longest increasing subsequence. Agent runs are tens of steps.
    best = [1] * len(values)
    previous = [-1] * len(values)
    for i in range(len(values)):
        for j in range(i):
            if values[j] < values[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                previous[i] = j

    end = max(range(len(values)), key=lambda i: best[i])
    backbone = set()
    while end != -1:
        backbone.add(ordered[end])
        end = previous[end]
    return set(ordered) - backbone


def _rebuild(
    left: list[Span],
    right: list[Span],
    pairs: dict[int, int],
    moved: set[int],
) -> list[AlignedStep]:
    """Lay the new correspondence back out as an ordered column list.

    Ordered by the left run, which stops a move from scrambling the display:
    the reordered step is shown where the left run did it, pointing at where the
    right run did it. Right-only steps are placed after whichever left step
    precedes them on the right-hand side.
    """
    taken = set(pairs.values())
    pending: dict[int | None, list[int]] = {}
    for index in range(len(right)):
        if index in taken:
            continue
        anchor = max(
            (i for i, j in pairs.items() if j < index),
            key=lambda i: pairs[i],
            default=None,
        )
        pending.setdefault(anchor, []).append(index)

    columns: list[AlignedStep] = [
        _gap(Op.INSERTED, right[j], j) for j in pending.get(None, [])
    ]
    for index in range(len(left)):
        if index in pairs:
            partner = pairs[index]
            column = _pair(left[index], right[partner], index, partner)
            column.moved = index in moved
            columns.append(column)
        else:
            columns.append(_gap(Op.REMOVED, left[index], index))
        for j in pending.get(index, []):
            columns.append(_gap(Op.INSERTED, right[j], j))
    return columns


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
