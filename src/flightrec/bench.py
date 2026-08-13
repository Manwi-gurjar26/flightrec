"""The measurement harness. Every number in the README comes out of here.

The rule this project was written under is that a claim without a committed,
reproducible measurement is not a claim. So each metric below states what it
compares, against what baseline, and what it does *not* establish -- because a
benchmark that only reports its good news is marketing with a test suite.

Run it with ``flightrec bench``. It needs no API key and no network: the demo
agent runs against a seeded stub model, so anyone can reproduce these numbers.
"""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterator

from flightrec.demo.agent import ResearchAgent
from flightrec.demo.tools import FaultConfig
from flightrec import tracer as tracer_module
from flightrec.determinism import SystemClock
from flightrec.diff import Op, diff_runs, diff_runs_by_index
from flightrec.replay import replay_run
from flightrec.sinks import MemorySink
from flightrec.tracer import Tracer
from flightrec.spans import (
    FR_INPUT,
    FR_OUTPUT,
    FR_SERVED,
    GEN_AI_TOOL_NAME,
    Run,
    Span,
    SpanKind,
    first_divergence,
    step_signature,
    trajectory,
)

DEFAULT_RUNS = 40


@dataclass
class Measurement:
    """One row of the README's measurement table, with its baseline attached."""

    name: str
    value: float
    unit: str
    baseline: float | None = None
    baseline_label: str = ""
    detail: str = ""
    breakdown: list[str] = field(default_factory=list)
    caveat: str = ""

    def format_value(self) -> str:
        return _format(self.value, self.unit)

    def format_baseline(self) -> str:
        return "-" if self.baseline is None else _format(self.baseline, self.unit)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "baseline": self.baseline,
            "baseline_label": self.baseline_label,
            "detail": self.detail,
            "breakdown": self.breakdown,
            "caveat": self.caveat,
        }


def _shape(span: Span) -> tuple[str, str]:
    """What a step *is*, ignoring what it was given and what it returned."""
    return (span.kind.value, str(span.attr(GEN_AI_TOOL_NAME) or span.name))


def _format(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "x":
        return f"{value:.2f}x"
    if unit == "ms":
        return f"{value:.3f}ms"
    return f"{value:.3f}"


def record(seed: int, faults: FaultConfig | None = None) -> Run:
    return ResearchAgent(seed=seed, faults=faults or FaultConfig.realistic()).run().run


# --- 1. replay fidelity -------------------------------------------------------


def measure_replay_fidelity(seeds: range) -> Measurement:
    """Does replaying a recording reproduce its step sequence?

    The baseline is the thing people actually do without this tool: run the task
    again and watch. That is measured by re-running the agent with fresh RNG
    streams -- no recorded seed, no recorded tool results -- and comparing what
    comes out against the original recording.

    **This baseline is generous to the no-tooling case.** The demo agent is
    seeded end to end, so "run it again with the same seed" would reproduce the
    run perfectly and prove nothing about replay. A real provider offers no such
    seed, so the honest comparison is against a re-run that does not have the
    recording's state -- and against a real provider the baseline would be worse
    than this, not better.
    """
    faithful = 0
    rerun_matched = 0
    rng = random.Random(20240811)

    for seed in seeds:
        recorded = record(seed)
        replayed = replay_run(recorded).run
        if first_divergence(recorded, replayed) is None:
            faithful += 1

        # "Run it again and watch", with none of the run's state restored.
        fresh = record(rng.randrange(10_000, 1_000_000))
        if first_divergence(recorded, fresh) is None:
            rerun_matched += 1

    total = len(seeds)
    return Measurement(
        name="Replay fidelity",
        value=100.0 * faithful / total,
        unit="%",
        baseline=100.0 * rerun_matched / total,
        baseline_label="re-running the task without the recording",
        detail=f"{faithful}/{total} replays reproduced the recorded step sequence",
        caveat=(
            "trajectory-identical, not byte-identical: a replay cannot recover "
            "the recording's span IDs or wall clock"
        ),
    )


def measure_partial_replay_fidelity(seeds: range) -> Measurement:
    """The stronger claim: replay from *any* step, not just from the start.

    The metric above replays each recording one way. This one replays it at
    every possible cut point and asks three things of each:

    * the steps before the cut reproduce the recording exactly;
    * exactly that many steps were served, so nothing was quietly re-executed;
    * replaying the same cut twice is byte-identical.

    All three, or the cut point does not count. "Replay from step N" is the
    sentence the tool is sold on, and it was being measured at N=0.

    The baseline is the same one as above, applied per cut: re-run the task
    without the recording and see whether its first N steps happen to match.
    """
    good = 0
    total = 0
    baseline_good = 0
    rng = random.Random(20240811)

    for seed in seeds:
        recorded = record(seed)
        prefix = trajectory(recorded)
        fresh = trajectory(record(rng.randrange(10_000, 1_000_000)))

        for cut in range(len(prefix) + 1):
            total += 1
            if fresh[:cut] == prefix[:cut]:
                baseline_good += 1

            replayed = replay_run(recorded, from_step=cut)
            if trajectory(replayed.run)[:cut] != prefix[:cut]:
                continue
            if replayed.served != cut:
                continue
            if replay_run(recorded, from_step=cut).run.canonical_json() != (
                replayed.run.canonical_json()
            ):
                continue
            good += 1

    return Measurement(
        name="Partial replay fidelity",
        value=100.0 * good / total if total else 0.0,
        unit="%",
        baseline=100.0 * baseline_good / total if total else 0.0,
        baseline_label="re-running the task and hoping the first N steps match",
        detail=(
            f"{good}/{total} cut points reproduced their prefix exactly, served "
            f"exactly that many steps, and repeated byte-identically"
        ),
        caveat=(
            "says nothing about the steps *after* the cut, which are supposed to "
            "differ -- that is the point of cutting. The baseline scores above "
            "zero only because a cut of 0 steps trivially matches, and because "
            "runs that fire no faults share a prefix"
        ),
    )


# --- 3. divergence localization ----------------------------------------------


@dataclass
class KindResult:
    """How both pairings did against one class of mutation."""

    kind: str
    total: int = 0
    aligned: int = 0
    naive: int = 0
    aligned_pairing: float = 0.0
    naive_pairing: float = 0.0
    blamed: int = 0
    blame_total: int = 0
    structure: float = 0.0
    structure_total: int = 0

    def rate(self, hits: int) -> float:
        return 100.0 * hits / self.total if self.total else 0.0

    def line(self) -> str:
        # Blame and structure carry their denominators because they do not
        # apply to every mutant. "structure 100%" over eight of thirty-eight
        # cases is a different claim from "structure 100%", and printing the
        # first as the second is how a number gets quoted without its limits.
        blame = (
            f"   blame {100.0 * self.blamed / self.blame_total:5.1f}% "
            f"({self.blame_total}/{self.total})"
            if self.blame_total
            else ""
        )
        structure = (
            f"   structure {self.structure:5.1f}% "
            f"({self.structure_total}/{self.total})"
            if self.structure_total
            else ""
        )
        return (
            f"{self.kind:<14} localized {self.rate(self.aligned):5.1f}% "
            f"(zip {self.rate(self.naive):5.1f}%)   "
            f"pairing {self.aligned_pairing:5.1f}% (zip {self.naive_pairing:5.1f}%)"
            f"{blame}{structure}"
        )


def localization_by_kind(
    seeds: range, detect_moves: bool = True
) -> list[KindResult]:
    """Score both pairings against every class of mutation, separately.

    Separately, because an average over mutation classes is a number that can
    hide a total failure behind five easy passes -- and the failure is the part
    worth knowing about.
    """
    block = recovery_block()
    results = []

    for kind in MUTATORS:
        # One RNG per kind, seeded identically, so adding a mutation class does
        # not shift the mutants every other class gets.
        rng = random.Random(20240812)
        result = KindResult(kind=kind)
        aligned_pairing, naive_pairing, structure_sum = 0.0, 0.0, 0.0

        for seed in seeds:
            mutation = mutate(record(seed), block, rng, kind=kind)
            if mutation is None:
                continue
            result.total += 1
            aligned = diff_runs(
                mutation.original, mutation.mutant, detect_moves=detect_moves
            )
            naive = diff_runs_by_index(mutation.original, mutation.mutant)

            result.aligned += int(mutation.localized_by(aligned))
            result.naive += int(mutation.localized_by(naive))
            aligned_pairing += mutation.pairing_accuracy(aligned)
            naive_pairing += mutation.pairing_accuracy(naive)

            blamed = mutation.blamed_correctly(aligned)
            if blamed is not None:
                result.blame_total += 1
                result.blamed += int(blamed)

            structure = mutation.structure_accuracy(aligned)
            if structure is not None:
                result.structure_total += 1
                structure_sum += structure

        if result.total:
            result.aligned_pairing = 100.0 * aligned_pairing / result.total
            result.naive_pairing = 100.0 * naive_pairing / result.total
        if result.structure_total:
            result.structure = 100.0 * structure_sum / result.structure_total
        results.append(result)

    return results


def measure_divergence_localization(seeds: range) -> Measurement:
    """Given a known injected change, does the diff pair the right two steps?

    Every mutant carries a structural edit and at least one changed tool result.
    The structural edit is what makes this hard: it shifts the steps after it,
    so index-by-index pairing compares the changed step against somebody else's.

    Measured as *pairing*, not as "reported a difference at index k". Naive
    pairing also reports a difference at index k -- it reports one nearly
    everywhere after an insertion -- so scoring that would hand it a pass for
    being wrong in the right place.
    """
    results = localization_by_kind(seeds)
    total = sum(r.total for r in results)
    aligned = sum(r.aligned for r in results)
    naive = sum(r.naive for r in results)
    worst = min(results, key=lambda r: r.rate(r.aligned))

    return Measurement(
        name="Divergence localization",
        value=100.0 * aligned / total if total else 0.0,
        unit="%",
        baseline=100.0 * naive / total if total else 0.0,
        baseline_label="index-by-index zip() pairing",
        detail=(
            f"{aligned}/{total} mutants across {len(results)} mutation classes had "
            f"every changed step paired with its counterpart"
            + (
                ""
                if worst.rate(worst.aligned) >= 100.0
                else f"; worst class is {worst.kind} at {worst.rate(worst.aligned):.1f}%"
            )
        ),
        breakdown=[r.line() for r in results],
        caveat=(
            "the headline averages one metric over unequal classes -- read the "
            "per-class lines. Four things are scored and they disagree: "
            "'localized' is the changed step paired with its counterpart, "
            "'pairing' is every surviving step, 'blame' is whether the first "
            "reported divergence is the real one (only where nothing structural "
            "happened), and 'structure' is whether added and removed steps are "
            "reported as added and removed. Structure is the one that still "
            "fails: adjacent-edit scores 0% because a replacement is "
            "indistinguishable from a tool substitution, which the diff is "
            "asked to pair. Blank columns mean the metric has no answer for "
            "that class, not a perfect score"
        ),
    )


@dataclass
class Mutation:
    """A recording, a copy of it edited in a known way, and the ground truth.

    ``expected`` is the correspondence a perfect diff would find: original step
    index to mutant step index, for every step that survived. A single offset
    was enough while the only edit was an insertion; deletions and reorderings
    need the whole map, and needing the whole map is what makes the harder
    mutations scoreable at all.
    """

    kind: str
    original: Run
    mutant: Run
    expected: dict[int, int]
    changed: set[int]
    removed: set[int] = field(default_factory=set)

    def _equivalent(self, got: int | None, truth: int | None) -> bool:
        """Did the diff pick a mutant step indistinguishable from the right one?

        Duplicating a step verbatim leaves two identical steps, and no amount of
        analysis says which one the other run "meant" -- they are the same
        content in the same run. Scoring the diff on that choice measures a coin
        toss, so an identical signature counts as the same answer.
        """
        if got == truth:
            return True
        if got is None or truth is None:
            return False
        steps = self.mutant.steps()
        return step_signature(steps[got]) == step_signature(steps[truth])

    def localized_by(self, diff: Any) -> bool:
        """Is every changed step paired with its counterpart and marked changed?

        This is the README's metric. It deliberately says nothing about the
        steps that were *not* changed -- a diff can pair those badly and still
        point a human at the right place.
        """
        for step in self.changed:
            column = next((c for c in diff.columns if c.left_index == step), None)
            if (
                column is None
                or not self._equivalent(column.right_index, self.expected.get(step))
                # REPLACED counts: the substitute class turns one tool call into
                # a different one, and "this is a different step" is a correct
                # report of that, not a miss.
                or column.op not in (Op.CHANGED, Op.REPLACED)
            ):
                return False
        return True

    @property
    def structural(self) -> bool:
        """Did this mutation add, remove or reorder anything?"""
        if self.removed or len(self.mutant.steps()) != len(self.expected):
            return True
        order = [self.expected[k] for k in sorted(self.expected)]
        return any(a > b for a, b in zip(order, order[1:]))

    def blamed_correctly(self, diff: Any) -> bool | None:
        """Does the diff point a human at the step that actually changed?

        ``None`` when the mutation is structural, because then an insertion or a
        reordering legitimately comes first and blaming it is not wrong. Only
        where the *sole* genuine difference is a changed result is there a right
        answer -- and that is the case worth checking, because it is where a
        cosmetic rewording can be blamed instead.
        """
        if self.structural:
            return None
        column = diff.first_divergence
        return column is not None and column.left_index in self.changed

    def structure_accuracy(self, diff: Any) -> float | None:
        """Are added and removed steps reported as added and removed?

        Neither of the other two metrics can see this. ``pairing_accuracy`` only
        scores steps that *survived*, and ``localized_by`` only looks at the
        changed one -- so a diff that pairs two unrelated steps as "changed"
        where the truth is "one removed, one added" scores 100% on both while
        telling a human that two steps correspond when they have nothing to do
        with each other.

        Only *decidable* additions and removals are scored. Three separate
        rounds of this metric reported failures that were its own fault, all the
        same mistake -- demanding one description where two are equally true:

        * a step injected as a copy of one deleted elsewhere is a **move**, and
          the diff is right to call it one;
        * a step duplicated verbatim leaves two identical steps, and which twin
          is "the extra one" has no answer;
        * a ``fetch_page`` deleted here and a *different* ``fetch_page`` added
          there is either "one call removed and another added" or "one call
          whose URL changed", and nothing distinguishes them.

        So a removed step is scored only when no added step could be taken for
        it, and vice versa -- same kind and same tool name is enough to be
        confusable. That is a structural test rather than a tuned threshold, and
        deliberately not the diff's own 0.9, which would be scoring the diff
        against its own opinion. It leaves the genuinely unambiguous case
        scored: a step whose tool appears nowhere in the other run has a fact of
        the matter, and ``adjacent-edit`` still fails on it.

        ``None`` comes back when nothing decidable is left.
        """
        mutant_steps = self.mutant.steps()
        original_steps = self.original.steps()
        injected = set(range(len(mutant_steps))) - set(self.expected.values())

        # Everything the *other* run contains, plus everything that survived.
        # An injected step is only accountable if nothing anywhere could be
        # mistaken for it -- including a step it was copied from before that
        # step was edited, which is why the original run is scanned whole.
        elsewhere = {step_signature(step) for step in original_steps}
        elsewhere |= {step_signature(mutant_steps[j]) for j in self.expected.values()}

        # Shape = kind plus tool name. Two steps of the same shape on opposite
        # sides of an edit are interchangeable descriptions of the same event.
        removed_shapes = {_shape(original_steps[i]) for i in self.removed}
        injected_shapes = {_shape(mutant_steps[j]) for j in injected}
        injected_signatures = {step_signature(mutant_steps[j]) for j in injected}

        accountable_injected = {
            j
            for j in injected
            if step_signature(mutant_steps[j]) not in elsewhere
            and _shape(mutant_steps[j]) not in removed_shapes
        }
        accountable_removed = {
            i
            for i in self.removed
            if step_signature(original_steps[i]) not in injected_signatures
            and _shape(original_steps[i]) not in injected_shapes
        }

        expected_gaps = len(accountable_injected) + len(accountable_removed)
        if not expected_gaps:
            return None

        # A step counts as "reported gone" when the diff gaps it, and also when
        # the diff pairs it but says REPLACED -- which states outright that the
        # two steps do not correspond. That is the same claim a gap makes, in a
        # form that keeps the position visible, and refusing to accept it would
        # be scoring a presentation choice rather than a correctness one.
        disowned_left = {
            c.left_index
            for c in diff.columns
            if c.right_index is None or c.op is Op.REPLACED
        }
        disowned_right = {
            c.right_index
            for c in diff.columns
            if c.left_index is None or c.op is Op.REPLACED
        }
        hits = len(accountable_removed & disowned_left) + len(
            accountable_injected & disowned_right
        )
        return hits / expected_gaps

    def pairing_accuracy(self, diff: Any) -> float:
        """Fraction of surviving steps paired with their true counterpart.

        The harsher metric, and the one that exposes what localization hides: a
        diff can find the changed step while mispairing everything around it,
        and a human reading that column list is being misled about the rest of
        the run even though the headline number says it worked.
        """
        if not self.expected:
            return 1.0
        paired = {
            c.left_index: c.right_index
            for c in diff.columns
            if c.left_index is not None
        }
        hits = sum(
            1 for k, v in self.expected.items() if self._equivalent(paired.get(k), v)
        )
        return hits / len(self.expected)


def recovery_block(faults: FaultConfig | None = None) -> list[Span]:
    """Two real recorded steps: the agent's second URL guess and the model call
    that decided on it.

    Real steps rather than synthetic filler, because the alignment's job is to
    handle the insertions that actually occur. Fabricating a block that the
    similarity function happens to find easy would be scoring the exam it wrote.
    """
    for seed in range(60):
        run = record(seed, faults)
        steps = run.steps()
        for index, step in enumerate(steps):
            # Model steps carry a prompt string here, tool steps an argument
            # dict, so this cannot assume either shape.
            inputs = step.attr(FR_INPUT)
            url = inputs.get("url", "") if isinstance(inputs, dict) else ""
            if (
                step.kind is SpanKind.TOOL
                and step.attr(GEN_AI_TOOL_NAME) == "fetch_page"
                and str(url).endswith("-annual")
                and index >= 1
            ):
                return [s.model_copy(deep=True) for s in steps[index - 1 : index + 1]]
    raise RuntimeError("no recorded run contained a second-guess recovery block")


#: A row of a mutant under construction: which original step it came from (or
#: ``None`` if it was injected), and the span itself.
Row = tuple[int | None, Span]


def _inject(block: list[Span]) -> list[Row]:
    return [(None, span.model_copy(deep=True)) for span in block]


def _change_a_tool(
    rows: list[Row],
    rng: random.Random,
    after: int = 0,
    exclude: set[int] | None = None,
) -> int | None:
    """Perturb one surviving tool step's output; return which original it was.

    Tool steps rather than model steps: a tool result is where a divergence
    actually originates, and it is the kind of change somebody would be trying
    to find.
    """
    barred = exclude or set()
    candidates = [
        i
        for i in range(after, len(rows))
        if rows[i][0] is not None
        and rows[i][0] not in barred
        and rows[i][1].kind is SpanKind.TOOL
    ]
    if not candidates:
        return None
    index = rng.choice(candidates)
    origin, span = rows[index]
    span.attributes[FR_OUTPUT] = _perturb(span.attr(FR_OUTPUT))
    return origin


def _mutate_insert(rows: list[Row], block: list[Span], rng: random.Random):
    """One recovery block spliced in. The original, easiest case."""
    at = rng.randrange(1, len(rows) // 2)
    rows = rows[:at] + _inject(block) + rows[at:]
    changed = _change_a_tool(rows, rng, after=at + len(block))
    return rows, {changed} if changed is not None else set()


def _mutate_delete(rows: list[Row], block: list[Span], rng: random.Random):
    """Two steps removed: the mutant is *shorter*, which shifts the other way."""
    at = rng.randrange(1, len(rows) // 2)
    rows = rows[:at] + rows[at + 2 :]
    changed = _change_a_tool(rows, rng, after=at)
    return rows, {changed} if changed is not None else set()


def _mutate_insert_and_delete(rows: list[Row], block: list[Span], rng: random.Random):
    """Both, in different places, so the offset is not even constant.

    The injected steps are made unmatchable, and that is what the class is worth
    measuring. Splicing a real recovery block in at one point while deleting two
    ordinary steps at another produces two edits the diff is entitled to read as
    a single *move* -- a ``fetch_page`` gone from here and a ``fetch_page``
    arrived there is a move by any reasonable account. Ground truth had to
    exclude those as undecidable, which left the structure metric scoring 8 of
    38 mutants: a real number, measured over almost nothing.

    Making the insertion distinguishable removes the ambiguity rather than
    ruling on it. The class exists to test a *non-constant offset* between two
    structural edits, which does not depend on what was inserted; the realistic
    case, where the block is a genuine recorded recovery, is what ``insert``
    covers.
    """
    at = rng.randrange(1, len(rows) // 3)
    rows = rows[:at] + _inject([_make_distinct(s, at) for s in block]) + rows[at:]
    cut = rng.randrange(at + len(block) + 1, max(at + len(block) + 2, len(rows) - 2))
    rows = rows[:cut] + rows[cut + 2 :]
    changed = _change_a_tool(rows, rng, after=cut)
    return rows, {changed} if changed is not None else set()


def _mutate_double_change(rows: list[Row], block: list[Span], rng: random.Random):
    """Two divergences and no structural edit at all.

    Index pairing gets this one right, and it is here to prove the benchmark is
    capable of saying so.
    """
    first = _change_a_tool(rows, rng)
    second = _change_a_tool(rows, rng, after=len(rows) // 2)
    return rows, {c for c in (first, second) if c is not None}


def _mutate_reorder(rows: list[Row], block: list[Span], rng: random.Random):
    """A pair of steps moved later in the run: the agent did B before A.

    No monotonic alignment can represent this, which is the point of including
    it. See the write-up in the README.
    """
    at = rng.randrange(1, len(rows) // 2)
    moved = rows[at : at + 2]
    rest = rows[:at] + rows[at + 2 :]
    to = rng.randrange(at + 2, len(rest))
    rows = rest[:to] + moved + rest[to:]
    changed = _change_a_tool(rows, rng)
    return rows, {changed} if changed is not None else set()


def _reword(span: Span) -> bool:
    """Change a step's arguments without changing anything it produced.

    The demo model does this for real above temperature 0. A diff that blames a
    rewording for a failure sends somebody hunting through a prompt for a bug
    that is in a tool result three steps later.
    """
    inputs = span.attr(FR_INPUT)
    if not isinstance(inputs, dict):
        return False
    for key, value in inputs.items():
        if isinstance(value, str) and value:
            words = value.split()
            span.attributes[FR_INPUT] = {
                **inputs,
                key: " ".join(reversed(words)) if len(words) > 1 else value + "?",
            }
            return True
    return False


def _mutate_adjacent_edit(rows: list[Row], block: list[Span], rng: random.Random):
    """An insertion with a deletion immediately after it.

    Aimed at a documented restriction in the alignment: it never transitions
    straight from a gap on one side to a gap on the other, so an inserted block
    butted directly against a removed one is not expressible as what it is. That
    was a deliberate choice and it had never been tested.

    The injected steps are made unmatchable first, and that is the whole
    difficulty. Splicing in a *copy* of real steps does not test this at all:
    the copies match their neighbours, the aligner slips a match between the two
    gaps, and the adjacency it was supposed to be forced into never happens.
    The first version of this mutation did exactly that and scored 100% while
    measuring nothing.
    """
    at = rng.randrange(1, max(2, len(rows) - 4))
    rows = rows[:at] + _inject([_make_distinct(s, at) for s in block]) + rows[at:]
    cut = at + len(block)
    rows = rows[:cut] + rows[cut + 2 :]
    changed = _change_a_tool(rows, rng, after=cut)
    return rows, {changed} if changed is not None else set()


def _make_distinct(span: Span, tag: int) -> Span:
    """A step that cannot be matched to anything else in either run."""
    copy = span.model_copy(deep=True)
    copy.name = f"{copy.name}.injected{tag}"
    if copy.attributes.get(GEN_AI_TOOL_NAME):
        copy.attributes[GEN_AI_TOOL_NAME] = f"injected_tool_{tag}"
    copy.attributes[FR_INPUT] = {"injected": f"unmatchable-{tag}"}
    copy.attributes[FR_OUTPUT] = f"unmatchable output {tag}"
    return copy


def _mutate_duplicate(rows: list[Row], block: list[Span], rng: random.Random):
    """A step copied verbatim to somewhere else in the run.

    Aimed at the move pass, which rescues identical steps first and assumes a
    signature identifies one step. Two identical candidates at different
    distances make that assumption false, and the wrong choice invents a move
    that never happened.
    """
    source = rng.randrange(1, len(rows) - 1)
    target = rng.randrange(source + 1, len(rows))
    copy = rows[source][1].model_copy(deep=True)
    rows = rows[:target] + [(None, copy)] + rows[target:]

    # The duplicated step is barred from also being the changed one. Editing one
    # twin leaves the run holding both the old and the new version of a step,
    # and then "which one does the other run mean" has two equally coherent
    # answers -- the diff can pair the unedited twin and call the edited one an
    # insertion, which describes the same pair of runs. Scoring that measures
    # nothing, and a mutation should test one thing at a time.
    changed = _change_a_tool(rows, rng, exclude={rows[source][0]})
    return rows, {changed} if changed is not None else set()


def _mutate_compound(rows: list[Row], block: list[Span], rng: random.Random):
    """An insertion, a reordering and two edits in one run.

    Every other class isolates a single kind of difference, which is convenient
    for attributing a failure and unlike anything that happens in practice: real
    runs diverge in several ways at once, and the edits interact.
    """
    at = rng.randrange(1, max(2, len(rows) // 3))
    rows = rows[:at] + _inject(block) + rows[at:]

    start = rng.randrange(at + len(block), max(at + len(block) + 1, len(rows) - 3))
    moved = rows[start : start + 2]
    rest = rows[:start] + rows[start + 2 :]
    to = rng.randrange(start, len(rest)) if start < len(rest) else len(rest)
    rows = rest[:to] + moved + rest[to:]

    first = _change_a_tool(rows, rng)
    second = _change_a_tool(rows, rng, after=len(rows) // 2)
    return rows, {c for c in (first, second) if c is not None}


def _mutate_cosmetic_noise(rows: list[Row], block: list[Span], rng: random.Random):
    """Several arguments reworded to no effect, and one result genuinely changed.

    No structural edit, so there is exactly one thing here worth blaming and the
    diff has to blame it rather than the nearest rewording.
    """
    tools = [
        i
        for i, (origin, span) in enumerate(rows)
        if origin is not None and span.kind is SpanKind.TOOL
    ]
    changed = _change_a_tool(rows, rng)
    for index in rng.sample(tools, min(3, len(tools))):
        if rows[index][0] != changed:
            _reword(rows[index][1])
    return rows, {changed} if changed is not None else set()


def _mutate_truncate(rows: list[Row], block: list[Span], rng: random.Random):
    """The run stops early, the way a crashed agent's does.

    Partial runs are first-class in this tool -- the run whose process died
    mid-flight is the one you opened it to look at -- so diffing a complete run
    against a truncated one is an ordinary thing to ask for and was untested.
    """
    keep = rng.randrange(len(rows) // 2, len(rows) - 1)
    changed = _change_a_tool(rows[:keep], rng)
    return rows[:keep], {changed} if changed is not None else set()


def _mutate_repeat_block(rows: list[Row], block: list[Span], rng: random.Random):
    """A pair of steps repeated several times, the shape a retry loop leaves.

    Aimed at the move pass, which resolves ties by distance. One duplicate has
    one alternative; four copies of the same two steps have many, all equally
    good, and the metric has to treat them as the same answer rather than score
    a coin toss.
    """
    at = rng.randrange(1, max(2, len(rows) - 3))
    repeated = [
        (None, span.model_copy(deep=True)) for _ in range(3) for _, span in rows[at : at + 2]
    ]
    rows = rows[: at + 2] + repeated + rows[at + 2 :]
    changed = _change_a_tool(rows, rng, after=at + 8)
    return rows, {changed} if changed is not None else set()


def _mutate_swap_adjacent(rows: list[Row], block: list[Span], rng: random.Random):
    """Two neighbouring steps in the other order.

    The smallest possible reordering, and the hardest to attribute: moving a
    block leaves a long backbone to measure against, while swapping neighbours
    leaves two equally good readings and nothing to break the tie.
    """
    at = rng.randrange(1, len(rows) - 2)
    rows = rows[:at] + [rows[at + 1], rows[at]] + rows[at + 2 :]
    changed = _change_a_tool(rows, rng)
    return rows, {changed} if changed is not None else set()


def _mutate_substitute(rows: list[Row], block: list[Span], rng: random.Random):
    """One tool call becomes a different tool call in the same slot."""
    candidates = [
        i
        for i, (origin, span) in enumerate(rows)
        if origin is not None
        and span.kind is SpanKind.TOOL
        and span.attr(GEN_AI_TOOL_NAME) != "calculator"
    ]
    if not candidates:
        return rows, set()
    index = rng.choice(candidates)
    origin, span = rows[index]
    span.name = "tool.calculator"
    span.attributes[GEN_AI_TOOL_NAME] = "calculator"
    span.attributes[FR_INPUT] = {"expression": "1 + 1"}
    span.attributes[FR_OUTPUT] = 2.0
    return rows, {origin}


MUTATORS = {
    "insert": _mutate_insert,
    "delete": _mutate_delete,
    "insert+delete": _mutate_insert_and_delete,
    "double-change": _mutate_double_change,
    "reorder": _mutate_reorder,
    "substitute": _mutate_substitute,
    "adjacent-edit": _mutate_adjacent_edit,
    "duplicate": _mutate_duplicate,
    "compound": _mutate_compound,
    "cosmetic-noise": _mutate_cosmetic_noise,
    "truncate": _mutate_truncate,
    "repeat-block": _mutate_repeat_block,
    "swap-adjacent": _mutate_swap_adjacent,
}


def mutate(
    run: Run, block: list[Span], rng: random.Random, kind: str = "insert"
) -> Mutation | None:
    """Build one mutant of ``run`` with a known ground-truth correspondence."""
    steps = [span.model_copy(deep=True) for span in run.steps()]
    if len(steps) < 8:
        return None

    rows: list[Row] = [(i, span.model_copy(deep=True)) for i, span in enumerate(steps)]
    rows, changed = MUTATORS[kind](rows, block, rng)
    if not changed:
        return None

    expected = {
        origin: position
        for position, (origin, _) in enumerate(rows)
        if origin is not None
    }
    if any(step not in expected for step in changed):
        return None  # the change landed on a step this mutation also removed

    return Mutation(
        kind=kind,
        original=_as_run(steps, "original"),
        mutant=_as_run([span for _, span in rows], "mutant"),
        expected=expected,
        changed=changed,
        removed={i for i in range(len(steps)) if i not in expected},
    )


def _as_run(steps: list[Span], label: str) -> Run:
    """Renumber a list of steps into a standalone run.

    Sequence numbers are the ordering authority and span IDs are the storage
    key, so a spliced-in step carrying its donor's numbers would sort into the
    wrong place and collide with the step it was copied from.
    """
    for index, step in enumerate(steps):
        step.sequence = index
        step.span_id = f"{label}-{index}"
        step.parent_span_id = None
    return Run(run_id=label, spans=steps)


def _perturb(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return "injected divergence"
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, list):
        return value + ["https://weather.example/injected"]
    return f"{value} [injected divergence]"


# --- 3. replay cost saving ----------------------------------------------------


@dataclass
class CostPoint:
    """What replaying from one point in the run costs, aggregated over the corpus."""

    cut_fraction: float
    steps_skipped: int
    steps_total: int
    spent: int
    full: int

    @property
    def saving_pct(self) -> float:
        return 100.0 * (1.0 - self.spent / self.full) if self.full else 0.0

    @property
    def skipped_pct(self) -> float:
        return 100.0 * self.steps_skipped / self.steps_total if self.steps_total else 0.0

    def line(self) -> str:
        return (
            f"cut at {self.cut_fraction * 100:>4.0f}%   skips {self.skipped_pct:>3.0f}% "
            f"of steps   spends {self.spent:>7,} tokens   saves {self.saving_pct:>5.1f}%"
        )


def cost_saving_curve(seeds: range) -> tuple[list[CostPoint], int, int]:
    """Sweep every cut point, then aggregate into deciles.

    Two numbers were not enough. The saving is not proportional to the steps
    skipped -- a model call carries the whole transcript, so the second half of
    a run costs far more than the first -- and reporting a midpoint and a 90%
    figure showed that only as a pair of dots with a straight line implied
    between them. It is not straight, and it is not even monotonic.

    Returns the curve, the number of adjacent cut points where cutting *later*
    cost *more*, and how many transitions were examined.
    """
    buckets = {d: CostPoint(d / 10, 0, 0, 0, 0) for d in range(11)}
    regressions = 0
    transitions = 0

    for seed in seeds:
        recorded = record(seed)
        steps = len(recorded.steps())
        spent_at = {}

        for cut in range(steps + 1):
            replayed = replay_run(recorded, from_step=cut)
            spent_at[cut] = sum(
                step.total_tokens
                for step in replayed.run.steps()
                if not step.attr(FR_SERVED)
            )

        for cut in range(steps):
            transitions += 1
            if spent_at[cut + 1] > spent_at[cut]:
                regressions += 1

        for decile, point in buckets.items():
            cut = max(0, min(steps, round(decile / 10 * steps)))
            point.steps_skipped += cut
            point.steps_total += steps
            point.spent += spent_at[cut]
            point.full += recorded.total_tokens

    return [buckets[d] for d in range(11)], regressions, transitions


def measure_replay_cost_saving(seeds: range) -> Measurement:
    """Tokens actually spent replaying from each point vs. re-running the task.

    A step served from the recording bills nothing, so only the live steps past
    the edit point count. This is the number that decides whether anybody edits
    step 7 more than once.
    """
    curve, regressions, transitions = cost_saving_curve(seeds)
    midpoint = curve[5]
    late = curve[9]

    return Measurement(
        name="Replay cost saving",
        value=midpoint.saving_pct,
        unit="%",
        baseline=0.0,
        baseline_label="re-running from step 0",
        detail=(
            f"{midpoint.spent:,} tokens spent vs {midpoint.full:,} to re-run, "
            f"cutting at the midpoint; cutting at 90% saves "
            f"{late.saving_pct:.1f}%. Cutting later cost *more* at "
            f"{regressions} of {transitions} adjacent cut points"
        ),
        breakdown=[point.line() for point in curve],
        caveat=(
            "far less than the steps skipped would suggest, for a structural "
            "reason: a model call's prompt carries the whole transcript, so the "
            "second half of a run costs much more than the first -- skipping "
            "half the steps saves about a quarter of the tokens. And the curve "
            "is not monotonic. Past the cut the agent runs live and can take a "
            "*different, longer* path, so cutting one step later sometimes costs "
            "more than cutting earlier. Tokens only -- not latency, and not the "
            "agent's own logic, which is re-executed either way"
        ),
    )


# --- 4. instrumentation overhead ---------------------------------------------


class _BlankSpan:
    """Stand-in for a span, for the uninstrumented baseline.

    Enough surface for an agent to write to and read back within a single step,
    and nothing else -- no validation, no serialisation, no sink. That gap is
    exactly the cost being measured.
    """

    __slots__ = ("attributes", "events", "status", "status_message")

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.events: list[Any] = []
        self.status = None
        self.status_message = None

    def attr(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    @property
    def input_tokens(self) -> int:
        return int(self.attributes.get("gen_ai.usage.input_tokens", 0) or 0)

    @property
    def output_tokens(self) -> int:
        return int(self.attributes.get("gen_ai.usage.output_tokens", 0) or 0)


class NoOpTracer:
    """The same agent with the instrumentation taken out.

    Not a null *sink* -- that would measure only the cost of storing spans and
    would flatter the SDK by leaving span construction in place, which is where
    the real cost is.
    """

    def __init__(self) -> None:
        self.clock = SystemClock()
        self.sink = None
        self.trace_id = "uninstrumented"

    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Iterator[_BlankSpan]:
        # The real ContextVar, because instrumented code reaches for the current
        # span through ``Tracer.current_span()`` and has to find something. An
        # agent that silently loses its span here would take a different path
        # from the instrumented one, and the two arms would stop comparing the
        # same program.
        span = _BlankSpan()
        token = tracer_module._current_span.set(span)  # type: ignore[arg-type]
        try:
            yield span
        finally:
            tracer_module._current_span.reset(token)

    def call(
        self,
        name: str,
        fn: Any,
        *,
        restore: Any = None,
        **kwargs: Any,
    ) -> Any:
        with self.span(name) as span:
            result = fn()
            return restore(result, span) if restore else result

    def record_usage(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    def set_output(*args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    def current_span() -> None:
        return None


@dataclass
class OverheadPoint:
    """Relative SDK overhead at one step duration."""

    step_us: float
    traced_ms: float
    bare_ms: float

    @property
    def overhead_pct(self) -> float:
        return 100.0 * (self.traced_ms - self.bare_ms) / self.bare_ms if self.bare_ms else 0.0

    def line(self) -> str:
        return (
            f"step of {self.step_us:>6.0f}us   traced {self.traced_ms:7.3f}ms   "
            f"bare {self.bare_ms:7.3f}ms   overhead {self.overhead_pct:6.1f}%"
        )


def _burn(microseconds: float) -> None:
    """Spin for a fixed span of CPU time.

    A spin rather than a sleep, deliberately. A sleeping step hands the CPU back
    and the SDK's cost disappears into time the process was not using anyway,
    which flatters the result. Burning CPU is the harsher and more honest
    stand-in for work a real step does.
    """
    end = time.perf_counter() + microseconds / 1e6
    while time.perf_counter() < end:
        pass


def overhead_curve(
    step_workloads: tuple[float, ...] = (10, 50, 100, 250, 500, 1000),
    spans: int = 12,
    repeats: int = 15,
) -> list[OverheadPoint]:
    """Measure relative overhead against how long a step takes.

    This replaces an arithmetic claim with a measurement. The previous version
    divided the per-span cost by 0.05 and printed "stays under 5% for any run
    longer than 2ms" -- true by construction, and never checked. Overhead is a
    ratio, so the honest way to report it is the curve, and the honest way to
    get the curve is to run it.

    A bare loop is the baseline rather than a no-op tracer: this is measuring
    what instrumenting costs somebody who currently has none.
    """
    points = []
    for work in step_workloads:

        def traced() -> float:
            tracer = Tracer(sink=MemorySink())
            start = time.perf_counter()
            for index in range(spans):
                tracer.call(
                    "step", lambda: _burn(work), kind=SpanKind.TOOL, inputs=index
                )
            return (time.perf_counter() - start) * 1000.0

        def bare() -> float:
            start = time.perf_counter()
            for _ in range(spans):
                _burn(work)
            return (time.perf_counter() - start) * 1000.0

        # Interleaved, so a machine that gets busy partway through does not load
        # the whole penalty onto whichever arm ran second.
        traced_samples, bare_samples = [], []
        for _ in range(repeats):
            traced_samples.append(traced())
            bare_samples.append(bare())
        # Minimum, not median. The fastest observed run is the one least
        # interrupted by the scheduler, and on a 12ms sample a single quantum
        # lands squarely in the middle of the distribution -- which showed up
        # as a curve that went *up* at the long end, where the SDK's fixed cost
        # should be vanishing. What is wanted is the cost of the work, not the
        # cost of the machine having something else to do.
        points.append(
            OverheadPoint(
                step_us=work,
                traced_ms=min(traced_samples),
                bare_ms=min(bare_samples),
            )
        )
    return points


def measure_overhead(repeats: int = 40, seed: int = 1) -> Measurement:
    """Wall-clock cost the SDK adds to a run.

    Token overhead is not measured because it is zero by construction: the SDK
    reads prompts and responses and never writes them, so an instrumented run
    sends the provider exactly what an uninstrumented one does.
    """

    spans = 0

    def timed(instrumented: bool) -> float:
        nonlocal spans
        agent = ResearchAgent(seed=seed, faults=FaultConfig.realistic())
        if not instrumented:
            agent.tracer = NoOpTracer()  # type: ignore[assignment]
        start = time.perf_counter()
        result = agent.run()
        elapsed = (time.perf_counter() - start) * 1000.0
        if instrumented:
            spans = len(result.run.spans)
        return elapsed

    # Interleaved and median-of-N: a machine that gets busy halfway through
    # would otherwise load the whole penalty onto whichever arm ran second.
    with_sdk, without_sdk = [], []
    for _ in range(repeats):
        with_sdk.append(timed(True))
        without_sdk.append(timed(False))

    # Minimum on both arms, for the reason given in overhead_curve: the fastest
    # run is the least interrupted, and what is being measured is the cost of
    # the instrumentation rather than the cost of the machine being busy. The
    # spread below is what says how much that assumption is worth today.
    traced, bare = min(with_sdk), min(without_sdk)
    added_ms = traced - bare
    per_span_us = 1000.0 * added_ms / spans if spans else 0.0
    # Spread, because a timing claim without one is a claim about a single
    # afternoon on a single machine. Quoting the median alone hides whether the
    # number is stable or whether it is being pulled around by scheduling.
    spread = sorted(with_sdk)
    low, high = spread[0], spread[-1 - len(spread) // 10]

    curve = overhead_curve()
    crossing = next((p for p in curve if p.overhead_pct < 5.0), None)

    return Measurement(
        name="Instrumentation overhead",
        value=100.0 * added_ms / bare if bare else 0.0,
        unit="%",
        baseline=0.0,
        baseline_label="uninstrumented agent",
        detail=(
            f"{traced:.3f}ms traced ({low:.3f}-{high:.3f} across {repeats} runs) vs "
            f"{bare:.3f}ms bare: +{added_ms:.3f}ms over {spans} spans, "
            f"{per_span_us:.1f}us per span. Measured below 5% once a step takes "
            + (f"{crossing.step_us:.0f}us or more" if crossing else "more than tested")
        ),
        breakdown=[point.line() for point in curve],
        caveat=(
            "the headline misses the <5% target and is the wrong number to read. "
            "Overhead is a ratio, so it depends entirely on what a step does, "
            "and every step of this agent is local and finishes in microseconds. "
            "The curve is the answer: fixed cost per span, measured against step "
            "duration rather than divided out of it -- an earlier version "
            "computed the crossover arithmetically and never checked it. Cost is "
            "dominated by span validation and UUID generation, both deliberate. "
            "Timing varies by machine; the counts above do not"
        ),
    )


# --- the harness --------------------------------------------------------------


def run_bench(runs: int = DEFAULT_RUNS, repeats: int = 40) -> list[Measurement]:
    seeds = range(runs)
    return [
        measure_replay_fidelity(seeds),
        measure_partial_replay_fidelity(seeds),
        measure_divergence_localization(seeds),
        measure_replay_cost_saving(seeds),
        measure_overhead(repeats),
    ]
