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
from flightrec.determinism import SystemClock
from flightrec.diff import Op, diff_runs, diff_runs_by_index
from flightrec.replay import replay_run
from flightrec.spans import (
    FR_INPUT,
    FR_OUTPUT,
    FR_SERVED,
    GEN_AI_TOOL_NAME,
    Run,
    Span,
    SpanKind,
    first_divergence,
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


# --- 2. divergence localization ----------------------------------------------


@dataclass
class KindResult:
    """How both pairings did against one class of mutation."""

    kind: str
    total: int = 0
    aligned: int = 0
    naive: int = 0
    aligned_pairing: float = 0.0
    naive_pairing: float = 0.0

    def rate(self, hits: int) -> float:
        return 100.0 * hits / self.total if self.total else 0.0

    def line(self) -> str:
        return (
            f"{self.kind:<14} localized {self.rate(self.aligned):5.1f}% "
            f"(zip {self.rate(self.naive):5.1f}%)   "
            f"pairing {self.aligned_pairing:5.1f}% (zip {self.naive_pairing:5.1f}%)"
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
        aligned_pairing, naive_pairing = 0.0, 0.0

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

        if result.total:
            result.aligned_pairing = 100.0 * aligned_pairing / result.total
            result.naive_pairing = 100.0 * naive_pairing / result.total
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
            "an average over mutation classes, and the classes are not equally "
            "hard -- read the per-class lines, not this number. Insertions are "
            "real recorded recovery blocks; the rest are synthetic edits, and "
            "where each one lands is chosen by a seeded RNG"
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
                or column.right_index != self.expected.get(step)
                or column.op is not Op.CHANGED
            ):
                return False
        return True

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
        hits = sum(1 for k, v in self.expected.items() if paired.get(k) == v)
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


def _change_a_tool(rows: list[Row], rng: random.Random, after: int = 0) -> int | None:
    """Perturb one surviving tool step's output; return which original it was.

    Tool steps rather than model steps: a tool result is where a divergence
    actually originates, and it is the kind of change somebody would be trying
    to find.
    """
    candidates = [
        i
        for i in range(after, len(rows))
        if rows[i][0] is not None and rows[i][1].kind is SpanKind.TOOL
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
    """Both, in different places, so the offset is not even constant."""
    at = rng.randrange(1, len(rows) // 3)
    rows = rows[:at] + _inject(block) + rows[at:]
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


def measure_replay_cost_saving(seeds: range) -> Measurement:
    """Tokens actually spent replaying from the midpoint vs. re-running the task.

    A step served from the recording bills nothing, so only the live steps past
    the edit point count. This is the number that decides whether anybody edits
    step 7 more than once.
    """
    def spend(fraction: float) -> tuple[int, int]:
        spent = full = 0
        for seed in seeds:
            recorded = record(seed)
            steps = len(recorded.steps())
            cut = max(1, min(steps - 1, int(steps * fraction)))
            replayed = replay_run(recorded, from_step=cut)
            spent += sum(
                step.total_tokens
                for step in replayed.run.steps()
                if not step.attr(FR_SERVED)
            )
            full += recorded.total_tokens
        return spent, full

    spent, full = spend(0.5)
    late_spent, late_full = spend(0.9)
    saving = 100.0 * (1.0 - spent / full) if full else 0.0
    late_saving = 100.0 * (1.0 - late_spent / late_full) if late_full else 0.0

    return Measurement(
        name="Replay cost saving",
        value=saving,
        unit="%",
        baseline=0.0,
        baseline_label="re-running from step 0",
        detail=(
            f"{spent:,} tokens spent vs {full:,} to re-run, cutting at the midpoint; "
            f"cutting at 90% saves {late_saving:.1f}%"
        ),
        caveat=(
            "less than the half you would expect from a midpoint cut, and the "
            "reason is structural: a model call's prompt carries the whole "
            "transcript so far, so the second half of a run is far more "
            "expensive than the first. Saving scales with where you cut, not "
            "with how many steps you skip. Tokens only -- not latency, and not "
            "the agent's own logic, which is re-executed either way"
        ),
    )


# --- 4. instrumentation overhead ---------------------------------------------


class _BlankSpan:
    """Stand-in for a span, for the uninstrumented baseline."""

    __slots__ = ("attributes", "events", "status", "status_message")

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.events: list[Any] = []
        self.status = None
        self.status_message = None


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
        yield _BlankSpan()

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

    traced, bare = median(with_sdk), median(without_sdk)
    added_ms = traced - bare
    per_span_us = 1000.0 * added_ms / spans if spans else 0.0
    # Below this run duration the SDK costs more than 5% of the run. Above it,
    # the same absolute cost disappears into the noise.
    break_even_ms = added_ms / 0.05 if added_ms else 0.0

    return Measurement(
        name="Instrumentation overhead",
        value=100.0 * added_ms / bare if bare else 0.0,
        unit="%",
        baseline=0.0,
        baseline_label="uninstrumented agent",
        detail=(
            f"{traced:.3f}ms traced vs {bare:.3f}ms bare (median of {repeats}): "
            f"+{added_ms:.3f}ms over {spans} spans, {per_span_us:.1f}us per span. "
            f"Stays under 5% for any run longer than {break_even_ms:.0f}ms"
        ),
        caveat=(
            "this misses the <5% target and the percentage is the wrong number "
            "to read. Every step of this agent is local and finishes in "
            "microseconds, so a fixed per-span cost lands on a run that does "
            "almost nothing. The portable figure is the per-span cost: one "
            "real model call takes longer than the entire instrumented run "
            "measured here"
        ),
    )


# --- the harness --------------------------------------------------------------


def run_bench(runs: int = DEFAULT_RUNS, repeats: int = 40) -> list[Measurement]:
    seeds = range(runs)
    return [
        measure_replay_fidelity(seeds),
        measure_divergence_localization(seeds),
        measure_replay_cost_saving(seeds),
        measure_overhead(repeats),
    ]
