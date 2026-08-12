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
from dataclasses import dataclass
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


def measure_divergence_localization(seeds: range) -> Measurement:
    """Given a known injected change, does the diff pair the right two steps?

    Each mutant gets two edits: a real two-step recovery block spliced in, and
    one later step's output changed. The insertion is what makes this hard --
    it shifts every following step, so index-by-index pairing compares the
    changed step against somebody else's.

    Measured as *pairing*, not as "reported a difference at index k". Naive
    pairing also reports a difference at index k -- it reports one nearly
    everywhere after the insertion -- so scoring that would hand it a pass for
    being wrong in the right place.
    """
    block = recovery_block()
    rng = random.Random(20240812)
    aligned_hits = naive_hits = total = 0

    for seed in seeds:
        mutation = mutate(record(seed), block, rng)
        if mutation is None:
            continue
        total += 1
        if _pairs_correctly(diff_runs(mutation.original, mutation.mutant), mutation):
            aligned_hits += 1
        if _pairs_correctly(
            diff_runs_by_index(mutation.original, mutation.mutant), mutation
        ):
            naive_hits += 1

    return Measurement(
        name="Divergence localization",
        value=100.0 * aligned_hits / total if total else 0.0,
        unit="%",
        baseline=100.0 * naive_hits / total if total else 0.0,
        baseline_label="index-by-index zip() pairing",
        detail=f"{aligned_hits}/{total} mutants had the changed step paired with its counterpart",
        caveat=(
            "the injected insertion is a real recorded recovery block, but where "
            "it goes and which step is changed are chosen by a seeded RNG"
        ),
    )


@dataclass
class Mutation:
    """A recording, a copy of it with a known change, and where the change is."""

    original: Run
    mutant: Run
    changed_step: int
    offset: int

    @property
    def counterpart(self) -> int:
        return self.changed_step + self.offset


def _pairs_correctly(diff, mutation: Mutation) -> bool:
    column = next(
        (c for c in diff.columns if c.left_index == mutation.changed_step), None
    )
    return (
        column is not None
        and column.right_index == mutation.counterpart
        and column.op is Op.CHANGED
    )


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


def mutate(run: Run, block: list[Span], rng: random.Random) -> Mutation | None:
    """Splice ``block`` into a copy of ``run`` and change one later step."""
    steps = [s.model_copy(deep=True) for s in run.steps()]
    if len(steps) < 6:
        return None

    insert_at = rng.randrange(1, len(steps) // 2)
    candidates = [
        i for i in range(insert_at, len(steps)) if steps[i].kind is SpanKind.TOOL
    ]
    if not candidates:
        return None
    changed_at = rng.choice(candidates)

    mutated = [s.model_copy(deep=True) for s in steps]
    mutated[changed_at].attributes[FR_OUTPUT] = _perturb(
        mutated[changed_at].attr(FR_OUTPUT)
    )
    spliced = (
        mutated[:insert_at]
        + [s.model_copy(deep=True) for s in block]
        + mutated[insert_at:]
    )

    return Mutation(
        original=_as_run(steps, "original"),
        mutant=_as_run(spliced, "mutant"),
        changed_step=changed_at,
        offset=len(block),
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
