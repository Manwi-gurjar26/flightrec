"""Tests for the measurement harness.

The benchmark is the only thing standing behind the numbers in the README, so
the thing worth testing is not that it produces numbers -- it is that the
numbers mean what they say. That means checking the *baselines* are real
comparisons and not strawmen, and that the whole thing is reproducible, since a
benchmark that drifts is a benchmark whose results cannot be checked by anyone
else.
"""

from __future__ import annotations

import random

import pytest

from flightrec.bench import (
    MUTATORS,
    NoOpTracer,
    localization_by_kind,
    measure_divergence_localization,
    measure_overhead,
    measure_replay_cost_saving,
    measure_replay_fidelity,
    mutate,
    record,
    recovery_block,
    run_bench,
)
from flightrec.demo.agent import ResearchAgent
from flightrec.demo.tools import GROUND_TRUTH, FaultConfig
from flightrec.diff import Op, diff_runs, diff_runs_by_index
from flightrec.spans import FR_OUTPUT, SpanKind, step_signature

SEEDS = range(8)


# --- the measurements ---------------------------------------------------------


def test_replay_fidelity_is_total_and_the_baseline_is_not() -> None:
    result = measure_replay_fidelity(SEEDS)

    assert result.value == 100.0
    assert result.baseline < 100.0, "a baseline that also scores 100% compares nothing"


def test_divergence_localization_beats_index_pairing() -> None:
    result = measure_divergence_localization(SEEDS)

    assert result.value > result.baseline
    assert result.breakdown, "the per-class lines are the point; the average hides them"


def test_the_mutation_classes_are_not_all_easy() -> None:
    """A benchmark every arm passes is a benchmark that is not measuring.

    Two classes are here specifically because index pairing handles them fine --
    they keep the comparison honest by showing the baseline is capable of
    winning where alignment buys nothing.
    """
    results = {r.kind: r for r in localization_by_kind(SEEDS)}

    assert results["double-change"].rate(results["double-change"].naive) == 100.0
    assert results["insert"].rate(results["insert"].naive) == 0.0


def test_reordering_defeats_the_monotonic_pass_on_its_own() -> None:
    """The structural limit that the move pass exists to answer.

    Needleman-Wunsch produces a monotonic correspondence: step order is
    preserved on both sides. A reordering is by definition non-monotonic, so no
    amount of penalty tuning represents one -- which is why the fix is a second
    pass rather than a better score function. This pins the *reason*: turn move
    recovery off and the failure comes straight back.
    """
    monotonic = next(
        r for r in localization_by_kind(SEEDS, detect_moves=False) if r.kind == "reorder"
    )

    assert monotonic.total > 0
    assert monotonic.rate(monotonic.aligned) < 100.0
    assert monotonic.aligned_pairing < 100.0


def test_move_recovery_fixes_reordering_without_costing_the_other_classes() -> None:
    """The second pass has to earn its place on every class, not just its own.

    A pass that re-pairs steps after the fact can just as easily invent
    correspondences that were not there -- the first version of it did exactly
    that, dropping deletions from 100% to 70% by letting a deleted step claim
    the changed step's partner.
    """
    monotonic = {r.kind: r for r in localization_by_kind(SEEDS, detect_moves=False)}
    recovered = {r.kind: r for r in localization_by_kind(SEEDS)}

    assert recovered["reorder"].rate(recovered["reorder"].aligned) == 100.0
    for kind, result in recovered.items():
        before = monotonic[kind]
        assert result.rate(result.aligned) >= before.rate(before.aligned), kind
        assert result.aligned_pairing >= before.aligned_pairing - 1e-9, kind


def test_replay_saves_tokens_against_re_running() -> None:
    result = measure_replay_cost_saving(SEEDS)

    assert 0.0 < result.value < 100.0
    assert "90%" in result.detail, "the saving depends on where you cut; say so"


def test_overhead_is_reported_with_a_portable_figure() -> None:
    """The percentage is unusable on an agent this fast; the per-span cost is not."""
    result = measure_overhead(repeats=3)

    assert result.value > 0.0
    assert "per span" in result.detail
    assert "5%" in result.caveat


def test_every_measurement_states_its_limits() -> None:
    """A number quoted without its caveat is the failure mode this guards."""
    for measurement in run_bench(runs=3, repeats=2):
        assert measurement.caveat, f"{measurement.name} has no stated limits"
        assert measurement.baseline_label
        assert measurement.detail


def test_the_benchmark_is_reproducible() -> None:
    """Anyone must get these numbers back, or they are not evidence of anything."""
    first = {m.name: m.value for m in run_bench(runs=4, repeats=2) if "overhead" not in m.name.lower()}
    second = {m.name: m.value for m in run_bench(runs=4, repeats=2) if "overhead" not in m.name.lower()}

    assert first == second


# --- the mutation, which the localization number depends on -------------------


def test_an_insertion_mutant_differs_in_exactly_two_ways() -> None:
    """One insertion and one changed output -- no accidental third difference.

    If mutation introduced anything else, the localization score would be
    measuring the aligner against a problem nobody characterised.
    """
    mutation = mutate(record(0), recovery_block(), random.Random(1), kind="insert")
    assert mutation is not None

    diff = diff_runs(mutation.original, mutation.mutant)

    assert diff.count(Op.INSERTED) == 2
    assert diff.count(Op.REMOVED) == 0
    assert diff.count(Op.CHANGED) == len(mutation.changed) == 1
    assert len(mutation.mutant.steps()) == len(mutation.original.steps()) + 2


@pytest.mark.parametrize("kind", list(MUTATORS))
def test_every_mutation_records_a_usable_ground_truth(kind: str) -> None:
    """The scoring is only as trustworthy as the correspondence it scores against."""
    mutation = mutate(record(0), recovery_block(), random.Random(4), kind=kind)
    assert mutation is not None

    assert mutation.changed <= set(mutation.expected), "a changed step must survive"
    assert set(mutation.expected) | mutation.removed == set(
        range(len(mutation.original.steps()))
    )
    assert len(set(mutation.expected.values())) == len(mutation.expected), (
        "two original steps cannot map to one mutant step"
    )
    for step in mutation.changed:
        before = mutation.original.steps()[step]
        after = mutation.mutant.steps()[mutation.expected[step]]
        assert before.attr(FR_OUTPUT) != after.attr(FR_OUTPUT)


def test_index_pairing_really_does_miss_the_injected_step() -> None:
    """The baseline has to fail for a reason, not by construction.

    It pairs the changed step with a step two positions away, which is a
    different step -- so it reports a difference, at the right index, about the
    wrong thing.
    """
    mutation = mutate(record(0), recovery_block(), random.Random(3), kind="insert")
    assert mutation is not None
    changed = next(iter(mutation.changed))

    naive = diff_runs_by_index(mutation.original, mutation.mutant)
    column = next(c for c in naive.columns if c.left_index == changed)

    assert column.right_index == changed
    assert column.right_index != mutation.expected[changed]


def test_the_adjacent_edit_injects_something_unmatchable() -> None:
    """Guard against the way this mutation was vacuous when first written.

    It splices a block in and deletes the steps immediately after, to force the
    alignment into a gap-on-one-side-then-gap-on-the-other it does not express.
    Splicing a *copy* of real steps does not do that: the copies match their
    neighbours, the aligner slips a match between the two gaps, and the case it
    was built for never arises. It scored 100% while testing nothing.
    """
    mutation = mutate(record(0), recovery_block(), random.Random(7), kind="adjacent-edit")
    assert mutation is not None

    original = {step_signature(s) for s in mutation.original.steps()}
    injected = [
        s
        for j, s in enumerate(mutation.mutant.steps())
        if j not in set(mutation.expected.values())
    ]

    assert injected, "the mutation must actually inject something"
    assert all(step_signature(s) not in original for s in injected)
    assert mutation.removed, "and must actually delete something"


def test_the_structure_metric_still_scores_something() -> None:
    """Excluding undecidable cases must not empty the metric out.

    Ground truth is only allowed to score added and removed steps that could
    not be mistaken for each other, and that exclusion has been widened three
    times. Widen it once more and every case disappears, at which point the
    remaining classes report a flattering 100% over nothing at all -- which is
    exactly the failure this whole benchmark exists to avoid.
    """
    results = {r.kind: r for r in localization_by_kind(SEEDS)}

    assert results["delete"].structure_total == results["delete"].total
    assert results["adjacent-edit"].structure_total == results["adjacent-edit"].total
    assert results["insert"].structure_total > 0
    assert results["insert+delete"].structure_total > 0


def test_an_unmatchable_step_is_never_reported_as_a_changed_result() -> None:
    """``adjacent-edit`` is scored over all of its cases, and must stay that way.

    It injects a tool that appears nowhere in the other run, so "nothing here
    corresponds" is the only true description rather than one of two -- which is
    what makes it the one class the exclusion rule must never swallow. It passes
    now because the diff says ``replaced`` rather than ``changed``: the steps
    stay paired so their position is visible, while the report stops claiming
    they are the same step with a new result.
    """
    adjacent = next(r for r in localization_by_kind(SEEDS) if r.kind == "adjacent-edit")

    assert adjacent.structure_total == adjacent.total, "every case must be scoreable"
    assert adjacent.structure == 100.0


def test_structure_has_no_answer_for_a_duplicated_step() -> None:
    """Ground truth has to admit when it does not know.

    Duplicating a step verbatim leaves two identical steps, and which twin is
    "the extra one" is undecidable -- so scoring the diff on it would penalise a
    correct answer. The metric reports no result rather than a failure.
    """
    mutation = mutate(record(0), recovery_block(), random.Random(8), kind="duplicate")
    assert mutation is not None

    assert mutation.structure_accuracy(diff_runs(mutation.original, mutation.mutant)) is None


def test_cosmetic_noise_blames_the_real_change_not_the_rewording() -> None:
    """Several arguments reworded to no effect, one result genuinely changed.

    Blaming a rewording sends somebody hunting through a prompt for a bug that
    is in a tool result three steps away, which is this tool's most likely way
    of wasting an afternoon.
    """
    result = next(r for r in localization_by_kind(SEEDS) if r.kind == "cosmetic-noise")

    assert result.blame_total == result.total, "every case must be blame-scoreable"
    assert result.blamed == result.blame_total


def test_a_deletion_mutant_makes_the_mutant_shorter() -> None:
    """The shift has to be able to go the other way, or only one case is tested."""
    mutation = mutate(record(0), recovery_block(), random.Random(5), kind="delete")
    assert mutation is not None

    assert len(mutation.mutant.steps()) == len(mutation.original.steps()) - 2
    assert len(mutation.removed) == 2


def test_the_recovery_block_is_real_recorded_steps() -> None:
    block = recovery_block()

    assert len(block) == 2
    assert block[0].kind is SpanKind.LLM
    assert block[1].kind is SpanKind.TOOL


# --- the uninstrumented baseline ----------------------------------------------


def test_the_bare_agent_does_the_same_work_without_recording_it() -> None:
    """Otherwise the overhead measurement compares two different programs."""
    traced = ResearchAgent(seed=1, faults=FaultConfig.realistic()).run()

    bare = ResearchAgent(seed=1, faults=FaultConfig.realistic())
    bare.tracer = NoOpTracer()
    result = bare.run()

    assert result.answer == traced.answer
    assert result.steps == traced.steps
    assert result.faults_fired == traced.faults_fired
    assert not result.run.spans, "the uninstrumented arm must not record anything"


def test_a_clean_bare_run_still_gets_the_right_answer() -> None:
    bare = ResearchAgent(seed=1, faults=FaultConfig())
    bare.tracer = NoOpTracer()

    assert bare.run().answer == GROUND_TRUTH


# --- the CLI ------------------------------------------------------------------


def test_cli_bench_emits_json(capsys) -> None:
    import json

    from flightrec.cli import main

    assert main(["bench", "--runs", "3", "--repeats", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert len(payload) == 4
    assert all(row["caveat"] for row in payload)


def test_cli_bench_prints_caveats_with_the_numbers(capsys) -> None:
    from flightrec.cli import main

    assert main(["bench", "--runs", "3", "--repeats", "2"]) == 0
    out = capsys.readouterr().out

    assert "Replay fidelity" in out
    assert "index-by-index" in out
    assert "trajectory-identical" in out
