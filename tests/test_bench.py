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
    Mutation,
    NoOpTracer,
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
from flightrec.spans import FR_OUTPUT, SpanKind

SEEDS = range(8)


# --- the measurements ---------------------------------------------------------


def test_replay_fidelity_is_total_and_the_baseline_is_not() -> None:
    result = measure_replay_fidelity(SEEDS)

    assert result.value == 100.0
    assert result.baseline < 100.0, "a baseline that also scores 100% compares nothing"


def test_divergence_localization_beats_index_pairing() -> None:
    result = measure_divergence_localization(SEEDS)

    assert result.value == 100.0
    assert result.baseline == 0.0


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


def test_a_mutant_differs_from_its_original_in_exactly_two_ways() -> None:
    """One insertion and one changed output -- no accidental third difference.

    If mutation introduced anything else, the localization score would be
    measuring the aligner against a problem nobody characterised.
    """
    mutation = mutate(record(0), recovery_block(), random.Random(1))
    assert mutation is not None

    diff = diff_runs(mutation.original, mutation.mutant)
    gaps = diff.count(Op.INSERTED) + diff.count(Op.REMOVED)

    assert gaps == mutation.offset
    assert diff.count(Op.CHANGED) == 1
    assert len(mutation.mutant.steps()) == len(mutation.original.steps()) + mutation.offset


def test_the_injected_change_is_where_the_mutation_says_it_is() -> None:
    mutation = mutate(record(3), recovery_block(), random.Random(2))
    assert mutation is not None

    original = mutation.original.steps()[mutation.changed_step]
    changed = mutation.mutant.steps()[mutation.counterpart]

    assert original.attr(FR_OUTPUT) != changed.attr(FR_OUTPUT)


def test_index_pairing_really_does_miss_the_injected_step() -> None:
    """The baseline has to fail for a reason, not by construction.

    It pairs the changed step with a step two positions away, which is a
    different step -- so it reports a difference, at the right index, about the
    wrong thing.
    """
    mutation = mutate(record(0), recovery_block(), random.Random(3))
    assert mutation is not None

    naive = diff_runs_by_index(mutation.original, mutation.mutant)
    column = next(c for c in naive.columns if c.left_index == mutation.changed_step)

    assert column.right_index == mutation.changed_step
    assert column.right_index != mutation.counterpart


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
