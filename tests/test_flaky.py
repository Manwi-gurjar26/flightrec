"""Tests for the flaky-step report.

The report is only as good as the alignment underneath it. If steps are grouped
by position rather than by what they correspond to, the numbers describe where
the trouble sat rather than which step caused it -- so most of these check the
grouping, not the arithmetic.
"""

from __future__ import annotations

import pytest

from flightrec.demo.agent import ResearchAgent
from flightrec.demo.tools import FaultConfig
from flightrec.flaky import flaky_report, reference_run
from flightrec.spans import SpanKind

CLEAN = FaultConfig()
REALISTIC = FaultConfig.realistic()


def corpus(count: int = 30, faults: FaultConfig | None = None):
    return [
        ResearchAgent(seed=seed, faults=faults or REALISTIC).run().run
        for seed in range(count)
    ]


def test_a_task_with_no_faults_has_nothing_flaky() -> None:
    """The control. Every run identical means every step is settled."""
    report = flaky_report(corpus(10, CLEAN))

    assert report
    assert all(entry.disagreement == 0.0 for entry in report)
    assert all(entry.absent == 0 for entry in report)


def test_the_flaky_steps_are_the_ones_with_the_injected_faults() -> None:
    """Scored against a known answer: the searches are where the coin is tossed.

    ``empty_search_rate`` is 0.25, so each search should disagree with itself
    roughly a quarter of the time. Anything that recovers that number from the
    traces alone is measuring the right thing.
    """
    report = {entry.position: entry for entry in flaky_report(corpus(40))}

    searches = [e for e in report.values() if e.name == "tool.web_search"]
    assert len(searches) == 2
    for search in searches:
        assert 0.1 < search.disagreement < 0.45, search.line()

    assert report[0].disagreement == 0.0, "the opening step cannot vary"


def test_damage_accumulates_towards_the_end_of_the_run() -> None:
    """What the report is for: the last step is the least trustworthy.

    Every fault upstream lands in the final total, so the answer disagrees with
    itself more often than any single step that caused it. A report that ranked
    only by error rate would miss this -- the calculator never fails, it just
    adds up different numbers.
    """
    report = {entry.position: entry for entry in flaky_report(corpus(40))}
    calculator = next(e for e in report.values() if e.name == "tool.calculator")
    searches = [e for e in report.values() if e.name == "tool.web_search"]

    assert calculator.error_rate == 0.0, "it never fails, it just gets a different sum"
    assert calculator.disagreement > max(s.disagreement for s in searches)


def test_steps_are_grouped_by_correspondence_not_position() -> None:
    """The reason this needs the aligner at all.

    Runs are 11, 13 or 15 steps long, so the step at index 8 of one run is not
    the step at index 8 of another. Grouping by position would scatter each
    step's outcomes across several rows and report variance that is really just
    the length difference.
    """
    runs = corpus(40)
    assert len({len(run.steps()) for run in runs}) > 1, "corpus must vary in length"

    report = flaky_report(runs)

    # Every reference step found a counterpart in every run. Before the move
    # pass was fixed, weak-but-real pairings were dropped and these came back
    # as absences.
    assert all(entry.absent == 0 for entry in report), [
        e.line() for e in report if e.absent
    ]
    assert all(entry.runs_present == len(runs) for entry in report)


def test_the_reference_is_the_commonest_shape_not_the_first_run() -> None:
    """Picking run zero works until run zero is the unusual one."""
    runs = corpus(40)
    reference = reference_run(runs)

    lengths = [len(run.steps()) for run in runs]
    commonest = max(set(lengths), key=lengths.count)

    assert len(reference.steps()) == commonest


def test_a_step_missing_from_half_the_runs_is_not_called_settled() -> None:
    """Presence is itself a coin toss, and the ranking has to say so.

    A step that appears in half the runs and does the same thing whenever it
    appears has a disagreement of zero. Sorting on that alone would rank it as
    the most stable thing in the report.
    """
    from flightrec.flaky import StepVariance

    always = StepVariance(position=0, name="a", runs_present=10, absent=0)
    always.outcomes.update(["x"] * 10)
    sometimes = StepVariance(position=1, name="b", runs_present=5, absent=5)
    sometimes.outcomes.update(["x"] * 5)

    assert sometimes.disagreement == always.disagreement == 0.0
    assert sometimes.instability > always.instability


def test_an_empty_corpus_reports_nothing_rather_than_failing() -> None:
    assert flaky_report([]) == []


def test_cli_flaky_ranks_and_explains(capsys) -> None:
    from flightrec.cli import main

    assert main(["flaky", "--runs", "12"]) == 0
    out = capsys.readouterr().out

    assert "most variable" in out
    assert "tool.web_search" in out
    assert "settled" in out, "the output has to say how to read the number"


def test_cli_flaky_reads_stored_runs(tmp_path, capsys) -> None:
    from flightrec.cli import main
    from flightrec.storage import RunStore

    db = str(tmp_path / "runs.db")
    store = RunStore(db)
    for run in corpus(6):
        store.add_run(run)
    store.close()

    assert main(["flaky", "--db", db]) == 0
    assert "stored run(s)" in capsys.readouterr().out


def test_cli_flaky_needs_something_to_compare(tmp_path, capsys) -> None:
    from flightrec.cli import main
    from flightrec.storage import RunStore

    db = str(tmp_path / "empty.db")
    RunStore(db).close()

    assert main(["flaky", "--db", db]) == 1
    assert "at least two runs" in capsys.readouterr().out
