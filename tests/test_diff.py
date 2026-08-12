"""Diff tests, built around the one failure mode that matters.

The alignment is not there to be clever. It is there so that a run with two
extra retry steps does not report every subsequent step as changed -- which is
what index-by-index pairing does, and which would make the diff view worse than
useless because it would be confidently wrong about where things went wrong.

So most of these tests are the same shape: construct two runs whose difference
is known exactly, then assert the diff points at that difference and nothing
else. Several of them assert the naive pairing gets it wrong, because "the
alignment earns its complexity" is a claim, and a claim needs a control.
"""

from __future__ import annotations

import pytest

from flightrec.demo.agent import ResearchAgent
from flightrec.demo.tools import FaultConfig
from flightrec.diff import (
    GAP_OPEN,
    Op,
    RunDiff,
    align,
    diff_runs,
    diff_runs_by_index,
    similarity,
)
from flightrec.spans import (
    FR_INPUT,
    FR_OUTPUT,
    GEN_AI_TOOL_NAME,
    Run,
    Span,
    SpanKind,
    SpanStatus,
)


def make_run(specs: list[tuple], run_id: str = "run") -> Run:
    """Build a run from ``(label, input, output[, status])`` tuples.

    Hand-built rather than recorded: these tests need two runs differing in one
    known way, and getting a real agent to differ in exactly one known way is
    the harder problem the diff is supposed to solve.
    """
    spans = []
    for index, spec in enumerate(specs):
        label, inputs, output = spec[0], spec[1], spec[2]
        status = spec[3] if len(spec) > 3 else SpanStatus.OK
        is_model = label == "chat"
        spans.append(
            Span(
                span_id=f"{run_id}-{index}",
                trace_id=run_id,
                name="chat" if is_model else f"tool.{label}",
                kind=SpanKind.LLM if is_model else SpanKind.TOOL,
                sequence=index,
                start_time=float(index),
                end_time=float(index) + 0.5,
                status=status,
                attributes={
                    FR_INPUT: inputs,
                    FR_OUTPUT: output,
                    **({} if is_model else {GEN_AI_TOOL_NAME: label}),
                },
            )
        )
    return Run(run_id=run_id, spans=spans)


#: The happy path, as a spec. Every test below is this with one thing changed.
CLEAN = [
    ("chat", "find seattle", "searching"),
    ("web_search", {"query": "seattle rainy days 2024"}, ["https://w.example/s"]),
    ("chat", "fetch it", "fetching"),
    ("fetch_page", {"url": "https://w.example/s"}, "recorded on 152 days"),
    ("chat", "add them", "adding"),
    ("calculator", {"expression": "152 + 144"}, 296.0),
    ("chat", "done", "combined total of 296 days"),
]

#: A transient failure and its retry: two steps present in one run only.
RETRY_BLOCK = [
    ("fetch_page", {"url": "https://w.example/s"}, None, SpanStatus.ERROR),
    ("chat", "that failed, retrying", "retrying"),
]


# --- similarity ---------------------------------------------------------------


def test_a_step_is_identical_to_itself() -> None:
    run = make_run(CLEAN)
    assert similarity(run.steps()[1], run.steps()[1]) == pytest.approx(1.0)


def test_steps_of_different_kinds_share_nothing() -> None:
    run = make_run(CLEAN)
    assert similarity(run.steps()[0], run.steps()[1]) == 0.0


def test_similarity_ignores_the_output() -> None:
    """The property the whole diff rests on.

    Two runs that called the same tool with the same arguments and got
    different results are the same step with a divergent result. If output
    counted toward similarity, those two would be pushed apart and each aligned
    against something else -- and the diff would lose the one difference the
    user is looking for.
    """
    left = make_run(CLEAN).steps()[3]
    right = make_run(CLEAN, run_id="b").steps()[3]
    right.attributes[FR_OUTPUT] = "recorded on 3 days"

    assert similarity(left, right) == pytest.approx(1.0)


def test_rewording_an_argument_costs_less_than_changing_the_tool() -> None:
    run = make_run(CLEAN)
    reworded = make_run(
        [("web_search", {"query": "how many rainy days seattle 2024"}, ["x"])]
    ).steps()[0]
    other_tool = make_run([("fetch_page", {"query": "seattle rainy days 2024"}, ["x"])]).steps()[0]

    assert similarity(run.steps()[1], reworded) > similarity(run.steps()[1], other_tool)


# --- alignment ----------------------------------------------------------------


def test_a_run_against_itself_is_all_matches() -> None:
    diff = diff_runs(make_run(CLEAN), make_run(CLEAN, run_id="b"))

    assert diff.identical
    assert diff.first_divergence is None
    assert diff.divergence_index is None


def test_an_inserted_retry_block_is_the_only_difference_reported() -> None:
    """The headline case. Index pairing reports a wall of changes instead."""
    left = make_run(CLEAN)
    right = make_run(CLEAN[:3] + RETRY_BLOCK + CLEAN[3:], run_id="b")

    aligned = diff_runs(left, right)
    naive = diff_runs_by_index(left, right)

    assert aligned.count(Op.INSERTED) == 2
    assert aligned.count(Op.MATCH) == len(CLEAN)
    assert aligned.count(Op.CHANGED) == 0
    # The control: without alignment, everything after the insertion is garbage.
    assert naive.count(Op.CHANGED) >= 4


def gap_blocks(diff, op: Op) -> list[list[int]]:
    """Group a diff's gap columns into contiguous runs."""
    positions = [i for i, c in enumerate(diff.columns) if c.op is op]
    blocks: list[list[int]] = []
    for position in positions:
        if blocks and position == blocks[-1][-1] + 1:
            blocks[-1].append(position)
        else:
            blocks.append([position])
    return blocks


def test_an_inserted_block_stays_in_one_piece() -> None:
    left = make_run(CLEAN)
    block = RETRY_BLOCK + [("chat", "trying once more", "retrying")]
    right = make_run(CLEAN[:3] + block + CLEAN[3:], run_id="b")

    assert [len(b) for b in gap_blocks(diff_runs(left, right), Op.INSERTED)] == [3]


def test_the_affine_penalty_breaks_a_tie_that_a_linear_one_cannot() -> None:
    """What the affine gap penalty actually buys, which is less than advertised.

    On the test above it buys nothing: a linear penalty keeps that block
    together too, because the steps around it match perfectly and there is
    nothing to gain by splitting. Asserting contiguity there and calling it
    evidence for affine gaps would be a test that passes no matter which
    penalty is in use.

    This is the case where they genuinely differ. Three identical candidate
    partners means every pairing scores the same, so the two alignments -- one
    gap of two, or two gaps of one with a match between them -- are *exactly*
    tied under a linear penalty, and which one comes out is down to evaluation
    order. Opening a gap costing something makes the answer principled.
    """
    left = make_run([("chat", "start", "starting"), CLEAN[3], ("chat", "end", "done")])
    right = make_run(
        [("chat", "start", "starting")] + [CLEAN[3]] * 3 + [("chat", "end", "done")],
        run_id="b",
    )

    affine = align(left.steps(), right.steps())
    linear = align(left.steps(), right.steps(), gap_open=0.0, gap_extend=-0.4)

    assert [len(b) for b in gap_blocks(RunDiff(left, right, affine), Op.INSERTED)] == [2]
    assert [len(b) for b in gap_blocks(RunDiff(left, right, linear), Op.INSERTED)] == [1, 1]


def test_the_gap_penalty_never_outbids_a_real_similarity_signal() -> None:
    """The bug the first tuning of these penalties had.

    With ``GAP_OPEN`` at -0.6, the aligner preferred pairing a *tool* step with
    a *chat* step -- similarity 0.0, nothing whatsoever in common -- over
    opening a second gap. Contiguity is a tie-breaker; when it starts outvoting
    evidence it is just a thumb on the scale, and the diff quietly reports the
    wrong pairing. The ceiling is the smallest similarity component, 0.2, which
    is 0.4 in score terms.
    """
    tool = ("web_search", {"query": "seattle rainy days 2024"}, ["u1"])
    other_tool = ("fetch_page", {"url": "https://w.example/x"}, "page x")
    noise = ("chat", "thinking about it", "hmm")

    left = make_run([("chat", "start", "go"), tool, ("chat", "end", "done")])
    right = make_run(
        [("chat", "start", "go"), noise, other_tool, noise, ("chat", "end", "done")],
        run_id="b",
    )

    paired = next(c for c in diff_runs(left, right).columns if c.op is Op.CHANGED)

    assert paired.right.kind is SpanKind.TOOL
    assert paired.right.attr(GEN_AI_TOOL_NAME) == "fetch_page"
    assert abs(GAP_OPEN) < 0.4


def test_a_changed_result_is_paired_with_its_counterpart_not_its_neighbour() -> None:
    """Divergence localization, which is the metric the README claims.

    The run has both an insertion *and* a genuinely changed step after it. The
    changed step is what a human is looking for; index pairing compares it
    against a step two positions away and never finds it.
    """
    left = make_run(CLEAN)
    changed = list(CLEAN)
    changed[5] = ("calculator", {"expression": "152 + 144"}, 289.0)
    right = make_run(changed[:3] + RETRY_BLOCK + changed[3:], run_id="b")

    aligned = diff_runs(left, right)
    calculator = next(
        c for c in aligned.columns if c.left is not None and c.left_index == 5
    )

    assert calculator.op is Op.CHANGED
    assert calculator.right_index == 7  # shifted by the two inserted steps
    assert calculator.right.attr(FR_OUTPUT) == 289.0

    naive = diff_runs_by_index(left, right)
    mispaired = naive.columns[5]
    assert mispaired.right.attr(GEN_AI_TOOL_NAME) != "calculator"


def test_a_removed_step_is_reported_as_removed() -> None:
    left = make_run(CLEAN)
    right = make_run(CLEAN[:5] + CLEAN[6:], run_id="b")

    diff = diff_runs(left, right)

    assert diff.count(Op.REMOVED) == 1
    assert diff.first_divergence.left_index == 5


def test_alignment_handles_an_empty_run() -> None:
    diff = diff_runs(make_run(CLEAN), make_run([], run_id="b"))
    assert diff.count(Op.REMOVED) == len(CLEAN)


# --- cosmetic vs. genuine -----------------------------------------------------


def test_a_reworded_query_with_the_same_result_is_not_a_divergence() -> None:
    """The failure this classification exists to prevent.

    Above temperature 0 the demo model rephrases its search query. The
    alternate phrasings all work and all return the same URLs, so reporting the
    rewording as the first divergence would send someone hunting through a
    prompt for a bug that is in a tool result three steps later.
    """
    left = make_run(CLEAN)
    reworded = list(CLEAN)
    reworded[1] = ("web_search", {"query": "seattle 2024 precipitation days"}, ["https://w.example/s"])
    right = make_run(reworded, run_id="b")

    diff = diff_runs(left, right)

    assert diff.count(Op.COSMETIC) == 1
    assert not diff.identical  # the difference is reported...
    assert diff.first_divergence is None  # ...but it is not blamed


def test_a_reworded_query_that_changed_the_result_is_a_divergence() -> None:
    """The other half: rewording is only cosmetic when nothing came of it."""
    left = make_run(CLEAN)
    reworded = list(CLEAN)
    reworded[1] = ("web_search", {"query": "seattle 2024 precipitation days"}, [])
    right = make_run(reworded, run_id="b")

    diff = diff_runs(left, right)

    assert diff.count(Op.COSMETIC) == 0
    assert diff.divergence_index == 1


def test_the_same_tool_failing_instead_of_succeeding_is_a_divergence() -> None:
    left = make_run(CLEAN)
    failed = list(CLEAN)
    failed[3] = ("fetch_page", {"url": "https://w.example/s"}, None, SpanStatus.ERROR)
    right = make_run(failed, run_id="b")

    assert diff_runs(left, right).divergence_index == 3


# --- against real recordings --------------------------------------------------


def test_two_recordings_of_the_same_task_diff_without_crashing() -> None:
    left = ResearchAgent(seed=0, faults=FaultConfig.realistic()).run().run
    right = ResearchAgent(seed=4, faults=FaultConfig.realistic()).run().run

    diff = diff_runs(left, right)

    assert diff.columns
    assert sum(diff.counts.values()) == len(diff.columns)
    assert len(diff.columns) >= max(len(left.steps()), len(right.steps()))


# --- the CLI ------------------------------------------------------------------


def two_stored_runs(tmp_path):
    from flightrec.storage import RunStore

    left = ResearchAgent(seed=0, faults=FaultConfig.realistic()).run().run
    right = ResearchAgent(seed=4, faults=FaultConfig.realistic()).run().run
    db = str(tmp_path / "runs.db")
    store = RunStore(db)
    store.add_run(left)
    store.add_run(right)
    store.close()
    return db, left, right


def test_cli_diff_reports_a_divergence(tmp_path, capsys) -> None:
    from flightrec.cli import main

    db, left, right = two_stored_runs(tmp_path)

    assert main(["diff", left.run_id, right.run_id, "--db", db]) == 0
    out = capsys.readouterr().out
    assert "first genuine divergence" in out
    assert "identical," in out


def test_cli_diff_of_a_run_against_itself_reports_no_difference(tmp_path, capsys) -> None:
    from flightrec.cli import main

    db, left, _ = two_stored_runs(tmp_path)

    assert main(["diff", left.run_id, left.run_id, "--db", db]) == 0
    assert "the two runs are identical" in capsys.readouterr().out


def test_cli_diff_by_index_is_available_as_the_baseline(tmp_path, capsys) -> None:
    from flightrec.cli import main

    db, left, right = two_stored_runs(tmp_path)

    assert main(["diff", left.run_id, right.run_id, "--db", db, "--by-index"]) == 0
    assert "index-by-index" in capsys.readouterr().out


def test_cli_describes_a_step_by_what_it_actually_produced() -> None:
    """An empty result is the demo's root cause; a failed step is its error.

    A failed tool call records ``FR_OUTPUT = None`` on purpose, so neither
    "truthy output wins" nor "present output wins" gets both of these right.
    """
    from flightrec.cli import _outcome_text

    empty = make_run([("web_search", {"query": "seattle"}, [])]).steps()[0]
    failed = make_run(
        [("fetch_page", {"url": "u"}, None, SpanStatus.ERROR)], run_id="b"
    ).steps()[0]
    failed.status_message = "ToolError: 404: no such page"

    assert _outcome_text(empty) == "[]"
    assert _outcome_text(failed) == "ToolError: 404: no such page"


def test_cli_diff_reports_a_missing_run(tmp_path, capsys) -> None:
    from flightrec.cli import main

    db, left, _ = two_stored_runs(tmp_path)

    assert main(["diff", left.run_id, "nope", "--db", db]) == 1
    assert "no run" in capsys.readouterr().out


def test_a_recording_diffed_against_its_own_replay_is_identical() -> None:
    """Ties step 8 back to step 7: a faithful replay must diff as unchanged."""
    from flightrec.replay import replay_run

    recorded = ResearchAgent(seed=0, faults=FaultConfig.realistic()).run().run
    replayed = replay_run(recorded).run

    assert diff_runs(recorded, replayed).identical
