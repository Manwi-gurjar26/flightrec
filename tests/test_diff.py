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
    # Counted as changed *or* replaced, since a mispairing that lands on a
    # different tool is reported as a replacement -- still garbage, still the
    # point.
    assert naive.count(Op.CHANGED) + naive.count(Op.REPLACED) >= 4


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

    paired = next(
        c
        for c in diff_runs(left, right).columns
        if c.op in (Op.CHANGED, Op.REPLACED) and c.left_index == 1
    )

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


# --- reorderings, which the monotonic pass cannot express ---------------------


def moved_run(order: list[int], run_id: str = "b") -> Run:
    return make_run([CLEAN[i] for i in order], run_id=run_id)


def test_a_reordered_block_is_paired_with_itself_and_marked_moved() -> None:
    """Steps 1-2 done later. Both runs contain them; only the order differs.

    Which *side* of a swap gets labelled "moved" is genuinely ambiguous -- "1-2
    happened later" and "3-4 happened earlier" describe one event, and the two
    backbones are the same length, so the tie-break is arbitrary and the test
    does not pretend otherwise. What is not ambiguous, and is what the diff is
    for, is that every step is paired with itself and nothing is reported as
    inserted, removed or changed.
    """
    left = make_run(CLEAN)
    right = moved_run([0, 3, 4, 1, 2, 5, 6])

    diff = diff_runs(left, right)
    pairs = {c.left_index: c.right_index for c in diff.columns}

    assert pairs == {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 5, 6: 6}
    assert diff.count(Op.REMOVED) == 0
    assert diff.count(Op.INSERTED) == 0
    assert diff.count(Op.CHANGED) == 0
    assert {c.left_index for c in diff.columns if c.moved} in ({1, 2}, {3, 4})


def test_the_monotonic_pass_alone_cannot_do_it() -> None:
    """The control. Without the second pass the same input comes out as gaps or
    substitutions, which is the structural limit the second pass exists for."""
    left = make_run(CLEAN)
    right = moved_run([0, 3, 4, 1, 2, 5, 6])

    columns = align(left.steps(), right.steps(), detect_moves=False)

    assert not any(c.moved for c in columns)
    assert {c.left_index: c.right_index for c in columns} != {1: 3, 2: 4, 0: 0}
    assert any(c.op is not Op.MATCH for c in columns)


def test_a_move_is_a_genuine_divergence() -> None:
    """Doing the same steps in a different order is a real difference.

    It is reported even though every step matched, because "same steps, other
    order" is exactly the kind of thing that changes an outcome.
    """
    diff = diff_runs(make_run(CLEAN), moved_run([0, 3, 4, 1, 2, 5, 6]))

    assert not diff.identical
    assert diff.first_divergence is not None
    assert diff.first_divergence.moved


def test_a_step_that_moved_and_changed_is_reported_as_both() -> None:
    """The case pass one cannot see: its content differs, so it is not identical
    to anything, and only the similarity pass can recover it."""
    left = make_run(CLEAN)
    reordered = [CLEAN[i] for i in [0, 3, 4, 1, 2, 5, 6]]
    reordered[3] = ("web_search", {"query": "seattle rainy days 2024"}, ["other-url"])
    right = make_run(reordered, run_id="b")

    diff = diff_runs(left, right)
    column = next(c for c in diff.columns if c.left_index == 1)

    assert column.op is Op.CHANGED
    assert column.right_index == 3, "the similarity pass has to find it despite the edit"
    assert diff.moved_count, "and the reordering still has to be reported"


def test_a_spliced_in_copy_does_not_steal_its_originals_pairing() -> None:
    """Position has to break the tie when content cannot.

    Splice in a copy of a step that also exists later, and edit the later one.
    The copy is now an *exact* match for the left step while the real
    counterpart differs by its output -- but similarity ignores output, so both
    score 1.0 and only position separates them. Ranking exact matches ahead of
    everything else made the copy win, which is six positions from where the
    surrounding alignment says the step belongs.
    """
    left = make_run(CLEAN)
    original_fetch = CLEAN[3]
    changed_fetch = ("fetch_page", {"url": "https://w.example/s"}, "recorded on 3 days")
    right = make_run(
        CLEAN[:1] + [original_fetch] + CLEAN[1:3] + [changed_fetch] + CLEAN[4:],
        run_id="b",
    )

    column = next(c for c in diff_runs(left, right).columns if c.left_index == 3)

    assert column.right_index == 4, "should pair with the counterpart, not the copy"
    assert column.op is Op.CHANGED


def test_move_recovery_does_not_invent_moves_in_an_ordinary_insertion() -> None:
    """A pass that re-pairs after the fact can invent correspondences.

    An insertion has no move in it, so nothing here may be marked as one --
    otherwise every retry would be reported as a reordering.
    """
    left = make_run(CLEAN)
    right = make_run(CLEAN[:3] + RETRY_BLOCK + CLEAN[3:], run_id="b")

    diff = diff_runs(left, right)

    assert not any(c.moved for c in diff.columns)
    assert diff.count(Op.INSERTED) == 2


def test_move_recovery_leaves_a_deletion_alone() -> None:
    """A removed step has no counterpart, and pairing it with a survivor to
    avoid an unexplained gap loses both the deletion and whatever it stole."""
    left = make_run(CLEAN)
    right = make_run(CLEAN[:1] + CLEAN[3:], run_id="b")

    diff = diff_runs(left, right)

    assert diff.count(Op.REMOVED) == 2
    assert not any(c.moved for c in diff.columns)


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


def test_a_different_tool_is_replaced_not_changed() -> None:
    """"Changed" claims two steps are the same step. Sometimes they are not.

    A ``fetch_page`` becoming a ``calculator`` is not that call returning
    something new -- it is the agent doing something else. The columns stay
    paired, because their position is worth seeing, but the report says so.
    """
    left = make_run(CLEAN)
    other = list(CLEAN)
    other[3] = ("calculator", {"expression": "1 + 1"}, 2.0)
    right = make_run(other, run_id="b")

    column = next(c for c in diff_runs(left, right).columns if c.left_index == 3)

    assert column.op is Op.REPLACED
    assert column.right_index == 3, "still paired -- position is information"
    assert column.genuine


def test_the_same_tool_with_a_new_result_is_still_changed() -> None:
    """The other side of it: the distinction is worthless if everything is replaced."""
    left = make_run(CLEAN)
    other = list(CLEAN)
    other[3] = ("fetch_page", {"url": "https://w.example/s"}, "recorded on 3 days")
    right = make_run(other, run_id="b")

    column = next(c for c in diff_runs(left, right).columns if c.left_index == 3)

    assert column.op is Op.CHANGED


def test_two_different_failures_are_not_the_same_result() -> None:
    """A failed step records an output of ``None``, which makes every failure
    look alike unless the message is compared.

    Two fetches of two *different* URLs that both 404 were being reported as one
    step with reworded arguments -- a genuine difference described as cosmetic,
    on exactly the steps where a run went wrong. It also hid them from the move
    pass, which treats a cosmetic column as settled and never looks again.
    """
    left = make_run(CLEAN)
    right = make_run(CLEAN, run_id="b")
    for run, url in ((left, "seattle-climate"), (right, "portland-annual")):
        step = run.steps()[3]
        step.attributes[FR_INPUT] = {"url": f"https://w.example/{url}"}
        step.attributes[FR_OUTPUT] = None
        step.status = SpanStatus.ERROR
        step.status_message = f"ToolError: 404: no such page https://w.example/{url}"

    diff = diff_runs(left, right)

    assert diff.count(Op.COSMETIC) == 0
    assert diff.divergence_index == 3


def test_the_same_failure_twice_is_still_a_match() -> None:
    """The other half: identical failures must not become spurious divergences."""
    runs = []
    for label in ("a", "b"):
        run = make_run(CLEAN, run_id=label)
        step = run.steps()[3]
        step.attributes[FR_OUTPUT] = None
        step.status = SpanStatus.ERROR
        step.status_message = "ToolError: 404: no such page"
        runs.append(run)

    assert diff_runs(*runs).identical


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


# --- banding ------------------------------------------------------------------


def long_pair(steps: int = 120):
    """Two long runs built by chaining recordings, for the banding tests."""
    from flightrec.demo.agent import ResearchAgent
    from flightrec.demo.tools import CITY_DAYS

    cities = list(CITY_DAYS)[:6]

    def build(first_seed: int, label: str) -> Run:
        spans = []
        seed = first_seed
        while len(spans) < steps:
            run = ResearchAgent(
                seed=seed, cities=cities, faults=FaultConfig.realistic()
            ).run().run
            for span in run.steps():
                copy = span.model_copy(deep=True)
                copy.sequence = len(spans)
                copy.span_id = f"{label}-{len(spans)}"
                copy.parent_span_id = None
                spans.append(copy)
            seed += 1
        return Run(run_id=label, spans=spans)

    return build(1, "left"), build(41, "right")


def columns_of(left, right, banded):
    return [
        (c.op, c.left_index, c.right_index, c.moved)
        for c in align(left.steps(), right.steps(), banded=banded)
    ]


def test_banding_returns_exactly_what_the_full_table_would() -> None:
    """The point of the widening: banding is a speed-up, not an approximation.

    A band that clips the true path returns a worse alignment and says nothing,
    which is the quiet wrongness this project exists to avoid. So the traceback
    reports how far it strayed, and a path that reaches the edge is recomputed
    twice as wide.
    """
    left, right = long_pair()

    assert columns_of(left, right, True) == columns_of(left, right, False)


def test_banding_is_faster_on_a_long_run() -> None:
    """Otherwise it is complexity with nothing bought."""
    import time

    left, right = long_pair(steps=200)

    def elapsed(banded: bool) -> float:
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            align(left.steps(), right.steps(), banded=banded)
            best = min(best, time.perf_counter() - start)
        return best

    assert elapsed(True) < elapsed(False)


def test_short_runs_skip_the_band_entirely() -> None:
    """Below the threshold the retry can only cost time, so it is not attempted."""
    left = make_run(CLEAN)
    right = make_run(CLEAN[:3] + RETRY_BLOCK + CLEAN[3:], run_id="b")

    assert columns_of(left, right, True) == columns_of(left, right, False)


def test_a_reordering_beyond_the_move_window_is_not_found() -> None:
    """The one real limit banding introduced, pinned as a limit.

    The move pass looks within a fixed window of where a step belongs, because
    searching the whole run for every unexplained step is quadratic. A block
    moved 64 positions is still found; one moved 70 is not, and a reader
    deserves to know that from a test rather than from a surprise.
    """
    from flightrec.diff import _MOVE_WINDOW

    left, _ = long_pair(steps=200)
    steps = left.steps()

    def moved_by(distance: int) -> bool:
        copies = [s.model_copy(deep=True) for s in steps]
        block, rest = copies[10:12], copies[:10] + copies[12:]
        target = 10 + distance
        reordered = rest[:target] + block + rest[target:]
        for index, span in enumerate(reordered):
            span.sequence = index
            span.span_id = f"r-{index}"
        return any(
            c.moved for c in align(steps, Run(run_id="r", spans=reordered).steps())
        )

    assert moved_by(_MOVE_WINDOW)
    assert not moved_by(_MOVE_WINDOW + 6)
