"""Replay has one job: be a reproduction, or say loudly that it is not.

So these tests are mostly about the ways a replay could *look* right while being
a second run in disguise -- results quietly re-executed, retries dropped, a
trajectory that drifted after an edit and was reported as faithful anyway.
"""

from __future__ import annotations

import pytest

from flightrec.demo.agent import ResearchAgent
from flightrec.demo.tools import FaultConfig
from flightrec.replay import ReplayMismatch, replay_run
from flightrec.spans import (
    first_divergence,
    trajectory,
    FR_DIVERGENT,
    FR_INPUT,
    FR_OUTPUT,
    FR_REPLAYED,
    FR_SERVED,
    GEN_AI_TOOL_NAME,
    Run,
    SpanKind,
)


def record(seed: int = 0, faults: FaultConfig | None = None):
    """One recorded run, made the way the CLI makes them."""
    agent = ResearchAgent(seed=seed, faults=faults or FaultConfig.realistic())
    return agent.run()


def tool_steps(run: Run) -> list[tuple[int, object]]:
    return [(i, s) for i, s in enumerate(run.steps()) if s.kind is SpanKind.TOOL]


# --- the core claim -----------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_full_replay_reproduces_the_recorded_trajectory(seed: int) -> None:
    recorded = record(seed)
    replay = replay_run(recorded.run)

    assert replay.divergence_step is None
    assert replay.faithful
    assert replay.outcome.answer == recorded.answer
    assert replay.outcome.text == recorded.text


def test_a_full_replay_executes_nothing_at_all() -> None:
    """Every step, model calls included, comes out of the recording.

    Serving only tool results would still reproduce the trajectory -- and would
    re-bill every model call to get there, which is the cost the feature exists
    to avoid.
    """
    recorded = record(seed=0)
    replay = replay_run(recorded.run)

    assert replay.served == len(recorded.run.steps())
    assert replay.live == 0


def test_tool_results_come_from_the_recording_not_from_the_tools() -> None:
    """The test that proves the previous one means something.

    A replay that re-executed the tools would also reproduce the trajectory,
    because the tools are seeded too -- identical output, wrong mechanism, and
    the whole feature silently absent. Tampering with the recording separates
    the two: only a replay that actually reads it can return the tampered value.

    The tampered step is the calculator, the last tool call in the run, and the
    cut is placed on the step straight after it so the model call that consumes
    the tampered value runs live. Tamper any earlier step and the replay
    correctly refuses: the transcript no longer matches the recording, which is
    a different behaviour worth testing and is tested below.
    """
    recorded = record(seed=0)
    steps = recorded.run.steps()
    calculator_at = next(
        i for i, s in enumerate(steps) if s.attr(GEN_AI_TOOL_NAME) == "calculator"
    )
    steps[calculator_at].attributes[FR_OUTPUT] = 999.0

    replay = replay_run(recorded.run, from_step=calculator_at + 1)

    assert replay.outcome.answer == 999
    assert recorded.answer != 999


def test_two_replays_of_one_recording_are_bit_identical() -> None:
    """Not just the same trajectory -- the same bytes, span IDs included."""
    recorded = record(seed=3)

    first = replay_run(recorded.run).run
    second = replay_run(recorded.run).run

    assert first.canonical_json() == second.canonical_json()


def test_replay_keeps_the_retries_that_made_the_run_expensive() -> None:
    """A flaky run replayed as a clean one is the failure this tool exists to stop."""
    for seed in range(20):
        recorded = record(seed)
        retries = sum(
            1 for s in recorded.run.spans for e in s.events if e.name == "retry"
        )
        if retries:
            break
    else:  # pragma: no cover - the realistic fault mix always produces one
        pytest.skip("no seed in range produced a retry")

    replay = replay_run(recorded.run)
    replayed_retries = sum(
        1 for s in replay.run.spans for e in s.events if e.name == "retry"
    )
    assert replayed_retries == retries


def test_replayed_and_served_are_recorded_as_different_claims() -> None:
    """Every span in a replay is replayed; only pre-cut steps are *served*.

    The distinction is what stops a live-executed step being presented as
    recorded data, so it has to survive on the run that mixes both.
    """
    recorded = record(seed=0)
    cut = tool_steps(recorded.run)[1][0]
    replay = replay_run(recorded.run, from_step=cut)

    assert all(s.attr(FR_REPLAYED) for s in replay.run.spans)
    steps = replay.run.steps()
    assert all(s.attr(FR_SERVED) for s in steps[:cut])
    assert not any(s.attr(FR_SERVED) for s in steps[cut:])


def test_a_tampered_recording_is_refused_rather_than_half_served() -> None:
    """Change a served result and the prompts downstream no longer match.

    Serving the recorded model response to a prompt the recording never saw
    would be answering a question nobody asked, and the answer would look
    perfectly ordinary on the timeline.
    """
    recorded = record(seed=0)
    calculator = next(
        s for _, s in tool_steps(recorded.run) if s.attr(GEN_AI_TOOL_NAME) == "calculator"
    )
    calculator.attributes[FR_OUTPUT] = 999.0

    with pytest.raises(ReplayMismatch, match="but the replay asked for"):
        replay_run(recorded.run)


# --- diverging on purpose -----------------------------------------------------


def test_from_step_executes_forward_live_and_marks_it() -> None:
    recorded = record(seed=0)
    cut = tool_steps(recorded.run)[1][0]

    replay = replay_run(recorded.run, from_step=cut)

    assert replay.served > 0 and replay.live > 0
    before = [s for s in replay.run.ordered_spans() if s.sequence < cut]
    after = replay.run.steps()[cut:]
    assert not any(s.attr(FR_DIVERGENT) for s in before)
    assert all(s.attr(FR_DIVERGENT) for s in after)


def test_strict_stops_at_the_edit_point_instead_of_guessing() -> None:
    recorded = record(seed=0)
    cut = tool_steps(recorded.run)[1][0]

    replay = replay_run(recorded.run, from_step=cut, strict=True)

    assert replay.stopped
    assert replay.live == 0
    assert "strict" in (replay.outcome.error or "")
    assert len(replay.run.steps()) <= cut + 1


def test_editing_the_task_diverges_from_the_first_step() -> None:
    recorded = record(seed=0)

    replay = replay_run(recorded.run, task="How many days of rain did Seattle record?")

    assert replay.edits["task"]
    assert replay.from_step == 0
    assert replay.live > 0 and replay.served == 0
    assert replay.divergence_step is not None


def test_replaying_a_clean_run_past_the_cut_still_uses_the_recorded_faults() -> None:
    """Live steps continue the same world, not a fault-free one.

    Fault rates are not in the code path being replayed -- they are recorded on
    the root span precisely so a step re-executed after the edit point behaves
    like a continuation rather than like a different agent.
    """
    recorded = record(seed=0, faults=FaultConfig())
    root = next(s for s in recorded.run.spans if s.kind is SpanKind.AGENT)

    assert root.attr("flightrec.faults") == {
        "empty_search_rate": 0.0,
        "flaky_fetch_rate": 0.0,
        "stale_page_rate": 0.0,
    }
    replay = replay_run(recorded.run, from_step=0)
    assert replay.outcome.faults_fired == []


# --- refusing to lie ----------------------------------------------------------


def test_a_recording_that_does_not_match_raises_rather_than_improvising() -> None:
    recorded = record(seed=0)
    _, first_tool = tool_steps(recorded.run)[0]
    first_tool.name = "tool.not_a_real_tool"

    with pytest.raises(ReplayMismatch, match="not_a_real_tool"):
        replay_run(recorded.run)


def test_mismatch_beats_the_agents_own_exception_handling() -> None:
    """The agent swallows tool exceptions by design; the replay error must not be."""
    recorded = record(seed=0)
    _, first_tool = tool_steps(recorded.run)[0]
    first_tool.attributes[FR_INPUT] = {"query": "something else entirely"}

    with pytest.raises(ReplayMismatch):
        replay_run(recorded.run)


# --- the fidelity primitive ---------------------------------------------------


def test_replays_of_different_recordings_do_not_share_span_ids() -> None:
    """Span IDs are the storage primary key, and writes are INSERT OR REPLACE.

    Seeding the ID generator from the *agent's* seed -- which every run made
    with that seed shares -- would make two replays overwrite each other's
    steps in the database instead of colliding loudly.
    """
    first = replay_run(record(seed=0).run).run
    second = replay_run(record(seed=1).run).run

    assert not {s.span_id for s in first.spans} & {s.span_id for s in second.spans}


def test_an_edited_replay_gets_its_own_identity() -> None:
    recorded = record(seed=0)

    plain = replay_run(recorded.run).run
    edited = replay_run(recorded.run, from_step=0).run

    assert plain.run_id != edited.run_id
    assert not {s.span_id for s in plain.spans} & {s.span_id for s in edited.spans}


def test_first_divergence_points_at_the_step_that_changed() -> None:
    recorded = record(seed=0)
    other = Run(run_id="x", spans=[s.model_copy(deep=True) for s in recorded.run.spans])
    target = other.steps()[2]
    target.attributes[FR_OUTPUT] = "something different"

    assert first_divergence(recorded.run, other) == 2


def test_first_divergence_reports_a_truncated_run_at_the_cut() -> None:
    recorded = record(seed=0)
    short = Run(
        run_id="x",
        spans=[s.model_copy(deep=True) for s in recorded.run.ordered_spans()[:4]],
    )

    assert first_divergence(recorded.run, short) == len(trajectory(short))


# --- the CLI ------------------------------------------------------------------


def stored_run(tmp_path, seed: int = 0):
    from flightrec.storage import RunStore

    recorded = record(seed)
    db = str(tmp_path / "runs.db")
    store = RunStore(db)
    store.add_run(recorded.run)
    store.close()
    return db, recorded


def test_cli_replay_reports_a_faithful_replay(tmp_path, capsys) -> None:
    from flightrec.cli import main

    db, recorded = stored_run(tmp_path)

    assert main(["replay", recorded.run.run_id, "--db", db]) == 0
    out = capsys.readouterr().out
    assert "FAITHFUL" in out
    assert "executed live" in out


def test_cli_replay_from_step_labels_each_step_by_source(tmp_path, capsys) -> None:
    from flightrec.cli import main

    db, recorded = stored_run(tmp_path)
    cut = tool_steps(recorded.run)[1][0]

    assert main(["replay", recorded.run.run_id, "--db", db, "--from-step", str(cut)]) == 0
    out = capsys.readouterr().out
    assert "recorded" in out and "live" in out


def test_cli_replay_stores_the_replay_alongside_the_original(tmp_path, capsys) -> None:
    from flightrec.cli import main
    from flightrec.storage import RunStore

    db, recorded = stored_run(tmp_path)

    assert main(["replay", recorded.run.run_id, "--db", db, "--store"]) == 0
    store = RunStore(db)
    run_ids = {s.run_id for s in store.list_runs(limit=10)}
    store.close()

    assert len(run_ids) == 2
    assert recorded.run.run_id in run_ids


def test_cli_replay_fails_loudly_on_a_mismatch(tmp_path, capsys) -> None:
    from flightrec.cli import main
    from flightrec.storage import RunStore

    recorded = record(seed=0)
    _, first_tool = tool_steps(recorded.run)[0]
    first_tool.name = "tool.not_a_real_tool"
    db = str(tmp_path / "runs.db")
    store = RunStore(db)
    store.add_run(recorded.run)
    store.close()

    assert main(["replay", recorded.run.run_id, "--db", db]) == 2
    assert "REPLAY FAILED" in capsys.readouterr().out


def test_cli_replay_reports_a_missing_run(tmp_path, capsys) -> None:
    from flightrec.cli import main

    db, _ = stored_run(tmp_path)
    assert main(["replay", "nope", "--db", db]) == 1
    assert "no run" in capsys.readouterr().out


# --- an agent this library has never seen -------------------------------------
#
# The point of the engine being general. If these only ever ran against the demo
# agent, "works for any agent" would be an architectural intention rather than a
# tested property.


class PageMissing(RuntimeError):
    """This agent's own failure type. flightrec has never heard of it."""


class HaikuAgent:
    """Somebody else's agent. Nothing here knows what flightrec is for.

    Two tools and a "model" that is a dictionary. It shares no code with the
    demo agent, has no seed on its root span, uses its own exception type, and
    was written to be replayed without being told how.
    """

    def __init__(self, tracer, pages: dict[str, str] | None = None) -> None:
        self.tracer = tracer
        self.pages = pages if pages is not None else {"a": "alpha", "b": "beta"}
        self.calls: list[str] = []

    def lookup(self, key: str) -> str:
        self.calls.append(key)
        if key not in self.pages:
            raise PageMissing(f"no page {key}")
        return self.pages[key]

    def run(self, task: str) -> str:
        from flightrec.spans import SpanKind

        with self.tracer.span("haiku_agent", kind=SpanKind.AGENT, inputs=task):
            parts = []
            for key in task.split():
                plan = self.tracer.call(
                    "chat",
                    lambda key=key: f"I will look up {key}.",
                    kind=SpanKind.LLM,
                    inputs=key,
                )
                parts.append(plan)
                try:
                    parts.append(
                        self.tracer.call(
                            "tool.lookup",
                            lambda key=key: self.lookup(key),
                            kind=SpanKind.TOOL,
                            inputs={"key": key},
                        )
                    )
                except Exception as exc:  # the agent handles its own failures
                    parts.append(f"failed: {exc}")
            return " ".join(parts)


def record_foreign(pages: dict[str, str] | None = None, task: str = "a b"):
    from flightrec.sinks import MemorySink
    from flightrec.tracer import Tracer

    tracer = Tracer(sink=MemorySink(), trace_id="foreign")
    agent = HaikuAgent(tracer, pages)
    answer = agent.run(task)
    return tracer.collect(), answer, agent


def test_an_unknown_agent_replays_without_running_its_tools() -> None:
    """The whole claim, in one test.

    The engine never imports this agent, cannot construct it, and knows nothing
    about its tools. It drives the callable it was given and answers that
    agent's tracer from the recording.
    """
    from flightrec.replay import replay

    recording, answer, original = record_foreign()
    assert original.calls == ["a", "b"], "the recording really did call the tools"

    replayed_agent = {}

    def run_agent(tracer, task):
        agent = HaikuAgent(tracer)
        replayed_agent["it"] = agent
        return agent.run(task)

    result = replay(recording, run_agent)

    assert result.outcome == answer
    assert result.faithful
    assert result.served == len(recording.steps())
    assert result.live == 0
    assert replayed_agent["it"].calls == [], "no tool may be executed on a full replay"


def test_an_unknown_agent_runs_live_past_the_edit_point() -> None:
    """And the other half: past the cut it really is the agent running."""
    from flightrec.replay import replay

    recording, _, _ = record_foreign()
    cut = 2

    seen = {}

    def run_agent(tracer, task):
        agent = HaikuAgent(tracer, pages={"a": "alpha", "b": "CHANGED"})
        seen["it"] = agent
        return agent.run(task)

    result = replay(recording, run_agent, from_step=cut)

    assert result.served == cut
    assert result.live > 0
    assert seen["it"].calls == ["b"], "only the steps past the cut are executed"
    assert "CHANGED" in result.outcome


def test_an_unknown_agents_failure_is_replayed_as_a_failure() -> None:
    """A recording stores an exception's name and message, not its class.

    Without being told which types matter, a replayed failure comes back as
    ReplayedError -- correct about what happened, honest that the type is gone.
    Callers that branch on the type pass ``exceptions=`` and get it back.
    """
    from flightrec.replay import ReplayedError, replay

    recording, answer, _ = record_foreign(pages={"a": "alpha"}, task="a missing")
    assert "failed: no page missing" in answer

    caught: list[BaseException] = []

    def run_agent(tracer, task):
        agent = HaikuAgent(tracer)
        original = agent.lookup

        def watched(key):
            try:
                return original(key)
            except BaseException as exc:  # pragma: no cover - never reached on replay
                caught.append(exc)
                raise

        agent.lookup = watched
        return agent.run(task)

    # Unmapped: the failure comes back as a ReplayedError subclass that *records*
    # exactly as the original did, so the replay is still faithful.
    generic = replay(recording, run_agent)
    assert generic.faithful
    assert generic.outcome == answer
    assert not caught, "the tool never ran, so nothing real was raised"

    # Mapped: the agent gets its own class back, for code that branches on type.
    typed = replay(
        recording,
        run_agent,
        exceptions={"PageMissing": PageMissing},
    )
    assert typed.faithful
    assert typed.outcome == answer


def test_the_engine_does_not_import_the_demo_agent() -> None:
    """Module layout has to back the claim up, not just the docstring.

    ``replay_run`` imports the demo inside the function precisely so that the
    engine does not depend on the example it ships with.
    """
    import ast
    import pathlib

    source = pathlib.Path("src/flightrec/replay.py").read_text()
    tree = ast.parse(source)
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    modules = {n.module for n in top_level if isinstance(n, ast.ImportFrom)}

    assert not any(m and "demo" in m for m in modules), modules


def test_the_worked_example_still_works() -> None:
    """The example is the documentation for replaying your own agent.

    An example that stopped running would be worse than none: it is the thing
    somebody copies, and it is the only end-to-end proof in the repo that the
    integration really is two calls to ``tracer.call``.
    """
    import importlib.util
    import pathlib

    path = pathlib.Path("examples/replay_your_own_agent.py")
    spec = importlib.util.spec_from_file_location("_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main()  # must not raise
