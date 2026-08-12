"""Tests for the demo agent.

The agent is measurement apparatus, so these tests are stricter than they would
be for an example. If the agent's failures are not reproducible, every number
in the README is noise.
"""

import pytest

from flightrec.demo.agent import FR_CONFABULATED, ResearchAgent, run_agent
from flightrec.demo.model import CITIES, FABRICATED_PRIOR, StubModel
from flightrec.demo.tools import GROUND_TRUTH, Fault, FaultConfig, Toolbox, ToolError
from flightrec.determinism import SeededIdGenerator, VirtualClock
from flightrec.retry import TransientError, make_rng
from flightrec.sinks import MemorySink
from flightrec.spans import FR_INPUT, SpanKind, SpanStatus

CLEAN = FaultConfig()
ALWAYS_EMPTY = FaultConfig(empty_search_rate=1.0)
ALWAYS_FLAKY = FaultConfig(flaky_fetch_rate=1.0)
ALWAYS_STALE = FaultConfig(stale_page_rate=1.0)


def pinned_agent(seed: int = 0, **kwargs) -> ResearchAgent:
    """An agent with every source of variation pinned."""
    kwargs.setdefault("faults", CLEAN)
    return ResearchAgent(
        seed=seed,
        sink=MemorySink(),
        clock=VirtualClock(start=1_000.0, step=0.001),
        id_generator=SeededIdGenerator(seed=seed),
        trace_id=f"trace-{seed}",
        **kwargs,
    )


# --- the happy path ----------------------------------------------------------


def test_clean_run_gets_the_right_answer():
    result = pinned_agent(seed=1).run()
    assert result.answer == GROUND_TRUTH
    assert result.correct
    assert result.error is None
    assert not result.confabulated


def test_clean_run_has_the_expected_trajectory():
    result = pinned_agent(seed=1).run()
    assert [s.name for s in result.run.steps()] == [
        "chat",
        "tool.web_search",
        "chat",
        "tool.fetch_page",
        "chat",
        "tool.web_search",
        "chat",
        "tool.fetch_page",
        "chat",
        "tool.calculator",
        "chat",
    ]


class CoarseClock:
    """A clock with terrible resolution, like ``time.time()`` on Windows.

    Returns the same value for a run of calls before ticking, so whole groups
    of spans share a start time.
    """

    def __init__(self, start: float = 1_000.0, tick: float = 0.0156, every: int = 8):
        self._t = start
        self._tick = tick
        self._every = every
        self._calls = 0

    def now(self) -> float:
        self._calls += 1
        if self._calls % self._every == 0:
            self._t += self._tick
        return self._t


def test_steps_are_ordered_correctly_under_a_coarse_clock():
    """Regression guard for the ordering bug.

    The first version sorted the trajectory by ``(start_time, span_id)``. On
    Windows ``time.time()`` resolves to ~15.6ms and agent steps finish in
    microseconds, so most spans shared a start time and the tiebreaker was a
    random UUID -- which put the calculator call ahead of the fetch that
    produced its operands.

    ``SystemClock`` is high-resolution now, which means a test using the real
    clock would pass even if time-based sorting came back. So this one forces a
    coarse clock: ordering must survive a clock that cannot tell these spans
    apart at all.
    """
    agent = ResearchAgent(
        seed=1,
        faults=CLEAN,
        sink=MemorySink(),
        clock=CoarseClock(),
        id_generator=SeededIdGenerator(seed=1),
        trace_id="coarse",
    )
    result = agent.run()
    steps = result.run.steps()

    # Precondition: the clock really is too coarse to order these spans.
    assert len({s.start_time for s in steps}) < len(steps)

    assert [s.name for s in steps] == [
        "chat",
        "tool.web_search",
        "chat",
        "tool.fetch_page",
        "chat",
        "tool.web_search",
        "chat",
        "tool.fetch_page",
        "chat",
        "tool.calculator",
        "chat",
    ]

    sequences = [s.sequence for s in result.run.spans]
    assert len(set(sequences)) == len(sequences), "sequence numbers must be unique"

    # The tree must survive it too, not just the flat step list.
    root = result.run.tree()[0]
    assert [c.span.sequence for c in root.children] == sorted(
        c.span.sequence for c in root.children
    )


def test_run_is_recorded_as_a_tree_under_one_agent_span():
    result = pinned_agent(seed=1).run()
    roots = result.run.tree()

    assert len(roots) == 1
    assert roots[0].span.kind is SpanKind.AGENT
    assert roots[0].span.name == "research_agent"
    assert len(roots[0].children) == len(result.run.spans) - 1
    assert {c.span.kind for c in roots[0].children} == {SpanKind.LLM, SpanKind.TOOL}


# --- determinism -------------------------------------------------------------


@pytest.mark.parametrize("faults", [CLEAN, ALWAYS_EMPTY, ALWAYS_FLAKY, ALWAYS_STALE])
def test_same_seed_reproduces_the_run_exactly(faults):
    a = pinned_agent(seed=3, faults=faults).run()
    b = pinned_agent(seed=3, faults=faults).run()

    assert a.answer == b.answer
    assert a.faults_fired == b.faults_fired
    assert a.run.model_dump_json() == b.run.model_dump_json()


def test_different_seeds_produce_different_outcomes():
    """There has to be variation, or there is nothing to diff."""
    outcomes = {
        run_agent(seed=seed, faults=FaultConfig.realistic()).answer
        for seed in range(12)
    }
    assert len(outcomes) > 1, "the corpus must contain both good and bad runs"
    assert GROUND_TRUTH in outcomes, "some runs must succeed"
    assert outcomes - {GROUND_TRUTH}, "some runs must fail"


def test_temperature_above_zero_changes_the_trajectory():
    """Why temperature has to be pinned before replaying, demonstrated."""

    def queries(temperature: float, seed: int) -> list:
        agent = pinned_agent(seed=seed, temperature=temperature)
        result = agent.run()
        return [
            s.attr("flightrec.input")
            for s in result.run.steps()
            if s.name == "tool.web_search"
        ]

    assert queries(0.0, 5) == queries(0.0, 5)

    hot = [queries(1.0, seed) for seed in range(8)]
    assert len({str(q) for q in hot}) > 1, "temperature must actually vary behaviour"


# --- the three failure modes -------------------------------------------------


def test_empty_search_makes_the_agent_invent_a_recovery():
    """The canonical agent failure: no crash, no error, a confident wrong answer."""
    result = pinned_agent(seed=2, faults=ALWAYS_EMPTY).run()

    assert Fault.EMPTY_SEARCH in result.faults_fired
    assert result.error is None, "the agent must not crash -- that is the whole problem"
    assert result.confabulated
    assert result.answer == FABRICATED_PRIOR["seattle"] + FABRICATED_PRIOR["portland"]
    assert not result.correct

    # The invented URL 404s, so the failure is visible on the timeline...
    fetches = [s for s in result.run.steps() if s.name == "tool.fetch_page"]
    assert all(s.status is SpanStatus.ERROR for s in fetches)

    # ...and the exact step where it stopped being grounded is marked.
    confabulated = [s for s in result.run.spans if s.attr(FR_CONFABULATED)]
    assert confabulated, "the ungrounded step must be identifiable"


def test_the_agent_tries_a_second_url_before_giving_up():
    """Two guesses, both wrong, and the give-up is what makes runs differ in length."""
    result = pinned_agent(seed=2, faults=ALWAYS_EMPTY).run()

    attempted = [
        s.attr(FR_INPUT)["url"]
        for s in result.run.steps()
        if s.name == "tool.fetch_page"
    ]

    assert len(attempted) == 2 * len(CITIES)
    assert len(set(attempted)) == len(attempted), "a retry of the same URL is not a recovery"
    assert result.confabulated, "the second guess must not rescue the run"


def test_trajectory_length_varies_across_the_corpus():
    """The property the diff measurement depends on, asserted so it cannot regress.

    Until the agent could give up on a city, every seed produced an 11-step run.
    With every run the same length, index-by-index pairing is accidentally
    correct and the sequence alignment cannot be shown to be worth anything --
    so a fixed-length corpus silently invalidates the divergence-localization
    number in the README.
    """
    lengths = {
        len(ResearchAgent(seed=seed, faults=FaultConfig.realistic()).run().run.steps())
        for seed in range(40)
    }

    assert len(lengths) >= 3, f"corpus is too uniform to exercise alignment: {lengths}"


def test_flaky_fetch_retries_and_still_succeeds():
    """The cost anomaly: right answer, more calls, and nothing looks wrong."""
    result = pinned_agent(seed=4, faults=ALWAYS_FLAKY).run()

    assert Fault.FLAKY_FETCH in result.faults_fired
    assert result.answer == GROUND_TRUTH, "retries should recover a correct answer"

    retries = [
        event
        for span in result.run.spans
        for event in span.events
        if event.name == "retry"
    ]
    assert retries, "retries must be recorded, or the cost spike is unexplainable"
    assert all(e.attributes["retry.delay_s"] > 0 for e in retries)


def test_stale_page_is_the_silent_failure():
    """Every step green, every tool succeeded, answer still wrong."""
    result = pinned_agent(seed=6, faults=ALWAYS_STALE).run()

    assert Fault.STALE_PAGE in result.faults_fired
    assert result.error is None
    assert not result.confabulated, "the agent was grounded -- in the wrong data"
    assert result.answer is not None and result.answer != GROUND_TRUTH
    assert not result.run.has_error, (
        "no span is marked failed, which is what makes this class of bug hard "
        "and what the diff view is for"
    )


def test_tool_failure_is_recorded_but_does_not_stop_the_loop():
    result = pinned_agent(seed=2, faults=ALWAYS_EMPTY).run()
    assert result.run.has_error
    assert result.steps > 1, "the loop continued past the failure"
    assert result.text, "the agent still produced an answer"


# --- the toolbox in isolation ------------------------------------------------


def test_calculator_rejects_malformed_expressions():
    tools = Toolbox(rng=make_rng(0))
    with pytest.raises(ToolError):
        tools.calculator("152 + ")
    with pytest.raises(ToolError):
        tools.calculator("__import__('os')")
    assert tools.calculator("152 + 144") == 296


def test_fetch_page_404s_on_an_invented_url():
    tools = Toolbox(rng=make_rng(0))
    with pytest.raises(ToolError, match="404"):
        tools.fetch_page("https://weather.example/seattle-climate")


def test_flaky_fetch_raises_a_retryable_error():
    tools = Toolbox(rng=make_rng(0), faults=ALWAYS_FLAKY)
    with pytest.raises(TransientError):
        tools.fetch_page("https://weather.example/seattle-2024")
    # Budget spent; the retry succeeds.
    assert "152 days" in tools.fetch_page("https://weather.example/seattle-2024")


def test_stub_model_is_a_pure_function_at_temperature_zero():
    messages = [{"role": "user", "content": "how many rainy days"}]
    one = StubModel(rng=make_rng(1)).complete(messages, temperature=0.0)
    two = StubModel(rng=make_rng(99)).complete(messages, temperature=0.0)
    assert one.tool_call.arguments == two.tool_call.arguments
