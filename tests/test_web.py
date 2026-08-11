"""Tests for the timeline view models and the HTML pages.

View-model assertions go against :mod:`flightrec.web`, not against parsed HTML.
Tests that scrape markup break every time the layout changes and tell you
nothing about whether the numbers were right.
"""

import pytest
from fastapi.testclient import TestClient

from flightrec.collector import create_app
from flightrec.demo.agent import ResearchAgent
from flightrec.demo.tools import FaultConfig
from flightrec.determinism import SeededIdGenerator, VirtualClock
from flightrec.sinks import MemorySink
from flightrec.spans import FR_OUTPUT, Run, Span, SpanKind, SpanStatus
from flightrec.storage import RunStore
from flightrec.tracer import Tracer
from flightrec.web import build_timeline, format_duration


@pytest.fixture
def store(tmp_path):
    s = RunStore(tmp_path / "web.db")
    yield s
    s.close()


@pytest.fixture
def client(store):
    return TestClient(create_app(store=store))


def make_run(seed: int = 1, faults: FaultConfig | None = None) -> Run:
    return ResearchAgent(
        seed=seed,
        faults=faults or FaultConfig(),
        sink=MemorySink(),
        clock=VirtualClock(start=1_000.0, step=0.001),
        id_generator=SeededIdGenerator(seed=seed),
        trace_id=f"run-{seed}",
    ).run().run


# --- view model --------------------------------------------------------------


def test_timeline_summarises_the_run():
    run = make_run(seed=1)
    view = build_timeline(run)

    assert view.run_id == run.run_id
    assert view.step_count == len(run.steps())
    assert view.total_tokens == run.total_tokens
    assert view.error_count == 0
    assert view.complete is True
    assert view.status == "ok"
    assert view.root_name == "research_agent"


def test_steps_are_in_execution_order():
    view = build_timeline(make_run(seed=1))
    assert [s.sequence for s in view.steps] == sorted(s.sequence for s in view.steps)
    assert view.steps[0].name == "chat"
    assert view.steps[1].name == "tool.web_search"


def test_bars_are_scaled_to_the_largest_value_in_this_run():
    """Relative bars, not absolute: equal steps must show equal bars."""
    view = build_timeline(make_run(seed=1))
    token_pcts = [s.token_pct for s in view.steps if s.tokens]

    assert max(token_pcts) == 100.0, "the largest step defines full width"
    assert all(0.0 <= p <= 100.0 for p in token_pcts)


def test_bar_scaling_survives_a_run_where_nothing_used_tokens():
    """Guards a division by zero on tool-only runs."""
    tracer = Tracer(sink=MemorySink(), clock=VirtualClock(), trace_id="t")
    with tracer.span("agent", kind=SpanKind.AGENT):
        with tracer.span("tool.noop", kind=SpanKind.TOOL):
            pass

    view = build_timeline(tracer.collect())
    assert view.total_tokens == 0
    assert view.steps[0].token_pct == 0.0


def test_failed_step_summary_shows_the_error_not_an_empty_output():
    run = make_run(seed=2, faults=FaultConfig(empty_search_rate=1.0))
    view = build_timeline(run)

    failed = [s for s in view.steps if s.is_error]
    assert failed, "the fixture must contain a failure"
    for step in failed:
        assert "404" in step.summary, "the collapsed row must say where to look"
        assert step.events, "the exception detail must be available on expand"


def test_retries_are_counted_on_the_step_and_the_run():
    run = make_run(seed=4, faults=FaultConfig(flaky_fetch_rate=1.0))
    view = build_timeline(run)

    assert view.retry_count > 0
    assert sum(s.retry_count for s in view.steps) == view.retry_count


def test_confabulated_step_is_flagged():
    run = make_run(seed=2, faults=FaultConfig(empty_search_rate=1.0))
    view = build_timeline(run)
    assert any(s.confabulated for s in view.steps)


def test_partial_run_is_reported_as_partial():
    run = make_run(seed=1)
    run.spans = [s for s in run.spans if s.parent_span_id is not None]

    view = build_timeline(run)
    assert view.complete is False
    assert view.status == "partial"


def test_promoted_attributes_are_not_repeated_in_the_attribute_table():
    view = build_timeline(make_run(seed=1))
    step = view.steps[0]
    keys = [k for k, _ in step.detail_attributes]

    assert "flightrec.input" not in keys
    assert "flightrec.output" not in keys
    assert "gen_ai.request.model" in keys, "everything else is still shown"


@pytest.mark.parametrize(
    "ms, expected",
    [(None, "-"), (0.4, "400us"), (12.5, "12.5ms"), (2500.0, "2.50s")],
)
def test_duration_formatting(ms, expected):
    assert format_duration(ms) == expected


# --- HTML pages --------------------------------------------------------------


def test_run_list_page_renders(client, store):
    store.add_run(make_run(seed=1))
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "research_agent" in response.text
    assert "run-1"[:12] in response.text


def test_run_list_is_helpful_when_empty(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "No runs recorded yet" in response.text
    assert "flightrec demo" in response.text, "an empty state should say what to do"


def test_timeline_page_renders_every_step(client, store):
    run = make_run(seed=1)
    store.add_run(run)
    response = client.get(f"/runs/{run.run_id}")

    assert response.status_code == 200
    for step in run.steps():
        assert step.name in response.text
    assert "tool.calculator" in response.text


def test_timeline_page_marks_errors(client, store):
    run = make_run(seed=2, faults=FaultConfig(empty_search_rate=1.0))
    store.add_run(run)
    response = client.get(f"/runs/{run.run_id}")

    assert response.status_code == 200
    assert "404" in response.text
    assert 'class="pill error"' in response.text


def test_timeline_page_for_a_missing_run_is_404(client):
    assert client.get("/runs/nope").status_code == 404


def test_page_is_self_contained_with_no_external_requests(client, store):
    """The collector must work offline. No CDN scripts, no remote fonts."""
    store.add_run(make_run(seed=1))
    html = client.get(f"/runs/run-1").text

    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "https://cdn" not in html
    assert "<script" not in html, "the timeline needs no JavaScript at all"


def test_tool_output_is_escaped(store, client):
    """A tool that returns HTML must not be able to inject it into the page."""
    tracer = Tracer(sink=MemorySink(), clock=VirtualClock(), trace_id="xss")
    payload = "<img src=x onerror=alert(1)>"
    with tracer.span("agent", kind=SpanKind.AGENT):
        with tracer.span("tool.evil", kind=SpanKind.TOOL) as span:
            span.attributes[FR_OUTPUT] = payload

    store.add_run(tracer.collect())
    html = client.get("/runs/xss").text

    assert payload not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
