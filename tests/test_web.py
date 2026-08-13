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
from flightrec.spans import FR_OUTPUT, GEN_AI_TOOL_NAME, Run, Span, SpanKind, SpanStatus
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


def test_a_replay_cannot_be_mistaken_for_a_recording(client, store):
    """The rule the CLI has enforced since replay landed, now on the page people read.

    A replayed run rendered exactly like an original one: same steps, same
    everything, no indication that half of them were re-executed live minutes
    ago. That is the confusion this project exists to prevent, sitting in the
    one view a human actually looks at.
    """
    from flightrec.replay import replay_run

    original = make_run(seed=1)
    replay = replay_run(original, from_step=4).run
    store.add_run(original)
    store.add_run(replay)

    recorded_page = client.get(f"/runs/{original.run_id}").text
    replay_page = client.get(f"/runs/{replay.run_id}").text

    assert "This is a replay, not a recording" in replay_page
    assert ">live<" in replay_page
    assert ">recorded<" in replay_page

    assert "This is a replay" not in recorded_page
    assert ">live<" not in recorded_page


def test_a_replayed_step_says_which_kind_it_is(store):
    """Per step, not just a banner: the banner is a footnote once you scroll."""
    from flightrec.replay import replay_run
    from flightrec.web import build_timeline

    view = build_timeline(replay_run(make_run(seed=1), from_step=4).run)

    assert view.is_replay
    assert view.recorded_count and view.live_count
    assert all(step.provenance in ("recorded", "live", "stopped") for step in view.steps)


def test_an_ordinary_run_carries_no_provenance_noise(store):
    """A recording is not a replay of anything, and must not imply it is."""
    from flightrec.web import build_timeline

    view = build_timeline(make_run(seed=1))

    assert not view.is_replay
    assert not any(step.provenance for step in view.steps)


# --- replay from the browser --------------------------------------------------


@pytest.fixture
def no_redirect(store):
    return TestClient(create_app(store=store), follow_redirects=False)


def test_replaying_from_the_browser_stores_a_run_and_redirects(no_redirect, store):
    run = make_run(seed=1)
    store.add_run(run)

    response = no_redirect.post(f"/runs/{run.run_id}/replay", data={"from_step": "4"})

    assert response.status_code == 303, "POST-redirect-GET, so refresh cannot re-run it"
    location = response.headers["location"]
    assert location != f"/runs/{run.run_id}", "a replay is its own run"

    replay = store.get_run(location.rsplit("/", 1)[-1])
    assert replay is not None
    assert no_redirect.get(location).status_code == 200


def test_a_browser_replay_is_labelled_like_any_other(no_redirect, store):
    """It has to arrive marked, or the UI has a second way to be misled."""
    run = make_run(seed=1)
    store.add_run(run)

    location = no_redirect.post(
        f"/runs/{run.run_id}/replay", data={"from_step": "4"}
    ).headers["location"]
    page = no_redirect.get(location).text

    assert "This is a replay, not a recording" in page
    assert ">recorded<" in page and ">live<" in page
    assert any(s.is_replay for s in store.list_runs())


def test_a_run_the_engine_cannot_rebuild_is_not_offered_a_replay(no_redirect, store):
    """The engine only knows the demo agent, so the button must not lie.

    A run ingested from somebody else's agent has no seed to reconstruct from.
    Replaying it anyway would build the wrong program, serve it this recording,
    and store something that looks like a replay of nothing.
    """
    tracer = Tracer(sink=MemorySink(), clock=VirtualClock(), trace_id="foreign")
    with tracer.span("my_own_agent", kind=SpanKind.AGENT):
        with tracer.span("chat", kind=SpanKind.LLM):
            pass
    store.add_run(tracer.collect())

    page = no_redirect.get("/runs/foreign").text
    assert 'method="post"' not in page
    assert "cannot be replayed" in page

    # And the guard holds against a hand-made POST, not just a hidden button.
    before = store.count_runs()
    assert no_redirect.post("/runs/foreign/replay", data={}).status_code == 400
    assert store.count_runs() == before


def test_a_bad_replay_form_explains_itself(no_redirect, store):
    run = make_run(seed=1)
    store.add_run(run)

    response = no_redirect.post(f"/runs/{run.run_id}/replay", data={"from_step": "abc"})

    assert response.status_code == 400
    assert "have to be numbers" in response.text
    assert store.count_runs() == 1, "a rejected form must not store anything"


def test_replaying_a_missing_run_is_404(no_redirect):
    assert no_redirect.post("/runs/nope/replay", data={}).status_code == 404


def test_the_replay_form_needs_no_javascript(client, store):
    store.add_run(make_run(seed=1))
    html = client.get("/runs/run-1").text

    assert 'action="/runs/run-1/replay"' in html
    assert 'method="post"' in html
    assert "<script" not in html


def test_diff_page_aligns_two_runs(client, store):
    """The diff was command-line only for two build steps.

    It is the feature that gained the most work and the one a person was least
    able to see, which is a poor combination for a tool whose whole argument is
    that the readable version is the point.
    """
    left = make_run(seed=1)
    right = make_run(seed=4)
    store.add_run(left)
    store.add_run(right)

    response = client.get("/diff", params={"left": left.run_id, "right": right.run_id})

    assert response.status_code == 200
    assert "First divergence" in response.text
    assert left.run_id[:12] in response.text
    assert right.run_id[:12] in response.text


def test_diff_page_says_when_two_runs_are_the_same(client, store):
    run = make_run(seed=1)
    store.add_run(run)

    response = client.get("/diff", params={"left": run.run_id, "right": run.run_id})

    assert "identical" in response.text
    assert "diverged" not in response.text


def test_diff_page_distinguishes_changed_from_replaced(store):
    """The two words the diff learned to tell apart must survive into the UI.

    Collapsing them in the template would undo the distinction at the last step,
    where it is the only thing a reader actually sees.
    """
    from flightrec.web import build_diff

    left = make_run(seed=1)
    right = make_run(seed=1)
    step = right.steps()[3]
    step.name = "tool.calculator"
    step.attributes[GEN_AI_TOOL_NAME] = "calculator"

    rows = {row.op for row in build_diff(left, right).rows}

    assert "replaced" in rows
    assert "a different step entirely" in [
        row.label for row in build_diff(left, right).rows
    ]


def test_a_reordered_step_is_not_rendered_as_quiet_background(client, store):
    """A moved step's op is ``match`` -- its content is identical.

    Styling the table by op alone would paint a reordering as unchanged
    background, which is the one thing it is not.
    """
    import random

    from flightrec.bench import mutate, recovery_block

    mutation = mutate(make_run(seed=1), recovery_block(), random.Random(2), kind="reorder")
    assert mutation is not None
    for run, name in ((mutation.original, "l"), (mutation.mutant, "r")):
        for span in run.spans:
            span.trace_id, span.span_id = name, f"{name}-{span.sequence}"
        store.add_run(run)

    html = client.get("/diff", params={"left": "l", "right": "r"}).text

    assert "d-moved" in html
    assert "d-moved d-quiet" not in html


def test_diff_page_for_a_missing_run_is_404(client, store):
    run = make_run(seed=1)
    store.add_run(run)

    assert client.get("/diff", params={"left": run.run_id, "right": "nope"}).status_code == 404
    assert client.get("/diff", params={"left": "nope", "right": run.run_id}).status_code == 404


def test_the_run_list_offers_a_comparison_without_javascript(client, store):
    """A plain GET form is the whole mechanism, because there is no JS to lean on."""
    store.add_run(make_run(seed=1))
    store.add_run(make_run(seed=4))

    html = client.get("/").text

    assert 'action="/diff"' in html
    assert 'method="get"' in html
    assert "<script" not in html


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
