"""Tests for storage and the collector HTTP API."""

import pytest
from fastapi.testclient import TestClient

from flightrec.collector import create_app
from flightrec.demo.agent import ResearchAgent
from flightrec.demo.tools import FaultConfig
from flightrec.determinism import SeededIdGenerator, VirtualClock
from flightrec.sinks import HTTPSink, MemorySink, TeeSink
from flightrec.spans import Run, Span, SpanKind, SpanStatus
from flightrec.storage import RunStore


@pytest.fixture
def store(tmp_path):
    s = RunStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def client(store):
    return TestClient(create_app(store=store))


def make_run(seed: int = 1, faults: FaultConfig | None = None) -> Run:
    agent = ResearchAgent(
        seed=seed,
        faults=faults or FaultConfig(),
        sink=MemorySink(),
        clock=VirtualClock(start=1_000.0, step=0.001),
        id_generator=SeededIdGenerator(seed=seed),
        trace_id=f"run-{seed}",
    )
    return agent.run().run


# --- storage -----------------------------------------------------------------


def test_run_round_trips_through_sqlite(store):
    original = make_run(seed=1)
    store.add_run(original)

    loaded = store.get_run(original.run_id)
    assert loaded is not None
    assert len(loaded.spans) == len(original.spans)
    assert [s.name for s in loaded.steps()] == [s.name for s in original.steps()]
    assert loaded.total_tokens == original.total_tokens
    # Compared canonically: the store returns spans ordered by sequence, while
    # the in-memory run holds them in emission order (children close first).
    assert loaded.canonical_json() == original.canonical_json()


def test_tree_survives_the_round_trip(store):
    original = make_run(seed=1)
    store.add_run(original)
    loaded = store.get_run(original.run_id)

    before, after = original.tree(), loaded.tree()
    assert len(after) == 1
    assert after[0].span.name == before[0].span.name
    assert [c.span.name for c in after[0].children] == [
        c.span.name for c in before[0].children
    ]


def test_errors_and_events_survive_the_round_trip(store):
    run = make_run(seed=2, faults=FaultConfig(empty_search_rate=1.0))
    store.add_run(run)
    loaded = store.get_run(run.run_id)

    failed = [s for s in loaded.spans if s.status is SpanStatus.ERROR]
    assert failed, "the fixture must contain a failure"
    assert all("404" in (s.status_message or "") for s in failed)
    assert [e.name for e in failed[0].events] == ["exception"]
    assert failed[0].events[0].attributes["exception.type"] == "ToolError"


def test_ingesting_the_same_spans_twice_is_idempotent(store):
    """An SDK retry must not be able to double a run's token count."""
    run = make_run(seed=1)
    store.add_spans(run.spans)
    store.add_spans(run.spans)

    loaded = store.get_run(run.run_id)
    assert len(loaded.spans) == len(run.spans)
    assert loaded.total_tokens == run.total_tokens
    assert store.count_runs() == 1


def test_a_partial_run_is_stored_and_reported_as_partial(store):
    """The run whose process died is the one you most want to look at."""
    run = make_run(seed=1)
    without_root = [s for s in run.spans if s.parent_span_id is not None]
    store.add_spans(without_root)

    summary = store.list_runs()[0]
    assert summary.complete is False
    assert summary.status == "partial"
    assert summary.span_count == len(without_root)

    loaded = store.get_run(run.run_id)
    assert len(loaded.spans) == len(without_root), "no span may be dropped"


def test_summary_counts_match_the_run(store):
    run = make_run(seed=2, faults=FaultConfig(empty_search_rate=1.0))
    store.add_run(run)

    summary = store.list_runs()[0]
    assert summary.span_count == len(run.spans)
    assert summary.step_count == len(run.steps())
    assert summary.error_count == sum(1 for s in run.spans if s.is_error)
    assert summary.total_tokens == run.total_tokens
    assert summary.status == "error"
    assert summary.root_name == "research_agent"


def test_list_runs_is_newest_first_and_paginates(store):
    for seed in range(5):
        store.add_run(make_run(seed=seed))

    assert store.count_runs() == 5
    all_runs = store.list_runs(limit=10)
    assert len(all_runs) == 5
    assert all_runs == sorted(all_runs, key=lambda s: s.created_at, reverse=True)

    page = store.list_runs(limit=2, offset=2)
    assert [s.run_id for s in page] == [s.run_id for s in all_runs[2:4]]


def test_get_missing_run_returns_none(store):
    assert store.get_run("nope") is None


def test_delete_removes_spans_and_the_run(store):
    run = make_run(seed=1)
    store.add_run(run)
    store.delete_run(run.run_id)

    assert store.get_run(run.run_id) is None
    assert store.count_runs() == 0


def test_store_reopens_an_existing_database(tmp_path):
    path = tmp_path / "persist.db"
    first = RunStore(path)
    run = make_run(seed=1)
    first.add_run(run)
    first.close()

    second = RunStore(path)
    assert second.get_run(run.run_id) is not None
    second.close()


# --- the HTTP API ------------------------------------------------------------


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "runs": 0}


def test_ingest_then_fetch_over_http(client):
    run = make_run(seed=1)
    payload = {"spans": [s.model_dump(mode="json") for s in run.spans]}

    response = client.post("/v1/spans", json=payload)
    assert response.status_code == 200
    assert response.json()["accepted"] == len(run.spans)

    fetched = client.get(f"/v1/runs/{run.run_id}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert len(body["spans"]) == len(run.spans)
    assert body["run_id"] == run.run_id


def test_run_list_endpoint(client):
    run = make_run(seed=1)
    client.post("/v1/spans", json={"spans": [s.model_dump(mode="json") for s in run.spans]})

    response = client.get("/v1/runs")
    assert response.status_code == 200
    summaries = response.json()
    assert len(summaries) == 1
    assert summaries[0]["run_id"] == run.run_id
    assert summaries[0]["step_count"] == len(run.steps())


def test_tree_endpoint_returns_nested_spans(client):
    run = make_run(seed=1)
    client.post("/v1/spans", json={"spans": [s.model_dump(mode="json") for s in run.spans]})

    response = client.get(f"/v1/runs/{run.run_id}/tree")
    assert response.status_code == 200
    tree = response.json()
    assert len(tree) == 1
    assert tree[0]["span"]["name"] == "research_agent"
    assert len(tree[0]["children"]) == len(run.spans) - 1


def test_missing_run_is_404(client):
    assert client.get("/v1/runs/nope").status_code == 404
    assert client.get("/v1/runs/nope/tree").status_code == 404
    assert client.delete("/v1/runs/nope").status_code == 404


def test_empty_batch_is_accepted(client):
    """The SDK flushes on close even with nothing buffered; that is not an error."""
    response = client.post("/v1/spans", json={"spans": []})
    assert response.status_code == 200
    assert response.json()["accepted"] == 0


def test_delete_over_http(client):
    run = make_run(seed=1)
    client.post("/v1/spans", json={"spans": [s.model_dump(mode="json") for s in run.spans]})
    assert client.delete(f"/v1/runs/{run.run_id}").status_code == 200
    assert client.get(f"/v1/runs/{run.run_id}").status_code == 404


# --- the HTTP sink -----------------------------------------------------------


def test_http_sink_never_raises_when_the_collector_is_down():
    """The rule that makes this sink safe to leave switched on in production."""
    sink = HTTPSink(url="http://127.0.0.1:9", batch_size=2, timeout=0.05)
    agent = ResearchAgent(seed=1, faults=FaultConfig(), sink=TeeSink(MemorySink(), sink))

    result = agent.run()  # must not raise

    sink.close()
    assert result.answer == 296, "the agent completes normally with a dead backend"
    assert sink.dropped > 0
    assert sink.sent == 0


def test_http_sink_batches_and_flushes_on_close(client, store):
    """Drive the real sink against the real app via a patched transport."""
    import httpx

    sink = HTTPSink(url="http://testserver", batch_size=4)
    real_post = httpx.post

    def routed(url, **kwargs):
        return client.post(url.replace("http://testserver", ""), **kwargs)

    httpx.post = routed
    try:
        run = make_run(seed=1)
        for span in run.spans:
            sink.emit(span)
        sink.close()
    finally:
        httpx.post = real_post

    assert sink.dropped == 0
    assert sink.sent == len(run.spans)
    assert store.get_run(run.run_id) is not None
    assert len(store.get_run(run.run_id).spans) == len(run.spans)
