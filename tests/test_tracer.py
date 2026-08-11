import pytest

from flightrec.determinism import SeededIdGenerator, VirtualClock
from flightrec.sinks import JSONLSink, MemorySink, read_jsonl
from flightrec.spans import (
    FR_INPUT,
    FR_OUTPUT,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    Run,
    SpanKind,
    SpanStatus,
)
from flightrec.tracer import Tracer


def make_tracer(sink=None):
    """A fully pinned tracer: virtual clock, seeded IDs, fixed trace id."""
    return Tracer(
        sink=sink or MemorySink(),
        clock=VirtualClock(start=1_000.0, step=0.5),
        id_generator=SeededIdGenerator(seed=42),
        trace_id="test-trace",
    )


# --- basic span recording ----------------------------------------------------


def test_span_records_timing_and_status():
    tracer = make_tracer()
    with tracer.span("call_model", kind=SpanKind.LLM):
        pass

    run = tracer.collect()
    assert len(run.spans) == 1
    span = run.spans[0]
    assert span.name == "call_model"
    assert span.kind is SpanKind.LLM
    assert span.status is SpanStatus.OK
    assert span.start_time == 1_000.0
    assert span.end_time == 1_000.5
    assert span.duration_ms == pytest.approx(500.0)


def test_nesting_sets_parent_ids():
    tracer = make_tracer()
    with tracer.span("agent", kind=SpanKind.AGENT) as root:
        with tracer.span("think", kind=SpanKind.LLM) as child:
            with tracer.span("search", kind=SpanKind.TOOL) as grandchild:
                pass

    assert root.parent_span_id is None
    assert child.parent_span_id == root.span_id
    assert grandchild.parent_span_id == child.span_id


def test_sibling_spans_share_a_parent():
    tracer = make_tracer()
    with tracer.span("agent", kind=SpanKind.AGENT) as root:
        with tracer.span("a", kind=SpanKind.TOOL) as a:
            pass
        with tracer.span("b", kind=SpanKind.TOOL) as b:
            pass

    assert a.parent_span_id == root.span_id
    assert b.parent_span_id == root.span_id
    assert a.span_id != b.span_id


# --- errors ------------------------------------------------------------------


def test_exception_is_recorded_and_re_raised():
    tracer = make_tracer()

    with pytest.raises(ValueError, match="boom"):
        with tracer.span("failing_tool", kind=SpanKind.TOOL):
            raise ValueError("boom")

    span = tracer.collect().spans[0]
    assert span.status is SpanStatus.ERROR
    assert "boom" in span.status_message
    assert span.end_time is not None, "failed spans must still be closed"
    assert [e.name for e in span.events] == ["exception"]
    assert span.events[0].attributes["exception.type"] == "ValueError"


def test_error_in_child_does_not_mark_parent_ok_incorrectly():
    tracer = make_tracer()
    with tracer.span("agent", kind=SpanKind.AGENT) as root:
        try:
            with tracer.span("tool", kind=SpanKind.TOOL):
                raise RuntimeError("inner")
        except RuntimeError:
            pass  # the agent caught it and carried on, as agents do

    run = tracer.collect()
    assert root.status is SpanStatus.OK
    assert run.has_error is True, "a caught inner failure is still a recorded failure"


# --- decorator ---------------------------------------------------------------


def test_trace_decorator_captures_inputs_and_output():
    tracer = make_tracer()

    @tracer.trace(kind=SpanKind.TOOL)
    def search(query: str, limit: int = 3):
        return [f"result for {query}"] * limit

    assert search("pdfs", limit=2) == ["result for pdfs"] * 2

    span = tracer.collect().spans[0]
    assert span.name == "search"
    assert span.kind is SpanKind.TOOL
    assert span.attributes[FR_INPUT] == {"args": ["pdfs"], "kwargs": {"limit": 2}}
    assert span.attributes[FR_OUTPUT] == ["result for pdfs", "result for pdfs"]


def test_decorator_preserves_function_metadata():
    tracer = make_tracer()

    @tracer.trace()
    def documented():
        """A docstring that must survive instrumentation."""

    assert documented.__name__ == "documented"
    assert "must survive" in documented.__doc__


def test_unserialisable_arguments_do_not_break_instrumentation():
    tracer = make_tracer()

    class Opaque:
        def __repr__(self):
            return "<Opaque>"

    @tracer.trace(kind=SpanKind.TOOL)
    def takes_object(obj):
        return obj

    takes_object(Opaque())
    span = tracer.collect().spans[0]
    assert span.attributes[FR_INPUT]["args"] == ["<Opaque>"]


# --- run tree and steps ------------------------------------------------------


def build_agent_run() -> Run:
    tracer = make_tracer()
    with tracer.span("research_agent", kind=SpanKind.AGENT):
        for i in range(2):
            with tracer.span(f"step_{i}", kind=SpanKind.STEP):
                with tracer.span("chat", kind=SpanKind.LLM) as llm:
                    llm.attributes[GEN_AI_USAGE_INPUT_TOKENS] = 100
                    llm.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = 20
                with tracer.span("web_search", kind=SpanKind.TOOL):
                    pass
    return tracer.collect()


def test_tree_reconstructs_the_hierarchy():
    run = build_agent_run()
    roots = run.tree()

    assert len(roots) == 1
    agent = roots[0]
    assert agent.span.name == "research_agent"
    assert [c.span.name for c in agent.children] == ["step_0", "step_1"]
    assert [c.span.name for c in agent.children[0].children] == ["chat", "web_search"]


def test_tree_promotes_orphans_instead_of_dropping_them():
    run = build_agent_run()
    # Simulate the collector losing the root span, which happens when a process
    # is killed before the outermost span closes.
    run.spans = [s for s in run.spans if s.name != "research_agent"]

    roots = run.tree()
    assert {r.span.name for r in roots} == {"step_0", "step_1"}
    total = sum(1 + len(r.children) for r in roots)
    assert total == len(run.spans), "no span may be silently dropped"


def test_steps_are_only_llm_and_tool_calls_in_order():
    run = build_agent_run()
    steps = run.steps()

    assert [s.name for s in steps] == ["chat", "web_search", "chat", "web_search"]
    starts = [s.start_time for s in steps]
    assert starts == sorted(starts)


def test_token_and_cost_rollups():
    run = build_agent_run()
    assert run.total_tokens == 240  # 2 model calls x (100 + 20)
    assert run.total_cost_usd == 0.0  # no cost table wired up yet


# --- determinism seams -------------------------------------------------------


def test_two_pinned_runs_are_byte_identical():
    """The core promise, tested at the SDK level before replay even exists."""
    a = build_agent_run().model_dump_json()
    b = build_agent_run().model_dump_json()
    assert a == b


def test_unpinned_runs_differ():
    """Guards the test above from passing for the wrong reason."""
    one = Tracer(sink=MemorySink()).trace_id
    two = Tracer(sink=MemorySink()).trace_id
    assert one != two


# --- sinks -------------------------------------------------------------------


def test_jsonl_sink_round_trips(tmp_path):
    path = tmp_path / "run.jsonl"
    tracer = make_tracer(sink=JSONLSink(path))
    with tracer.span("agent", kind=SpanKind.AGENT):
        with tracer.span("tool", kind=SpanKind.TOOL):
            pass
    tracer.sink.close()

    spans = read_jsonl(path)
    assert [s.name for s in spans] == ["tool", "agent"]  # children close first
    assert spans[0].kind is SpanKind.TOOL
    assert spans[1].parent_span_id is None


def test_jsonl_sink_survives_a_truncated_final_line(tmp_path):
    path = tmp_path / "crashed.jsonl"
    tracer = make_tracer(sink=JSONLSink(path))
    with tracer.span("tool", kind=SpanKind.TOOL):
        pass
    tracer.sink.close()

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"span_id": "half-writ')  # process died mid-emit

    spans = read_jsonl(path)
    assert len(spans) == 1, "a crashed run must still yield its complete spans"
