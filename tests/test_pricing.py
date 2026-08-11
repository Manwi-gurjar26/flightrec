"""Tests for pricing and cost rollups."""

import json

import pytest

from flightrec.demo.agent import ResearchAgent
from flightrec.demo.tools import FaultConfig
from flightrec.determinism import SeededIdGenerator, VirtualClock
from flightrec.pricing import (
    FR_PRICE_INPUT,
    FR_PRICE_SOURCE,
    FR_UNPRICED,
    BUILTIN_PRICES,
    ModelPrice,
    PriceTable,
    format_usd,
)
from flightrec.rollup import CostComparison, build_rollup
from flightrec.sinks import MemorySink
from flightrec.spans import FR_COST_USD, Run, SpanKind
from flightrec.tracer import Tracer

CLEAN = FaultConfig()


def make_run(seed: int = 1, faults: FaultConfig | None = None, **kwargs) -> Run:
    return ResearchAgent(
        seed=seed,
        faults=faults or CLEAN,
        sink=MemorySink(),
        clock=VirtualClock(start=1_000.0, step=0.001),
        id_generator=SeededIdGenerator(seed=seed),
        trace_id=f"run-{seed}",
        **kwargs,
    ).run().run


# --- price lookup ------------------------------------------------------------


def test_cost_is_per_million_tokens():
    price = ModelPrice(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0)
    assert price.cost(1_000_000, 0) == pytest.approx(3.0)
    assert price.cost(0, 1_000_000) == pytest.approx(15.0)
    assert price.cost(1_000, 500) == pytest.approx(0.003 + 0.0075)


def test_exact_match_wins():
    table = PriceTable({"a-model": ModelPrice(1.0, 2.0)})
    assert table.lookup("a-model") == ModelPrice(1.0, 2.0)


def test_longest_prefix_match_handles_dated_model_ids():
    table = PriceTable(
        {
            "example-medium": ModelPrice(3.0, 15.0),
            "example": ModelPrice(99.0, 99.0),
        }
    )
    # The dated id must resolve to the more specific entry, not the generic one.
    assert table.lookup("example-medium-20260101").input_usd_per_mtok == 3.0


def test_unknown_model_returns_none_rather_than_zero():
    """Zero would be a confident wrong answer, which is the thing we are against."""
    assert PriceTable().cost_for("no-such-model", 1000, 1000) is None


def test_apply_records_the_rate_used_for_auditability():
    tracer = Tracer(sink=MemorySink(), clock=VirtualClock())
    with tracer.span("chat", kind=SpanKind.LLM) as span:
        tracer.record_usage(span, model="stub-1", input_tokens=1000, output_tokens=100)

    assert span.attributes[FR_COST_USD] == pytest.approx(0.003 + 0.0015)
    assert span.attributes[FR_PRICE_INPUT] == 3.00
    assert span.attributes[FR_PRICE_SOURCE] == "builtin"


def test_recorded_cost_does_not_change_when_the_price_table_does():
    """A recorded run is a historical fact, not a view over today's prices."""
    tracer = Tracer(sink=MemorySink(), clock=VirtualClock())
    with tracer.span("chat", kind=SpanKind.LLM) as span:
        tracer.record_usage(span, model="stub-1", input_tokens=1000, output_tokens=0)
    recorded = span.attributes[FR_COST_USD]

    tracer.prices.prices["stub-1"] = ModelPrice(300.0, 1500.0)  # prices went up
    assert span.attributes[FR_COST_USD] == recorded
    assert build_rollup(tracer.collect()).total_cost_usd == pytest.approx(recorded)


def test_unpriced_span_is_marked_and_not_costed():
    tracer = Tracer(sink=MemorySink(), clock=VirtualClock())
    with tracer.span("chat", kind=SpanKind.LLM) as span:
        tracer.record_usage(span, model="mystery", input_tokens=500, output_tokens=500)

    assert span.attributes[FR_UNPRICED] is True
    assert FR_COST_USD not in span.attributes
    assert span.total_tokens == 1000, "usage is still recorded"


def test_price_table_loads_from_json(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"my-model": {"input": 1.0, "output": 4.0}}))

    table = PriceTable.from_json(path)
    assert table.cost_for("my-model", 1_000_000, 1_000_000) == pytest.approx(5.0)
    assert table.source == str(path)


@pytest.mark.parametrize(
    "amount, expected",
    [(0, "$0"), (0.000162, "$0.000162"), (0.0345, "$0.0345"), (12.5, "$12.50")],
)
def test_money_formatting_does_not_round_small_costs_to_nothing(amount, expected):
    assert format_usd(amount) == expected


# --- rollups -----------------------------------------------------------------


def test_rollup_totals_match_the_run():
    run = make_run(seed=1)
    rollup = build_rollup(run)

    assert rollup.total_tokens == run.total_tokens
    assert rollup.total_cost_usd == pytest.approx(run.total_cost_usd)
    assert rollup.total_cost_usd > 0, "the demo model is priced"
    assert rollup.priced_completely


def test_breakdowns_partition_the_run():
    run = make_run(seed=1)
    rollup = build_rollup(run)

    assert sum(line.calls for line in rollup.by_kind) == len(run.steps())
    assert sum(line.tokens for line in rollup.by_kind) == rollup.total_tokens
    assert sum(line.cost_usd for line in rollup.by_model) == pytest.approx(
        rollup.total_cost_usd
    )
    assert {line.label for line in rollup.by_tool} == {
        "web_search",
        "fetch_page",
        "calculator",
    }


def test_breakdowns_are_sorted_most_expensive_first():
    rollup = build_rollup(make_run(seed=1))
    costs = [line.cost_usd for line in rollup.by_kind]
    assert costs == sorted(costs, reverse=True)


def test_tools_are_not_hidden_by_a_zero_cost_tie():
    """Tools cost no tokens; they must still sort by the time they ate."""
    rollup = build_rollup(make_run(seed=1))
    durations = [line.duration_ms for line in rollup.by_tool]
    assert durations == sorted(durations, reverse=True)


def test_unpriced_calls_make_the_total_visibly_incomplete():
    tracer = Tracer(sink=MemorySink(), clock=VirtualClock())
    with tracer.span("agent", kind=SpanKind.AGENT):
        with tracer.span("chat", kind=SpanKind.LLM) as a:
            tracer.record_usage(a, model="stub-1", input_tokens=100, output_tokens=10)
        with tracer.span("chat", kind=SpanKind.LLM) as b:
            tracer.record_usage(b, model="mystery", input_tokens=900, output_tokens=90)

    rollup = build_rollup(tracer.collect())
    assert rollup.priced_completely is False
    assert rollup.unpriced_calls == 1
    assert rollup.unpriced_tokens == 990


def test_errors_and_retries_are_counted():
    run = make_run(seed=4, faults=FaultConfig(flaky_fetch_rate=1.0))
    rollup = build_rollup(run)

    assert rollup.retry_count > 0
    assert rollup.retry_delay_ms > 0


def test_post_failure_spend_is_attributed():
    run = make_run(seed=2, faults=FaultConfig(empty_search_rate=1.0))
    rollup = build_rollup(run)

    assert rollup.error_count > 0
    assert rollup.post_failure_calls > 0
    assert rollup.post_failure_cost_usd > 0
    assert 0 < rollup.post_failure_share <= 100


def test_a_clean_run_attributes_nothing_to_failure():
    rollup = build_rollup(make_run(seed=1))
    assert rollup.error_count == 0
    assert rollup.post_failure_calls == 0
    assert rollup.post_failure_share == 0.0


def test_rollup_of_an_empty_run_does_not_divide_by_zero():
    rollup = build_rollup(Run(run_id="empty"))
    assert rollup.total_tokens == 0
    assert rollup.post_failure_share == 0.0
    assert rollup.priced_completely


# --- comparison --------------------------------------------------------------


def test_comparison_explains_which_component_moved():
    """The 'why did this run cost more?' question, answered per component."""
    cheap = build_rollup(make_run(seed=1))
    expensive = build_rollup(make_run(seed=2, faults=FaultConfig(empty_search_rate=1.0)))
    comparison = CostComparison(baseline=cheap, candidate=expensive)

    assert comparison.token_delta != 0
    movers = comparison.by_model_delta()
    assert movers, "a cost difference must be attributable to something"
    assert movers[0][0] == "stub-1"


def test_comparison_ratio_handles_a_free_baseline():
    zero = build_rollup(Run(run_id="empty"))
    real = build_rollup(make_run(seed=1))
    assert CostComparison(baseline=zero, candidate=real).cost_ratio is None


def test_comparing_a_run_with_itself_shows_no_movement():
    rollup = build_rollup(make_run(seed=1))
    comparison = CostComparison(baseline=rollup, candidate=rollup)

    assert comparison.token_delta == 0
    assert comparison.cost_delta == 0
    assert comparison.cost_ratio == 1.0
    assert all(cost == 0 and tokens == 0 for _, cost, tokens in comparison.by_model_delta())
