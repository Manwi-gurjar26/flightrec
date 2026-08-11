"""Aggregating a run's tokens, cost and time into something you can act on.

A single total tells you a run was expensive. It does not tell you *why*, which
is the only reason anyone opens a cost view. So the run is broken down three
ways -- by span kind, by model, by tool -- and then the parts of the spend that
are plausibly waste are separated out and labelled honestly.

The honesty matters. "Wasted spend" cannot be computed exactly: nobody can say
what the run would have cost had the tool not failed, because the agent would
have taken a different path. So this module reports two things it *can* stand
behind -- spend on steps that failed, and spend that occurred after the first
failure -- and names the second one an upper bound rather than pretending it is
a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flightrec.pricing import FR_UNPRICED
from flightrec.spans import GEN_AI_TOOL_NAME, GEN_AI_REQUEST_MODEL, Run, SpanKind


@dataclass
class CostLine:
    """One row of a breakdown."""

    label: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    errors: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def share_of(self, total: float) -> float:
        return round((self.cost_usd / total) * 100.0, 1) if total else 0.0

    def token_share_of(self, total: int) -> float:
        return round((self.tokens / total) * 100.0, 1) if total else 0.0


@dataclass
class RunCost:
    """A run's spend, broken down and partly attributed."""

    run_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0

    by_kind: list[CostLine] = field(default_factory=list)
    by_model: list[CostLine] = field(default_factory=list)
    by_tool: list[CostLine] = field(default_factory=list)

    unpriced_calls: int = 0
    unpriced_tokens: int = 0

    error_count: int = 0
    error_cost_usd: float = 0.0
    error_duration_ms: float = 0.0

    retry_count: int = 0
    retry_delay_ms: float = 0.0

    post_failure_calls: int = 0
    post_failure_tokens: int = 0
    post_failure_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def priced_completely(self) -> bool:
        """False when any call used a model missing from the price table.

        The UI must say so rather than showing a total that looks authoritative
        and silently excludes calls.
        """
        return self.unpriced_calls == 0

    @property
    def post_failure_share(self) -> float:
        """Upper bound on the share of spend attributable to going wrong.

        An upper bound, not an estimate: everything after the first failure is
        counted, and some of it would have been spent anyway.
        """
        if not self.total_cost_usd:
            return 0.0
        return round((self.post_failure_cost_usd / self.total_cost_usd) * 100.0, 1)


def build_rollup(run: Run) -> RunCost:
    rollup = RunCost(run_id=run.run_id)

    kinds: dict[str, CostLine] = {}
    models: dict[str, CostLine] = {}
    tools: dict[str, CostLine] = {}

    ordered = run.ordered_spans()
    first_error_sequence = next(
        (s.sequence for s in ordered if s.is_error), None
    )

    for span in ordered:
        duration = span.duration_ms or 0.0
        rollup.total_input_tokens += span.input_tokens
        rollup.total_output_tokens += span.output_tokens
        rollup.total_cost_usd += span.cost_usd

        if span.kind in (SpanKind.LLM, SpanKind.TOOL):
            rollup.total_duration_ms += duration
            _accumulate(kinds, span.kind.value, span, duration)

        if span.kind is SpanKind.LLM:
            model = str(span.attr(GEN_AI_REQUEST_MODEL) or "unknown")
            _accumulate(models, model, span, duration)
            if span.attr(FR_UNPRICED):
                rollup.unpriced_calls += 1
                rollup.unpriced_tokens += span.total_tokens

        if span.kind is SpanKind.TOOL:
            tool = str(span.attr(GEN_AI_TOOL_NAME) or span.name)
            _accumulate(tools, tool, span, duration)

        if span.is_error:
            rollup.error_count += 1
            rollup.error_cost_usd += span.cost_usd
            rollup.error_duration_ms += duration

        for event in span.events:
            if event.name == "retry":
                rollup.retry_count += 1
                rollup.retry_delay_ms += (
                    float(event.attributes.get("retry.delay_s", 0.0)) * 1000.0
                )

        if (
            first_error_sequence is not None
            and span.sequence > first_error_sequence
            and span.kind is SpanKind.LLM
        ):
            rollup.post_failure_calls += 1
            rollup.post_failure_tokens += span.total_tokens
            rollup.post_failure_cost_usd += span.cost_usd

    rollup.by_kind = _sorted(kinds)
    rollup.by_model = _sorted(models)
    rollup.by_tool = _sorted(tools)
    return rollup


def _accumulate(
    bucket: dict[str, CostLine], label: str, span, duration: float
) -> None:
    line = bucket.setdefault(label, CostLine(label=label))
    line.calls += 1
    line.input_tokens += span.input_tokens
    line.output_tokens += span.output_tokens
    line.cost_usd += span.cost_usd
    line.duration_ms += duration
    if span.is_error:
        line.errors += 1


def _sorted(bucket: dict[str, CostLine]) -> list[CostLine]:
    """Most expensive first, falling back to tokens then time.

    Cost alone would sort every tool to the bottom in a tie at zero, hiding the
    tool that ate thirty seconds of wall clock.
    """
    return sorted(
        bucket.values(),
        key=lambda line: (line.cost_usd, line.tokens, line.duration_ms),
        reverse=True,
    )


@dataclass
class CostComparison:
    """Two runs side by side. The 'why did this cost 8x more?' view."""

    baseline: RunCost
    candidate: RunCost

    @property
    def token_delta(self) -> int:
        return self.candidate.total_tokens - self.baseline.total_tokens

    @property
    def cost_delta(self) -> float:
        return self.candidate.total_cost_usd - self.baseline.total_cost_usd

    @property
    def cost_ratio(self) -> float | None:
        if not self.baseline.total_cost_usd:
            return None
        return round(self.candidate.total_cost_usd / self.baseline.total_cost_usd, 2)

    def by_model_delta(self) -> list[tuple[str, float, int]]:
        """Per-model (label, cost delta, token delta), biggest mover first.

        This is the answer to the question. A run that cost more did so because
        some specific model or tool was called more, or with more context, and a
        single total can never say which.
        """
        return self._delta(self.baseline.by_model, self.candidate.by_model)

    def by_tool_delta(self) -> list[tuple[str, float, int]]:
        return self._delta(self.baseline.by_tool, self.candidate.by_tool)

    @staticmethod
    def _delta(
        baseline: list[CostLine], candidate: list[CostLine]
    ) -> list[tuple[str, float, int]]:
        before = {line.label: line for line in baseline}
        after = {line.label: line for line in candidate}
        rows = []
        for label in sorted(set(before) | set(after)):
            b = before.get(label, CostLine(label=label))
            a = after.get(label, CostLine(label=label))
            rows.append((label, a.cost_usd - b.cost_usd, a.tokens - b.tokens))
        return sorted(rows, key=lambda row: (abs(row[1]), abs(row[2])), reverse=True)
