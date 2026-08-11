"""View models for the timeline.

Templates should not compute anything. Every width, percentage, label and
truncation is worked out here, where it can be tested without parsing HTML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from flightrec.spans import (
    FR_INPUT,
    FR_OUTPUT,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_TOOL_NAME,
    Run,
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
)

#: Attributes rendered in dedicated places, so they are not repeated in the
#: raw attribute table.
_PROMOTED = {FR_INPUT, FR_OUTPUT}


@dataclass
class StepView:
    """One row of the timeline."""

    sequence: int
    span_id: str
    name: str
    kind: str
    kind_label: str
    status: str
    is_error: bool
    duration_ms: float
    duration_pct: float
    tokens: int
    token_pct: float
    cost_usd: float
    input_text: str
    output_text: str
    summary: str
    detail_attributes: list[tuple[str, str]] = field(default_factory=list)
    events: list[SpanEvent] = field(default_factory=list)
    retry_count: int = 0
    confabulated: bool = False

    @property
    def has_detail(self) -> bool:
        return bool(
            self.input_text or self.output_text or self.detail_attributes or self.events
        )


@dataclass
class TimelineView:
    run_id: str
    steps: list[StepView]
    span_count: int
    step_count: int
    error_count: int
    retry_count: int
    total_tokens: int
    total_cost_usd: float
    duration_ms: float | None
    complete: bool
    root_name: str | None
    metadata: dict[str, Any]

    @property
    def status(self) -> str:
        if not self.complete:
            return "partial"
        return "error" if self.error_count else "ok"


_KIND_LABELS = {
    SpanKind.LLM: "model",
    SpanKind.TOOL: "tool",
    SpanKind.AGENT: "agent",
    SpanKind.STEP: "step",
}


def build_timeline(run: Run) -> TimelineView:
    steps = run.steps()

    # Bars are scaled against the largest value in *this* run, not a global
    # constant. A run where every step costs the same should show equal bars,
    # not a row of slivers.
    max_duration = max((s.duration_ms or 0.0) for s in steps) if steps else 0.0
    max_tokens = max((s.total_tokens for s in steps), default=0)

    views = [_build_step(s, max_duration, max_tokens) for s in steps]

    roots = [s for s in run.spans if s.parent_span_id is None]
    ends = [s.end_time for s in run.spans if s.end_time is not None]
    starts = [s.start_time for s in run.spans]
    duration = (max(ends) - min(starts)) * 1000.0 if ends and starts else None

    return TimelineView(
        run_id=run.run_id,
        steps=views,
        span_count=len(run.spans),
        step_count=len(steps),
        error_count=sum(1 for s in run.spans if s.is_error),
        retry_count=sum(
            1 for s in run.spans for e in s.events if e.name == "retry"
        ),
        total_tokens=run.total_tokens,
        total_cost_usd=run.total_cost_usd,
        duration_ms=duration,
        complete=bool(roots) and roots[0].end_time is not None,
        root_name=roots[0].name if roots else None,
        metadata=run.metadata,
    )


def _build_step(span: Span, max_duration: float, max_tokens: int) -> StepView:
    duration = span.duration_ms or 0.0
    tokens = span.total_tokens

    detail = [
        (key, _stringify(value))
        for key, value in sorted(span.attributes.items())
        if key not in _PROMOTED
    ]

    return StepView(
        sequence=span.sequence,
        span_id=span.span_id,
        name=span.name,
        kind=span.kind.value,
        kind_label=_KIND_LABELS.get(span.kind, span.kind.value),
        status=span.status.value,
        is_error=span.status is SpanStatus.ERROR,
        duration_ms=duration,
        duration_pct=_pct(duration, max_duration),
        tokens=tokens,
        token_pct=_pct(tokens, max_tokens),
        cost_usd=span.cost_usd,
        input_text=_stringify(span.attr(FR_INPUT)),
        output_text=_stringify(span.attr(FR_OUTPUT)),
        summary=_summarise(span),
        detail_attributes=detail,
        events=list(span.events),
        retry_count=sum(1 for e in span.events if e.name == "retry"),
        confabulated=bool(span.attr("flightrec.confabulated")),
    )


def _summarise(span: Span) -> str:
    """The one line shown before a step is expanded.

    A failed step shows its error, never its (absent) output -- the collapsed
    view has to tell you where to look, or you are back to scrolling.
    """
    if span.status is SpanStatus.ERROR and span.status_message:
        return span.status_message
    output = span.attr(FR_OUTPUT)
    if output not in (None, ""):
        return _stringify(output)
    tool = span.attr(GEN_AI_TOOL_NAME)
    return f"{tool}(...)" if tool else ""


def _pct(value: float, maximum: float) -> float:
    if not maximum or maximum <= 0:
        return 0.0
    return round(min(100.0, (value / maximum) * 100.0), 2)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(ms: float | None) -> str:
    if ms is None:
        return "-"
    if ms < 1:
        return f"{ms * 1000:.0f}us"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


def model_of(span: Span) -> str:
    model = span.attr(GEN_AI_REQUEST_MODEL)
    temperature = span.attr(GEN_AI_REQUEST_TEMPERATURE)
    if model and temperature is not None:
        return f"{model} @ T={temperature}"
    return str(model or "")
