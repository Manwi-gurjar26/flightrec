"""The span model.

Follows the OpenTelemetry GenAI semantic conventions so this is a real tracing
format rather than a private invention: a run is a trace, each model call or
tool call is a span, and spans form a tree via ``parent_span_id``.

Attribute names come from the OTel GenAI conventions (``gen_ai.*``). Anything
specific to this tool is namespaced ``flightrec.*`` so the two never collide and
a standard OTel consumer can ignore ours.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# --- OpenTelemetry GenAI semantic convention attribute keys -------------------

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# --- flightrec-specific attributes -------------------------------------------

FR_INPUT = "flightrec.input"
FR_OUTPUT = "flightrec.output"
FR_COST_USD = "flightrec.cost_usd"
FR_STEP_INDEX = "flightrec.step_index"
FR_REPLAYED = "flightrec.replayed"
FR_DIVERGENT = "flightrec.divergent"


class SpanKind(str, Enum):
    """What sort of work a span represents.

    Maps onto ``gen_ai.operation.name`` where the convention defines a value,
    and stays as our own label where it does not.
    """

    AGENT = "agent"  # the whole run, or a sub-agent
    LLM = "chat"  # a model call
    TOOL = "execute_tool"  # a tool call
    STEP = "step"  # one iteration of the agent loop


class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanEvent(BaseModel):
    """A point-in-time event on a span, e.g. an exception or a retry."""

    name: str
    timestamp: float
    attributes: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    """One recorded unit of work."""

    span_id: str
    trace_id: str
    parent_span_id: str | None = None

    name: str
    kind: SpanKind

    start_time: float
    end_time: float | None = None

    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None

    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    @property
    def is_error(self) -> bool:
        return self.status is SpanStatus.ERROR

    def attr(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    @property
    def input_tokens(self) -> int:
        return int(self.attr(GEN_AI_USAGE_INPUT_TOKENS, 0) or 0)

    @property
    def output_tokens(self) -> int:
        return int(self.attr(GEN_AI_USAGE_OUTPUT_TOKENS, 0) or 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return float(self.attr(FR_COST_USD, 0.0) or 0.0)


class SpanNode(BaseModel):
    """A span plus its children, for rendering the run as a tree."""

    span: Span
    children: list["SpanNode"] = Field(default_factory=list)


class Run(BaseModel):
    """A complete recorded agent run: a flat list of spans plus its metadata."""

    run_id: str
    spans: list[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def root_spans(self) -> list[Span]:
        """Spans with no parent, or whose parent is missing from this run."""
        known = {s.span_id for s in self.spans}
        return [
            s
            for s in self.spans
            if s.parent_span_id is None or s.parent_span_id not in known
        ]

    def tree(self) -> list[SpanNode]:
        """Rebuild the span tree, ordered by start time at every level.

        Orphans -- spans whose parent never arrived -- are promoted to roots
        rather than dropped. Losing a step silently is worse than showing it in
        the wrong place, because the whole point of the tool is that you can
        trust what you see.
        """
        nodes = {s.span_id: SpanNode(span=s) for s in self.spans}
        roots: list[SpanNode] = []

        for span in self.spans:
            node = nodes[span.span_id]
            parent = nodes.get(span.parent_span_id) if span.parent_span_id else None
            if parent is None:
                roots.append(node)
            else:
                parent.children.append(node)

        def sort_children(node: SpanNode) -> None:
            node.children.sort(key=lambda n: n.span.start_time)
            for child in node.children:
                sort_children(child)

        roots.sort(key=lambda n: n.span.start_time)
        for root in roots:
            sort_children(root)
        return roots

    def steps(self) -> list[Span]:
        """The linear sequence of significant steps, in execution order.

        This -- not the raw span list -- is what the timeline renders and what
        the diff aligns. Structural spans (the agent wrapper, loop iterations)
        are excluded; model calls and tool calls are the steps a human reasons
        about.
        """
        significant = [
            s for s in self.spans if s.kind in (SpanKind.LLM, SpanKind.TOOL)
        ]
        return sorted(significant, key=lambda s: (s.start_time, s.span_id))

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def has_error(self) -> bool:
        return any(s.is_error for s in self.spans)


SpanNode.model_rebuild()
