"""The instrumentation SDK: one context manager and one decorator.

Design goal: instrumenting an existing agent should be a handful of lines and
should never change its behaviour. An observability tool that alters the thing
it observes is worse than no tool, so exceptions are always re-raised after
being recorded.

Parent/child nesting is tracked with a ``ContextVar``, which means it works
correctly across ``async`` code and threads without the caller passing anything
around.
"""

from __future__ import annotations

import functools
import itertools
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, TypeVar

from flightrec.determinism import (
    Clock,
    IdGenerator,
    RandomIdGenerator,
    SystemClock,
)
from flightrec.sinks import MemorySink, Sink
from flightrec.spans import (
    FR_INPUT,
    FR_OUTPUT,
    GEN_AI_OPERATION_NAME,
    Run,
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
)

_current_span: ContextVar[Span | None] = ContextVar("flightrec_current_span", default=None)

F = TypeVar("F", bound=Callable[..., Any])


class Tracer:
    """Creates spans and hands finished ones to a sink."""

    def __init__(
        self,
        sink: Sink | None = None,
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
        trace_id: str | None = None,
        prices: "PriceTable | None" = None,
    ) -> None:
        from flightrec.pricing import PriceTable

        self.sink: Sink = sink if sink is not None else MemorySink()
        self.clock: Clock = clock if clock is not None else SystemClock()
        self.ids: IdGenerator = (
            id_generator if id_generator is not None else RandomIdGenerator()
        )
        self.prices = prices if prices is not None else PriceTable()
        self.trace_id = trace_id or self.ids.new_id()
        # itertools.count().__next__ is atomic in CPython, so concurrent spans
        # still get distinct, ordered sequence numbers without a lock.
        self._sequence = itertools.count()
        self.replay_source: "ReplaySource | None" = None
        """Set by the replay engine. ``None`` in every ordinary run.

        When present, :meth:`call` asks it for a recorded result before running
        anything. This is the whole reason :meth:`call` exists: a tracer that
        only wraps a function can observe it, but cannot stand in for it.
        """

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.STEP,
        inputs: Any = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        """Record a unit of work.

        The yielded span is mutable: set ``flightrec.output`` or token counts on
        it from inside the block via :meth:`set_output` / ``span.attributes``.
        """
        parent = _current_span.get()
        span = Span(
            span_id=self.ids.new_id(),
            trace_id=self.trace_id,
            parent_span_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            sequence=next(self._sequence),
            start_time=self.clock.now(),
            attributes={GEN_AI_OPERATION_NAME: kind.value, **attributes},
        )
        if inputs is not None:
            span.attributes[FR_INPUT] = _safe(inputs)

        token = _current_span.set(span)
        try:
            yield span
        except BaseException as exc:
            self.record_exception(span, exc)
            raise  # never swallow: the tool must not change what it observes
        finally:
            _current_span.reset(token)
            if span.status is SpanStatus.UNSET:
                span.status = SpanStatus.OK
            span.end_time = self.clock.now()
            self.sink.emit(span)

    def call(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        kind: SpanKind = SpanKind.STEP,
        inputs: Any = None,
        restore: Callable[[Any, Span], Any] | None = None,
        **attributes: Any,
    ) -> Any:
        """Run one unit of work *through* the tracer, and record what it produced.

        The difference from :meth:`span` is that the tracer invokes ``fn``
        instead of standing around it, and that difference is the entire reason
        replay can work for an agent this library has never seen. A wrapper can
        watch a call; only the caller can decline to make it and hand back what
        happened last time.

        So: use ``span`` to bracket something you are running yourself, and
        ``call`` for anything you would ever want replayed -- every model call
        and every tool call.

        ``fn`` must return something serialisable, because what comes back on a
        replay is what was written down, not the object that produced it. Where
        a step really returns something richer -- a model response with a tool
        call attached -- record the readable part, put the rest on the span, and
        pass ``restore`` to rebuild it. ``restore(value, span)`` runs on both
        paths, so the agent gets the same type whether the step ran or not.
        """
        with self.span(name, kind=kind, inputs=inputs, **attributes) as span:
            source = self.replay_source
            if source is not None:
                served, value = source.serve(span, name, kind, inputs)
                if served:
                    span.attributes[FR_OUTPUT] = _safe(value)
                    return restore(value, span) if restore else value
            result = fn()
            span.attributes[FR_OUTPUT] = _safe(result)
            return restore(result, span) if restore else result

    def trace(
        self,
        name: str | None = None,
        kind: SpanKind = SpanKind.STEP,
        capture_args: bool = True,
    ) -> Callable[[F], F]:
        """Decorator form of :meth:`span`.

        The return value is recorded as the span output automatically, which is
        the common case for tool functions.
        """

        def decorator(func: F) -> F:
            span_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                inputs = {"args": args, "kwargs": kwargs} if capture_args else None
                # Routed through call(), so a decorated tool is replayable
                # without its author doing anything else.
                return self.call(
                    span_name,
                    lambda: func(*args, **kwargs),
                    kind=kind,
                    inputs=inputs,
                )

            return wrapper  # type: ignore[return-value]

        return decorator

    def record_exception(self, span: Span, exc: BaseException) -> None:
        """Mark a span as failed and attach the exception detail.

        Called automatically when an exception escapes a span, and *manually* by
        callers that catch a failure themselves. Agent frameworks routinely
        catch tool errors and hand them back to the model rather than letting
        them propagate -- that is not an edge case, it is the main path, and it
        is exactly how an agent talks itself into a wrong answer. Both routes
        have to produce identically detailed spans, or the timeline shows a red
        marker with nothing behind it on the failures that matter most.
        """
        span.status = SpanStatus.ERROR
        span.status_message = f"{type(exc).__name__}: {exc}"
        span.events.append(
            SpanEvent(
                name="exception",
                timestamp=self.clock.now(),
                attributes={
                    "exception.type": type(exc).__name__,
                    "exception.message": str(exc),
                    "exception.stacktrace": "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip(),
                },
            )
        )

    def record_usage(
        self,
        span: Span,
        model: str | None,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record token usage on a model span and price it.

        Cost is fixed at record time from the table in force then, and the rate
        applied is written alongside it. See :mod:`flightrec.pricing` for why
        deriving cost at read time would quietly rewrite history.
        """
        self.prices.apply(span, model, input_tokens, output_tokens)

    @staticmethod
    def set_output(span: Span, value: Any) -> None:
        span.attributes[FR_OUTPUT] = _safe(value)

    @staticmethod
    def current_span() -> Span | None:
        return _current_span.get()

    def collect(self) -> Run:
        """Build a :class:`Run` from an in-memory sink. Tests and replay use this."""
        if not isinstance(self.sink, MemorySink):
            raise TypeError(
                "collect() requires a MemorySink; other sinks do not retain spans"
            )
        return Run(run_id=self.trace_id, spans=list(self.sink.spans))


def _safe(value: Any) -> Any:
    """Make a value safe to serialise, without ever raising.

    Instrumentation that crashes on an unserialisable argument would take the
    agent down with it, so anything unknown degrades to its ``repr``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    try:
        return repr(value)
    except Exception:  # pragma: no cover - repr() itself failing is pathological
        return f"<unrepresentable {type(value).__name__}>"
