"""Retry with exponential backoff and jitter.

This module exists mostly because of the jitter. Retry jitter is one of the
sources of non-determinism that is easy to miss: it does not change *what* the
agent does, it changes *when*, which is enough to make two runs' timings differ
and -- if a retry budget is exhausted differently -- to change the trajectory
itself. So the RNG is injected, never global.
"""

from __future__ import annotations

import random
from typing import Any, Callable, TypeVar

from flightrec.spans import Span, SpanEvent

T = TypeVar("T")


class TransientError(RuntimeError):
    """An error worth retrying, as opposed to a bug."""


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.05,
    rng: random.Random,
    sleep: Callable[[float], None] = lambda _: None,
    span: Span | None = None,
    now: Callable[[], float] | None = None,
) -> T:
    """Call ``func``, retrying on :class:`TransientError`.

    ``sleep`` defaults to a no-op so tests and replays do not spend real time.
    Each retry is recorded as an event on ``span`` when one is supplied, which
    is what puts the red markers on the timeline.
    """
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            return func()
        except TransientError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            delay = base_delay * (2**attempt) * (0.5 + rng.random())
            if span is not None:
                span.events.append(
                    SpanEvent(
                        name="retry",
                        timestamp=now() if now else 0.0,
                        attributes={
                            "retry.attempt": attempt + 1,
                            "retry.delay_s": round(delay, 6),
                            "retry.reason": str(exc),
                        },
                    )
                )
            sleep(delay)

    assert last_error is not None
    raise last_error


def make_rng(seed: int) -> random.Random:
    """The only sanctioned way to get randomness in this project."""
    return random.Random(seed)


def _unused(*_: Any) -> None:  # pragma: no cover
    pass
