"""Where finished spans go.

Kept behind a tiny interface so the SDK does not care whether spans are being
held in memory for a test, appended to a file, or posted to the collector.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

from flightrec.spans import Span


class Sink(Protocol):
    def emit(self, span: Span) -> None: ...

    def close(self) -> None: ...


class MemorySink:
    """Collects spans in a list. Used by tests and by the in-process replay."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def emit(self, span: Span) -> None:
        self.spans.append(span)

    def close(self) -> None:
        pass


class JSONLSink:
    """Appends one JSON object per span to a file.

    Line-delimited on purpose: a run that crashes halfway still leaves a
    readable partial recording, which is exactly the run you most want to look
    at. A single JSON array would leave an unparseable file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = self.path.open("a", encoding="utf-8")

    def emit(self, span: Span) -> None:
        line = json.dumps(span.model_dump(mode="json"), ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> "JSONLSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class HTTPSink:
    """Buffers spans and posts them to the collector in batches.

    **This sink must never take the agent down with it.** An observability tool
    that raises when its backend is unreachable has made the system it observes
    strictly less reliable, which is an unarguable case for ripping it out. So
    every failure here is swallowed and counted in ``dropped``, and the count is
    exposed rather than hidden -- silently losing data is bad, but crashing a
    production agent because a dashboard is down is worse.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000",
        batch_size: int = 32,
        timeout: float = 2.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.dropped = 0
        self.sent = 0
        self._buffer: list[Span] = []
        self._lock = threading.Lock()

    def emit(self, span: Span) -> None:
        with self._lock:
            self._buffer.append(span)
            ready = len(self._buffer) >= self.batch_size
        if ready:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return

        try:
            import httpx

            payload = {"spans": [s.model_dump(mode="json") for s in batch]}
            response = httpx.post(
                f"{self.url}/v1/spans", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            self.sent += len(batch)
        except Exception:
            # Deliberately broad. See the class docstring: there is no exception
            # from a telemetry backend that justifies breaking the caller.
            self.dropped += len(batch)

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "HTTPSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TeeSink:
    """Fans spans out to several sinks. Used to record locally *and* ship."""

    def __init__(self, *sinks: Sink) -> None:
        self.sinks = list(sinks)

    def emit(self, span: Span) -> None:
        for sink in self.sinks:
            sink.emit(span)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


def read_jsonl(path: str | Path) -> list[Span]:
    """Load spans back from a JSONL recording, skipping any truncated tail."""
    spans: list[Span] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                spans.append(Span.model_validate_json(line))
            except ValueError:
                # A partially written final line means the process died mid-emit.
                # Everything before it is still valid, so keep it.
                continue
    return spans
