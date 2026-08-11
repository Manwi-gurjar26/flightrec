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
