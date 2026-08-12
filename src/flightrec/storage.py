"""SQLite storage for run trees.

Plain ``sqlite3``, no ORM. The schema is six columns of structure and two of
JSON, and an ORM would obscure the two decisions that actually matter here.

**Spans are stored idempotently.** ``span_id`` is the primary key and writes are
``INSERT OR REPLACE``. An SDK that retries a failed batch must not be able to
double a run's token count -- and it will retry, because the network is what it
is. Idempotency belongs at the store, not in the client's good intentions.

**Partial runs are first-class.** A run row is created by whichever span arrives
first, not by a "start run" call, and a run with no closed root is reported as
``partial`` rather than hidden. The run whose process died mid-flight is
precisely the one you opened this tool to look at; a collector that only shows
cleanly finished runs is useless for the job.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from flightrec.spans import FR_REPLAYED, Run, Span, SpanKind, SpanStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS spans (
    span_id        TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    parent_span_id TEXT,
    sequence       INTEGER NOT NULL,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    start_time     REAL NOT NULL,
    end_time       REAL,
    status         TEXT NOT NULL,
    status_message TEXT,
    attributes     TEXT NOT NULL DEFAULT '{}',
    events         TEXT NOT NULL DEFAULT '[]',
    received_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
"""


class RunSummary(BaseModel):
    """One row of the run list. Deliberately cheap to compute."""

    run_id: str
    created_at: float
    root_name: str | None = None
    span_count: int = 0
    step_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float | None = None
    complete: bool = False
    is_replay: bool = False
    """Marked in the list as well as on the page.

    Finding out that a run was a replay only after opening it is finding out too
    late -- the run list is where somebody picks which run to trust.
    """

    @property
    def status(self) -> str:
        if not self.complete:
            return "partial"
        return "error" if self.error_count else "ok"


class RunStore:
    """Reads and writes run trees."""

    def __init__(self, path: str | Path = "flightrec.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # A single shared connection, guarded by SQLite's own locking. WAL keeps
        # the UI's reads from blocking the collector's writes, which matters
        # because you will be refreshing the timeline while a run is still going.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    def close(self) -> None:
        self._conn.close()

    # -- writing --------------------------------------------------------------

    def add_spans(self, spans: list[Span], metadata: dict | None = None) -> int:
        """Store a batch of spans. Safe to call twice with the same spans."""
        if not spans:
            return 0

        now = time.time()
        with self._tx() as conn:
            for run_id in {s.trace_id for s in spans}:
                conn.execute(
                    "INSERT OR IGNORE INTO runs (run_id, created_at, metadata) "
                    "VALUES (?, ?, ?)",
                    (run_id, now, json.dumps(metadata or {})),
                )
            conn.executemany(
                """
                INSERT OR REPLACE INTO spans (
                    span_id, run_id, parent_span_id, sequence, name, kind,
                    start_time, end_time, status, status_message,
                    attributes, events, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.span_id,
                        s.trace_id,
                        s.parent_span_id,
                        s.sequence,
                        s.name,
                        s.kind.value,
                        s.start_time,
                        s.end_time,
                        s.status.value,
                        s.status_message,
                        json.dumps(s.attributes),
                        json.dumps([e.model_dump(mode="json") for e in s.events]),
                        now,
                    )
                    for s in spans
                ],
            )
        return len(spans)

    def add_run(self, run: Run) -> int:
        return self.add_spans(run.spans, metadata=run.metadata)

    def delete_run(self, run_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM spans WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    # -- reading --------------------------------------------------------------

    def get_run(self, run_id: str) -> Run | None:
        rows = self._conn.execute(
            "SELECT * FROM spans WHERE run_id = ? ORDER BY sequence", (run_id,)
        ).fetchall()
        meta_row = self._conn.execute(
            "SELECT metadata FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not rows and meta_row is None:
            return None

        return Run(
            run_id=run_id,
            spans=[_row_to_span(r) for r in rows],
            metadata=json.loads(meta_row["metadata"]) if meta_row else {},
        )

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[RunSummary]:
        """Summaries for the run list.

        Token and cost totals live inside the attributes JSON, so they are
        summed in Python rather than SQL. At the scale this tool is for -- one
        developer's laptop, thousands of runs -- that is the right trade: no
        denormalised counters to drift out of step with the spans they describe.
        """
        run_rows = self._conn.execute(
            "SELECT run_id, created_at FROM runs ORDER BY created_at DESC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        summaries = []
        for run_row in run_rows:
            run = self.get_run(run_row["run_id"])
            if run is None:  # pragma: no cover - deleted between the two reads
                continue
            summaries.append(_summarise(run, run_row["created_at"]))
        return summaries

    def count_runs(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])


def _row_to_span(row: sqlite3.Row) -> Span:
    return Span(
        span_id=row["span_id"],
        trace_id=row["run_id"],
        parent_span_id=row["parent_span_id"],
        sequence=row["sequence"],
        name=row["name"],
        kind=SpanKind(row["kind"]),
        start_time=row["start_time"],
        end_time=row["end_time"],
        status=SpanStatus(row["status"]),
        status_message=row["status_message"],
        attributes=json.loads(row["attributes"]),
        events=json.loads(row["events"]),
    )


def _summarise(run: Run, created_at: float) -> RunSummary:
    roots = [s for s in run.spans if s.parent_span_id is None]
    root = roots[0] if roots else None
    complete = root is not None and root.end_time is not None

    duration = None
    if run.spans:
        starts = [s.start_time for s in run.spans]
        ends = [s.end_time for s in run.spans if s.end_time is not None]
        if ends:
            duration = (max(ends) - min(starts)) * 1000.0

    return RunSummary(
        run_id=run.run_id,
        created_at=created_at,
        root_name=root.name if root else None,
        span_count=len(run.spans),
        step_count=len(run.steps()),
        error_count=sum(1 for s in run.spans if s.is_error),
        total_tokens=run.total_tokens,
        total_cost_usd=run.total_cost_usd,
        duration_ms=duration,
        complete=complete,
        is_replay=any(s.attr(FR_REPLAYED) for s in run.spans),
    )
