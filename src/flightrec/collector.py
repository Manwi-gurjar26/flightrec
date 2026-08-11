"""The collector: an HTTP endpoint that receives spans and stores them.

Small on purpose. The collector's only jobs are to accept spans quickly, never
lose them, and never reject a batch for a reason the SDK cannot act on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from flightrec.spans import Run, Span, SpanNode
from flightrec.storage import RunStore, RunSummary


class SpanBatch(BaseModel):
    """A batch of spans from an SDK. Batched because per-span HTTP is wasteful."""

    spans: list[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    accepted: int


def create_app(store: RunStore | None = None, db_path: str | Path = "flightrec.db") -> FastAPI:
    """Build the app around a store.

    The store is injected rather than constructed at import time so tests can
    use an in-memory database and the app can be mounted more than once.
    """
    resolved = store if store is not None else RunStore(db_path)
    app = FastAPI(
        title="flightrec collector",
        description="Receives agent spans and serves recorded runs.",
        version="0.1.0",
    )
    app.state.store = resolved

    def get_store() -> RunStore:
        return app.state.store

    @app.get("/healthz")
    def healthz(store: RunStore = Depends(get_store)) -> dict[str, Any]:
        return {"status": "ok", "runs": store.count_runs()}

    @app.post("/v1/spans", response_model=IngestResult)
    def ingest(batch: SpanBatch, store: RunStore = Depends(get_store)) -> IngestResult:
        return IngestResult(accepted=store.add_spans(batch.spans, batch.metadata))

    @app.get("/v1/runs", response_model=list[RunSummary])
    def list_runs(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        store: RunStore = Depends(get_store),
    ) -> list[RunSummary]:
        return store.list_runs(limit=limit, offset=offset)

    @app.get("/v1/runs/{run_id}", response_model=Run)
    def get_run(run_id: str, store: RunStore = Depends(get_store)) -> Run:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        return run

    @app.get("/v1/runs/{run_id}/tree", response_model=list[SpanNode])
    def get_tree(run_id: str, store: RunStore = Depends(get_store)) -> list[SpanNode]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        return run.tree()

    @app.delete("/v1/runs/{run_id}")
    def delete_run(run_id: str, store: RunStore = Depends(get_store)) -> dict[str, str]:
        if store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        store.delete_run(run_id)
        return {"deleted": run_id}

    return app
