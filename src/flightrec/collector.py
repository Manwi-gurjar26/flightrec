"""The collector: an HTTP endpoint that receives spans and stores them.

Small on purpose. The collector's only jobs are to accept spans quickly, never
lose them, and never reject a batch for a reason the SDK cannot act on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from flightrec.pricing import format_usd
from flightrec.rollup import RunCost, build_rollup
from flightrec.spans import Run, Span, SpanNode
from flightrec.storage import RunStore, RunSummary
from flightrec.web import (
    build_diff,
    build_timeline,
    format_duration,
    format_timestamp,
    is_replayable,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


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
    app.state.db_path = str(db_path)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals.update(
        format_duration=format_duration,
        format_timestamp=format_timestamp,
        format_usd=format_usd,
    )

    def get_store() -> RunStore:
        return app.state.store

    # -- the UI ---------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, store: RunStore = Depends(get_store)) -> Any:
        return templates.TemplateResponse(
            request,
            "runs.html",
            {"summaries": store.list_runs(limit=100), "db_path": app.state.db_path},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def timeline(
        request: Request, run_id: str, store: RunStore = Depends(get_store)
    ) -> Any:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        return templates.TemplateResponse(
            request,
            "timeline.html",
            {"view": build_timeline(run), "cost": build_rollup(run)},
        )

    @app.post("/runs/{run_id}/replay")
    async def start_replay(
        request: Request, run_id: str, store: RunStore = Depends(get_store)
    ) -> Any:
        """Replay a run from the browser, store the result, and redirect to it.

        POST then redirect, deliberately: a replay past the edit point executes
        real calls and writes a new run, so a page refresh must not quietly do
        it again. The 303 makes the browser land on the replay with a GET.
        """
        from flightrec.replay import ReplayMismatch, replay_run

        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")

        form = await request.form()

        def field(name: str) -> str | None:
            value = str(form.get(name, "")).strip()
            return value or None

        def render_error(message: str) -> Any:
            return templates.TemplateResponse(
                request,
                "timeline.html",
                {
                    "view": build_timeline(run),
                    "cost": build_rollup(run),
                    "error": message,
                },
                status_code=400,
            )

        if not is_replayable(run):
            return render_error(
                "This run cannot be replayed: the engine rebuilds the demo agent "
                "from the seed on its root span, and this recording does not "
                "carry one."
            )

        try:
            from_step = int(field("from_step")) if field("from_step") else None
            temperature = (
                float(field("temperature")) if field("temperature") else None
            )
        except ValueError:
            return render_error("Step and temperature have to be numbers.")

        try:
            result = replay_run(
                run,
                from_step=from_step,
                strict=bool(form.get("strict")),
                task=field("task"),
                temperature=temperature,
            )
        except ReplayMismatch as exc:
            # Loud here too. A mismatch means the pinning is broken, and the
            # browser should say so rather than store a run that is not a replay.
            return render_error(f"Replay failed: {exc}")

        store.add_run(result.run)
        return RedirectResponse(f"/runs/{result.run.run_id}", status_code=303)

    # Query parameters rather than path segments so a plain <form method="get">

    # can reach this. The UI has no JavaScript, so the form has to be the whole
    # mechanism -- and a GET keeps the result linkable and back-button safe.
    @app.get("/diff", response_class=HTMLResponse)
    def diff(
        request: Request,
        left: str = Query(...),
        right: str = Query(...),
        store: RunStore = Depends(get_store),
    ) -> Any:
        left_run = store.get_run(left)
        right_run = store.get_run(right)
        for run_id, run in ((left, left_run), (right, right_run)):
            if run is None:
                raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        return templates.TemplateResponse(
            request, "diff.html", {"view": build_diff(left_run, right_run)}
        )

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
