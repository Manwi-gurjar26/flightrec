"""Deterministic replay: re-run a recording with every source of variation pinned.

The contract is narrow on purpose. A replay serves **tool results** from the
recording and never re-executes them, forces the recorded sampling parameters,
and drives the run from a virtual clock and a seeded ID generator. What it does
*not* do is serve model responses -- those are re-executed, because the whole
point of replaying is to see what the model does when you change something.
That only works if the model is a function of its inputs, which is exactly what
temperature 0 buys and what the recorded temperature makes checkable.

Two fidelity claims, and they are deliberately different:

* **replay vs. replay is bit-identical** -- same span IDs, same timestamps, same
  canonical JSON. Everything feeding those is seeded from the recording.
* **replay vs. recording is trajectory-identical** -- same steps, inputs,
  outputs and statuses, but not the same span IDs, because the recording used
  real UUIDs and a real clock. Nothing can recover those, and claiming to would
  be a lie of the silent kind this tool exists to prevent.

Past the edit point the recording no longer describes what the run would do, so
those steps are re-executed live and marked ``flightrec.divergent``. ``strict``
stops there instead. See the README for why guessing forward is the default.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from flightrec.demo.agent import AgentResult, ResearchAgent
from flightrec.demo.tools import FaultConfig, ToolError
from flightrec.determinism import SeededIdGenerator, VirtualClock
from flightrec.retry import TransientError
from flightrec.spans import (
    FR_DIVERGENT,
    FR_INPUT,
    FR_OUTPUT,
    FR_REPLAYED,
    FR_SERVED,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_TOOL_NAME,
    Run,
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
    first_divergence,
    stable_key,
)
from flightrec.tracer import Tracer


class ReplayMismatch(RuntimeError):
    """The replayed agent asked for a tool call the recording does not contain.

    Raised loudly rather than papered over. Before the edit point a replay is
    supposed to be a reproduction, so a mismatch means the pinning is broken --
    and a debugging tool that quietly serves the wrong data is worse than one
    that stops.
    """


class ReplayStopped(RuntimeError):
    """Raised in ``strict`` mode when the replay reaches the edit point."""


# Tool failures come back from the recording as ``"Type: message"`` on the span.
# Re-raising the same class matters: the agent treats transient and permanent
# failures differently, so collapsing them would change the trajectory.
_EXCEPTIONS: dict[str, type[Exception]] = {
    "ToolError": ToolError,
    "TransientError": TransientError,
}


class RecordedTools:
    """Serves recorded tool results in order, then hands over to live execution.

    Ordering, not matching, is the authority: the *n*-th tool call of the replay
    is answered by the *n*-th tool call of the recording. Name and arguments are
    then checked, and a disagreement is a :class:`ReplayMismatch` rather than a
    lookup miss. Keying by ``(name, arguments)`` instead would quietly serve the
    first ``fetch_page`` result to a later retry of the same URL and hide the
    retry entirely.
    """

    def __init__(
        self,
        run: Run,
        agent: ResearchAgent,
        *,
        from_step: int | None = None,
        strict: bool = False,
    ) -> None:
        self._agent = agent
        self._recorded: list[tuple[int, Span]] = [
            (index, span)
            for index, span in enumerate(run.steps())
            if span.kind is SpanKind.TOOL
        ]
        self._cursor = 0
        self.from_step = from_step
        self.strict = strict
        self.served = 0
        self.live = 0
        self.stopped = False
        self.mismatch: ReplayMismatch | None = None

    @property
    def _gone_live(self) -> bool:
        return self.live > 0

    def __call__(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._gone_live:
            return self._live(name, arguments)

        if self._cursor >= len(self._recorded):
            raise self._diverged(
                f"the replay called {name!r} but the recording has no further tool calls"
            )

        step_index, recorded = self._recorded[self._cursor]

        if self.from_step is not None and step_index >= self.from_step:
            if self.strict:
                self.stopped = True
                raise ReplayStopped(
                    f"stopped at step {step_index} (--strict): "
                    "the recording no longer describes this run"
                )
            return self._live(name, arguments)

        recorded_name = recorded.attr(GEN_AI_TOOL_NAME) or recorded.name
        recorded_args = recorded.attr(FR_INPUT)
        if recorded_name != name or recorded_args != arguments:
            raise self._diverged(
                f"step {step_index}: recorded {recorded_name}({recorded_args!r}) "
                f"but the replay asked for {name}({arguments!r})"
            )

        self._cursor += 1
        self.served += 1
        return self._serve(recorded)

    # -- the two paths --------------------------------------------------------

    def _serve(self, recorded: Span) -> Any:
        """Reproduce one recorded tool outcome, including how it got there."""
        span = Tracer.current_span()
        if span is not None:
            span.attributes[FR_SERVED] = True
            # Retries are part of what the step *did* -- they are why it was slow
            # and what it cost. Serving only the final outcome would replay a
            # flaky run as a clean one and lose the whole reason to look at it.
            span.events.extend(
                SpanEvent(**event.model_dump())
                for event in recorded.events
                if event.name == "retry"
            )

        if recorded.status is SpanStatus.ERROR:
            raise _reconstruct(recorded)
        return recorded.attr(FR_OUTPUT)

    def _live(self, name: str, arguments: dict[str, Any]) -> Any:
        self.live += 1
        return self._agent.invoke_live(name, arguments, Tracer.current_span())

    def _diverged(self, detail: str) -> ReplayMismatch:
        self.mismatch = ReplayMismatch(detail)
        return self.mismatch


def _reconstruct(span: Span) -> Exception:
    message = span.status_message or "recorded tool failure"
    type_name, _, detail = message.partition(": ")
    return _EXCEPTIONS.get(type_name, ToolError)(detail or message)


# --- the engine ---------------------------------------------------------------


@dataclass
class ReplayResult:
    """A replayed run, alongside the recording it came from."""

    run: Run
    original: Run
    outcome: AgentResult
    from_step: int | None = None
    strict: bool = False
    served: int = 0
    live: int = 0
    stopped: bool = False
    edits: dict[str, Any] = field(default_factory=dict)

    @property
    def faithful(self) -> bool:
        """Did this replay reproduce the recording's step sequence exactly?

        Only meaningful for an unedited full replay. With an edit point, steps
        after it are *supposed* to differ, so a ``False`` here says nothing.
        """
        return first_divergence(self.original, self.run) is None

    @property
    def divergence_step(self) -> int | None:
        return first_divergence(self.original, self.run)


def replay_run(
    run: Run,
    *,
    from_step: int | None = None,
    strict: bool = False,
    task: str | None = None,
    temperature: float | None = None,
) -> ReplayResult:
    """Replay ``run``, optionally re-executing live from ``from_step`` onward.

    ``task`` and ``temperature`` are the edits: change one and everything from
    the first affected step is re-executed and marked divergent.
    """
    root = _root(run)
    seed = int(root.attr("flightrec.seed", 0) or 0) if root else 0
    recorded_task = root.attr(FR_INPUT) if root else None
    faults = _faults(root)

    edits: dict[str, Any] = {}
    if task is not None and task != recorded_task:
        edits["task"] = task
    if temperature is not None:
        edits["temperature"] = temperature
    # An edit with no explicit cut point takes effect from the very first step:
    # a different task changes the run from the top.
    if edits and from_step is None:
        from_step = 0

    agent = ResearchAgent(
        seed=seed,
        temperature=_temperature(run) if temperature is None else temperature,
        faults=faults,
        clock=VirtualClock(start=_start_time(run)),
        id_generator=SeededIdGenerator(_identity(run, from_step, strict, edits)),
        max_steps=int(root.attr("flightrec.max_steps", 12) or 12) if root else 12,
    )
    oracle = RecordedTools(run, agent, from_step=from_step, strict=strict)
    agent.tool_override = oracle

    outcome = agent.run(task if task is not None else (recorded_task or ""))

    # The agent catches its own exceptions -- that is the behaviour under test --
    # so a mismatch would otherwise end up as a string in ``outcome.error``.
    if oracle.mismatch is not None:
        raise oracle.mismatch

    _mark(outcome.run, from_step)
    return ReplayResult(
        run=outcome.run,
        original=run,
        outcome=outcome,
        from_step=from_step,
        strict=strict,
        served=oracle.served,
        live=oracle.live,
        stopped=oracle.stopped,
        edits=edits,
    )


def _mark(run: Run, from_step: int | None) -> None:
    """Flag every span as replayed, and everything past the edit point divergent.

    Divergence is a property of the span, not a note in the UI: a user must
    never be able to mistake a live-executed step for a recorded one.
    """
    for span in run.spans:
        span.attributes[FR_REPLAYED] = True

    if from_step is None:
        return
    steps = run.steps()
    if from_step >= len(steps):
        return
    cut = steps[from_step].sequence
    for span in run.spans:
        if span.sequence >= cut:
            span.attributes[FR_DIVERGENT] = True


def _identity(
    run: Run, from_step: int | None, strict: bool, edits: dict[str, Any]
) -> int:
    """Seed for the replay's span IDs: a hash of the recording plus the edit.

    Not the agent's own seed, which is a property of the *recorded* run and is
    shared by every run made with it. Span IDs are the storage primary key and
    writes are ``INSERT OR REPLACE``, so two replays that collided would not
    conflict -- they would silently overwrite each other's steps. Deriving the
    ID space from what makes this replay distinct keeps replays of one recording
    bit-identical while keeping different ones apart.
    """
    material = stable_key([run.run_id, from_step, strict, edits])
    return int.from_bytes(hashlib.blake2b(material.encode(), digest_size=8).digest())


def _root(run: Run) -> Span | None:
    for span in run.ordered_spans():
        if span.kind is SpanKind.AGENT:
            return span
    return None


def _faults(root: Span | None) -> FaultConfig:
    recorded = root.attr("flightrec.faults") if root else None
    if not isinstance(recorded, dict):
        # An older recording, or one made by an agent that never had faults
        # configured. Assume none rather than inventing a fault rate that would
        # make live steps behave unlike the run being continued.
        return FaultConfig()
    known = FaultConfig().__dict__
    return FaultConfig(**{k: float(v) for k, v in recorded.items() if k in known})


def _temperature(run: Run) -> float:
    for span in run.ordered_spans():
        value = span.attr(GEN_AI_REQUEST_TEMPERATURE)
        if value is not None:
            return float(value)
    return 0.0


def _start_time(run: Run) -> float:
    ordered = run.ordered_spans()
    return ordered[0].start_time if ordered else 0.0
