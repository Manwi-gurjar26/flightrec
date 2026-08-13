"""Deterministic replay: re-run a recording with every source of variation pinned.

The contract splits at the edit point, and both halves matter.

**Before it, every step is served from the recording** -- tool results *and*
model responses. That is what makes a replay cheap: re-running the first six
steps against a real provider to get to the seventh costs exactly what it cost
the first time, and nobody edits step 7 twenty times if each attempt is billed
from step 0.

**After it, everything is re-executed live**, because seeing what happens next
is the entire reason to edit a step. Re-execution only produces an answer worth
having if the model is a function of its inputs, which is what temperature 0
buys and what the recorded temperature makes checkable.

Two fidelity claims, deliberately different:

* **replay vs. replay is bit-identical** -- same span IDs, same timestamps.
  Everything feeding those is seeded from the recording.
* **replay vs. recording is trajectory-identical** -- same steps, inputs,
  outputs and statuses, but not the same span IDs, because the recording used
  real UUIDs and a real clock. Nothing can recover those, and claiming to would
  be a lie of the silent kind this tool exists to prevent.

**This engine has never seen your agent and does not need to.** It drives a
callable you supply, hands it a tracer, and answers that tracer's
:meth:`~flightrec.tracer.Tracer.call` invocations from the recording. Anything
instrumented with ``tracer.call`` or ``@tracer.trace`` is replayable; anything
using the bare ``span`` context manager is not, because a wrapper cannot decline
to run the code it wraps. See :func:`replay`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from flightrec.determinism import SeededIdGenerator, VirtualClock
from flightrec.sinks import MemorySink
from flightrec.spans import (
    FR_DIVERGENT,
    FR_INPUT,
    FR_OUTPUT,
    FR_REPLAYED,
    FR_SERVED,
    GEN_AI_REQUEST_TEMPERATURE,
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
    """The replayed agent asked for something the recording does not contain.

    Raised loudly rather than papered over. Before the edit point a replay is
    supposed to be a reproduction, so a mismatch means the pinning is broken --
    and a debugging tool that quietly serves the wrong data is worse than one
    that stops.
    """


class ReplayStopped(RuntimeError):
    """Raised in ``strict`` mode when the replay reaches the edit point."""


class ReplayedError(RuntimeError):
    """Stands in for a failure the recording describes but cannot reconstruct.

    A recording stores an exception's *type name and message*, not the class.
    Rebuilding the real class needs the caller to say which ones matter, via
    ``exceptions=``. Without that this is what a failed step raises -- correct
    about what happened, and honest that the type is gone.
    """


#: Attributes the engine and tracer set themselves, never copied from the
#: recording onto a served span.
_ENGINE_OWNED = frozenset(
    {FR_INPUT, FR_OUTPUT, FR_REPLAYED, FR_SERVED, FR_DIVERGENT}
)


class ReplaySource(Protocol):
    """What a tracer consults before running anything, during a replay."""

    def serve(
        self, span: Span, name: str, kind: SpanKind, inputs: Any
    ) -> tuple[bool, Any]:
        """``(True, value)`` to serve from the recording, ``(False, None)`` to run live."""


@dataclass
class Cutover:
    """Shared state for the recording: where it stops applying.

    One decision for the whole run rather than one per step kind. If model calls
    and tool calls each kept their own idea of where the edit point was, a tool
    could still be served from the recording after the model had been let loose
    -- a run half in each world, which is worse than either.
    """

    from_step: int | None = None
    strict: bool = False
    gone_live: bool = False
    stopped: bool = False
    served: int = 0
    live: int = 0
    mismatch: ReplayMismatch | None = None

    def is_live(self, step_index: int) -> bool:
        """Decide, once, whether this step and everything after it runs live."""
        if self.gone_live:
            return True
        if self.from_step is None or step_index < self.from_step:
            return False
        if self.strict:
            self.stopped = True
            raise ReplayStopped(
                f"stopped at step {step_index} (--strict): "
                "the recording no longer describes this run"
            )
        self.gone_live = True
        return True

    def diverged(self, detail: str) -> ReplayMismatch:
        self.mismatch = ReplayMismatch(detail)
        return self.mismatch


class Recording(ReplaySource):
    """Serves a recorded run back to whatever agent is asking for it.

    Ordering, not matching, is the authority: the *n*-th call of a given kind in
    the replay is answered by the *n*-th call of that kind in the recording. The
    name and inputs are then checked, and a disagreement is a
    :class:`ReplayMismatch` rather than a lookup miss. Keying by ``(name,
    inputs)`` instead would quietly serve the first result to a later retry of
    the same call and hide the retry entirely.

    Separate cursors per kind, because a model call and a tool call are
    different sequences that happen to interleave, and an agent may make a
    different number of each once past the edit point.
    """

    def __init__(
        self,
        run: Run,
        *,
        from_step: int | None = None,
        strict: bool = False,
        exceptions: dict[str, type[BaseException]] | None = None,
    ) -> None:
        self.cutover = Cutover(from_step=from_step, strict=strict)
        self._exceptions = exceptions or {}
        self._steps: dict[SpanKind, list[tuple[int, Span]]] = {}
        for index, span in enumerate(run.steps()):
            self._steps.setdefault(span.kind, []).append((index, span))
        self._cursors: dict[SpanKind, int] = {}

    def serve(
        self, span: Span, name: str, kind: SpanKind, inputs: Any
    ) -> tuple[bool, Any]:
        if self.cutover.gone_live:
            self.cutover.live += 1
            return False, None

        recorded_steps = self._steps.get(kind, [])
        cursor = self._cursors.get(kind, 0)
        if cursor >= len(recorded_steps):
            raise self.cutover.diverged(
                f"the replay called {name!r} but the recording has no further "
                f"{kind.value} steps"
            )

        step_index, recorded = recorded_steps[cursor]
        if self.cutover.is_live(step_index):
            self.cutover.live += 1
            return False, None

        if recorded.name != name or stable_key(recorded.attr(FR_INPUT)) != stable_key(
            inputs
        ):
            raise self.cutover.diverged(
                f"step {step_index}: recorded {recorded.name}({recorded.attr(FR_INPUT)!r}) "
                f"but the replay asked for {name}({inputs!r})"
            )

        self._cursors[kind] = cursor + 1
        self.cutover.served += 1
        return True, self._serve(span, recorded)

    def _serve(self, span: Span, recorded: Span) -> Any:
        """Reproduce one recorded outcome, including how it got there."""
        span.attributes[FR_SERVED] = True
        # Retries are part of what the step *did* -- they are why it was slow
        # and what it cost. Serving only the final outcome would replay a flaky
        # run as a clean one and lose the whole reason to look at it.
        span.events.extend(
            SpanEvent(**event.model_dump())
            for event in recorded.events
            if event.name == "retry"
        )
        # Everything the original step recorded about itself: token counts, the
        # cost it incurred then, whatever the agent attached. A served step
        # spends nothing now but still describes what the original spent, and an
        # agent rebuilding a rich result via ``restore`` reads it from here.
        for key, value in recorded.attributes.items():
            if key not in _ENGINE_OWNED:
                span.attributes.setdefault(key, value)

        if recorded.status is SpanStatus.ERROR:
            raise self._reconstruct(recorded)
        return recorded.attr(FR_OUTPUT)

    def _reconstruct(self, span: Span) -> BaseException:
        message = span.status_message or "recorded failure"
        type_name, _, detail = message.partition(": ")
        cls = self._exceptions.get(type_name) or _synthetic_error(type_name)
        return cls(detail or message)


#: Synthetic exception classes, one per recorded type name, made once.
_SYNTHETIC: dict[str, type[BaseException]] = {}


def _synthetic_error(type_name: str) -> type[BaseException]:
    """A stand-in class that *records* like the original failure did.

    The obvious implementation -- raise ``ReplayedError(message)`` for anything
    unmapped -- makes a replay of a failing step permanently unfaithful. The
    tracer writes ``f"{type(exc).__name__}: {exc}"`` into the span, so a
    recording that said ``KeyError: 'no page'`` replays as
    ``ReplayedError: KeyError: 'no page'`` and the trajectories differ on every
    run that failed. Which is most of the interesting ones.

    Borrowing the recorded *name* fixes that: the span comes back byte-identical
    while the class stays a ``ReplayedError`` subclass, so ``except
    ReplayedError`` catches every replayed failure. What it deliberately is not
    is the real class -- ``except KeyError`` will not catch this, because the
    recording never stored a class to resurrect. Agents that branch on the type
    pass ``exceptions=`` and get the real one.

    One thing neither route survives: exceptions whose ``str()`` is not what
    they were constructed with. ``KeyError('x')`` stringifies as ``"'x'"``, so
    round-tripping it through a recorded message adds a layer of quotes every
    time. Anything with ordinary string semantics is exact.
    """
    if type_name not in _SYNTHETIC:
        # __module__ says where this came from, so a traceback reads
        # "flightrec.replay.recorded.PricingError" and nobody spends an
        # afternoon wondering why their own class is not being caught.
        _SYNTHETIC[type_name] = type(
            type_name, (ReplayedError,), {"__module__": "flightrec.replay.recorded"}
        )
    return _SYNTHETIC[type_name]


# --- the engine ---------------------------------------------------------------


#: What the caller supplies: something that builds and runs their agent using
#: the tracer it is handed. The tracer arrives pre-wired -- virtual clock,
#: seeded IDs, and a recording to answer from -- so the agent needs no idea it
#: is being replayed.
AgentRunner = Callable[[Tracer, str], Any]


@dataclass
class ReplayResult:
    """A replayed run, alongside the recording it came from."""

    run: Run
    original: Run
    outcome: Any = None
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


def replay(
    run: Run,
    run_agent: AgentRunner,
    *,
    from_step: int | None = None,
    strict: bool = False,
    task: str | None = None,
    exceptions: dict[str, type[BaseException]] | None = None,
    edits: dict[str, Any] | None = None,
) -> ReplayResult:
    """Replay ``run`` by driving ``run_agent``, whatever agent that is.

    ``run_agent(tracer, task)`` builds and runs the agent against the tracer it
    is given. That tracer is already pinned to a virtual clock and seeded IDs,
    and already knows how to answer from the recording, so the agent itself
    needs no replay-specific code at all -- only instrumentation via
    ``tracer.call`` or ``@tracer.trace``.

    ``exceptions`` maps recorded exception type names to classes, for agents
    that branch on the type of a failure. Anything unlisted comes back as
    :class:`ReplayedError`, because a recording stores a name and a message,
    not a class.
    """
    edits = dict(edits or {})
    recorded_task = _task_of(run)
    if task is not None and task != recorded_task:
        edits["task"] = task
    # An edit with no explicit cut point takes effect from the very first step:
    # a different task changes the run from the top.
    if edits and from_step is None:
        from_step = 0

    recording = Recording(
        run, from_step=from_step, strict=strict, exceptions=exceptions
    )
    sink = MemorySink()
    tracer = Tracer(
        sink=sink,
        clock=VirtualClock(start=_start_time(run)),
        id_generator=SeededIdGenerator(_identity(run, from_step, strict, edits)),
    )
    tracer.replay_source = recording

    outcome = run_agent(tracer, task if task is not None else (recorded_task or ""))

    # Agents routinely catch their own exceptions -- that is the behaviour under
    # test -- so a mismatch would otherwise end up as a string in their result.
    if recording.cutover.mismatch is not None:
        raise recording.cutover.mismatch

    replayed = Run(run_id=tracer.trace_id, spans=list(sink.spans))
    _mark(replayed, from_step)
    return ReplayResult(
        run=replayed,
        original=run,
        outcome=outcome,
        from_step=from_step,
        strict=strict,
        served=recording.cutover.served,
        live=recording.cutover.live,
        stopped=recording.cutover.stopped,
        edits=edits,
    )


def replay_run(
    run: Run,
    *,
    from_step: int | None = None,
    strict: bool = False,
    task: str | None = None,
    temperature: float | None = None,
) -> ReplayResult:
    """Replay a recording of the demo agent.

    A thin adapter over :func:`replay`: it knows how to rebuild *this one*
    agent, and everything after that is the general engine. Anyone replaying
    their own agent writes the equivalent for theirs -- twenty lines that say
    how to construct it -- rather than modifying anything here.

    The demo imports are deliberately local. The engine must not depend on the
    example, or "works for any agent" is a claim the module layout contradicts.
    """
    from flightrec.demo.agent import ResearchAgent
    from flightrec.demo.tools import FaultConfig, ToolError
    from flightrec.retry import TransientError

    root = root_span(run)
    seed = int(root.attr("flightrec.seed", 0) or 0) if root else 0
    max_steps = int(root.attr("flightrec.max_steps", 12) or 12) if root else 12
    recorded = root.attr("flightrec.faults") if root else None
    known = FaultConfig().__dict__
    faults = (
        FaultConfig(**{k: float(v) for k, v in recorded.items() if k in known})
        if isinstance(recorded, dict)
        # An older recording, or an agent that never had faults configured.
        # Assume none rather than inventing a rate that would make live steps
        # behave unlike the run being continued.
        else FaultConfig()
    )
    settings = recorded_temperature(run) if temperature is None else temperature

    def run_agent(tracer: Tracer, task_text: str) -> Any:
        agent = ResearchAgent(
            seed=seed, temperature=settings, faults=faults, max_steps=max_steps
        )
        agent.tracer = tracer
        return agent.run(task_text)

    return replay(
        run,
        run_agent,
        from_step=from_step,
        strict=strict,
        task=task,
        # The demo's loop branches on whether a failure is retryable, so the
        # types have to come back as themselves rather than as ReplayedError.
        exceptions={"ToolError": ToolError, "TransientError": TransientError},
        edits={"temperature": temperature} if temperature is not None else {},
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


def root_span(run: Run) -> Span | None:
    for span in run.ordered_spans():
        if span.kind is SpanKind.AGENT:
            return span
    return None


def _task_of(run: Run) -> str | None:
    root = root_span(run)
    value = root.attr(FR_INPUT) if root else None
    return value if isinstance(value, str) else None


def recorded_temperature(run: Run) -> float:
    """The sampling temperature the recording was made at, for forcing on replay."""
    for span in run.ordered_spans():
        value = span.attr(GEN_AI_REQUEST_TEMPERATURE)
        if value is not None:
            return float(value)
    return 0.0


def _start_time(run: Run) -> float:
    ordered = run.ordered_spans()
    return ordered[0].start_time if ordered else 0.0
