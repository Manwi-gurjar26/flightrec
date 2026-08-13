"""Record and replay an agent flightrec has never seen.

Run it:

    python examples/replay_your_own_agent.py

There is no flightrec-specific code in the agent below beyond two calls to
``tracer.call``. That is the whole integration: wrap the two things you would
ever want served from a recording -- the model call and the tool call -- and the
replay engine can drive the rest.
"""

from __future__ import annotations

from flightrec.replay import replay
from flightrec.sinks import MemorySink
from flightrec.spans import SpanKind
from flightrec.tracer import Tracer

PRICES = {"widget": 4, "gasket": 11}


class PricingError(RuntimeError):
    """This agent's own failure type."""


class QuoteAgent:
    """Totals up a list of parts, asking a "model" what to do at each step."""

    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self.looked_up: list[str] = []

    # -- the two things worth replaying ---------------------------------------

    def decide(self, part: str) -> str:
        """A model call. Wrapped in ``call`` so a replay can answer it."""
        return self.tracer.call(
            "chat",
            lambda: f"Looking up the price of {part}.",
            kind=SpanKind.LLM,
            inputs=part,
        )

    def price_of(self, part: str) -> int:
        """A tool call. Same wrapper, same reason."""

        def lookup() -> int:
            self.looked_up.append(part)
            if part not in PRICES:
                raise PricingError(f"no price for {part}")
            return PRICES[part]

        return self.tracer.call(
            "tool.price_of", lookup, kind=SpanKind.TOOL, inputs={"part": part}
        )

    # -- an ordinary agent loop ------------------------------------------------

    def run(self, task: str) -> str:
        with self.tracer.span("quote_agent", kind=SpanKind.AGENT, inputs=task):
            total = 0
            notes = []
            for part in task.split():
                notes.append(self.decide(part))
                try:
                    total += self.price_of(part)
                except PricingError as exc:
                    notes.append(f"skipped: {exc}")
            return f"total {total} ({'; '.join(notes)})"


def record(task: str) -> tuple:
    tracer = Tracer(sink=MemorySink(), trace_id="quote-1")
    agent = QuoteAgent(tracer)
    answer = agent.run(task)
    return tracer.collect(), answer, agent


def main() -> None:
    task = "widget gasket sprocket"

    recording, answer, live_agent = record(task)
    print(f"recorded : {answer}")
    print(f"           tools really called: {live_agent.looked_up}")

    # The engine needs one thing from you: how to build and run your agent
    # against the tracer it hands over. That tracer is already pinned to a
    # virtual clock and seeded IDs, and already knows how to answer from the
    # recording.
    replayed_agents = []

    def run_agent(tracer: Tracer, task_text: str) -> str:
        agent = QuoteAgent(tracer)
        replayed_agents.append(agent)
        return agent.run(task_text)

    # ``exceptions`` matters as soon as your agent catches its own failure
    # types, as this one does. A recording stores an exception's *name and
    # message*, not its class, so without this the replayed PricingError is a
    # stand-in class of the same name -- it records identically, but
    # ``except PricingError`` in the agent will not catch it and the run dies
    # somewhere it never died originally.
    known = {"PricingError": PricingError}

    full = replay(recording, run_agent, exceptions=known)
    print(f"\nreplayed : {full.outcome}")
    print(f"           faithful: {full.faithful}")
    print(f"           served {full.served} steps, executed {full.live}")
    print(f"           tools really called: {replayed_agents[-1].looked_up}")

    # Past the edit point it is genuinely running again -- with a different
    # price list, to prove the steps after the cut are live.
    PRICES["gasket"] = 99
    forked = replay(recording, run_agent, from_step=2, exceptions=known)
    print(f"\nfrom step 2: {forked.outcome}")
    print(f"           served {forked.served}, executed {forked.live}")
    print(f"           tools really called: {replayed_agents[-1].looked_up}")
    print(f"           first divergence at step {forked.divergence_step}")


if __name__ == "__main__":
    main()
