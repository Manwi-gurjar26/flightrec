"""The demo agent loop, instrumented with the flightrec SDK.

This is what an ordinary agent looks like: call the model, run the tool it asks
for, append the observation, repeat until it answers or you hit the step cap.
It is written to be unremarkable on purpose -- the interesting part is that
every source of variation in it is injected, so the same seed produces the same
run every time.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from flightrec.demo.model import ModelClient, ModelResponse, StubModel, ToolCall
from flightrec.demo.tools import Fault, FaultConfig, GROUND_TRUTH, Toolbox, ToolError
from flightrec.determinism import Clock, IdGenerator, SystemClock, RandomIdGenerator
from flightrec.replay import ReplayedError
from flightrec.retry import TransientError, make_rng, retry_call
from flightrec.spans import (
    FR_OUTPUT,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_NAME,
    Run,
    SpanKind,
)
from flightrec.sinks import MemorySink, Sink
from flightrec.tracer import Tracer

TASK = (
    "How many days with measurable precipitation did Seattle and Portland "
    "record in 2024, combined?"
)

#: Marks the step where the agent stopped being grounded in tool output.
FR_CONFABULATED = "flightrec.confabulated"


@dataclass
class AgentResult:
    """The outcome of one run, alongside the recording of it."""

    run: Run
    answer: int | None
    text: str
    steps: int
    faults_fired: list[Fault]
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.answer == GROUND_TRUTH

    @property
    def confabulated(self) -> bool:
        return any(s.attr(FR_CONFABULATED) for s in self.run.spans)


class ResearchAgent:
    """A three-tool research agent that answers by searching, fetching and adding."""

    def __init__(
        self,
        *,
        seed: int = 0,
        temperature: float = 0.0,
        faults: FaultConfig | None = None,
        model: ModelClient | None = None,
        sink: Sink | None = None,
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
        trace_id: str | None = None,
        max_steps: int = 12,
    ) -> None:
        self.seed = seed
        self.temperature = temperature
        self.max_steps = max_steps

        # One seed, three independent streams. Sharing a single RNG between the
        # model and the tools would make a change in one silently shift the
        # other, which makes a bisect impossible.
        self._model_rng = make_rng(seed * 7919 + 1)
        self._tool_rng = make_rng(seed * 7919 + 2)
        self._retry_rng = make_rng(seed * 7919 + 3)

        self.model = model or StubModel(rng=self._model_rng)
        self.faults = faults or FaultConfig()
        self.tools = Toolbox(rng=self._tool_rng, faults=self.faults)
        self.tracer = Tracer(
            sink=sink or MemorySink(),
            clock=clock or SystemClock(),
            id_generator=id_generator or RandomIdGenerator(),
            trace_id=trace_id,
        )

    # -- the loop -------------------------------------------------------------

    def run(self, task: str = TASK) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        answer: int | None = None
        text = ""
        error: str | None = None
        steps = 0

        with self.tracer.span(
            "research_agent",
            kind=SpanKind.AGENT,
            inputs=task,
            **{
                "flightrec.seed": self.seed,
                "flightrec.max_steps": self.max_steps,
                # Recorded because replay may have to re-execute tools live past
                # the edit point, and a live step run under a different fault
                # configuration is not a continuation of this run.
                "flightrec.faults": asdict(self.faults),
            },
        ) as root:
            try:
                for step in range(self.max_steps):
                    steps = step + 1
                    response = self._call_model(messages)
                    messages.append(
                        {"role": "assistant", "content": response.text}
                    )

                    if response.tool_call is None:
                        text = response.text
                        answer = _extract_total(response.text)
                        break

                    observation = self._call_tool(
                        response.tool_call.name, response.tool_call.arguments
                    )
                    messages.append(observation)
                else:
                    error = f"step cap of {self.max_steps} reached without an answer"
            except Exception as exc:  # the agent gave up; still a recorded run
                error = f"{type(exc).__name__}: {exc}"

            self.tracer.set_output(root, {"answer": answer, "text": text})
            root.attributes["flightrec.correct"] = answer == GROUND_TRUTH

        return AgentResult(
            run=self._collect(),
            answer=answer,
            text=text,
            steps=steps,
            faults_fired=list(self.tools.fired),
            error=error,
        )

    # -- instrumented calls ---------------------------------------------------

    def _call_model(self, messages: list[dict[str, Any]]) -> ModelResponse:
        """One model call, routed through the tracer so it can be replayed.

        The step records its *text* as the output, because that is what a human
        reads on the timeline, and puts the structured part -- the tool call it
        asked for, the token counts -- on the span. ``restore`` rebuilds the
        response from those two halves, and runs whether the call happened or
        was served, so the loop below never learns which.
        """

        def complete() -> str:
            response = self.model.complete(messages, temperature=self.temperature)
            span = Tracer.current_span()
            if span is not None:
                self.tracer.record_usage(
                    span,
                    model=getattr(self.model, "name", None),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                span.attributes[GEN_AI_RESPONSE_FINISH_REASONS] = [
                    response.finish_reason
                ]
                if response.tool_call:
                    span.attributes["flightrec.tool_call"] = {
                        "name": response.tool_call.name,
                        "arguments": response.tool_call.arguments,
                    }
                if response.confabulated:
                    span.attributes[FR_CONFABULATED] = True
            return response.text

        return self.tracer.call(
            "chat",
            complete,
            kind=SpanKind.LLM,
            inputs=messages[-1].get("content"),
            restore=_response_from_span,
            **{
                GEN_AI_SYSTEM: "stub",
                GEN_AI_REQUEST_MODEL: getattr(self.model, "name", "unknown"),
                GEN_AI_REQUEST_TEMPERATURE: self.temperature,
            },
        )

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool, recording the result -- including failure -- as an observation.

        Tool failures are returned to the model rather than raised, because that
        is what real agent frameworks do and it is precisely how an agent talks
        itself into a wrong answer. The span is still marked as an error, so the
        timeline shows red even though the loop carried on.
        """
        try:
            result = self.tracer.call(
                "tool." + name,
                lambda: self.invoke_live(name, arguments, Tracer.current_span()),
                kind=SpanKind.TOOL,
                inputs=arguments,
                **{GEN_AI_TOOL_NAME: name},
            )
        except (ToolError, TransientError, ReplayedError) as exc:
            return {
                "role": "tool",
                "name": name,
                "arguments": arguments,
                "ok": False,
                "content": str(exc),
            }

        return {
            "role": "tool",
            "name": name,
            "arguments": arguments,
            "ok": True,
            "content": result,
        }

    def invoke_live(self, name: str, arguments: dict[str, Any], span: Any) -> Any:
        """Actually execute a tool. Never called for a step served from a recording."""
        if name == "web_search":
            return self.tools.web_search(str(arguments["query"]))
        if name == "calculator":
            return self.tools.calculator(str(arguments["expression"]))
        if name == "fetch_page":
            url = str(arguments["url"])
            return retry_call(
                lambda: self.tools.fetch_page(url),
                attempts=3,
                rng=self._retry_rng,
                span=span,
                now=self.tracer.clock.now,
            )
        raise ToolError(f"unknown tool {name!r}")

    def _collect(self) -> Run:
        if isinstance(self.tracer.sink, MemorySink):
            return self.tracer.collect()
        return Run(run_id=self.tracer.trace_id)


def _response_from_span(text: Any, span: Any) -> ModelResponse:
    """Rebuild a model response from what the span recorded about it.

    Used on both paths, which is the point: the agent loop gets a
    ``ModelResponse`` whether the model was called or the answer came out of a
    recording, and so contains no replay-specific code at all.
    """
    call = span.attr("flightrec.tool_call")
    reasons = span.attr(GEN_AI_RESPONSE_FINISH_REASONS) or ["stop"]
    return ModelResponse(
        text=str(text or ""),
        tool_call=ToolCall(str(call["name"]), dict(call["arguments"])) if call else None,
        input_tokens=span.input_tokens,
        output_tokens=span.output_tokens,
        finish_reason=str(reasons[0]),
        model=str(span.attr(GEN_AI_REQUEST_MODEL) or "unknown"),
        confabulated=bool(span.attr(FR_CONFABULATED)),
    )


def _extract_total(text: str) -> int | None:
    import re

    match = re.search(r"combined total of (\d+)", text)
    return int(match.group(1)) if match else None


def run_agent(
    seed: int = 0,
    temperature: float = 0.0,
    faults: FaultConfig | None = None,
    **kwargs: Any,
) -> AgentResult:
    """Convenience wrapper used by the CLI, tests and the benchmark."""
    return ResearchAgent(
        seed=seed, temperature=temperature, faults=faults, **kwargs
    ).run()
