"""Terminal rendering of a run.

A placeholder for the real timeline UI in step 5, but a useful one: if a run is
not legible as plain text, it will not become legible by adding CSS to it.
"""

from __future__ import annotations

from flightrec.demo.agent import TASK, AgentResult
from flightrec.demo.tools import GROUND_TRUTH
from flightrec.spans import FR_OUTPUT, SpanStatus


def format_run(result: AgentResult, seed: int, faults: str, temperature: float) -> str:
    lines = [
        f"task:  {TASK}",
        f"seed:  {seed}   faults: {faults}   temp: {temperature}",
        "",
    ]

    for step in result.run.steps():
        marker = "!" if step.status is SpanStatus.ERROR else " "
        retries = sum(1 for e in step.events if e.name == "retry")
        suffix = f"  ({retries} retr{'y' if retries == 1 else 'ies'})" if retries else ""
        output = str(step.attr(FR_OUTPUT) or step.status_message or "")
        lines.append(
            f" {marker} [{step.sequence:>2}] {step.name:<18} {output[:64]:<64}{suffix}"
        )

    lines += [
        "",
        f"answer:   {result.answer}   (correct answer: {GROUND_TRUTH})",
        f"verdict:  {'CORRECT' if result.correct else 'WRONG'}",
        f"tokens:   {result.run.total_tokens}",
        f"faults:   {[f.value for f in result.faults_fired] or 'none fired'}",
    ]
    if result.confabulated:
        lines.append("warning:  the agent invented data at least once")
    if result.error:
        lines.append(f"error:    {result.error}")
    return "\n".join(lines)
