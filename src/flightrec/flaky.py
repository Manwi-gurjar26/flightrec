"""Which steps is the agent getting right by luck?

Run the same task many times and most steps do the same thing every time. A few
do not, and those are the ones a prompt is handling by accident: the search that
sometimes comes back empty, the fetch that sometimes serves last year's page. A
single trace cannot show you that, because a single trace is one sample.

The hard part is not the counting, it is knowing which steps to count together.
Two runs of the same task do not line up by index -- one retried, one guessed a
second URL -- so "step 5" means different things in different runs, and a report
built on position would blame whichever step happened to sit where the trouble
was. So this aligns every run against a reference first, using the same machinery
as the diff, and groups steps by what they *correspond to* rather than where they
sit.

What comes out is a disagreement rate per step: the fraction of runs where that
step did something other than its usual thing. A step at 0% is settled. A step at
40% is a coin toss the agent is winning most of the time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from flightrec.diff import diff_runs
from flightrec.spans import FR_OUTPUT, Run, Span, SpanStatus, stable_key


@dataclass
class StepVariance:
    """How settled one step of a task is, across many runs of it."""

    position: int
    name: str
    runs_present: int
    absent: int
    outcomes: Counter = field(default_factory=Counter)
    errors: int = 0

    @property
    def distinct(self) -> int:
        return len(self.outcomes)

    @property
    def disagreement(self) -> float:
        """Fraction of runs where this step did something other than its usual thing.

        Deliberately not entropy. "Three runs in ten disagreed" is a sentence
        somebody can act on; 1.48 bits is not, and the ranking comes out much the
        same either way.
        """
        if not self.runs_present:
            return 0.0
        commonest = self.outcomes.most_common(1)[0][1]
        return (self.runs_present - commonest) / self.runs_present

    @property
    def error_rate(self) -> float:
        return self.errors / self.runs_present if self.runs_present else 0.0

    @property
    def instability(self) -> float:
        """What the report sorts on: disagreement, plus how often the step vanished.

        A step that is missing from half the runs is not stable just because it
        agreed with itself whenever it did appear -- its *presence* is the coin
        toss. Counting only the runs it showed up in would rank it as settled.
        """
        total = self.runs_present + self.absent
        return self.disagreement + (self.absent / total if total else 0.0)

    def line(self) -> str:
        summary = ", ".join(
            f"{count}x {_short(value)}" for value, count in self.outcomes.most_common(3)
        )
        return (
            f"[{self.position:>2}] {self.name:<18} "
            f"disagree {self.disagreement * 100:>5.1f}%  "
            f"absent {self.absent:>2}  "
            f"errors {self.error_rate * 100:>5.1f}%   {summary}"
        )


def _short(text: str, width: int = 28) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _outcome(span: Span) -> str:
    """What a step 'did', for the purpose of asking whether it does it every time.

    The error message counts, so two different failures are two outcomes rather
    than one. Duration and tokens do not: a step that returns the same thing more
    slowly is not flaky, it is slow, and conflating those would fill the report
    with noise.
    """
    if span.status is SpanStatus.ERROR:
        return span.status_message or "error"
    return stable_key(span.attr(FR_OUTPUT))


def reference_run(runs: list[Run]) -> Run:
    """The run to align the others against: the most typical shape available.

    The commonest trajectory length, and within that the first. Picking run zero
    would work until run zero happened to be the weird one, and then every step
    of every other run would be reported as disagreeing with it.
    """
    lengths = Counter(len(run.steps()) for run in runs)
    typical = lengths.most_common(1)[0][0]
    return next(run for run in runs if len(run.steps()) == typical)


def flaky_report(runs: list[Run], reference: Run | None = None) -> list[StepVariance]:
    """Rank the steps of a task by how much they vary across runs of it."""
    if not runs:
        return []
    reference = reference or reference_run(runs)
    steps = reference.steps()
    report = [
        StepVariance(position=index, name=span.name, runs_present=0, absent=0)
        for index, span in enumerate(steps)
    ]

    for run in runs:
        if run.run_id == reference.run_id:
            seen = {index: span for index, span in enumerate(steps)}
        else:
            # The alignment is the whole reason this is trustworthy: it says
            # which step of this run corresponds to which step of the reference,
            # rather than assuming the fifth is the fifth.
            seen = {
                column.left_index: column.right
                for column in diff_runs(reference, run).columns
                if column.left_index is not None and column.right is not None
            }

        for entry in report:
            span = seen.get(entry.position)
            if span is None:
                entry.absent += 1
                continue
            entry.runs_present += 1
            entry.outcomes[_outcome(span)] += 1
            if span.status is SpanStatus.ERROR:
                entry.errors += 1

    return sorted(report, key=lambda entry: (-entry.instability, entry.position))
