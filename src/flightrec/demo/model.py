"""A deterministic stand-in for a model provider.

Real providers are unreproducible, cost money, and require a key -- all three
of which would make the numbers in this README impossible for anyone else to
check. So the default model is a rule-based policy that behaves like an agent
loop: it reads the observations so far and decides the next tool call.

Two properties matter:

* **At temperature 0 it is a pure function of the conversation.** Same inputs,
  same decision, every time.
* **Above temperature 0 it genuinely varies**, using an injected RNG. This is
  not decoration -- it is what makes "pin the temperature before replaying" a
  requirement you can demonstrate rather than assert.

The important behaviour is what it does when a tool lets it down: it does not
stop and it does not crash. It invents a plausible recovery and keeps going,
which is exactly how real agents produce confidently wrong answers.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from flightrec.demo.tools import CITY_DAYS, DEFAULT_CITIES

CITIES = list(DEFAULT_CITIES)

#: What the model "remembers" when it cannot fetch a real page. Plausible,
#: close to the truth, and wrong -- the ingredients of a bad recovery.
#:
#: Derived from the real figure so it stays plausible for any city the task
#: happens to name: a few days out, never suspiciously exact.
#:
#: Always *under*, never alternating. The first version of this offset was
#: +2/-2 by position, which cancels: for an even number of cities the invented
#: total came out exactly equal to the truth, and a demo whose confabulated run
#: gets the right answer has no failure left to demonstrate. A one-sided offset
#: cannot sum back to the real figure however many cities are named.
FABRICATED_PRIOR = {
    city: current - 2 - (index % 3)
    for index, (city, (current, _)) in enumerate(CITY_DAYS.items())
}

_DAYS = re.compile(r"recorded on (\d+) days")


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    text: str
    tool_call: ToolCall | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    model: str = "stub-1"
    confabulated: bool = False
    """True when this response invented data instead of using a tool result.

    Recorded on the span so the timeline can mark the exact step where the
    agent stopped being grounded. Real systems do not hand you this flag; the
    demo does, so that the tool's ability to *find* the step can be scored
    against a known answer.
    """


class ModelClient(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], temperature: float = 0.0
    ) -> ModelResponse: ...


#: How many URLs the model will try for one city before giving up on it.
#:
#: This is what makes trajectories vary in length, and it is the whole reason
#: the diff needs sequence alignment rather than ``zip()``. With a fixed policy
#: every run is the same length, index-by-index pairing works by accident, and
#: the alignment cannot be shown to earn its complexity -- which is exactly the
#: state this corpus was in until it was measured.
MAX_FETCH_ATTEMPTS = 2


@dataclass
class _Progress:
    """What the model can infer about each city from the transcript so far."""

    searched: dict[str, list[str] | None] = field(default_factory=dict)
    fetched: dict[str, str] = field(default_factory=dict)
    fetch_attempts: dict[str, list[str]] = field(default_factory=dict)
    calculated: float | None = None

    def attempts(self, city: str) -> list[str]:
        return self.fetch_attempts.get(city, [])


class StubModel:
    """A rule-based agent policy. See module docstring."""

    name = "stub-1"

    def __init__(
        self, rng: random.Random | None = None, cities: list[str] | None = None
    ) -> None:
        self._rng = rng or random.Random(0)
        #: Which cities this task is about. Per-instance rather than a module
        #: constant so one agent can produce runs of very different lengths --
        #: four steps per city -- without a second agent to maintain.
        self.cities = list(cities) if cities else list(CITIES)

    # -- public API -----------------------------------------------------------

    def complete(
        self, messages: list[dict[str, Any]], temperature: float = 0.0
    ) -> ModelResponse:
        progress = self._read_transcript(messages)
        response = self._decide(progress, temperature)
        response.input_tokens = _count_tokens(messages)
        response.output_tokens = _estimate_output_tokens(response)
        return response

    # -- reading the transcript ----------------------------------------------

    def _city_of(self, text: str) -> str | None:
        lowered = text.lower()
        for city in self.cities:
            if city in lowered:
                return city
        return None

    def _read_transcript(self, messages: list[dict[str, Any]]) -> _Progress:
        progress = _Progress()
        for message in messages:
            if message.get("role") != "tool":
                continue
            name = message.get("name")
            args = message.get("arguments") or {}
            ok = bool(message.get("ok"))
            content = message.get("content")

            if name == "web_search":
                city = self._city_of(str(args.get("query", "")))
                if city:
                    progress.searched[city] = list(content) if ok and content else []
            elif name == "fetch_page":
                url = str(args.get("url", ""))
                city = self._city_of(url)
                if not city:
                    continue
                progress.fetch_attempts.setdefault(city, []).append(url)
                if ok:
                    progress.fetched[city] = str(content)
            elif name == "calculator" and ok:
                progress.calculated = float(content)
        return progress

    # -- the policy -----------------------------------------------------------

    def _decide(self, progress: _Progress, temperature: float) -> ModelResponse:
        for city in self.cities:
            if city not in progress.searched:
                return ModelResponse(
                    text=f"I need the {city.title()} figure first. Searching.",
                    tool_call=ToolCall(
                        "web_search", {"query": self._phrase_query(city, temperature)}
                    ),
                )

            if city in progress.fetched:
                continue

            attempts = progress.attempts(city)
            if len(attempts) >= MAX_FETCH_ATTEMPTS:
                continue  # out of ideas for this city; it will get invented later

            results = progress.searched[city] or []
            candidates = [u for u in _candidate_urls(city, results) if u not in attempts]
            if not candidates:
                continue

            if not attempts:
                if results:
                    return ModelResponse(
                        text=f"Found a source for {city.title()}. Fetching it.",
                        tool_call=ToolCall("fetch_page", {"url": candidates[0]}),
                    )
                # The recovery. Search came back empty, so the model guesses a
                # URL that looks right and is not. It does not flag any
                # uncertainty, because models generally do not.
                return ModelResponse(
                    text=(
                        f"Search returned nothing for {city.title()}. "
                        f"The site's URL scheme is predictable, so I'll go direct."
                    ),
                    tool_call=ToolCall("fetch_page", {"url": candidates[0]}),
                )

            # The second guess. Same confidence, same absence of doubt, two more
            # steps on the bill -- and it is where a run stops being the same
            # length as its neighbours.
            return ModelResponse(
                text=(
                    f"That URL 404'd for {city.title()}. This site uses a couple of "
                    f"different schemes; trying the other one."
                ),
                tool_call=ToolCall("fetch_page", {"url": candidates[0]}),
            )

        numbers, confabulated = self._numbers(progress)

        if progress.calculated is None:
            expression = " + ".join(str(numbers[city]) for city in self.cities)
            return ModelResponse(
                text=f"I have every figure. Adding them: {expression}.",
                tool_call=ToolCall("calculator", {"expression": expression}),
                confabulated=confabulated,
            )

        total = int(progress.calculated)
        summary = ", ".join(
            f"{city.title()} had {numbers[city]}" for city in self.cities
        )
        return ModelResponse(
            text=(
                f"{summary} days with measurable precipitation, "
                f"for a combined total of {total} days."
            ),
            finish_reason="stop",
            confabulated=confabulated,
        )

    def _numbers(self, progress: _Progress) -> tuple[dict[str, int], bool]:
        """Extract each city's figure, inventing one where the page is missing."""
        numbers: dict[str, int] = {}
        confabulated = False
        for city in self.cities:
            page = progress.fetched.get(city)
            match = _DAYS.search(page) if page else None
            if match:
                numbers[city] = int(match.group(1))
            else:
                numbers[city] = FABRICATED_PRIOR[city]
                confabulated = True
        return numbers, confabulated

    def _phrase_query(self, city: str, temperature: float) -> str:
        """Query wording. Stable at temperature 0, variable above it.

        The alternate phrasings all work, so this changes the trajectory without
        changing the answer -- the cosmetic divergence that run diffing has to
        avoid reporting as the cause of a failure.
        """
        default = f"{city} rainy days 2024"
        if temperature <= 0:
            return default
        if self._rng.random() < min(temperature, 1.0) * 0.5:
            return self._rng.choice(
                [
                    f"{city} 2024 precipitation days",
                    f"how many rainy days {city} 2024",
                    f"{city} annual rainfall days 2024",
                ]
            )
        return default


def _candidate_urls(city: str, results: list[str]) -> list[str]:
    """URLs the model will try for a city, best first.

    The two guesses are both plausible and both wrong. Letting the second one
    succeed would be a nicer story and would delete the demo's most important
    failure: an empty search leading to invented data. So the recovery costs two
    extra steps and still ends in a confabulation, which is what guessing at a
    site's URL scheme usually gets you.
    """
    return list(results) + [
        f"https://weather.example/{city}-climate",
        f"https://weather.example/{city}-annual",
    ]


def _count_tokens(messages: list[dict[str, Any]]) -> int:
    """A crude but deterministic token estimate: roughly four characters each.

    Good enough for cost rollups in a demo, and honest about being an estimate.
    A real integration would use the provider's reported usage.
    """
    total = sum(len(str(m.get("content", ""))) for m in messages)
    return max(1, total // 4)


def _estimate_output_tokens(response: ModelResponse) -> int:
    size = len(response.text)
    if response.tool_call:
        size += len(str(response.tool_call.arguments)) + len(response.tool_call.name)
    return max(1, size // 4)
