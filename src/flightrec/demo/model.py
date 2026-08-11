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

CITIES = ["seattle", "portland"]

#: What the model "remembers" when it cannot fetch a real page. Plausible,
#: close to the truth, and wrong -- the ingredients of a bad recovery.
FABRICATED_PRIOR = {"seattle": 150, "portland": 140}

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


@dataclass
class _Progress:
    """What the model can infer about each city from the transcript so far."""

    searched: dict[str, list[str] | None] = field(default_factory=dict)
    fetched: dict[str, str] = field(default_factory=dict)
    fetch_failed: set[str] = field(default_factory=set)
    calculated: float | None = None


class StubModel:
    """A rule-based agent policy. See module docstring."""

    name = "stub-1"

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random(0)

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
                city = _city_of(str(args.get("query", "")))
                if city:
                    progress.searched[city] = list(content) if ok and content else []
            elif name == "fetch_page":
                city = _city_of(str(args.get("url", "")))
                if not city:
                    continue
                if ok:
                    progress.fetched[city] = str(content)
                else:
                    progress.fetch_failed.add(city)
            elif name == "calculator" and ok:
                progress.calculated = float(content)
        return progress

    # -- the policy -----------------------------------------------------------

    def _decide(self, progress: _Progress, temperature: float) -> ModelResponse:
        for city in CITIES:
            if city not in progress.searched:
                return ModelResponse(
                    text=f"I need the {city.title()} figure first. Searching.",
                    tool_call=ToolCall(
                        "web_search", {"query": self._phrase_query(city, temperature)}
                    ),
                )

            results = progress.searched[city] or []
            already_tried = city in progress.fetched or city in progress.fetch_failed
            if not already_tried:
                if results:
                    url = results[0]
                else:
                    # The recovery. Search came back empty, so the model guesses
                    # a URL that looks right and is not. It does not flag any
                    # uncertainty, because models generally do not.
                    url = f"https://weather.example/{city}-climate"
                    return ModelResponse(
                        text=(
                            f"Search returned nothing for {city.title()}. "
                            f"The site's URL scheme is predictable, so I'll go direct."
                        ),
                        tool_call=ToolCall("fetch_page", {"url": url}),
                    )
                return ModelResponse(
                    text=f"Found a source for {city.title()}. Fetching it.",
                    tool_call=ToolCall("fetch_page", {"url": url}),
                )

        numbers, confabulated = self._numbers(progress)

        if progress.calculated is None:
            expression = f"{numbers[CITIES[0]]} + {numbers[CITIES[1]]}"
            return ModelResponse(
                text=f"I have both figures. Adding them: {expression}.",
                tool_call=ToolCall("calculator", {"expression": expression}),
                confabulated=confabulated,
            )

        total = int(progress.calculated)
        return ModelResponse(
            text=(
                f"{CITIES[0].title()} had {numbers[CITIES[0]]} days with measurable "
                f"precipitation and {CITIES[1].title()} had {numbers[CITIES[1]]}, "
                f"for a combined total of {total} days."
            ),
            finish_reason="stop",
            confabulated=confabulated,
        )

    def _numbers(self, progress: _Progress) -> tuple[dict[str, int], bool]:
        """Extract each city's figure, inventing one where the page is missing."""
        numbers: dict[str, int] = {}
        confabulated = False
        for city in CITIES:
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


def _city_of(text: str) -> str | None:
    lowered = text.lower()
    for city in CITIES:
        if city in lowered:
            return city
    return None


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
