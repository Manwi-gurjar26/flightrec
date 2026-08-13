"""A tiny fake web, plus three tools over it, plus seeded fault injection.

The faults are the point. Each one is named, deliberately chosen to mirror a
failure that really happens to agents in production, and reproducible from a
seed:

``EMPTY_SEARCH``
    The search returns nothing. The agent does not crash -- it invents a
    plausible recovery and confidently produces a wrong answer. This is the
    canonical agent failure and the one the timeline has to make obvious.

``FLAKY_FETCH``
    A tool fails transiently and succeeds on retry. Nothing is wrong with the
    answer; the run just silently costs more and takes longer. This is the
    "why did this run cost 8x more than the last one?" case.

``STALE_PAGE``
    The tool succeeds and returns data that is subtly wrong (last year's
    figures). Every step looks green. Only the answer is wrong, which is the
    hardest class of failure to debug and the best argument for the diff view.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from flightrec.retry import TransientError

# --- the fake web ------------------------------------------------------------

#: Rainy days per city, as (2024, 2023). The second figure is what the
#: STALE_PAGE fault serves instead of the first.
#:
#: More cities than the demo task uses, so a run can be made as long as it needs
#: to be. Each city costs an agent four steps -- decide, search, decide, fetch --
#: and the benchmark leans on that to build corpora of very different lengths
#: from one unchanged agent.
CITY_DAYS: dict[str, tuple[int, int]] = {
    "seattle": (152, 149),
    "portland": (144, 138),
    "boston": (127, 131),
    "denver": (89, 94),
    "miami": (135, 128),
    "chicago": (125, 119),
    "austin": (88, 91),
    "phoenix": (36, 41),
    "juneau": (222, 218),
    "honolulu": (98, 104),
    "buffalo": (167, 171),
    "fargo": (101, 97),
}

#: The cities the demo task asks about unless told otherwise.
DEFAULT_CITIES = ["seattle", "portland"]


def _page(city: str, year: int, days: int) -> str:
    return (
        f"{city.title()} annual climate summary {year}. "
        f"Measurable precipitation was recorded on {days} days."
    )


PAGES: dict[str, str] = {}
SEARCH_INDEX: dict[str, list[str]] = {}
for _city, (_current, _stale) in CITY_DAYS.items():
    PAGES[f"https://weather.example/{_city}-2024"] = _page(_city, 2024, _current)
    # Same page, previous year. Reads as perfectly valid -- nothing downstream
    # can tell it is the wrong one, which is the point of the STALE_PAGE fault.
    PAGES[f"https://weather.example/{_city}-2023"] = _page(_city, 2023, _stale)
    SEARCH_INDEX[_city] = [f"https://weather.example/{_city}-2024"]


def ground_truth(cities: list[str] | None = None) -> int:
    """The total a correct run produces for a given set of cities."""
    return sum(CITY_DAYS[city][0] for city in (cities or DEFAULT_CITIES))


#: The answer a correct run of the default task produces: 152 + 144.
GROUND_TRUTH = ground_truth()


class Fault(str, Enum):
    NONE = "none"
    EMPTY_SEARCH = "empty_search"
    FLAKY_FETCH = "flaky_fetch"
    STALE_PAGE = "stale_page"


@dataclass
class FaultConfig:
    """Which faults are live, and how often they fire.

    Probabilities are resolved against an injected, seeded RNG -- so a run is
    "random" in the sense that matters for realism, and exactly reproducible in
    the sense that matters for measurement.
    """

    empty_search_rate: float = 0.0
    flaky_fetch_rate: float = 0.0
    stale_page_rate: float = 0.0

    @classmethod
    def realistic(cls) -> "FaultConfig":
        """The mix used for the benchmark corpus: mostly fine, sometimes not."""
        return cls(empty_search_rate=0.25, flaky_fetch_rate=0.35, stale_page_rate=0.15)


class ToolError(RuntimeError):
    """A permanent tool failure. Not worth retrying."""


class Toolbox:
    """The three tools, sharing one seeded RNG and one fault configuration.

    Tracks which faults actually fired so tests and the benchmark can assert on
    the cause of a failure rather than guessing from the symptom.
    """

    def __init__(self, rng: random.Random, faults: FaultConfig | None = None) -> None:
        self._rng = rng
        self._faults = faults or FaultConfig()
        self.fired: list[Fault] = []
        self._flaky_budget: dict[str, int] = {}

    # -- tool 1 ---------------------------------------------------------------

    def web_search(self, query: str) -> list[str]:
        """Return URLs matching a query. Sometimes returns nothing at all."""
        if self._rng.random() < self._faults.empty_search_rate:
            self.fired.append(Fault.EMPTY_SEARCH)
            return []

        needle = query.lower()
        hits: list[str] = []
        for keyword, urls in SEARCH_INDEX.items():
            if keyword in needle:
                hits.extend(urls)
        return hits

    # -- tool 2 ---------------------------------------------------------------

    def fetch_page(self, url: str) -> str:
        """Fetch a page. Flaky, and occasionally returns last year's data."""
        if self._rng.random() < self._faults.flaky_fetch_rate:
            remaining = self._flaky_budget.get(url)
            if remaining is None:
                remaining = 1
                self._flaky_budget[url] = remaining
            if remaining > 0:
                self._flaky_budget[url] = remaining - 1
                self.fired.append(Fault.FLAKY_FETCH)
                raise TransientError(f"upstream timeout fetching {url}")

        if url not in PAGES:
            raise ToolError(f"404: no such page {url}")

        if url.endswith("-2024") and self._rng.random() < self._faults.stale_page_rate:
            stale_url = url.replace("-2024", "-2023")
            if stale_url in PAGES:
                self.fired.append(Fault.STALE_PAGE)
                return PAGES[stale_url]

        return PAGES[url]

    # -- tool 3 ---------------------------------------------------------------

    def calculator(self, expression: str) -> float:
        """Evaluate a simple arithmetic expression.

        Deliberately strict. When the model builds a malformed expression -- a
        real and common failure -- this raises rather than guessing, so the
        error shows up on the timeline instead of becoming a silent zero.
        """
        allowed = set("0123456789+-*/.() ")
        if not expression or not set(expression) <= allowed:
            raise ToolError(f"cannot evaluate expression: {expression!r}")
        try:
            return float(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
        except (SyntaxError, ZeroDivisionError, TypeError, NameError) as exc:
            raise ToolError(f"cannot evaluate expression: {expression!r}") from exc
