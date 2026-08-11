"""Turning token counts into money.

Two decisions here are worth defending.

**Cost is computed and stored when the span is recorded, not when it is read.**
Prices change. If cost were derived at read time from the current table, editing
that table would silently rewrite the cost of every run you had already
recorded, and last month's numbers would quietly stop matching last month's
invoice. A recorded run is a historical fact. So the rate used is written onto
the span alongside the money, and the run stays auditable after the table moves
on.

**An unknown model is never priced at zero.** Reporting `$0.00` for a model
missing from the table is not a small error, it is a confident wrong answer of
exactly the kind this whole project exists to catch. Unpriced calls are counted
separately and surfaced, so a total is either complete or visibly incomplete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from flightrec.spans import (
    FR_COST_USD,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    Span,
)

#: Written onto priced spans so a run can be audited against the rate that was
#: actually applied, rather than whatever the table says today.
FR_PRICE_INPUT = "flightrec.price.input_usd_per_mtok"
FR_PRICE_OUTPUT = "flightrec.price.output_usd_per_mtok"
FR_PRICE_SOURCE = "flightrec.price.source"
FR_UNPRICED = "flightrec.price.unknown_model"


@dataclass(frozen=True)
class ModelPrice:
    """US dollars per million tokens."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_mtok
            + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


#: The demo's stub model. Priced like a mid-tier hosted model purely so the
#: example numbers land in a realistic range -- it costs nothing to run.
STUB_PRICE = ModelPrice(input_usd_per_mtok=3.00, output_usd_per_mtok=15.00)

#: Illustrative entries only. **Check these against your provider's current
#: pricing page before quoting any figure they produce.** They are here so the
#: table has a shape to copy, not as a source of truth -- provider prices change
#: and this file does not. Override with `PriceTable.from_json(...)`.
BUILTIN_PRICES: dict[str, ModelPrice] = {
    "stub-1": STUB_PRICE,
    "example-small": ModelPrice(0.25, 1.25),
    "example-medium": ModelPrice(3.00, 15.00),
    "example-large": ModelPrice(15.00, 75.00),
}


class PriceTable:
    """Looks up a price for a model name.

    Matching is exact first, then longest-prefix, so a dated model id such as
    ``example-medium-20260101`` resolves against the ``example-medium`` entry
    without needing a row per release.
    """

    def __init__(
        self, prices: dict[str, ModelPrice] | None = None, source: str = "builtin"
    ) -> None:
        self.prices = dict(prices if prices is not None else BUILTIN_PRICES)
        self.source = source

    @classmethod
    def from_json(cls, path: str | Path) -> "PriceTable":
        """Load a table of ``{"model": {"input": x, "output": y}}`` per Mtok."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            {
                name: ModelPrice(float(row["input"]), float(row["output"]))
                for name, row in data.items()
            },
            source=str(path),
        )

    def lookup(self, model: str | None) -> ModelPrice | None:
        if not model:
            return None
        if model in self.prices:
            return self.prices[model]
        candidates = [name for name in self.prices if model.startswith(name)]
        if not candidates:
            return None
        return self.prices[max(candidates, key=len)]

    def cost_for(
        self, model: str | None, input_tokens: int, output_tokens: int
    ) -> float | None:
        """Cost in USD, or ``None`` when the model is not in the table."""
        price = self.lookup(model)
        if price is None:
            return None
        return price.cost(input_tokens, output_tokens)

    def apply(
        self, span: Span, model: str | None, input_tokens: int, output_tokens: int
    ) -> None:
        """Write usage and cost onto a span, recording the rate applied."""
        span.attributes[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens

        price = self.lookup(model)
        if price is None:
            # Marked, not zeroed. A total that quietly omits these would be
            # wrong in the one direction that matters.
            span.attributes[FR_UNPRICED] = True
            return

        span.attributes[FR_COST_USD] = price.cost(input_tokens, output_tokens)
        span.attributes[FR_PRICE_INPUT] = price.input_usd_per_mtok
        span.attributes[FR_PRICE_OUTPUT] = price.output_usd_per_mtok
        span.attributes[FR_PRICE_SOURCE] = self.source


def format_usd(amount: float) -> str:
    """Money, at a precision that does not round a real cost away to nothing."""
    if amount == 0:
        return "$0"
    if amount < 0.01:
        return f"${amount:.6f}".rstrip("0")
    if amount < 1:
        return f"${amount:.4f}"
    return f"${amount:,.2f}"
