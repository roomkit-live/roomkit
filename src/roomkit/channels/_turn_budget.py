"""What a turn may spend (RFC §6.4).

A token budget over every token the provider bills for the turn, a cost
budget at the model's catalogue price, or both. The tool loop counts each
generation's usage as it comes and stops at the first round boundary where
the turn has reached either.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from roomkit.providers.ai.base import BILLED_USAGE_COUNTERS, AIProvider, ModelPricing

logger = logging.getLogger("roomkit.channels.ai")


@dataclass(frozen=True)
class TurnBudget:
    """A turn's spending cap: billed tokens, a cost in the price's currency, or both."""

    tokens: int | None = None
    usd: float | None = None
    pricing: ModelPricing | None = None

    def tokens_of(self, usage: Mapping[str, Any]) -> int:
        """The tokens one generation's *usage* bills."""
        return sum(int(usage.get(counter) or 0) for counter in BILLED_USAGE_COUNTERS)

    def cost_of(self, usage: Mapping[str, Any]) -> float:
        """What one generation's *usage* costs, priced as one response."""
        return self.pricing.cost_for(usage) if self.pricing is not None else 0.0

    def reached(self, tokens: int, cost: float) -> bool:
        """Whether a turn that billed *tokens* and cost *cost* so far has reached it."""
        if self.tokens is not None and tokens >= self.tokens:
            return True
        return self.usd is not None and cost >= self.usd


def turn_budget(
    tokens: int | None,
    usd: float | None,
    provider: AIProvider | None,
) -> TurnBudget | None:
    """The budget *tokens* and *usd* set for a turn on *provider*, or ``None``
    when neither is set.

    Every generation of the turn is priced at *provider*'s rate, one a router
    hands to another provider included (RFC §6.4). Without a *provider* (a
    channel whose router picks it per turn), only the numbers are checked.

    Raises:
        ValueError: a budget that is not a positive number, or *usd* set for a
            model with no known price, whose cost cannot be counted.
    """
    if tokens is None and usd is None:
        return None
    _check_positive("turn_budget_tokens", tokens, int)
    _check_positive("turn_budget_usd", usd, (int, float))
    if provider is None:
        return TurnBudget(tokens=tokens, usd=usd)
    pricing = _pricing(provider)
    if usd is not None and pricing is None:
        raise ValueError(
            f"turn_budget_usd needs a model with a known price; {provider.model_name!r} "
            "has none in the catalogue. Set turn_budget_tokens instead."
        )
    return TurnBudget(tokens=tokens, usd=usd, pricing=pricing)


def _check_positive(name: str, value: object, kinds: type | tuple[type, ...]) -> None:
    """Raise ``ValueError`` unless *value* is ``None`` or a positive number of *kinds*."""
    if value is None:
        return
    positive = (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isinstance(value, kinds)
        and value > 0
    )
    if not positive:
        raise ValueError(f"{name} must be a positive number, got {value!r}")


def _pricing(provider: AIProvider) -> ModelPricing | None:
    """*provider*'s catalogue price, if it has one."""
    entry = provider.catalog_entry()
    return entry.pricing if entry is not None else None


# The models already warned about: once each.
_WARNED: set[str] = set()


def warn_if_priced_otherwise(budget: TurnBudget | None, other: AIProvider) -> None:
    """Log once that a cost *budget*, priced at the rate of the model its
    turn started on, prices *other*'s generations at that rate when *other*
    is priced otherwise."""
    if budget is None or budget.usd is None:
        return
    if _pricing(other) == budget.pricing or other.model_name in _WARNED:
        return
    _WARNED.add(other.model_name)
    logger.warning(
        "turn_budget_usd prices every generation of a turn at the rate of the model it "
        "started on; %s, which the router handed a generation, is priced otherwise",
        other.model_name,
    )
