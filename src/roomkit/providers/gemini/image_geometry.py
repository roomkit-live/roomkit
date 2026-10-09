"""Gemini image geometry: a portable ``"WIDTHxHEIGHT"`` size as a named aspect
ratio and a resolution tier, refused by the size's own name when the model
cannot produce it (RFC §25.2, RMK-655)."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from math import gcd

from roomkit.providers.image.base import parse_size

# Aspect ratios the Interactions image format accepts across Gemini's lineup.
# A requested size is reduced to its ratio and looked up; an unlisted one is
# refused rather than rounded to a neighbour, because a silently different
# geometry is a failure the caller can neither see nor correct (RFC §25.2).
ASPECT_RATIOS: tuple[str, ...] = (
    "1:1",
    "1:4",
    "1:8",
    "16:9",
    "2:3",
    "21:9",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
)

# Resolution tiers, keyed by the largest dimension the caller asked for. Google
# names tiers, not pixel counts, so the request is mapped to the smallest tier
# that covers it.
SIZE_TIERS: tuple[tuple[int, str], ...] = ((512, "512"), (1024, "1K"), (2048, "2K"), (4096, "4K"))
TIERS: tuple[str, ...] = tuple(tier for _, tier in SIZE_TIERS)


def reduce_size(size: str) -> tuple[str, str | None]:
    """A size as its reduced aspect ratio and the smallest tier covering it,
    ``None`` beyond the largest."""
    width, height = parse_size(size)
    divisor = gcd(width, height)
    largest = max(width, height)
    tier = next((name for ceiling, name in SIZE_TIERS if largest <= ceiling), None)
    return f"{width // divisor}:{height // divisor}", tier


def size_geometry(
    size: str, ratios: Collection[str], tiers: Sequence[str], owner: str
) -> tuple[str, str]:
    """The ratio and tier *owner* (a model, or Gemini's whole lineup) produces
    for *size*, refused by the size's own name when *owner* lacks either."""
    aspect_ratio, tier = reduce_size(size)
    if aspect_ratio not in ratios:
        raise ValueError(
            f"size {size!r} reduces to aspect ratio {aspect_ratio}, which {owner} "
            f"does not offer (its ratios: {', '.join(ratios) or 'none'})"
        )
    if tier is None or tier not in tiers:
        needed = f"Gemini's {tier} tier" if tier else "a tier beyond Gemini's largest tier (4K)"
        raise ValueError(
            f"size {size!r} needs {needed}, which {owner} does not offer "
            f"(its tiers: {', '.join(tiers) or 'none'})"
        )
    return aspect_ratio, tier
