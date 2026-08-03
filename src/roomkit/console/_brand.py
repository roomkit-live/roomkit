"""RoomKit brand palette and logo shared by the console surfaces.

Colors come from the roomkit.live website. Both the voice dashboard
(:mod:`roomkit.console._display`) and the CLI chat presenter
(:mod:`roomkit.console._chat`) render from this single source.
"""

from __future__ import annotations

from rich.text import Text

PRIMARY = "rgb(99,102,241)"
PRIMARY_LIGHT = "rgb(129,140,248)"
PRIMARY_DIM = "rgb(55,58,130)"
ACCENT = "rgb(6,182,212)"
MUTED = "rgb(100,116,139)"
SURFACE = "rgb(38,36,63)"
"""Subtle raised-background tint (user-line echo in the console shell)."""


def logo_lines() -> tuple[Text, Text]:
    """The two rows of the 2x2 block logo."""
    top = Text()
    top.append("██", style=PRIMARY)
    top.append(" ", style="")
    top.append("██", style=PRIMARY_LIGHT)

    bottom = Text()
    bottom.append("██", style=PRIMARY_LIGHT)
    bottom.append(" ", style="")
    bottom.append("██", style=PRIMARY_DIM)

    return top, bottom
