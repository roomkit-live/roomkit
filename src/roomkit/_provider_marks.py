"""The marks a realtime provider writes into its model's input (RFC §6.4).

A leaf, so that the provider that writes them and the cleaning that replaces
a copy of them (:mod:`roomkit.channels._mark_copies`) import them without
importing each other.
"""

from __future__ import annotations

SAID_BEFORE_MARK = "[Assistant previously said]"
"""How a provider that takes no assistant turn as text (Gemini Live) marks an
injection that stands for one."""

CONTEXT_UPDATE_MARK = "[Context update, do not respond to this"
"""How a provider that cannot add text without a reply (Gemini Live) opens a
silent injection; it ends ``]`` after the text, ``image]`` after an image."""
