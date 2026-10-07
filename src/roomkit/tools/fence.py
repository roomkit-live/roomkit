"""Data set apart inside text a model reads (RFC §6.4).

A tool's result, a worker's output, a retrieved knowledge passage or a
memory's summary of the conversation is external data. Framed in a tagged
block, it must not be able to end the block early: what followed would read
as the text around it, prompt or instruction.

The public path of :func:`fence` and :func:`named_blocks`. They live in
:mod:`roomkit._text`, beside the inline quote that names a block rather than
cut one, which every module can import.
"""

from __future__ import annotations

from roomkit._text import FENCED_TAGS, fence, named_blocks

__all__ = ["FENCED_TAGS", "fence", "named_blocks"]
