"""Data set apart inside text a model reads.

A tool's result, a worker's output or a retrieved knowledge passage is
external data. Framed in a tagged
block, it must not be able to end the block early: what followed would read
as the text around it, prompt or instruction.
"""

from __future__ import annotations

import re


def fence(tag: str, text: str) -> str:
    """*text* inside ``<tag>`` … ``</tag>``, with no closing tag of its own.

    Any closing tag of that name in *text*, in any case, with any spacing or
    trailing attributes (``</TOOL_RESULT >``, ``< / tool_result>``,
    ``</tool_result foo>``, ``</tool_result/>``), is neutralised, so the data
    cannot close the block.
    """
    closing = re.compile(rf"<\s*/\s*{re.escape(tag)}\b[^>]*>", re.IGNORECASE)
    neutral = f"</{tag}_>"
    body = closing.sub(lambda _match: neutral, text)
    return f"<{tag}>\n{body}\n</{tag}>"


# The tags RoomKit fences external data in: a tool's result, a worker's output,
# a knowledge passage, a memory's summary of the conversation.
FENCED_TAGS = ("tool_result", "worker_output", "knowledge", "conversation_summary")


def named_blocks(text: str, tags: tuple[str, ...] = FENCED_TAGS) -> str:
    """*text* with each block of *tags* replaced by its tag in brackets
    (``[tool_result]``), a block cut off before its end included.

    For text about to be cut short, such as a summary: quoting part of a block
    could leave it open, and what follows would then read as data.
    """
    for tag in tags:
        name = re.escape(tag)
        block = re.compile(
            rf"<\s*{name}\b[^>]*>.*?(?:<\s*/\s*{name}\b[^>]*>|\Z)", re.IGNORECASE | re.DOTALL
        )
        text = block.sub(f"[{tag}]", text)
    return text
