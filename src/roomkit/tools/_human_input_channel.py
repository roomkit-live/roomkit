"""The human-input tools one channel serves (RFC §9.3, §21.6).

A channel given a :class:`~roomkit.tools.human_input.HumanInputToolHandler`
(``human_input_handler=``) serves its tools itself, on every door it has: an
``AIChannel`` turn, a realtime voice session (the provider's call, a call
recovered from speech, a reasoning backend's), a conference. Each holds one
:class:`ChannelHumanInput`, which gives the tools the same rules wherever the
model calls them (a channel given none holds an empty one, which declares
and serves nothing):

* declared beside the host's tools and served before the host's handler;
* bounded by the handler's own ``timeout``, never by the channel's default
  call bound: a person takes the time they take;
* each request announced through ``ON_USER_INPUT_REQUIRED`` once the channel
  is registered with a kit, a BLOCK rejecting it, the request naming the
  channel type of the door it was asked on;
* the requests still open settled when the channel closes.
"""

from __future__ import annotations

import logging
from collections.abc import Container, Iterable
from typing import TYPE_CHECKING, Any

from roomkit.channels._served_tools import (
    dict_tool_name,
    refuse_given_twice,
    refuse_served_names,
)
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.tools.human_input import HumanInputToolHandler

if TYPE_CHECKING:
    from roomkit.models.enums import ChannelType
    from roomkit.providers.ai.base import AITool
    from roomkit.tools.human_input import OnInputRequiredCallback

logger = logging.getLogger("roomkit.tools.human_input")


def warn_plain_handler(tool_handler: object, channel_id: str) -> None:
    """Say so when a channel is given a ``HumanInputToolHandler`` as its plain
    ``tool_handler``: served as any host handler, it keeps none of its rules."""
    if isinstance(tool_handler, HumanInputToolHandler):
        logger.warning(
            "Channel %s serves a HumanInputToolHandler as its tool_handler: pass it as "
            "human_input_handler= so its own timeout, ON_USER_INPUT_REQUIRED and the "
            "channel's close apply to it",
            channel_id,
        )


class ChannelHumanInput:
    """The human-input tools of one channel object: what it declares, what it
    serves, and the scope of requests it owns."""

    def __init__(self, tools: HumanInputToolHandler | None, channel_type: ChannelType) -> None:
        self._tools = tools
        self._channel_type = channel_type
        # The token naming this channel object as the owner of its id's
        # requests, handed back on close: a channel displaced under the same
        # id and torn down later closes nothing its replacement holds.
        self._registration: int | None = None
        # Names already reported as served but never declared: a wiring
        # diagnostic, said once, not a per-turn event.
        self._warned_unoffered: set[str] = set()

    @property
    def given(self) -> bool:
        """Whether the channel was given a human-input handler."""
        return self._tools is not None

    @property
    def names(self) -> frozenset[str]:
        """Every tool name it serves: a call to one asks a person."""
        return frozenset(self._tools.tool_names) if self._tools is not None else frozenset()

    @property
    def definitions(self) -> list[AITool]:
        """The tools it declares to the model; a name it serves without a
        definition is declared by the host's tools."""
        return self._tools.tools if self._tools is not None else []

    @property
    def declared_names(self) -> frozenset[str]:
        """The names its definitions carry: no other tool may take one."""
        return frozenset(tool.name for tool in self.definitions)

    def refuse_collisions(self, served: Container[str], channel_id: str) -> None:
        """Refuse a definition given twice, or under a name the channel serves
        itself (*served*), as a host tool under it is refused (RFC §21.1)."""
        names = [tool.name for tool in self.definitions]
        refuse_given_twice(names, channel_id)
        refuse_served_names(names, served, channel_id)

    def warn_unoffered(self, offered: Iterable[AITool | dict[str, Any]], channel_id: str) -> None:
        """Say so, once per name, when a name it serves is in none of the
        tools the model is offered (*offered*, definitions or a realtime
        session's dicts): no definition of its own, and none among the host's
        tools. The model is never told the tool exists, so no person is ever
        asked, and nothing else says so."""
        names = {dict_tool_name(t) if isinstance(t, dict) else t.name for t in offered}
        missing = {name for name in self.names if name not in names}
        missing -= self._warned_unoffered
        if not missing:
            return
        self._warned_unoffered |= missing
        logger.warning(
            "Channel %s intercepts human-input tool(s) %s but never offers them to the "
            "model: declare them via HumanInputToolHandler(tool_definitions=...) or the "
            "channel's tools, or no human will ever be asked.",
            channel_id,
            sorted(missing),
        )

    def serves(self, name: str) -> bool:
        """Whether a call to *name* asks a person."""
        return name in self.names

    async def serve(self, name: str, arguments: dict[str, Any]) -> str:
        """The person's answer to the call, asked on this channel's door."""
        if self._tools is None:
            raise UnservedToolCallError(name)
        return await self._tools.ask(name, arguments, channel_type=self._channel_type)

    def register(self, channel_id: str, on_input_required: OnInputRequiredCallback) -> None:
        """Announce the requests of the channel *channel_id* through
        *on_input_required* (the kit's ``ON_USER_INPUT_REQUIRED`` hooks), this
        channel object owning them."""
        if self._tools is None:
            return
        handler = self._tools.handler
        self._registration = handler._set_on_input_required(channel_id, on_input_required)

    async def close(self, channel_id: str) -> None:
        """Settle the requests the channel still has open, and take no more
        until it registers again."""
        if self._tools is not None:
            await self._tools.handler.close(channel_id=channel_id, registration=self._registration)
