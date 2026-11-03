"""Whether a transport may read an agent's response as it streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.core.mixins.helpers import _source_block_reason
from roomkit.core.visibility import visibility_allows
from roomkit.models.enums import Access, ChannelCategory

if TYPE_CHECKING:
    from roomkit.models.channel import ChannelBinding


def reads_the_stream(
    source: ChannelBinding | None, binding: ChannelBinding, scope: str | None
) -> bool:
    """Whether *binding*'s transport may read the response of the agent bound
    by *source* as it streams, the answer scoped by *scope*.

    Read when the stream starts, for its target, and again at each row's
    commit: once the agent may no longer write (RFC §7.5 rule 2), or the
    transport may no longer read it, the target reads nothing more of it.
    """
    if _source_block_reason(source) is not None or source is None:
        return False
    if binding.category != ChannelCategory.TRANSPORT:
        return False
    if binding.channel_id == source.channel_id:
        return False
    if binding.access in (Access.WRITE_ONLY, Access.NONE):
        return False
    # A live stream is a delivery: an INBOUND-only transport is not
    # delivered to (RFC §10.2 step 3d).
    if not binding.sends:
        return False
    # The rows are committed with the answer's scope intersected with the
    # agent's binding (RFC §7.3): the live pipe reaches no more.
    if scope is not None and not visibility_allows(scope, binding):
        return False
    return visibility_allows(source.visibility, binding)
