"""Discussion reads a live ACP channel's local attributes before installation."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit import ACPChannel
from roomkit.orchestration.strategies.discussion import Discussion


@pytest.mark.parametrize(
    ("channel_id", "given", "expected"),
    [("agent:7c1e", None, "Developer"), ("dev", None, "dev"), ("agent:7c1e", "coder", "coder")],
)
async def test_a_live_acp_agent_is_named_before_discussion_installation(
    tmp_path: Path, channel_id: str, given: str | None, expected: str
) -> None:
    channel = ACPChannel(channel_id, ["agent-not-started"], cwd=tmp_path, name="Developer")
    discussion = Discussion([channel], handles={channel_id: given} if given else None)
    assert discussion.handles() == {channel_id: expected}
    assert channel.declaration().identity is None
    assert channel.declaration().takes_owner_tools is False
    await channel.close()


@pytest.mark.parametrize("name", [None, "", "  "])
async def test_a_live_acp_agent_without_a_usable_handle_is_refused(
    tmp_path: Path, name: str | None
) -> None:
    channel = ACPChannel("agent:7c1e", ["agent-not-started"], cwd=tmp_path, name=name)
    with pytest.raises(ValueError):
        Discussion([channel])
    await channel.close()
