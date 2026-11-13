"""ACP construction preserves the agent's public identity (RFC §6.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit import ACPChannel


@pytest.mark.parametrize(
    "identity",
    [
        {},
        {"name": None, "role": None, "description": None},
        {"name": "", "role": "", "description": ""},
        {"name": "  ", "role": "\t", "description": "\n"},
        {"name": "Développeuse", "role": "dev", "description": "reads the code 🔎"},
    ],
)
def test_an_acp_agent_preserves_its_identity(
    tmp_path: Path, identity: dict[str, str | None]
) -> None:
    channel = ACPChannel("agent:7c1e", command=["agent-not-started"], cwd=tmp_path, **identity)

    assert (channel.name, channel.role, channel.description) == (
        identity.get("name"),
        identity.get("role"),
        identity.get("description"),
    )
    assert channel.channel_id == "agent:7c1e"
    # The local attributes are not Agent.declaration()'s identity contract.
    assert channel.declaration().identity is None
