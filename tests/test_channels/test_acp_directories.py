"""Additional session directories are validated and declared without changing cwd."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit import ACPChannel
from tests.test_channels.test_acp import (
    _FakeACPConnection,
    _FakeTransport,
)


@pytest.mark.parametrize("directory", ["relative", "", Path("relative")])
def test_additional_directories_must_be_absolute(directory: str | Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="additional_directories must be an absolute path"):
        ACPChannel(
            "agent",
            ["unused-agent"],
            cwd=tmp_path,
            additional_directories=[tmp_path, directory],
        )


async def test_additional_directories_expand_home_and_accept_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    connection = _FakeACPConnection(None, emit_updates=False)
    channel = ACPChannel(
        "agent",
        transport=_FakeTransport(connection),
        cwd="/srv/remote-project",
        additional_directories=["~/shared", Path("/srv/remote-reference")],
    )
    connection.client = channel._client
    try:
        assert connection.new_session_calls == []
        await channel.set_config_option("room-1", "model", "opus")
        [request] = connection.new_session_calls
        assert request["cwd"] == "/srv/remote-project"
        assert request["additional_directories"] == [
            str(tmp_path / "shared"),
            "/srv/remote-reference",
        ]
    finally:
        await channel.close()
