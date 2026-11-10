"""A scenario's simulated processes own their calls and store connections."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit.store.sqlite import SQLiteStore
from tests.conformance._harness import Harness


@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_regeneration_uses_the_named_kit_without_closing_the_other(
    store: str, tmp_path: Path
) -> None:
    """A closed follower refuses its regeneration; the owner still reads the room.

    Routing regeneration to the default kit would return null instead, and
    sharing a SQLite connection would close the owner's store with the follower.
    """
    scenario = {
        "setup": {"rooms": [{"id": "r1"}], "processes": [{"id": "follower"}]},
        "steps": [
            {"call": "close", "kit": "follower"},
            {
                "call": "regenerate_response",
                "kit": "follower",
                "room_id": "r1",
                "expect": {"error": {"name": "kit_closed"}},
            },
            {"call": "get_room", "room_id": "r1", "expect": {"result": {"id": "r1"}}},
        ],
    }
    on = SQLiteStore(tmp_path / "store.db") if store == "sqlite" else None
    reader = SQLiteStore(tmp_path / "store.db") if store == "sqlite" else None
    try:
        problems = await Harness(scenario, on, observation_store=reader).run()
        assert not problems, "\n".join(problems)
    finally:
        if reader is not None:
            await reader.close()
