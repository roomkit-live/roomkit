"""A scripted memory preserves the metadata the real memory contract carries."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._memory import _built
from tests.conformance._shape import unknown_members


@pytest.mark.parametrize("content", [{"text": "labelled"}, {"parts": ["labelled"]}])
@pytest.mark.parametrize("speaker", ["runtime", None])
def test_a_memory_message_keeps_its_speaker_metadata(
    content: dict[str, Any], speaker: str | None
) -> None:
    message = _built({"role": "user", **content, "metadata": {"speaker": speaker}})
    assert message.metadata == {"speaker": speaker}


@pytest.mark.parametrize("metadata", [None, [], "", False])
def test_memory_metadata_requires_an_object(metadata: object) -> None:
    scenario = {
        "setup": {
            "channels": [
                {
                    "id": "ai",
                    "type": "ai",
                    "memory": {
                        "messages": [{"role": "user", "text": "text", "metadata": metadata}]
                    },
                }
            ]
        }
    }
    assert any("metadata: expected an object" in item for item in unknown_members(scenario))
