"""The one vocabulary a provider ends a generation in (RFC §6.7).

The contract carries a closed set of stop reasons. Each provider maps its own
vendor's words onto it in its adapter, so the core reads no vendor word.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from roomkit.providers.ai import (
    AIResponse,
    FinishReason,
    StreamDone,
    call_cut,
    call_garbled,
    is_truncation,
)


def test_the_contract_refuses_a_vendor_s_own_word() -> None:
    with pytest.raises(ValidationError):
        AIResponse(content="", finish_reason="end_turn")
    with pytest.raises(ValidationError):
        StreamDone(finish_reason="MAX_TOKENS")


@pytest.mark.parametrize(
    "word", ["max_tokens", "MAX_TOKENS", "model_context_window_exceeded", "model_length", "LENGTH"]
)
def test_the_core_reads_no_vendor_word(word: str) -> None:
    """Running out of room is ``length`` and nothing else: a vendor's word
    neither truncates a response nor cuts its last call."""
    fragment = '{"q": "par'
    assert not is_truncation(word)
    assert not call_cut(fragment, word) and call_garbled(fragment, word)
    assert is_truncation(FinishReason.LENGTH) and call_cut(fragment, FinishReason.LENGTH)
