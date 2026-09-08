"""G.711 wire formats preserve the realtime provider's PCM16 contract."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import struct
import sys
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest

from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.voice._g711 import _G711Codec
from roomkit.voice.base import VoiceSession

# Generated independently with CPython 3.12 audioop: encode every signed
# PCM16 value in ascending order; decode every byte 0..255 into little endian.
_REFERENCE_HASHES = {
    "mulaw": (
        "81d633c9e6972a18c74a58720b96cb8ca0bdd096d4060b646dd708c3b846019a",
        "3dab54339e520bb2c924826e3b72a917a2b612e9fd12fc867500f1d983a75827",
    ),
    "alaw": (
        "38488f6fd710f4686360edc4d38639f96c491595ef93f8eb8d62d5e07ca6ce7b",
        "e04788d110e58ff8c70c93b8480190d973e3b67876b6119abbaec766cc75c174",
    ),
}


@pytest.mark.parametrize("law", ["mulaw", "alaw"])
@pytest.mark.parametrize("without_numpy", [False, True])
def test_full_domain_matches_audioop(law: Literal["mulaw", "alaw"], without_numpy: bool) -> None:
    with patch.dict(sys.modules, {"numpy": None} if without_numpy else {}):
        codec = _G711Codec(law)
        pcm = struct.pack("<65536h", *range(-32768, 32768))
        assert hashlib.sha256(codec.encode(pcm)).hexdigest() == _REFERENCE_HASHES[law][0]
        assert (
            hashlib.sha256(codec.decode(bytes(range(256)))).hexdigest()
            == _REFERENCE_HASHES[law][1]
        )
        assert codec.encode(b"") == codec.decode(b"") == b""


@pytest.mark.parametrize("codec", ["pcmu", "pcma"])
@pytest.mark.parametrize(
    ("input_rate", "output_rate"), [(8000, 8000), (8000, 24000), (24000, 8000), (24000, 24000)]
)
async def test_realtime_converts_both_directions_and_preserves_duration(
    codec: str,
    input_rate: int,
    output_rate: int,
) -> None:
    provider = OpenAIRealtimeProvider(api_key="test")
    ws = AsyncMock()
    provider._import_websockets = lambda: SimpleNamespace(connect=AsyncMock(return_value=ws))

    async def receive(session: VoiceSession) -> None:
        await asyncio.Event().wait()

    provider._receive_loop = receive
    session = VoiceSession(id="s1", room_id="r", participant_id="p", channel_id="rt")
    received: list[bytes] = []
    provider.on_audio(lambda s, audio: received.append(audio))
    await provider.connect(
        session,
        input_sample_rate=input_rate,
        output_sample_rate=output_rate,
        provider_config={"codec": codec},
    )
    try:
        silence_byte = b"\xff" if codec == "pcmu" else b"\xd5"
        await provider.send_audio(session, b"\x00\x00" * (input_rate // 50))
        payload = json.loads(ws.send.await_args.args[0])
        expected = silence_byte * 160 if input_rate == 8000 else b"\x00\x00" * 480
        assert base64.b64decode(payload["audio"]) == expected

        wire = silence_byte * 160 if output_rate == 8000 else b"\x00\x00" * 480
        await provider._on_audio_delta(
            session,
            {
                "item_id": "item1",
                "content_index": 0,
                "delta": base64.b64encode(wire).decode(),
            },
        )
        sample = 8 if codec == "pcma" and output_rate == 8000 else 0
        assert received == [struct.pack("<h", sample) * (output_rate // 50)]
        await provider.truncate_audio(session, 9999)
        assert json.loads(ws.send.await_args.args[0])["audio_end_ms"] == 20
        if input_rate == 8000:
            with pytest.raises(ValueError, match="complete"):
                await provider.send_audio(session, b"\x00")
    finally:
        await provider.disconnect(session)
    assert not provider._audio_codecs


async def test_failed_session_update_releases_codec_state() -> None:
    provider = OpenAIRealtimeProvider(api_key="test")
    ws = AsyncMock()
    ws.send.side_effect = ConnectionError("peer left")
    provider._import_websockets = lambda: SimpleNamespace(connect=AsyncMock(return_value=ws))
    session = VoiceSession(id="s1", room_id="r", participant_id="p", channel_id="rt")
    with pytest.raises(ConnectionError):
        await provider.connect(session, input_sample_rate=8000, output_sample_rate=8000)
    assert not provider._audio_codecs
    assert not provider._connections
    ws.close.assert_awaited_once()
