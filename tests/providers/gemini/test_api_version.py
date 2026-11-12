"""Every Gemini Developer API client calls v1beta.

google-genai 3.0 made v1 its default version. The Live API, the voices
endpoint and the Decisions API answer 404 there (probed on the live API on
2026-11-12): a voice the store holds read as missing, and a Live session could
not open. A client RoomKit builds names v1beta, whatever the SDK's default.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

genai_live = pytest.importorskip("google.genai.live")

from roomkit.providers.gemini import GeminiLiveProvider  # noqa: E402
from roomkit.voice.base import VoiceSession  # noqa: E402
from roomkit.voice.tts import GeminiVoiceLibrary, GeminiVoiceLibraryConfig  # noqa: E402


async def test_a_voice_library_call_goes_to_v1beta(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def not_found(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(404, json={"error": {"code": 404, "message": "not found"}})

    real = httpx.AsyncClient

    def through_fake(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(not_found), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", through_fake)
    library = GeminiVoiceLibrary(GeminiVoiceLibraryConfig(api_key="k"))
    try:
        assert await library.get_voice("custom-1") is None
    finally:
        await library.close()
    assert [request.url.path for request in seen] == ["/v1beta/voices/custom-1"]


async def test_a_live_session_opens_on_v1beta(monkeypatch: pytest.MonkeyPatch) -> None:
    dialled: list[str] = []

    def refuse(uri: str, **kwargs: Any) -> Any:
        dialled.append(uri)
        raise ConnectionRefusedError("refused by the test")

    monkeypatch.setattr(genai_live, "ws_connect", refuse)
    provider = GeminiLiveProvider(api_key="k")
    session = VoiceSession(id="s1", room_id="r", participant_id="p", channel_id="rt")
    with pytest.raises(ConnectionRefusedError):
        await provider.connect(session)
    assert len(dialled) == 1
    assert ".generativelanguage.v1beta.GenerativeService.BidiGenerateContent" in dialled[0]
