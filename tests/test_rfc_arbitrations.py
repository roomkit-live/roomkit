"""Two invariants the RFC states and the code did not enforce.

§12.1 forbids any transition out of ENDED — the field was plain and mutable, so
the rule was unenforceable rather than enforced. §19.4 step 4 says a supervisor
always receives an agent's event; under ADDRESSED_ONLY it received nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.event_router import _solicits
from roomkit.core.exceptions import VoiceSessionEndedError
from roomkit.models.enums import AgentResponsePolicy, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


def _session() -> VoiceSession:
    return VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="voice-1")


class TestEndedIsTerminal:
    def test_the_documented_path_is_allowed(self) -> None:
        s = _session()
        s.state = VoiceSessionState.ACTIVE
        s.state = VoiceSessionState.PAUSED
        s.state = VoiceSessionState.ACTIVE
        s.state = VoiceSessionState.ENDED
        assert s.state is VoiceSessionState.ENDED

    @pytest.mark.parametrize(
        "target",
        [VoiceSessionState.ACTIVE, VoiceSessionState.CONNECTING, VoiceSessionState.PAUSED],
    )
    def test_leaving_ended_is_refused(self, target: VoiceSessionState) -> None:
        s = _session()
        s.state = VoiceSessionState.ENDED
        with pytest.raises(VoiceSessionEndedError):
            s.state = target

    def test_setting_ended_again_is_not_a_transition(self) -> None:
        s = _session()
        s.state = VoiceSessionState.ENDED
        s.state = VoiceSessionState.ENDED  # idempotent teardown must not raise
        assert s.state is VoiceSessionState.ENDED

    def test_an_undocumented_transition_warns_rather_than_raises(self, caplog) -> None:
        """The table does not model every provider's reality — a renegotiating
        realtime provider goes ACTIVE -> CONNECTING. Turning that into a crash
        would trade a documentation gap for an outage."""
        s = _session()
        s.state = VoiceSessionState.ACTIVE
        with caplog.at_level("WARNING"):
            s.state = VoiceSessionState.CONNECTING
        assert s.state is VoiceSessionState.CONNECTING
        assert "undocumented transition" in caplog.text

    def test_renegotiate_is_the_one_sanctioned_way_back(self) -> None:
        """A reconfigure tears the upstream connection down and rebuilds it
        while the participant's session continues — nobody hung up."""
        s = _session()
        s.state = VoiceSessionState.ACTIVE
        s.state = VoiceSessionState.ENDED

        s.renegotiate()

        assert s.state is VoiceSessionState.CONNECTING
        s.state = VoiceSessionState.ACTIVE  # the rebuilt connection

    async def test_a_handoff_renegotiates_the_session_it_ended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RFC §12.2: a realtime session is CONNECTING until its provider
        connects; the base reconfigure ends it, renegotiates it and connects
        it again, ACTIVE as before."""
        provider = MockRealtimeProvider()
        kit = RoomKit()
        kit.register_channel(
            RealtimeVoiceChannel("rt", provider=provider, transport=MockRealtimeTransport())
        )
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt")
        seen: list[VoiceSessionState] = []
        connect, disconnect = provider.connect, provider.disconnect

        async def spied_connect(session: VoiceSession, **kwargs: Any) -> Any:
            seen.append(session.state)
            return await connect(session, **kwargs)

        async def spied_disconnect(session: VoiceSession) -> None:
            await disconnect(session)
            seen.append(session.state)

        monkeypatch.setattr(provider, "connect", spied_connect)
        monkeypatch.setattr(provider, "disconnect", spied_disconnect)

        session = await kit.join(room.id, "rt", connection=object())
        assert (seen, session.state) == ([VoiceSessionState.CONNECTING], VoiceSessionState.ACTIVE)
        await provider.reconfigure(session, system_prompt="the next agent")
        handoff, after = seen[1:], session.state
        await kit.close()

        assert handoff == [VoiceSessionState.ENDED, VoiceSessionState.CONNECTING]
        assert after is VoiceSessionState.ACTIVE


def _agent_event() -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="agent-a", channel_type=ChannelType.AI),
        content=TextContent(body="thinking out loud"),
    )


class TestSupervisorAlwaysReceives:
    def test_addressed_only_still_reaches_the_supervisor(self) -> None:
        """ADDRESSED_ONLY governs who is solicited to act; a supervisor is not
        being asked to act, it is watching."""
        event = _agent_event()
        event.metadata["_always_process"] = ["supervisor"]

        assert (
            _solicits(
                event,
                "supervisor",
                source_is_agent=True,
                policy=AgentResponsePolicy.ADDRESSED_ONLY,
            )
            is True
        )

    def test_other_agents_are_still_not_solicited(self) -> None:
        event = _agent_event()
        event.metadata["_always_process"] = ["supervisor"]

        assert (
            _solicits(
                event,
                "other-agent",
                source_is_agent=True,
                policy=AgentResponsePolicy.ADDRESSED_ONLY,
            )
            is False
        )

    def test_without_a_supervisor_nothing_is_solicited(self) -> None:
        assert (
            _solicits(
                _agent_event(),
                "other-agent",
                source_is_agent=True,
                policy=AgentResponsePolicy.ADDRESSED_ONLY,
            )
            is False
        )
