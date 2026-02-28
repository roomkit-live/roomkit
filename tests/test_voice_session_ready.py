"""Tests for ON_VOICE_SESSION_READY hook and kit.send_greeting()."""

from __future__ import annotations

import asyncio

from roomkit import (
    Agent,
    HookExecution,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice import AudioPipelineConfig
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.events import VoiceSessionReadyEvent
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider


class TestVoiceSessionReady:
    """ON_VOICE_SESSION_READY hook tests."""

    async def test_hook_fires_after_connect_voice(self) -> None:
        """Hook fires when backend signals ready AND session is bound."""
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["hi"])
        pipeline = AudioPipelineConfig()

        kit = RoomKit(stt=stt, voice=backend)
        voice_channel = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        ready_events: list[VoiceSessionReadyEvent] = []

        @kit.hook(HookTrigger.ON_VOICE_SESSION_READY, HookExecution.ASYNC)
        async def on_ready(event: VoiceSessionReadyEvent, context: object) -> None:
            ready_events.append(event)

        # connect_voice calls backend.connect() which fires session_ready
        # immediately for MockVoiceBackend, AND bind_session().
        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        assert len(ready_events) == 1
        assert ready_events[0].session.id == session.id

        await kit.close()

    async def test_hook_fires_when_ready_before_bind(self) -> None:
        """When backend fires ready before bind_session, hook still fires."""
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        ready_events: list[VoiceSessionReadyEvent] = []

        @kit.hook(HookTrigger.ON_VOICE_SESSION_READY, HookExecution.ASYNC)
        async def on_ready(event: VoiceSessionReadyEvent, context: object) -> None:
            ready_events.append(event)

        # MockVoiceBackend fires ready in connect(), and connect_voice
        # calls bind_session() after. The dual-signal handles both orderings.
        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        assert len(ready_events) == 1
        assert ready_events[0].session.id == session.id

        await kit.close()

    async def test_hook_fires_when_bind_before_ready(self) -> None:
        """When bind_session runs before backend fires ready, hook still fires."""
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        ready_events: list[VoiceSessionReadyEvent] = []

        @kit.hook(HookTrigger.ON_VOICE_SESSION_READY, HookExecution.ASYNC)
        async def on_ready(event: VoiceSessionReadyEvent, context: object) -> None:
            ready_events.append(event)

        # Create session manually, bind, then simulate ready
        from roomkit.voice.base import VoiceSession, VoiceSessionState

        session = VoiceSession(
            id="test-session",
            room_id=room.id,
            participant_id="user-1",
            channel_id="voice-1",
            state=VoiceSessionState.ACTIVE,
        )
        binding = await kit._store.get_binding(room.id, "voice-1")
        assert binding is not None
        voice_channel.bind_session(session, room.id, binding)

        # Now simulate backend signalling ready
        await backend.simulate_session_ready(session)
        await asyncio.sleep(0.1)

        assert len(ready_events) == 1
        assert ready_events[0].session.id == session.id

        await kit.close()

    async def test_simulate_session_ready_on_mock_backend(self) -> None:
        """MockVoiceBackend.simulate_session_ready() fires callbacks."""
        backend = MockVoiceBackend()

        fired: list[str] = []

        def on_ready(session: object) -> None:
            fired.append("ready")

        backend.on_session_ready(on_ready)

        from roomkit.voice.base import VoiceSession, VoiceSessionState

        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="p1",
            channel_id="c1",
            state=VoiceSessionState.ACTIVE,
        )

        await backend.simulate_session_ready(session)
        assert fired == ["ready"]

    async def test_framework_event_emitted(self) -> None:
        """voice_session_ready framework event is emitted."""
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        fw_events: list[str] = []

        @kit.on("voice_session_ready")
        async def on_ready(event: object) -> None:
            fw_events.append("voice_session_ready")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        assert "voice_session_ready" in fw_events

        await kit.close()

    async def test_unbind_clears_pending(self) -> None:
        """unbind_session clears pending ready state."""
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        voice_channel.unbind_session(session)

        # Verify _session_ready_pending is clean
        assert session.id not in voice_channel._session_ready_pending

        await kit.close()


class TestSendGreeting:
    """kit.send_greeting() tests."""

    async def test_send_greeting_with_agent(self) -> None:
        """send_greeting injects a greeting message through the agent."""
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["hi"])
        tts = MockTTSProvider()

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
        )
        ai_provider = MockAIProvider(responses=["Hello! How can I help?"])
        agent = Agent(
            "agent-1",
            provider=ai_provider,
            greeting="Welcome to our service!",
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id)
        await asyncio.sleep(0.1)

        # Check that an event was stored in the room
        events = await kit._store.list_events(room.id)
        new_events = events[len(events_before) :]
        # The greeting should have been processed as an inbound message
        assert len(new_events) >= 1

        await kit.close()

    async def test_send_greeting_no_agent_is_noop(self) -> None:
        """send_greeting is a no-op when no agent is attached."""
        backend = MockVoiceBackend()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=AudioPipelineConfig())
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        # Should not raise
        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id)
        await asyncio.sleep(0.1)

        events_after = await kit._store.list_events(room.id)
        assert len(events_after) == len(events_before)

        await kit.close()

    async def test_send_greeting_no_greeting_text_is_noop(self) -> None:
        """send_greeting is a no-op when agent has no greeting."""
        backend = MockVoiceBackend()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=AudioPipelineConfig())
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Hi"]),
            # No greeting set
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id)
        await asyncio.sleep(0.1)

        events_after = await kit._store.list_events(room.id)
        assert len(events_after) == len(events_before)

        await kit.close()

    async def test_send_greeting_explicit_text(self) -> None:
        """send_greeting uses explicit greeting text over agent.greeting."""
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["hi"])
        tts = MockTTSProvider()

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
        )
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["reply"]),
            greeting="Default greeting",
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id, greeting="Custom hello!")
        await asyncio.sleep(0.1)

        events = await kit._store.list_events(room.id)
        new_events = events[len(events_before) :]
        assert len(new_events) >= 1
        # The greeting should use the explicit text
        assert new_events[0].content.body == "Custom hello!"

        await kit.close()

    async def test_send_greeting_with_language(self) -> None:
        """send_greeting prepends language hint when agent has language."""
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["hi"])
        tts = MockTTSProvider()

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
        )
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["reply"]),
            greeting="Bienvenue!",
            language="French",
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id)
        await asyncio.sleep(0.1)

        events = await kit._store.list_events(room.id)
        new_events = events[len(events_before) :]
        assert len(new_events) >= 1
        assert new_events[0].content.body.startswith("[Respond in French]")

        await kit.close()


class TestAutoGreet:
    """connect_voice(auto_greet=True) tests."""

    async def test_auto_greet_triggers_greeting_on_session_ready(self) -> None:
        """auto_greet=True registers a one-shot hook that calls send_greeting."""
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["hi"])
        tts = MockTTSProvider()

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
        )
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Hello!"]),
            greeting="Welcome!",
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1", auto_greet=True)
        await asyncio.sleep(0.2)

        # The greeting should have been sent
        events = await kit._store.list_events(room.id)
        assert len(events) >= 1

        await kit.close()

    async def test_auto_greet_false_does_not_greet(self) -> None:
        """auto_greet=False (default) does not register any greeting hook."""
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["hi"])
        tts = MockTTSProvider()

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
        )
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Hello!"]),
            greeting="Welcome!",
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        events = await kit._store.list_events(room.id)
        # No greeting events should be stored
        greeting_events = [
            e for e in events if hasattr(e.content, "body") and "Welcome" in (e.content.body or "")
        ]
        assert len(greeting_events) == 0

        await kit.close()
