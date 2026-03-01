"""Tests for ON_VOICE_SESSION_READY hook and kit.send_greeting()."""

from __future__ import annotations

import asyncio
import warnings

from roomkit import (
    Agent,
    HookExecution,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.models.enums import EventType
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
    """kit.send_greeting() tests — direct TTS path."""

    async def test_send_greeting_stores_assistant_event(self) -> None:
        """send_greeting stores a greeting as an assistant event with metadata."""
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["hi"])
        tts = MockTTSProvider()

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
        )
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Hello! How can I help?"]),
            greeting="Welcome to our service!",
            auto_greet=False,
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        # Connect a session so TTS has a target
        await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id)
        await asyncio.sleep(0.1)

        events = await kit._store.list_events(room.id)
        new_events = events[len(events_before) :]
        assert len(new_events) == 1

        greeting_event = new_events[0]
        assert greeting_event.content.body == "Welcome to our service!"
        assert greeting_event.source.channel_id == "agent-1"
        assert greeting_event.metadata.get("auto_greeting") is True
        assert greeting_event.type == EventType.MESSAGE

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
            auto_greet=False,
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
            auto_greet=False,
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id, greeting="Custom hello!")
        await asyncio.sleep(0.1)

        events = await kit._store.list_events(room.id)
        new_events = events[len(events_before) :]
        assert len(new_events) == 1
        assert new_events[0].content.body == "Custom hello!"

        await kit.close()

    async def test_send_greeting_no_language_prepend(self) -> None:
        """send_greeting does not prepend language hint (greeting is literal TTS)."""
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
            auto_greet=False,
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id)
        await asyncio.sleep(0.1)

        events = await kit._store.list_events(room.id)
        new_events = events[len(events_before) :]
        assert len(new_events) == 1
        # Greeting is literal — no [Respond in French] prefix
        assert new_events[0].content.body == "Bienvenue!"

        await kit.close()


class TestAutoGreet:
    """Agent(auto_greet=True) tests."""

    async def test_auto_greet_triggers_greeting_on_session_ready(self) -> None:
        """Agent with auto_greet=True greets automatically on session ready."""
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
            auto_greet=True,
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.2)

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 1
        assert greeting_events[0].content.body == "Welcome!"
        assert greeting_events[0].source.channel_id == "agent-1"

        await kit.close()

    async def test_auto_greet_false_does_not_greet(self) -> None:
        """Agent with auto_greet=False does not auto-greet."""
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
            auto_greet=False,
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 0

        await kit.close()

    async def test_auto_greet_no_greeting_text_skips(self) -> None:
        """Agent with auto_greet=True but no greeting text does not register hook."""
        backend = MockVoiceBackend()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=AudioPipelineConfig())
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Hi"]),
            auto_greet=True,
            # No greeting set
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 0

        await kit.close()

    async def test_auto_greet_dedup_per_session(self) -> None:
        """Auto-greet fires only once per session even if hook fires again."""
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
            auto_greet=True,
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.2)

        # Simulate a second session_ready for the same session
        await backend.simulate_session_ready(session)
        await asyncio.sleep(0.2)

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        # Only one greeting despite two ready signals
        assert len(greeting_events) == 1

        await kit.close()

    async def test_connect_voice_auto_greet_deprecated(self) -> None:
        """connect_voice(auto_greet=True) emits DeprecationWarning."""
        backend = MockVoiceBackend()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=AudioPipelineConfig())
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await kit.connect_voice(room.id, "user-1", "voice-1", auto_greet=True)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Agent" in str(w[0].message)

        await kit.close()
