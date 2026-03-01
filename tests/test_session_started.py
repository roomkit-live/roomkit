"""Tests for ON_SESSION_STARTED hook and kit.send_greeting()."""

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
from roomkit.channels import SMSChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import TextContent
from roomkit.models.session_event import SessionStartedEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.voice import AudioPipelineConfig
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider


class TestVoiceSessionStarted:
    """ON_SESSION_STARTED hook tests (voice channel)."""

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

        ready_events: list[SessionStartedEvent] = []

        @kit.hook(HookTrigger.ON_SESSION_STARTED, HookExecution.ASYNC)
        async def on_ready(event: SessionStartedEvent, context: object) -> None:
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

        ready_events: list[SessionStartedEvent] = []

        @kit.hook(HookTrigger.ON_SESSION_STARTED, HookExecution.ASYNC)
        async def on_ready(event: SessionStartedEvent, context: object) -> None:
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

        ready_events: list[SessionStartedEvent] = []

        @kit.hook(HookTrigger.ON_SESSION_STARTED, HookExecution.ASYNC)
        async def on_ready(event: SessionStartedEvent, context: object) -> None:
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
        """session_started framework event is emitted."""
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        fw_events: list[str] = []

        @kit.on("session_started")
        async def on_ready(event: object) -> None:
            fw_events.append("session_started")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        assert "session_started" in fw_events

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

    async def test_session_started_event_fields(self) -> None:
        """SessionStartedEvent carries correct channel-agnostic fields."""
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()

        kit = RoomKit(voice=backend)
        voice_channel = VoiceChannel("voice-1", backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        captured: list[SessionStartedEvent] = []

        @kit.hook(HookTrigger.ON_SESSION_STARTED, HookExecution.ASYNC)
        async def on_ready(event: SessionStartedEvent, context: object) -> None:
            captured.append(event)

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        await asyncio.sleep(0.1)

        assert len(captured) == 1
        evt = captured[0]
        assert evt.room_id == room.id
        assert evt.channel_id == "voice-1"
        assert evt.channel_type == "voice"
        assert evt.participant_id == "user-1"
        assert evt.session is session

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

    async def test_auto_greet_triggers_greeting_on_session_started(self) -> None:
        """Agent with auto_greet=True greets automatically on session started."""
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


class TestTextSessionStarted:
    """ON_SESSION_STARTED hook tests for text channels."""

    async def test_session_started_fires_on_room_auto_create(self) -> None:
        """First inbound message creates room and fires ON_SESSION_STARTED."""
        kit = RoomKit()
        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        kit.register_channel(sms)

        captured: list[SessionStartedEvent] = []

        @kit.hook(HookTrigger.ON_SESSION_STARTED, HookExecution.ASYNC)
        async def on_started(event: SessionStartedEvent, context: object) -> None:
            captured.append(event)

        msg = InboundMessage(
            channel_id="sms-1",
            sender_id="+15551234567",
            content=TextContent(body="Hello"),
        )
        result = await kit.process_inbound(msg)
        await asyncio.sleep(0.1)

        assert not result.blocked
        assert len(captured) == 1
        evt = captured[0]
        assert evt.channel_id == "sms-1"
        assert evt.channel_type == "sms"
        assert evt.participant_id == "+15551234567"
        assert evt.session is None

        await kit.close()

    async def test_session_started_not_fired_on_existing_room(self) -> None:
        """Subsequent messages to an existing room don't re-fire the hook."""
        kit = RoomKit()
        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        kit.register_channel(sms)

        captured: list[SessionStartedEvent] = []

        @kit.hook(HookTrigger.ON_SESSION_STARTED, HookExecution.ASYNC)
        async def on_started(event: SessionStartedEvent, context: object) -> None:
            captured.append(event)

        # First message — creates room
        msg1 = InboundMessage(
            channel_id="sms-1",
            sender_id="+15551234567",
            content=TextContent(body="Hello"),
        )
        result1 = await kit.process_inbound(msg1)
        await asyncio.sleep(0.1)
        room_id = result1.event.room_id

        # Second message — same room
        msg2 = InboundMessage(
            channel_id="sms-1",
            sender_id="+15551234567",
            content=TextContent(body="Follow up"),
        )
        await kit.process_inbound(msg2, room_id=room_id)
        await asyncio.sleep(0.1)

        # Hook fires only once (for room creation)
        assert len(captured) == 1

        await kit.close()

    async def test_auto_greet_text_channel(self) -> None:
        """Agent with auto_greet broadcasts greeting on text channel session start."""
        kit = RoomKit()
        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Hi!"]),
            greeting="Welcome via SMS!",
            auto_greet=True,
        )
        kit.register_channel(sms)
        kit.register_channel(agent)

        # Pre-create room with both channels attached
        room = await kit.create_room()
        await kit.attach_channel(room.id, "sms-1")
        await kit.attach_channel(room.id, "agent-1")

        # Send first inbound to a room that already exists — this won't trigger
        # ON_SESSION_STARTED because room already exists.  To test auto-greet on
        # text, we need auto-creation.  So use a NEW inbound without room_id.
        # But this room was pre-created, so let's test via a fresh setup.
        await kit.close()

        # Fresh setup: let process_inbound auto-create room
        kit2 = RoomKit()
        sms2 = SMSChannel("sms-1", provider=MockSMSProvider())
        agent2 = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Hi!"]),
            greeting="Welcome via SMS!",
            auto_greet=True,
        )
        kit2.register_channel(sms2)
        kit2.register_channel(agent2)

        # We need to pre-attach agent to the auto-created room.
        # The auto-greet handler checks if agent is attached to the room.
        # For text auto-greet to work, the agent must be attached after
        # room creation.  Let's hook ON_ROOM_CREATED to attach the agent.
        @kit2.hook(HookTrigger.ON_ROOM_CREATED, HookExecution.ASYNC)
        async def attach_agent(event: object, context: object) -> None:
            room_id = event.room_id  # type: ignore[attr-defined]
            await kit2.attach_channel(room_id, "agent-1")

        msg = InboundMessage(
            channel_id="sms-1",
            sender_id="+15551234567",
            content=TextContent(body="Hello"),
        )
        await kit2.process_inbound(msg)
        await asyncio.sleep(0.3)

        # Find the room that was auto-created
        rooms = await kit2._store.list_rooms()
        assert len(rooms) == 1
        events = await kit2._store.list_events(rooms[0].id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 1
        assert greeting_events[0].content.body == "Welcome via SMS!"

        await kit2.close()

    async def test_send_greeting_broadcasts_to_text_channel(self) -> None:
        """send_greeting delivers text to transport channels in non-voice rooms."""
        kit = RoomKit()
        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Hi!"]),
            greeting="Hello from the agent!",
            auto_greet=False,
        )
        kit.register_channel(sms)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "sms-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.send_greeting(room.id)
        await asyncio.sleep(0.1)

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 1
        assert greeting_events[0].content.body == "Hello from the agent!"

        await kit.close()
