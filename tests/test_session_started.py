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
from tests.conference.test_conference_realtime import until


def _started(kit: RoomKit) -> list[SessionStartedEvent]:
    """ON_SESSION_STARTED's events as an ASYNC hook sees them: once one is
    in, every ASYNC hook of that firing (an auto-greeting) is under way, and
    the kit's close waits for them."""
    seen: list[SessionStartedEvent] = []

    @kit.hook(HookTrigger.ON_SESSION_STARTED, HookExecution.ASYNC, name="test_started")
    async def started(event: SessionStartedEvent, context: object) -> None:
        seen.append(event)

    return seen


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
        await until(lambda: bool(ready_events))
        await kit.close()  # a second firing's hook would have run by now

        assert len(ready_events) == 1
        assert ready_events[0].session.id == session.id

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
        await until(lambda: bool(ready_events))
        await kit.close()  # a second firing's hook would have run by now

        assert len(ready_events) == 1
        assert ready_events[0].session.id == session.id

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
        await until(lambda: bool(ready_events))
        await kit.close()  # a second firing's hook would have run by now

        assert len(ready_events) == 1
        assert ready_events[0].session.id == session.id

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
        await until(lambda: bool(fw_events))

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
        await until(lambda: bool(captured))
        await kit.close()  # a second firing's hook would have run by now

        assert len(captured) == 1
        evt = captured[0]
        assert evt.room_id == room.id
        assert evt.channel_id == "voice-1"
        assert evt.channel_type == "voice"
        assert evt.participant_id == "user-1"
        assert evt.session is session


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

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id)  # stored before it returns

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
        await kit.close()  # anything it started has run by now

        events_after = await kit._store.list_events(room.id)
        assert len(events_after) == len(events_before)

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
        await kit.close()  # anything it started has run by now

        events_after = await kit._store.list_events(room.id)
        assert len(events_after) == len(events_before)

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

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id, greeting="Custom hello!")  # stored before it returns

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

        events_before = await kit._store.list_events(room.id)
        await kit.send_greeting(room.id)  # stored before it returns

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
        started = _started(kit)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await until(lambda: bool(started))
        await kit.close()  # the auto-greeting hook has run

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 1
        assert greeting_events[0].content.body == "Welcome!"
        assert greeting_events[0].source.channel_id == "agent-1"

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
        started = _started(kit)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await until(lambda: bool(started))
        await kit.close()  # an auto-greeting hook would have run

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 0

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
        started = _started(kit)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")
        await until(lambda: bool(started))
        await kit.close()  # an auto-greeting hook would have run

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 0

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
        started = _started(kit)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        await until(lambda: bool(started))

        # Simulate a second session_ready for the same session
        await backend.simulate_session_ready(session)
        await until(lambda: len(started) == 2)  # fired again, its hooks under way
        await kit.close()

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        # Only one greeting despite two ready signals
        assert len(greeting_events) == 1

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
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 2  # connect_voice + auto_greet
            auto_greet_w = [x for x in dep_warnings if "Agent" in str(x.message)]
            assert len(auto_greet_w) == 1

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
        await until(lambda: bool(captured))
        await kit.close()  # a second firing's hook would have run by now

        assert not result.blocked
        assert len(captured) == 1
        evt = captured[0]
        assert evt.channel_id == "sms-1"
        assert evt.channel_type == "sms"
        assert evt.participant_id == "+15551234567"
        assert evt.session is None

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
        await until(lambda: bool(captured))
        room_id = result1.event.room_id

        # Second message — same room
        msg2 = InboundMessage(
            channel_id="sms-1",
            sender_id="+15551234567",
            content=TextContent(body="Follow up"),
        )
        await kit.process_inbound(msg2, room_id=room_id)
        await kit.close()  # a second firing's hook would have run by now

        # Hook fires only once (for room creation)
        assert len(captured) == 1

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
        await kit2.close()  # its ASYNC hooks have run

        # Find the room that was auto-created
        rooms = await kit2._store.list_rooms()
        assert len(rooms) == 1
        events = await kit2._store.list_events(rooms[0].id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 1
        assert greeting_events[0].content.body == "Welcome via SMS!"

    async def test_another_agent_reading_the_greeting_holds_no_message(self) -> None:
        """A greeting's delivery to another intelligence channel waits on no gate
        of its own: the message is processed, not refused process_timeout (RMK-503)."""
        kit = RoomKit(process_timeout=0.5)
        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Noted"]),
            greeting="Welcome!",
            auto_greet=True,
        )
        helper = Agent("helper", provider=MockAIProvider(responses=["Hello"]))
        kit.register_channel(sms)
        kit.register_channel(agent)
        kit.register_channel(helper)

        @kit.hook(HookTrigger.ON_ROOM_CREATED, HookExecution.ASYNC)
        async def attach_agents(event: object, context: object) -> None:
            room_id = event.room_id  # type: ignore[attr-defined]
            await kit.attach_channel(room_id, "agent-1")
            await kit.attach_channel(room_id, "helper")

        msg = InboundMessage(
            channel_id="sms-1",
            sender_id="+15551234567",
            content=TextContent(body="Hello"),
        )

        result = await kit.process_inbound(msg)

        assert not result.blocked
        await kit.close()

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

        await kit.send_greeting(room.id)  # stored before it returns

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 1
        assert greeting_events[0].content.body == "Hello from the agent!"

        await kit.close()


class TestGreetingGate:
    """Greeting gate mechanism tests."""

    async def test_gate_timeout_releases(self) -> None:
        """Gate timeout clears the gate and allows processing to continue."""
        kit = RoomKit()

        room = await kit.create_room()
        kit._set_greeting_gate(room.id)

        # Wait with a very short timeout — should clear and not hang
        await kit._wait_greeting_gate(room.id, timeout=0.05)

        # Gate and refcount should both be cleared
        assert room.id not in kit._greeting_gates
        assert room.id not in kit._greeting_gate_counts

        await kit.close()

    async def test_voice_greeting_uses_say_not_send_tts(self) -> None:
        """Voice greeting dispatches via say() not _send_tts()."""
        from unittest.mock import AsyncMock, patch

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
            greeting="Hello!",
            auto_greet=False,
        )
        kit.register_channel(voice_channel)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "agent-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        from roomkit.models.enums import ChannelType

        with (
            patch.object(voice_channel, "say", new_callable=AsyncMock) as mock_say,
            patch.object(voice_channel, "_send_tts", new_callable=AsyncMock) as mock_raw,
        ):
            await kit.send_greeting(room.id, session=session, channel_type=ChannelType.VOICE)

            mock_say.assert_called_once()
            mock_raw.assert_not_called()

        await kit.close()

    async def test_text_session_in_multichannel_room_gets_text(self) -> None:
        """Text session in room with VoiceChannel gets text broadcast, not TTS."""
        backend = MockVoiceBackend()
        stt = MockSTTProvider(transcripts=["hi"])
        tts = MockTTSProvider()

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
        )
        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["reply"]),
            greeting="Welcome!",
            auto_greet=False,
        )
        kit.register_channel(voice_channel)
        kit.register_channel(sms)
        kit.register_channel(agent)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "sms-1")
        await kit.attach_channel(room.id, "agent-1")

        # Call send_greeting without a session (text path) —
        # should store + broadcast, NOT try TTS
        await kit.send_greeting(room.id)

        events = await kit._store.list_events(room.id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 1
        assert greeting_events[0].content.body == "Welcome!"

        await kit.close()

    async def test_auto_greet_passes_session_to_send_greeting(self) -> None:
        """_auto_greet_handler passes session and channel_type to send_greeting."""
        from unittest.mock import AsyncMock, patch

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

        with patch.object(kit, "send_greeting", new_callable=AsyncMock) as mock_greet:
            session = await kit.connect_voice(room.id, "user-1", "voice-1")
            await until(lambda: mock_greet.called)
            await kit.close()  # a second greeting would have been asked by now

            mock_greet.assert_called_once()
            call_kwargs = mock_greet.call_args
            assert call_kwargs.kwargs.get("session") is session
            assert call_kwargs.kwargs.get("channel_type") is not None

    async def test_voice_greeting_holds_the_gate_until_it_fails(self) -> None:
        """A voice session's greeting holds the gate, and its failure releases it."""
        from unittest.mock import patch

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

        held: list[bool] = []

        async def failing_greeting(room_id: str, **kwargs: object) -> None:
            held.append(room_id in kit._greeting_gates)
            raise RuntimeError("boom")

        with patch.object(kit, "send_greeting", side_effect=failing_greeting):
            await kit.connect_voice(room.id, "user-1", "voice-1")
            await until(lambda: bool(held))
            await until(lambda: room.id not in kit._greeting_gates)

        assert held == [True]
        await kit.close()

    async def test_two_agents_greet_before_the_message(self) -> None:
        """Two auto-greeting agents reading each other: both greet, then the message
        is processed, not refused process_timeout (RMK-503)."""
        kit = RoomKit()
        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        agent1 = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["AI-1 reply"]),
            greeting="Hello from agent 1!",
            auto_greet=True,
        )
        agent2 = Agent(
            "agent-2",
            provider=MockAIProvider(responses=["AI-2 reply"]),
            greeting="Hello from agent 2!",
            auto_greet=True,
        )
        kit.register_channel(sms)
        kit.register_channel(agent1)
        kit.register_channel(agent2)

        @kit.hook(HookTrigger.ON_ROOM_CREATED, HookExecution.ASYNC)
        async def attach_agents(event: object, context: object) -> None:
            room_id = event.room_id  # type: ignore[attr-defined]
            await kit.attach_channel(room_id, "agent-1")
            await kit.attach_channel(room_id, "agent-2")

        msg = InboundMessage(
            channel_id="sms-1",
            sender_id="+15551234567",
            content=TextContent(body="Hello"),
        )

        result = await kit.process_inbound(msg)
        assert not result.blocked

        rooms = await kit._store.list_rooms()
        assert len(rooms) == 1

        # The two agents answer each other past the default page (RMK-370).
        events = await kit._store.list_events(rooms[0].id, limit=500)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        user_inbound = [
            e for e in events if e.source.channel_id == "sms-1" and e.type == EventType.MESSAGE
        ]

        assert {e.content.body for e in greeting_events} == {
            "Hello from agent 1!",
            "Hello from agent 2!",
        }
        assert len(user_inbound) == 1
        assert max(e.index for e in greeting_events) < user_inbound[0].index

        await kit.close()

    async def test_slow_user_hook_does_not_block_inbound(self) -> None:
        """Slow user ON_SESSION_STARTED hook doesn't block process_inbound."""
        kit = RoomKit()
        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        agent = Agent(
            "agent-1",
            provider=MockAIProvider(responses=["Hi!"]),
            greeting="Welcome!",
            auto_greet=True,
        )
        kit.register_channel(sms)
        kit.register_channel(agent)

        @kit.hook(HookTrigger.ON_ROOM_CREATED, HookExecution.ASYNC)
        async def attach_agent(event: object, context: object) -> None:
            room_id = event.room_id  # type: ignore[attr-defined]
            await kit.attach_channel(room_id, "agent-1")

        slow_hook_started = asyncio.Event()
        slow_hook_finished = asyncio.Event()
        release = asyncio.Event()

        @kit.hook(HookTrigger.ON_SESSION_STARTED, HookExecution.ASYNC, name="slow_user_hook")
        async def slow_hook(event: object, context: object) -> None:
            slow_hook_started.set()
            await release.wait()  # as slow as the test keeps it
            slow_hook_finished.set()

        msg = InboundMessage(
            channel_id="sms-1",
            sender_id="+15551234567",
            content=TextContent(body="Hello"),
        )

        # process_inbound should return quickly (greeting stored, slow hook in background)
        result = await asyncio.wait_for(kit.process_inbound(msg), timeout=2.0)
        assert not result.blocked

        # Greeting should already be stored (internal hook completed)
        rooms = await kit._store.list_rooms()
        events = await kit._store.list_events(rooms[0].id)
        greeting_events = [e for e in events if e.metadata.get("auto_greeting") is True]
        assert len(greeting_events) == 1

        # The slow user hook still runs: process_inbound did not wait for it.
        assert not slow_hook_finished.is_set()
        release.set()
        await asyncio.wait_for(slow_hook_finished.wait(), timeout=2.0)

        await kit.close()
