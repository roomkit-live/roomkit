"""Tests for the roomkit.console module."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from roomkit import RoomKit
from roomkit.console._hooks import (
    HOOK_PREFIX,
    register_console_hooks,
    unregister_console_hooks,
)
from roomkit.console._state import ConsoleState, ConversationTurn, LogRingBuffer

# ---------------------------------------------------------------------------
# ConsoleState tests
# ---------------------------------------------------------------------------


class TestConsoleState:
    def test_defaults(self):
        state = ConsoleState()
        assert state.input_level_db == -60.0
        assert state.output_level_db == -60.0
        assert state.is_speech is False
        assert state.voice_state == "idle"
        assert state.session_id is None
        assert state.room_id is None
        assert state.transcription_count == 0
        assert len(state.input_level_history) == 20
        assert len(state.conversation) == 0

    def test_push_input_level(self):
        state = ConsoleState()
        state.push_input_level(-30.0)
        assert state.input_level_db == -30.0
        assert state.input_level_history[-1] == -30.0
        assert len(state.input_level_history) == 20

    def test_push_output_level(self):
        state = ConsoleState()
        state.push_output_level(-15.0)
        assert state.output_level_db == -15.0
        assert state.output_level_history[-1] == -15.0

    def test_history_overflow(self):
        state = ConsoleState()
        for i in range(30):
            state.push_input_level(float(-i))
        assert len(state.input_level_history) == 20
        assert state.input_level_history[-1] == -29.0

    def test_conversation_ring_buffer(self):
        state = ConsoleState()
        for i in range(60):
            state.conversation.append(ConversationTurn(role="user", text=f"msg {i}"))
        # maxlen=50
        assert len(state.conversation) == 50
        assert state.conversation[0].text == "msg 10"


# ---------------------------------------------------------------------------
# LogRingBuffer tests
# ---------------------------------------------------------------------------


class TestLogRingBuffer:
    def test_captures_records(self):
        buf = LogRingBuffer(max_records=10)
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        buf.emit(record)
        assert len(buf.records_buffer) == 1
        assert buf.records_buffer[0].getMessage() == "hello"

    def test_ring_overflow(self):
        buf = LogRingBuffer(max_records=5)
        for i in range(10):
            record = logging.LogRecord("test", logging.INFO, "", 0, f"msg {i}", (), None)
            buf.emit(record)
        assert len(buf.records_buffer) == 5
        assert buf.records_buffer[0].getMessage() == "msg 5"


# ---------------------------------------------------------------------------
# Hook registration tests
# ---------------------------------------------------------------------------


class TestHookRegistration:
    async def test_register_and_unregister(self):
        kit = RoomKit()
        state = ConsoleState()

        names = register_console_hooks(kit.hook_engine, state)
        assert len(names) == 12
        assert all(n.startswith(HOOK_PREFIX) for n in names)

        for name in names:
            found = any(h.name == name for h in kit.hook_engine._global_hooks)
            assert found, f"Hook {name} not found"

        unregister_console_hooks(kit.hook_engine, names)
        for name in names:
            found = any(h.name == name for h in kit.hook_engine._global_hooks)
            assert not found, f"Hook {name} still present"

    async def test_input_level_hook_updates_state(self):
        kit = RoomKit()
        state = ConsoleState()
        register_console_hooks(kit.hook_engine, state)

        hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}input_level"
        )
        await hook.fn(MagicMock(level_db=-25.0), None)

        assert state.input_level_db == -25.0
        assert state.input_level_history[-1] == -25.0

    async def test_vad_level_hook_updates_state(self):
        kit = RoomKit()
        state = ConsoleState()
        register_console_hooks(kit.hook_engine, state)

        hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}vad_level"
        )
        await hook.fn(MagicMock(level_db=-10.0, is_speech=True), None)

        assert state.input_level_db == -10.0
        assert state.is_speech is True

    async def test_session_started_hook(self):
        kit = RoomKit()
        state = ConsoleState()
        register_console_hooks(kit.hook_engine, state)

        hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}session_started"
        )
        mock_session = MagicMock(id="sess-abc123")
        await hook.fn(
            MagicMock(
                participant_id="user-1",
                room_id="room-1",
                channel_id="voice-1",
                session=mock_session,
                timestamp=MagicMock(),
            ),
            None,
        )

        assert state.session_id == "sess-abc123"
        assert state.participant_id == "user-1"
        assert state.room_id == "room-1"
        assert state.channel_id == "voice-1"

    async def test_speech_start_end_hooks(self):
        kit = RoomKit()
        state = ConsoleState()
        register_console_hooks(kit.hook_engine, state)

        start_hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}speech_start"
        )
        end_hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}speech_end"
        )

        await start_hook.fn(MagicMock(), None)
        assert state.voice_state == "listening"

        await end_hook.fn(MagicMock(), None)
        assert state.voice_state == "processing"

    async def test_transcription_hooks_user(self):
        """Standard TranscriptionEvent (VoiceChannel) — no role attr, always user."""
        kit = RoomKit()
        state = ConsoleState()
        register_console_hooks(kit.hook_engine, state)

        partial_hook = next(
            h
            for h in kit.hook_engine._global_hooks
            if h.name == f"{HOOK_PREFIX}partial_transcription"
        )
        final_hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}transcription"
        )

        await partial_hook.fn(MagicMock(text="hello wo"), None)
        assert state.partial_text == "hello wo"

        # TranscriptionEvent has no .role — defaults to "user"
        event = MagicMock(text="hello world", spec=["text"])
        await final_hook.fn(event, None)
        assert state.last_final_text == "hello world"
        assert state.partial_text == ""
        assert state.transcription_count == 1
        assert len(state.conversation) == 1
        assert state.conversation[0].role == "user"

    async def test_transcription_hooks_realtime(self):
        """RealtimeTranscriptionEvent — role=user and role=assistant."""
        kit = RoomKit()
        state = ConsoleState()
        register_console_hooks(kit.hook_engine, state)

        hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}transcription"
        )

        # User transcription
        user_event = MagicMock(text="bonjour", role="user", is_final=True)
        await hook.fn(user_event, None)
        assert state.transcription_count == 1
        assert state.conversation[-1].role == "user"
        assert state.conversation[-1].text == "bonjour"

        # Assistant transcription
        ai_event = MagicMock(text="salut!", role="assistant", is_final=True)
        await hook.fn(ai_event, None)
        assert state.tts_count == 1
        assert state.last_tts_text == "salut!"
        assert state.conversation[-1].role == "assistant"
        assert state.conversation[-1].text == "salut!"
        assert state.voice_state == "idle"

    async def test_tts_hooks_with_conversation(self):
        kit = RoomKit()
        state = ConsoleState()
        register_console_hooks(kit.hook_engine, state)

        before_hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}before_tts"
        )
        after_hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}after_tts"
        )

        await before_hook.fn("Hello there!", None)
        assert state.voice_state == "speaking"
        assert state.last_tts_text == "Hello there!"
        assert state.tts_count == 1
        # Should also add to conversation
        assert len(state.conversation) == 1
        assert state.conversation[0].role == "assistant"
        assert state.conversation[0].text == "Hello there!"

        await after_hook.fn("Hello there!", None)
        assert state.voice_state == "idle"

    async def test_barge_in_hook(self):
        kit = RoomKit()
        state = ConsoleState()
        register_console_hooks(kit.hook_engine, state)

        hook = next(h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}barge_in")
        await hook.fn(MagicMock(), None)
        assert state.barge_in_count == 1
        assert state.voice_state == "listening"

    async def test_tool_call_hook(self):
        kit = RoomKit()
        state = ConsoleState()
        register_console_hooks(kit.hook_engine, state)

        hook = next(
            h for h in kit.hook_engine._global_hooks if h.name == f"{HOOK_PREFIX}tool_call"
        )
        await hook.fn(MagicMock(), None)
        assert state.tool_call_count == 1


# ---------------------------------------------------------------------------
# Display rendering tests
# ---------------------------------------------------------------------------


class TestDisplay:
    def test_db_to_ratio(self):
        from roomkit.console._display import _db_to_ratio

        assert _db_to_ratio(-60.0) == pytest.approx(0.0)
        assert _db_to_ratio(0.0) == pytest.approx(1.0)
        assert _db_to_ratio(-30.0) == pytest.approx(0.5)
        assert _db_to_ratio(-100.0) == pytest.approx(0.0)
        assert _db_to_ratio(10.0) == pytest.approx(1.0)

    def test_render_meter(self):
        from roomkit.console._display import _render_meter

        silent = [-60.0] * 20
        text = _render_meter(silent, width=20)
        assert len(text.plain) == 20

        loud = [0.0] * 20
        text = _render_meter(loud, width=20)
        assert len(text.plain) == 20
        assert all(c == "\u2588" for c in text.plain)

    def test_truncate(self):
        from roomkit.console._display import _truncate

        assert _truncate("short", 10) == "short"
        assert _truncate("a very long string", 10) == "a very lo\u2026"
        assert _truncate("exact", 5) == "exact"

    def test_format_uptime(self):
        from datetime import UTC, datetime, timedelta

        from roomkit.console._display import _format_uptime

        now = datetime.now(UTC)
        assert _format_uptime(now - timedelta(seconds=5)) == "0:00:05"
        assert _format_uptime(now - timedelta(hours=1, minutes=23, seconds=45)) == "1:23:45"

    def test_build_dashboard(self):
        from roomkit.console._display import _build_dashboard

        state = ConsoleState()
        state.room_id = "test-room"
        state.voice_state = "listening"
        state.conversation.append(ConversationTurn(role="user", text="hello"))
        log_buf = LogRingBuffer()

        layout = _build_dashboard(state, log_buf)
        assert layout is not None

    def test_build_log_panel_with_records(self):
        from roomkit.console._display import _build_log_panel

        buf = LogRingBuffer()
        record = logging.LogRecord("test", logging.INFO, "", 0, "test message", (), None)
        buf.emit(record)

        panel = _build_log_panel(buf, content_lines=20)
        assert panel is not None

    def test_build_conversation_panel_empty(self):
        from roomkit.console._display import _build_conversation_panel

        state = ConsoleState()
        panel = _build_conversation_panel(state, content_lines=20)
        assert panel is not None

    def test_build_conversation_panel_with_turns(self):
        from roomkit.console._display import _build_conversation_panel

        state = ConsoleState()
        state.conversation.append(ConversationTurn(role="user", text="hello"))
        state.conversation.append(ConversationTurn(role="assistant", text="hi there!"))
        state.partial_text = "how are"

        panel = _build_conversation_panel(state, content_lines=20)
        assert panel is not None


# ---------------------------------------------------------------------------
# RoomKitConsole integration tests
# ---------------------------------------------------------------------------


class TestRoomKitConsole:
    @patch("roomkit.console._display.sys")
    async def test_console_non_tty(self, mock_sys):
        """When not a TTY, Live display is skipped but hooks still register."""
        mock_sys.stdout.isatty.return_value = False

        console = __import__("roomkit.console._display", fromlist=["RoomKitConsole"])
        kit = RoomKit()
        rkc = console.RoomKitConsole(kit)

        assert rkc._live is None
        assert rkc._refresh_task is None
        assert len(rkc._hook_names) == 12

        await rkc.stop()
        assert len(rkc._hook_names) == 0

    @patch("roomkit.console._display.sys")
    async def test_console_stop_cleans_up(self, mock_sys):
        """stop() removes hooks and restores logging."""
        mock_sys.stdout.isatty.return_value = False

        console = __import__("roomkit.console._display", fromlist=["RoomKitConsole"])
        kit = RoomKit()
        rkc = console.RoomKitConsole(kit)

        hook_count_before = len(kit.hook_engine._global_hooks)
        assert hook_count_before >= 12

        await rkc.stop()

        console_hooks = [
            h for h in kit.hook_engine._global_hooks if h.name.startswith(HOOK_PREFIX)
        ]
        assert len(console_hooks) == 0

    @patch("roomkit.console._display.sys")
    async def test_state_accessible(self, mock_sys):
        """The state property gives access to ConsoleState."""
        mock_sys.stdout.isatty.return_value = False

        console = __import__("roomkit.console._display", fromlist=["RoomKitConsole"])
        kit = RoomKit()
        rkc = console.RoomKitConsole(kit)

        assert isinstance(rkc.state, ConsoleState)
        assert rkc.state.voice_state == "idle"

        await rkc.stop()

    @patch("roomkit.console._display.sys")
    async def test_log_buffer_captures(self, mock_sys):
        """Log messages are captured into the ring buffer."""
        mock_sys.stdout.isatty.return_value = False

        console = __import__("roomkit.console._display", fromlist=["RoomKitConsole"])
        kit = RoomKit()
        rkc = console.RoomKitConsole(kit)

        test_logger = logging.getLogger("test.console")
        test_logger.info("captured message")

        assert any(r.getMessage() == "captured message" for r in rkc._log_buffer.records_buffer)

        await rkc.stop()


# ---------------------------------------------------------------------------
# Import guard test
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_module_exports(self):
        from roomkit.console import RoomKitConsole

        assert RoomKitConsole is not None
