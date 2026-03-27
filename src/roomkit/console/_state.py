"""Shared mutable state for the RoomKit console dashboard."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in the conversation history."""

    role: str  # "user" or "assistant"
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ConsoleState:
    """Mutable state updated by hooks, read by the dashboard.

    Single-threaded asyncio — simple attribute assignment is safe.
    """

    # Audio levels (updated ~10Hz)
    input_level_db: float = -60.0
    output_level_db: float = -60.0
    is_speech: bool = False

    # Voice state: idle | listening | processing | speaking
    voice_state: str = "idle"

    # Session info
    session_id: str | None = None
    room_id: str | None = None
    channel_id: str | None = None
    participant_id: str | None = None
    session_started_at: datetime | None = None

    # Transcription
    partial_text: str = ""
    last_final_text: str = ""

    # TTS / streaming assistant text
    last_tts_text: str = ""
    partial_assistant_text: str = ""

    # Counters
    transcription_count: int = 0
    tts_count: int = 0
    barge_in_count: int = 0
    tool_call_count: int = 0

    # History of recent audio levels for waveform display
    input_level_history: deque[float] = field(
        default_factory=lambda: deque([-60.0] * 20, maxlen=20)
    )
    output_level_history: deque[float] = field(
        default_factory=lambda: deque([-60.0] * 20, maxlen=20)
    )

    # Conversation history (ring buffer of recent turns)
    conversation: deque[ConversationTurn] = field(default_factory=lambda: deque(maxlen=50))

    # Start time for uptime display
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def push_input_level(self, level_db: float) -> None:
        """Record a new input level sample, shifting the history window."""
        self.input_level_db = level_db
        self.input_level_history.append(level_db)

    def push_output_level(self, level_db: float) -> None:
        """Record a new output level sample, shifting the history window."""
        self.output_level_db = level_db
        self.output_level_history.append(level_db)


class LogRingBuffer(logging.Handler):
    """Logging handler that captures records into a ring buffer for display."""

    def __init__(self, max_records: int = 100) -> None:
        super().__init__()
        self.records_buffer: deque[logging.LogRecord] = deque(maxlen=max_records)

    def emit(self, record: logging.LogRecord) -> None:
        self.records_buffer.append(record)
