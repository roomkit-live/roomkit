"""Deepgram Voice Agent provider (speech-to-speech).

Deepgram's transcription provider lives beside the STT contract it implements,
in :mod:`roomkit.voice.stt.deepgram`; this package holds the Voice Agent
implementation of :class:`~roomkit.voice.realtime.provider.RealtimeVoiceProvider`.
The two are self-contained — same vendor, different protocol.
"""

from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.deepgram.realtime import DeepgramAgentProvider

__all__ = ["DeepgramAgentConfig", "DeepgramAgentProvider"]
