"""The voice test bench: a simulated phone, WAV helpers and a hook timeline.

Three bricks a voice scenario is written with:

- :class:`ScenarioVoiceBackend`: a pure-transport backend that plays a WAV
  as cadenced 20 ms frames and captures what the bot sends, per session;
- :class:`VoiceTrace`: the timeline of the voice hooks, with ``wait_for`` in
  place of ``asyncio.sleep``;
- :func:`read_wav`, :func:`write_wav`, :func:`pcm_frames`, :func:`silence`,
  :func:`tone`: stdlib WAV and PCM helpers around :class:`PCMAudio`.
"""

from __future__ import annotations

from roomkit.voice.testing.backend import ScenarioVoiceBackend
from roomkit.voice.testing.trace import VOICE_TRIGGERS, TraceEntry, VoiceTrace
from roomkit.voice.testing.wav import (
    DEFAULT_SAMPLE_RATE,
    PCMAudio,
    pcm_frames,
    read_wav,
    silence,
    tone,
    write_wav,
)

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "VOICE_TRIGGERS",
    "PCMAudio",
    "ScenarioVoiceBackend",
    "TraceEntry",
    "VoiceTrace",
    "pcm_frames",
    "read_wav",
    "silence",
    "tone",
    "write_wav",
]
