#!/usr/bin/env python3
"""Audio path tracer for RoomKit debugging.

Monkey-patches key methods to log audio flow through the system using
periodic summaries instead of per-frame logging.  Run your example with
this module imported:

    python -c "import examples.trace_audio; exec(open('examples/voice_local_onnx_vllm.py').read())"

Or simply:  import examples.trace_audio  at the top of the example.

Traced points (logged as periodic summaries every ~2s):
  [INBOUND]    VoiceChannel._on_audio_received     — mic → pipeline
  [PIPELINE]   AudioPipeline.process_inbound        — inbound stage
  [PIPELINE]   AudioPipeline.process_outbound       — outbound stage
  [AEC]        AECProvider.process / feed_reference  — echo cancellation
  [VAD]        VADProvider.process                   — RMS, is_speech, state
  [STT-STREAM] VoiceChannel streaming STT lifecycle  — start/result/timing
  [DELIVER]    VoiceChannel._deliver_voice           — TTS → backend
  [BACKEND]    LocalAudioBackend.send_audio          — chunks to speaker
"""

from __future__ import annotations

import functools
import logging
import struct
import time
from collections import defaultdict

logger = logging.getLogger("roomkit.trace")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # prevent duplicate output via root logger

# Add a handler if none exist yet
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s TRACE %(message)s"))
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Counters and periodic summary
# ---------------------------------------------------------------------------
_counts: dict[str, int] = defaultdict(int)
_last_summary = time.monotonic()
_SUMMARY_INTERVAL = 2.0  # print summary every 2s

# Track state for change-only logging
_state: dict[str, object] = {}

# VAD-specific accumulator (richer than simple counters)
_vad_stats: dict[str, float] = {
    "frame_count": 0,
    "rms_sum": 0.0,
    "rms_max": 0.0,
    "speech_count": 0,
}

# Denoiser input/output RMS comparison
_denoiser_stats: dict[str, float] = {
    "frame_count": 0,
    "in_rms_sum": 0.0,
    "in_rms_max": 0.0,
    "out_rms_sum": 0.0,
    "out_rms_max": 0.0,
}


def _rms_int16(data: bytes) -> float:
    """Compute RMS of int16 little-endian PCM data."""
    n_samples = len(data) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", data[: n_samples * 2])
    sum_sq = sum(s * s for s in samples)
    return float((sum_sq / n_samples) ** 0.5)


def _flush_summary() -> None:
    """Print accumulated counters and reset."""
    global _last_summary
    now = time.monotonic()
    has_counts = any(_counts.values())
    has_vad = _vad_stats["frame_count"] > 0
    has_denoiser = _denoiser_stats["frame_count"] > 0
    if now - _last_summary < _SUMMARY_INTERVAL or not (has_counts or has_vad or has_denoiser):
        return

    elapsed = now - _last_summary
    parts = []
    for k in sorted(_counts):
        v = _counts[k]
        rate = v / elapsed
        parts.append(f"{k}={v} ({rate:.0f}/s)")
    if parts:
        logger.info("[SUMMARY] %s", " | ".join(parts))
    _counts.clear()

    # Emit denoiser summary (input vs output RMS)
    dfc = _denoiser_stats["frame_count"]
    if dfc > 0:
        in_avg = _denoiser_stats["in_rms_sum"] / dfc
        in_max = _denoiser_stats["in_rms_max"]
        out_avg = _denoiser_stats["out_rms_sum"] / dfc
        out_max = _denoiser_stats["out_rms_max"]
        logger.info(
            "[DENOISER] in_rms_avg=%.0f in_rms_max=%.0f out_rms_avg=%.0f out_rms_max=%.0f",
            in_avg,
            in_max,
            out_avg,
            out_max,
        )
        _denoiser_stats["frame_count"] = 0
        _denoiser_stats["in_rms_sum"] = 0.0
        _denoiser_stats["in_rms_max"] = 0.0
        _denoiser_stats["out_rms_sum"] = 0.0
        _denoiser_stats["out_rms_max"] = 0.0

    # Emit VAD summary alongside the counter summary
    fc = _vad_stats["frame_count"]
    if fc > 0:
        rms_avg = _vad_stats["rms_sum"] / fc
        rms_max = _vad_stats["rms_max"]
        sc = int(_vad_stats["speech_count"])
        # Read live state from the VAD provider (set by the wrapper)
        state = _state.get("vad_speaking", False)
        silence_ms = _state.get("vad_silence_ms", 0.0)
        speech_ms = _state.get("vad_speech_ms", 0.0)
        logger.info(
            "[VAD] state=%s is_speech=%d/%d rms_avg=%.0f rms_max=%.0f"
            " silence_ms=%.0f speech_ms=%.0f",
            "speaking" if state else "idle",
            sc,
            int(fc),
            rms_avg,
            rms_max,
            silence_ms,
            speech_ms,
        )
        _vad_stats["frame_count"] = 0
        _vad_stats["rms_sum"] = 0.0
        _vad_stats["rms_max"] = 0.0
        _vad_stats["speech_count"] = 0

    _last_summary = now


def _frame_info(frame: object) -> str:
    data = getattr(frame, "data", b"")
    sr = getattr(frame, "sample_rate", "?")
    ch = getattr(frame, "channels", "?")
    return f"{len(data)}B/{sr}Hz/{ch}ch"


# ---------------------------------------------------------------------------
# Patches — hot-path methods only count, state-change methods log
# ---------------------------------------------------------------------------


def _patch_voice_channel() -> None:
    from roomkit.channels.voice import VoiceChannel

    orig_on_audio = VoiceChannel._on_audio_received

    @functools.wraps(orig_on_audio)
    def traced_on_audio(self, session, frame):
        _counts["inbound"] += 1

        # Log once on first frame or when pipeline state changes
        key = f"inbound:{session.id[:8]}"
        has_pipeline = self._pipeline is not None
        if _state.get(key) != has_pipeline:
            _state[key] = has_pipeline
            logger.info(
                "[INBOUND] session=%s frame=%s pipeline=%s",
                session.id[:8],
                _frame_info(frame),
                has_pipeline,
            )

        _flush_summary()
        return orig_on_audio(self, session, frame)

    VoiceChannel._on_audio_received = traced_on_audio

    orig_deliver = VoiceChannel._deliver_voice

    @functools.wraps(orig_deliver)
    async def traced_deliver(self, event, binding, context):
        logger.info(
            "[DELIVER] _deliver_voice pipeline=%s tts=%s backend=%s",
            self._pipeline is not None,
            self._tts.name if self._tts else None,
            self._backend.name if self._backend else None,
        )
        return await orig_deliver(self, event, binding, context)

    VoiceChannel._deliver_voice = traced_deliver


def _patch_pipeline() -> None:
    from roomkit.voice.pipeline.engine import AudioPipeline

    orig_inbound = AudioPipeline.process_inbound

    @functools.wraps(orig_inbound)
    def traced_inbound(self, session, frame):
        _counts["pipeline_in"] += 1

        # Log once when AEC config changes
        has_aec = self._config.aec is not None
        aec_name = self._config.aec.name if has_aec else None
        key = f"pipeline_aec:{session.id[:8]}"
        if _state.get(key) != aec_name:
            _state[key] = aec_name
            logger.info(
                "[PIPELINE] process_inbound session=%s frame=%s aec=%s",
                session.id[:8],
                _frame_info(frame),
                aec_name,
            )

        return orig_inbound(self, session, frame)

    AudioPipeline.process_inbound = traced_inbound

    orig_outbound = AudioPipeline.process_outbound

    @functools.wraps(orig_outbound)
    def traced_outbound(self, session, frame):
        _counts["pipeline_out"] += 1

        # Log first outbound frame per session
        key = f"pipeline_out_seen:{session.id[:8]}"
        if key not in _state:
            _state[key] = True
            aec_name = self._config.aec.name if self._config.aec else None
            inbound_rate = getattr(self, "_inbound_sample_rate", None)
            needs_resample = inbound_rate is not None and frame.sample_rate != inbound_rate
            logger.info(
                "[PIPELINE] process_outbound session=%s frame=%s aec=%s "
                "inbound_rate=%s resample=%s",
                session.id[:8],
                _frame_info(frame),
                aec_name,
                inbound_rate,
                f"{frame.sample_rate}->{inbound_rate}" if needs_resample else "no",
            )

        result = orig_outbound(self, session, frame)
        return result

    AudioPipeline.process_outbound = traced_outbound


def _make_aec_wrapper(orig_process, orig_feed):
    """Create traced wrappers that delegate to the given originals."""

    @functools.wraps(orig_process)
    def traced_process(self, frame):
        _counts["aec.process"] += 1
        key = f"aec_process_seen:{self.name}"
        if key not in _state:
            _state[key] = True
            logger.info("[AEC] %s.process frame=%s", self.name, _frame_info(frame))
        return orig_process(self, frame)

    @functools.wraps(orig_feed)
    def traced_feed(self, frame):
        _counts["aec.feed_ref"] += 1
        # Log first call and then only on rate changes (not every call —
        # transport-level feeding runs at 50/s from the audio thread).
        key = f"aec_feed_seen:{self.name}"
        rate = getattr(frame, "sample_rate", None)
        prev = _state.get(key)
        if prev != rate:
            _state[key] = rate
            logger.info("[AEC-REF] %s.feed_reference frame=%s", self.name, _frame_info(frame))
        return orig_feed(self, frame)

    return traced_process, traced_feed


def _patch_aec() -> None:
    # Patch each concrete class with wrappers that call ITS original method
    # (not the abstract base). Only patch the base as a fallback for unknown
    # subclasses.
    import importlib

    from roomkit.voice.pipeline.aec.base import AECProvider

    for mod, cls_name in [
        ("roomkit.voice.pipeline.aec.speex", "SpeexAECProvider"),
        ("roomkit.voice.pipeline.aec.webrtc", "WebRTCAECProvider"),
    ]:
        try:
            m = importlib.import_module(mod)
            cls = getattr(m, cls_name)
            tp, tf = _make_aec_wrapper(cls.process, cls.feed_reference)
            cls.process = tp
            cls.feed_reference = tf
        except (ImportError, AttributeError):
            pass

    # Fallback: patch the ABC for any other subclass we haven't imported
    tp, tf = _make_aec_wrapper(AECProvider.process, AECProvider.feed_reference)
    AECProvider.process = tp
    AECProvider.feed_reference = tf


def _patch_local_backend() -> None:
    try:
        from roomkit.voice.backends.local import LocalAudioBackend
    except ImportError:
        return

    orig_send = LocalAudioBackend.send_audio

    @functools.wraps(orig_send)
    async def traced_send(self, session, audio):
        is_stream = not isinstance(audio, bytes)
        has_aec = self._aec is not None
        logger.info(
            "[BACKEND] send_audio session=%s streaming=%s backend_aec=%s",
            session.id[:8],
            is_stream,
            has_aec,
        )
        return await orig_send(self, session, audio)

    LocalAudioBackend.send_audio = traced_send

    # Trace transport-level AEC reference feeding (counter only)
    if hasattr(LocalAudioBackend, "_aec_feed_played"):
        orig_aec_feed = LocalAudioBackend._aec_feed_played

        @functools.wraps(orig_aec_feed)
        def traced_aec_feed(self, data):
            _counts["backend_aec_ref"] += 1
            return orig_aec_feed(self, data)

        LocalAudioBackend._aec_feed_played = traced_aec_feed


def _patch_vad() -> None:
    import importlib

    from roomkit.voice.pipeline.vad.base import VADProvider

    def _make_vad_wrapper(orig_process):
        @functools.wraps(orig_process)
        def traced_process(self, frame):
            # Compute RMS on the frame entering the VAD
            rms = _rms_int16(frame.data)
            _vad_stats["frame_count"] += 1
            _vad_stats["rms_sum"] += rms
            if rms > _vad_stats["rms_max"]:
                _vad_stats["rms_max"] = rms

            result = orig_process(self, frame)

            # Peek at is_speech: check sherpa detector or energy threshold
            detector = getattr(self, "_detector", None)
            if detector is not None and hasattr(detector, "is_speech_detected"):
                if detector.is_speech_detected():
                    _vad_stats["speech_count"] += 1
            elif hasattr(self, "_energy_threshold") and rms >= self._energy_threshold:
                _vad_stats["speech_count"] += 1

            # Snapshot live state for the summary line
            _state["vad_speaking"] = getattr(self, "_speaking", False)
            _state["vad_silence_ms"] = getattr(self, "_silence_ms", 0.0)
            _state["vad_speech_ms"] = getattr(self, "_speech_ms", 0.0)

            _counts["vad"] += 1
            return result

        return traced_process

    # Patch concrete VAD classes
    for mod, cls_name in [
        ("roomkit.voice.pipeline.vad.sherpa_onnx", "SherpaOnnxVADProvider"),
        ("roomkit.voice.pipeline.vad.energy", "EnergyVADProvider"),
    ]:
        try:
            m = importlib.import_module(mod)
            cls = getattr(m, cls_name)
            cls.process = _make_vad_wrapper(cls.process)
        except (ImportError, AttributeError):
            pass

    # Fallback on ABC for unknown subclasses
    VADProvider.process = _make_vad_wrapper(VADProvider.process)


def _patch_denoiser() -> None:
    import importlib

    from roomkit.voice.pipeline.denoiser.base import DenoiserProvider

    def _make_denoiser_wrapper(orig_process):
        @functools.wraps(orig_process)
        def traced_process(self, frame):
            _counts["denoiser"] += 1
            in_rms = _rms_int16(frame.data)
            result = orig_process(self, frame)
            out_rms = _rms_int16(result.data)
            _denoiser_stats["frame_count"] += 1
            _denoiser_stats["in_rms_sum"] += in_rms
            if in_rms > _denoiser_stats["in_rms_max"]:
                _denoiser_stats["in_rms_max"] = in_rms
            _denoiser_stats["out_rms_sum"] += out_rms
            if out_rms > _denoiser_stats["out_rms_max"]:
                _denoiser_stats["out_rms_max"] = out_rms
            return result

        return traced_process

    for mod, cls_name in [
        ("roomkit.voice.pipeline.denoiser.rnnoise", "RNNoiseDenoiserProvider"),
        ("roomkit.voice.pipeline.denoiser.sherpa_onnx", "SherpaOnnxDenoiserProvider"),
    ]:
        try:
            m = importlib.import_module(mod)
            cls = getattr(m, cls_name)
            cls.process = _make_denoiser_wrapper(cls.process)
        except (ImportError, AttributeError):
            pass

    DenoiserProvider.process = _make_denoiser_wrapper(DenoiserProvider.process)


def _patch_stt_stream() -> None:
    """Trace streaming STT lifecycle in VoiceChannel."""
    from roomkit.channels.voice import VoiceChannel

    # Track per-session timing
    _stt_stream_timing: dict[str, float] = {}  # session_id -> speech_start monotonic

    orig_start = VoiceChannel._start_stt_stream

    @functools.wraps(orig_start)
    def traced_start(self, session, room_id, pre_roll=None):
        sid = session.id[:8]
        _stt_stream_timing[session.id] = time.monotonic()
        supports = self._stt.supports_streaming if self._stt else False
        pre_roll_ms = 0
        if pre_roll:
            sr = session.metadata.get("input_sample_rate", 16000)
            pre_roll_ms = len(pre_roll) / (sr * 2) * 1000  # 16-bit mono
        logger.info(
            "[STT-STREAM] start session=%s provider=%s supports_streaming=%s "
            "pre_roll=%dB (%.0fms)",
            sid,
            self._stt.name if self._stt else None,
            supports,
            len(pre_roll) if pre_roll else 0,
            pre_roll_ms,
        )
        return orig_start(self, session, room_id, pre_roll=pre_roll)

    VoiceChannel._start_stt_stream = traced_start

    orig_flush = VoiceChannel._flush_stt_buffer

    @functools.wraps(orig_flush)
    def traced_flush(self, state, session_id):
        buf_bytes = len(state.frame_buffer)
        if buf_bytes > 0:
            _counts["stt_flush"] += 1
            _counts["stt_bytes"] += buf_bytes
        return orig_flush(self, state, session_id)

    VoiceChannel._flush_stt_buffer = traced_flush

    orig_speech_frame = VoiceChannel._on_pipeline_speech_frame

    @functools.wraps(orig_speech_frame)
    def traced_speech_frame(self, session, frame):
        _counts["stt_frames"] += 1
        return orig_speech_frame(self, session, frame)

    VoiceChannel._on_pipeline_speech_frame = traced_speech_frame

    orig_process_end = VoiceChannel._process_speech_end

    @functools.wraps(orig_process_end)
    async def traced_process_end(self, session, audio, room_id):
        sid = session.id[:8]
        t_start = time.monotonic()
        stream_state = self._stt_streams.get(session.id)
        had_stream = stream_state is not None
        stream_ok = had_stream and not stream_state.error and not stream_state.cancelled
        t_speech_start = _stt_stream_timing.pop(session.id, None)

        final_text = None
        error = None
        try:
            await orig_process_end(self, session, audio, room_id)
        except Exception as exc:
            error = exc
            raise
        finally:
            t_end = time.monotonic()
            stt_ms = (t_end - t_start) * 1000
            speech_dur_ms = (t_start - t_speech_start) * 1000 if t_speech_start else 0

            # Determine which path was used
            used_stream = False
            if stream_ok and stream_state is not None:
                final_text = stream_state.final_text
                used_stream = final_text is not None

            logger.info(
                "[STT-STREAM] result session=%s mode=%s stt_latency=%.0fms "
                "speech_dur=%.0fms audio=%dB had_stream=%s "
                "final_text=%r error=%s",
                sid,
                "stream" if used_stream else "batch",
                stt_ms,
                speech_dur_ms,
                len(audio),
                had_stream,
                final_text,
                error,
            )

    VoiceChannel._process_speech_end = traced_process_end


def install() -> None:
    """Install all trace patches."""
    logger.info("=== RoomKit audio trace installed ===")
    _patch_voice_channel()
    _patch_pipeline()
    _patch_aec()
    _patch_vad()
    _patch_denoiser()
    _patch_local_backend()
    _patch_stt_stream()
    logger.info("=== Tracing active (summaries every %.0fs) ===", _SUMMARY_INTERVAL)


# Auto-install on import
install()
