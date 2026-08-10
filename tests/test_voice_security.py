"""RFC §17.6 — DTMF redaction and recording encryption at rest.

Two MUSTs: configurable masking of DTMF digits wherever the framework
exposes them, and configurable encryption of stored recordings.
"""

from __future__ import annotations

from pathlib import Path

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import (
    AudioPipeline,
    AudioPipelineConfig,
    DTMFRedaction,
    MockDTMFDetector,
    MockVADProvider,
    RecordingConfig,
    RecordingEncryption,
    RecordingHandle,
    WavFileRecorder,
)
from roomkit.voice.pipeline.dtmf.base import DTMFEvent


def _session() -> VoiceSession:
    return VoiceSession(id="sess-1", room_id="r1", participant_id="p1", channel_id="voice-1")


class TestDTMFRedaction:
    def test_masks_everything_by_default(self) -> None:
        """A redaction that leaks the head of a PIN by default would be a
        worse trap than none at all."""
        r = DTMFRedaction()
        assert r.mask("4111111111111111") == "*" * 16
        assert r.mask("7") == "*"

    def test_card_shape_keeps_the_edges(self) -> None:
        r = DTMFRedaction(keep_first=4, keep_last=4)
        assert r.mask("4111111111111111") == "4111********1111"

    def test_short_sequence_is_masked_entirely(self) -> None:
        """Keeping edges is context on a long number, not a peephole onto a
        short secret."""
        r = DTMFRedaction(keep_first=4, keep_last=4)
        assert r.mask("1234") == "****"
        assert r.mask("12345") == "*****"

    def test_disabled_is_a_passthrough(self) -> None:
        assert DTMFRedaction(enabled=False).mask("1234") == "1234"

    def test_rejects_nonsense_configuration(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            DTMFRedaction(keep_first=-1)
        with pytest.raises(ValueError):
            DTMFRedaction(mask_char="**")

    def test_frame_metadata_carries_the_masked_digit(self) -> None:
        """RFC §17.6 — frame metadata reaches recorders, debug taps and logs,
        so that is where the raw digit must not survive."""
        detector = MockDTMFDetector(events=[DTMFEvent(digit="7", duration_ms=80.0)])
        pipeline = AudioPipeline(
            AudioPipelineConfig(
                vad=MockVADProvider(),
                dtmf=detector,
                dtmf_redaction=DTMFRedaction(),
            )
        )
        seen: list[DTMFEvent] = []
        pipeline.on_dtmf(lambda subject, event: seen.append(event))

        frame = AudioFrame(data=b"\x00\x00" * 160, sample_rate=16000)
        pipeline.process_inbound(_session(), frame)

        assert frame.metadata["dtmf"]["digit"] == "*"
        assert frame.metadata["dtmf"]["redacted"] is True
        # The detector's own event still carries the digit — the ON_DTMF hook
        # is how an IVR reads what it exists to collect.
        assert seen and seen[0].digit == "7"

    def test_no_redaction_configured_leaves_the_digit(self) -> None:
        detector = MockDTMFDetector(events=[DTMFEvent(digit="7", duration_ms=80.0)])
        pipeline = AudioPipeline(AudioPipelineConfig(vad=MockVADProvider(), dtmf=detector))
        frame = AudioFrame(data=b"\x00\x00" * 160, sample_rate=16000)
        pipeline.process_inbound(_session(), frame)
        assert frame.metadata["dtmf"]["digit"] == "7"
        assert frame.metadata["dtmf"]["redacted"] is False


class _ReversingEncryption(RecordingEncryption):
    """Stand-in cipher: the framework owns the *when*, not the *how*."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "reversing"

    def encrypt_file(self, path: str) -> str:
        self.calls.append(path)
        src = Path(path)
        dst = src.with_suffix(src.suffix + ".enc")
        dst.write_bytes(src.read_bytes()[::-1])
        src.unlink()
        return str(dst)


class _FailingEncryption(RecordingEncryption):
    @property
    def name(self) -> str:
        return "failing"

    def encrypt_file(self, path: str) -> str:
        raise RuntimeError("no key available")


def _record_briefly(config: RecordingConfig) -> tuple[WavFileRecorder, RecordingHandle]:
    recorder = WavFileRecorder()
    handle = recorder.start(_session(), config)
    recorder.tap_inbound(handle, AudioFrame(data=b"\x01\x02" * 800, sample_rate=16000))
    return recorder, handle


class TestRecordingEncryptionAtRest:
    def test_finished_recording_is_encrypted_and_plaintext_removed(self, tmp_path: Path) -> None:
        """RFC §17.6 — "Implementations MUST support configurable encryption
        for stored recordings"."""
        cipher = _ReversingEncryption()
        config = RecordingConfig(storage=str(tmp_path), encryption=cipher)
        recorder, handle = _record_briefly(config)

        result = recorder.stop(handle)
        recorder.close()

        assert cipher.calls, "the recorder never asked the cipher to encrypt"
        assert result.urls and all(url.endswith(".enc") for url in result.urls)
        assert result.metadata["encryption"] == "reversing"
        assert list(tmp_path.glob("*.wav")) == []
        assert list(tmp_path.glob("*.wav.enc"))

    def test_unencryptable_recording_is_discarded_not_left_in_the_clear(
        self, tmp_path: Path
    ) -> None:
        """A caller who asked for encryption at rest must not be handed a
        plaintext recording because the cipher failed."""
        config = RecordingConfig(storage=str(tmp_path), encryption=_FailingEncryption())
        recorder, handle = _record_briefly(config)

        result = recorder.stop(handle)
        recorder.close()

        assert result.urls == []
        assert list(tmp_path.glob("*.wav")) == []

    def test_without_encryption_nothing_changes(self, tmp_path: Path) -> None:
        config = RecordingConfig(storage=str(tmp_path))
        recorder, handle = _record_briefly(config)

        result = recorder.stop(handle)
        recorder.close()

        assert result.urls and all(url.endswith(".wav") for url in result.urls)
        assert "encryption" not in result.metadata
