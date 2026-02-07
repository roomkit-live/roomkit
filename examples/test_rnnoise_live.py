"""Quick test: record mic audio and show RNNoise noise reduction stats.

Records a few seconds of mic audio, processes each frame through
RNNoise, and prints before/after RMS energy to prove denoising works.

Usage:
    uv run python examples/test_rnnoise_live.py

Stay quiet for the first 3 seconds (measures noise floor reduction),
then speak for the next 3 seconds (measures speech preservation).
"""

from __future__ import annotations

import math
import struct
import time

import sounddevice as sd

from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

SAMPLE_RATE = 16000
BLOCK_MS = 10
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000  # 160
DURATION_S = 6


def rms(samples: tuple[int, ...]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def main() -> None:
    dn = RNNoiseDenoiserProvider(sample_rate=SAMPLE_RATE)

    frames_in: list[tuple[int, ...]] = []
    frames_out: list[tuple[int, ...]] = []

    print(f"Recording {DURATION_S}s at {SAMPLE_RATE}Hz...")
    print("  0-3s: stay QUIET (measures noise floor)")
    print("  3-6s: SPEAK (measures speech preservation)")
    print()

    def callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
        from roomkit.voice.audio_frame import AudioFrame

        n = len(indata) // 2
        samples = struct.unpack(f"<{n}h", bytes(indata))
        frames_in.append(samples)

        frame = AudioFrame(
            data=bytes(indata), sample_rate=SAMPLE_RATE, channels=1, sample_width=2
        )
        result = dn.process(frame)
        out_samples = struct.unpack(f"<{n}h", result.data)
        frames_out.append(out_samples)

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SAMPLES,
        channels=1,
        dtype="int16",
        callback=callback,
    )
    stream.start()
    time.sleep(DURATION_S)
    stream.stop()
    stream.close()
    dn.close()

    # Split into quiet (first half) and speech (second half).
    mid = len(frames_in) // 2

    def stats(label: str, raw: list[tuple[int, ...]], denoised: list[tuple[int, ...]]) -> None:
        raw_rms = [rms(f) for f in raw]
        den_rms = [rms(f) for f in denoised]
        avg_raw = sum(raw_rms) / len(raw_rms) if raw_rms else 0
        avg_den = sum(den_rms) / len(den_rms) if den_rms else 0
        if avg_raw > 0:
            reduction_db = 20 * math.log10(avg_den / avg_raw) if avg_den > 0 else -99
        else:
            reduction_db = 0
        print(f"  {label}:")
        print(f"    Raw avg RMS:      {avg_raw:8.1f}")
        print(f"    Denoised avg RMS: {avg_den:8.1f}")
        print(f"    Change:           {reduction_db:+.1f} dB")

    print(f"Processed {len(frames_in)} frames ({len(frames_in) * BLOCK_MS}ms total)")
    print()
    stats("Silence (noise floor)", frames_in[:mid], frames_out[:mid])
    print()
    stats("Speech", frames_in[mid:], frames_out[mid:])
    print()
    print("If denoiser works: silence should show negative dB (noise reduced),")
    print("speech should show ~0 dB or small negative (speech preserved).")


if __name__ == "__main__":
    main()
