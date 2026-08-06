"""RoomKit — Show an image to a live OpenAI Realtime session.

`inject_image` puts a picture inside the speech-to-speech conversation itself,
so the model sees it in the same context as the audio.  That is a different
thing from the `video/vision` providers, which make a separate model call and
hand back text.

The example is deliberately headless: the transport is `MockVoiceBackend`, so
no microphone and no speakers are involved.  The model still answers with
audio, and its own transcription is printed here — which is the readable proof
that it looked at the image.

Image input needs a model that accepts it: `gpt-realtime-2.1` or later.  PNG
and JPEG are the only formats the API reads.

Requirements:
    pip install roomkit[realtime-openai]

Run with:
    OPENAI_API_KEY=... IMAGE=path/to/picture.png \\
        uv run python examples/realtime_voice_openai_image.py

Environment variables:
    OPENAI_API_KEY  (required) OpenAI API key
    IMAGE           (required) Path to a PNG or JPEG file
    OPENAI_MODEL    Model name (default: the provider's own default)
    PROMPT          What to ask about the image
    IMAGE_DETAIL    auto | low | high — fidelity against token cost.  Unset
                    means the API's default, which is high detail.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# How long to keep the session open waiting for the model's answer.
_ANSWER_TIMEOUT_S = 30.0


def _load_image() -> tuple[bytes, str]:
    """Read the image and map its extension to a MIME type the API reads."""
    path = Path(os.environ["IMAGE"]).expanduser()
    mime_type = _MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        raise SystemExit(f"{path.name}: only PNG and JPEG are accepted, got {path.suffix!r}")
    return path.read_bytes(), mime_type


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("IMAGE"):
        print("Set OPENAI_API_KEY and IMAGE to run this example.")
        print(
            "  OPENAI_API_KEY=... IMAGE=picture.png "
            "uv run python examples/realtime_voice_openai_image.py"
        )
        return

    image_data, mime_type = _load_image()

    model = os.environ.get("OPENAI_MODEL")
    provider = OpenAIRealtimeProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        **({"model": model} if model else {}),
    )

    answered = asyncio.Event()

    def on_transcription(session: VoiceSession, text: str, role: str, is_final: bool) -> None:
        if role != "assistant" or not is_final:
            return
        print(f"\nThe model says: {text}\n")
        answered.set()

    provider.on_transcription(on_transcription)

    kit = RoomKit()
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockVoiceBackend(),
        system_prompt="You look at images and describe them out loud, briefly.",
    )
    kit.register_channel(channel)

    await kit.create_room(room_id="image-demo")
    await kit.attach_channel("image-demo", "voice")

    # provider_config travels in the session metadata, and image_detail is
    # read at injection time from what was fixed here at connect time.
    detail = os.environ.get("IMAGE_DETAIL")
    session = await channel.start_session(
        "image-demo",
        "viewer",
        connection=None,
        metadata={"provider_config": {"image_detail": detail}} if detail else None,
    )

    print(f"Sending {len(image_data)} bytes of {mime_type} to the model…")
    await channel.inject_image(
        session,
        image_data,
        mime_type,
        prompt=os.environ.get("PROMPT", "What do you see in this image?"),
    )

    try:
        await asyncio.wait_for(answered.wait(), timeout=_ANSWER_TIMEOUT_S)
    except TimeoutError:
        print(f"No answer within {_ANSWER_TIMEOUT_S:.0f}s — check the model accepts image input.")
    finally:
        await channel.end_session(session)


if __name__ == "__main__":
    asyncio.run(main())
