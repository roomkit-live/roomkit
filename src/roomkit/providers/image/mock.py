"""Mock image provider for testing without a vendor key."""

from __future__ import annotations

import base64

from roomkit.providers.ai.base import AIImagePart
from roomkit.providers.image.base import ImageProvider, ImageResult, parse_size, to_data_uri

# A real 1x1 transparent PNG. Real bytes rather than a placeholder string so a
# consumer's test exercises the whole path — decode, write, measure — and a
# provider returning something undecodable fails in a test rather than on the
# day a key is configured.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class MockImageProvider(ImageProvider):
    """Image provider that returns canned images.

    Attributes:
        calls: Every ``(prompt, size, n, reference_images)`` this provider was
            asked for, in order — what a test asserts against.
    """

    def __init__(
        self,
        images: list[bytes] | None = None,
        *,
        model: str = "mock-image",
        mime_type: str = "image/png",
        supports_editing: bool = True,
        revised_prompt: str | None = None,
    ) -> None:
        self._images = images or [_PNG_1X1]
        self._model = model
        self._mime_type = mime_type
        self._supports_editing = supports_editing
        self._revised_prompt = revised_prompt
        self._index = 0
        self.calls: list[tuple[str, str | None, int, list[AIImagePart]]] = []

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_editing(self) -> bool:
        return self._supports_editing

    async def generate(
        self,
        prompt: str,
        *,
        size: str | None = None,
        n: int = 1,
        reference_images: list[AIImagePart] | None = None,
    ) -> list[ImageResult]:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if size is not None:
            parse_size(size)
        references = list(reference_images or [])
        if references and not self._supports_editing:
            raise ValueError(f"{self.name} does not support editing")
        self.calls.append((prompt, size, n, references))

        results: list[ImageResult] = []
        for _ in range(n):
            data = self._images[self._index % len(self._images)]
            self._index += 1
            results.append(
                ImageResult(
                    data=to_data_uri(data, self._mime_type),
                    mime_type=self._mime_type,
                    revised_prompt=self._revised_prompt,
                    usage={
                        "input_tokens": len(prompt.split()),
                        "input_image_tokens": len(references) * 100,
                        "output_tokens": 0,
                        "output_image_tokens": 1024,
                    },
                )
            )
        return results
