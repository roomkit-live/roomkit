"""Image generation abstractions and mock implementation (RFC §25)."""

from roomkit.providers.image.base import (
    IMAGE_GEN_CAPABILITY,
    ImageAttempt,
    ImageGenerationError,
    ImageProgressCallback,
    ImageProvider,
    ImageResult,
    parse_data_uri,
    parse_size,
    payload_mime_type,
    sniff_mime_type,
    to_data_uri,
)
from roomkit.providers.image.mock import MockImageProvider
from roomkit.providers.image.options import ImageCapabilities, ImageModelInfo, ImageOptions

__all__ = [
    "IMAGE_GEN_CAPABILITY",
    "ImageAttempt",
    "ImageCapabilities",
    "ImageGenerationError",
    "ImageModelInfo",
    "ImageOptions",
    "ImageProgressCallback",
    "ImageProvider",
    "ImageResult",
    "MockImageProvider",
    "parse_data_uri",
    "parse_size",
    "payload_mime_type",
    "sniff_mime_type",
    "to_data_uri",
]
