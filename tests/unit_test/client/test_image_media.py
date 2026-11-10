# SPDX-License-Identifier: Apache-2.0
"""Inline image serialization and integrity at client boundaries."""

import base64
import hashlib
import io
from typing import Literal

import pytest
from PIL import Image

from sglang_omni.client.image_media import InlineImage, InlineMedia


@pytest.mark.parametrize(
    "image_format,mime_type",
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")],
)
def test_inline_image_round_trip(
    image_format: Literal["PNG", "JPEG", "WEBP"],
    mime_type: Literal["image/png", "image/jpeg", "image/webp"],
) -> None:
    with io.BytesIO() as buffer:
        Image.new("RGB", (4, 2), "red").save(buffer, format=image_format)
        image_bytes = buffer.getvalue()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    if image_format == "PNG":
        media = InlineImage.from_png(encoded)
    else:
        media = InlineImage.model_validate(
            {
                "kind": "image",
                "mime_type": mime_type,
                "size_bytes": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "url": f"data:{mime_type};base64,{encoded}",
            }
        )
    restored = InlineImage.model_validate_json(media.model_dump_json())
    assert restored.mime_type == mime_type
    assert base64.b64decode(restored.encoded_image(), validate=True) == image_bytes
    assert restored.size_bytes == len(image_bytes)
    assert restored.sha256 == hashlib.sha256(image_bytes).hexdigest()


@pytest.mark.parametrize("field", ["sha256", "size_bytes", "url", "mime_type"])
def test_inline_media_rejects_tampered_content(
    field: Literal["sha256", "size_bytes", "url", "mime_type"],
) -> None:
    media = InlineImage.from_png(base64.b64encode(b"image bytes").decode("ascii"))
    changes = {
        "sha256": "0" * 64,
        "size_bytes": media.size_bytes + 1,
        "url": "https://example.invalid/image.png",
        "mime_type": "image/jpeg",
    }
    altered = media.model_dump()
    altered[field] = changes[field]
    with pytest.raises(ValueError):
        InlineImage.model_validate(altered).encoded_image()


def test_inline_media_rejects_malformed_base64() -> None:
    with pytest.raises(ValueError):
        InlineImage.from_png("!invalid!")
    media = InlineImage.from_png("eA==")
    altered = media.model_dump()
    altered["url"] = "data:image/png;base64,!invalid!"
    with pytest.raises(ValueError):
        InlineImage.model_validate(altered).encoded_image()


@pytest.mark.parametrize(
    "mime_type", ["image/png", "audio/wav", "audio/mpeg", "audio/flac"]
)
def test_inline_media_round_trips_every_kind(mime_type: str) -> None:
    content = b"media bytes"
    media = InlineMedia.from_bytes(mime_type, content)
    restored = InlineMedia.from_data_url(media.url)
    assert restored == media
    assert restored.kind == mime_type.split("/")[0]
    assert base64.b64decode(restored.encoded_content()) == content


def test_inline_media_rejects_mismatched_kinds() -> None:
    media = InlineMedia.from_bytes("audio/wav", b"audio")
    with pytest.raises(ValueError, match="cannot have MIME type"):
        InlineMedia.model_validate({**media.model_dump(), "kind": "image"})
    with pytest.raises(ValueError):
        InlineImage.model_validate({**media.model_dump(), "kind": "image"})
    with pytest.raises(ValueError, match="Unsupported"):
        InlineMedia.from_bytes("text/plain", b"text")
