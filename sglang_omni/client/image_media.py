# SPDX-License-Identifier: Apache-2.0
"""Inline media validation at UMM boundaries."""

import base64
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MediaKind = Literal["image", "video", "audio"]
MEDIA_MIME_TYPES: dict[MediaKind, tuple[str, ...]] = {
    "image": ("image/png", "image/jpeg", "image/webp"),
    "video": ("video/mp4",),
    "audio": ("audio/wav", "audio/mpeg", "audio/flac"),
}


class InlineMedia(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: MediaKind
    mime_type: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str

    @model_validator(mode="after")
    def validate_mime_type(self) -> "InlineMedia":
        if self.mime_type not in MEDIA_MIME_TYPES[self.kind]:
            raise ValueError(f"UMM {self.kind} cannot have MIME type {self.mime_type}")
        else:
            return self

    @classmethod
    def from_bytes(cls, mime_type: str, content: bytes) -> "InlineMedia":
        kind = next(
            (kind for kind, types in MEDIA_MIME_TYPES.items() if mime_type in types),
            None,
        )
        if kind is None:
            raise ValueError(f"Unsupported inline media MIME type {mime_type}")
        else:
            encoded_content = base64.b64encode(content).decode("ascii")
        return cls.model_validate(
            {
                "kind": kind,
                "mime_type": mime_type,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "url": f"data:{mime_type};base64,{encoded_content}",
            }
        )

    @classmethod
    def from_data_url(cls, url: str) -> "InlineMedia":
        header, separator, encoded_content = url.partition(",")
        if not separator or not (
            header.startswith("data:") and header.endswith(";base64")
        ):
            raise ValueError("Inline media must be a base64 data URL")
        else:
            mime_type = header[len("data:") : -len(";base64")]
        return cls.from_bytes(
            mime_type, base64.b64decode(encoded_content, validate=True)
        )

    def encoded_content(self) -> str:
        prefix = f"data:{self.mime_type};base64,"
        if not self.url.startswith(prefix):
            raise ValueError(
                f"UMM {self.kind} must be an inline data URL matching its MIME type"
            )
        else:
            encoded_content = self.url[len(prefix) :]
        content = base64.b64decode(encoded_content, validate=True)
        if (
            len(content) != self.size_bytes
            or hashlib.sha256(content).hexdigest() != self.sha256
        ):
            raise ValueError(
                f"UMM {self.kind} size or SHA-256 does not match its content"
            )
        else:
            return encoded_content


class InlineImage(InlineMedia):
    kind: Literal["image"] = "image"
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]

    @classmethod
    def from_png(cls, encoded_image: str) -> "InlineImage":
        image_bytes = base64.b64decode(encoded_image, validate=True)
        return cls(
            mime_type="image/png",
            size_bytes=len(image_bytes),
            sha256=hashlib.sha256(image_bytes).hexdigest(),
            url=f"data:image/png;base64,{encoded_image}",
        )

    def encoded_image(self) -> str:
        # note (Wes): the Images API routes and SenseNova call this name.
        return self.encoded_content()
