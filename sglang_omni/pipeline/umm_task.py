# SPDX-License-Identifier: Apache-2.0
"""Client tasks of composition requests: conversation, inline media and options."""

import base64
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.client.image_media import InlineImage, InlineMedia, MediaKind
from sglang_omni.pipeline.umm_plan import (
    INPUT_IMAGE_ID,
    ContentPart,
    UMMArtifact,
    UMMCompositionSpec,
    UMMRequestOptions,
    media_part,
)
from sglang_omni.proto.request import OmniRequest

INPUT_AUDIO_MIME_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}
MAX_TASK_MESSAGES = 32
MAX_TASK_TEXT_BYTES = 16 * 1024


class MediaURL(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    url: str


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["text"]
    text: str


class ImageURLPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["image_url"]
    image_url: MediaURL | str


class AudioURLPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["audio_url"]
    audio_url: MediaURL | str


class InputAudio(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    data: str
    format: Literal["wav", "mp3"]


class InputAudioPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["input_audio"]
    input_audio: InputAudio


MediaPart = ImageURLPart | AudioURLPart | InputAudioPart
TaskPart = Annotated[TextPart | MediaPart, Field(discriminator="type")]


class TaskMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant"]
    content: str | list[TaskPart]


class TaskConversation(BaseModel):
    messages: list[TaskMessage] = Field(min_length=1, max_length=MAX_TASK_MESSAGES)


@dataclass(frozen=True, kw_only=True)
class ClientTask:
    """A validated request: planner content, step text and client input artifacts."""

    options: UMMRequestOptions
    content: list[ContentPart]
    text: str
    artifacts: dict[str, UMMArtifact]
    audio_instruction_ref: str | None = None


def part_media(part: MediaPart, encoded_limit: int) -> InlineMedia:
    """Decode one client media part, which must be inline and within encoded_limit."""
    if isinstance(part, InputAudioPart):
        if len(part.input_audio.data) > encoded_limit:
            raise ValueError("Input media exceeds the composition byte limit")
        else:
            return InlineMedia.from_bytes(
                INPUT_AUDIO_MIME_TYPES[part.input_audio.format],
                base64.b64decode(part.input_audio.data, validate=True),
            )
    elif isinstance(part, ImageURLPart):
        reference, kind = part.image_url, "image"
    else:
        reference, kind = part.audio_url, "audio"
    url = reference if isinstance(reference, str) else reference.url
    if not url.startswith("data:"):
        raise ValueError("Composition media must be inline data URLs")
    elif len(url) > encoded_limit:
        raise ValueError("Input media exceeds the composition byte limit")
    else:
        media = InlineMedia.from_data_url(url)
    if media.kind != kind:
        raise ValueError(f"A {part.type} part must carry {kind} data")
    else:
        return media


def parse_task(
    request: OmniRequest, composition: UMMCompositionSpec, stage_name: str
) -> ClientTask:
    """Validate a string task or chat conversation and its composition options."""
    if isinstance(request.inputs, str):
        conversation = TaskConversation(
            messages=[TaskMessage(role="user", content=request.inputs)]
        )
        options = request.metadata.get("composition", {})
    elif isinstance(request.inputs, list):
        conversation = TaskConversation.model_validate({"messages": request.inputs})
        stage_params = request.params.get("stage_params", {})
        if not isinstance(stage_params, dict):
            raise ValueError("stage_params must be an object")
        else:
            options = stage_params.get(stage_name, {})
    else:
        raise ValueError(
            "Composition takes a task string or chat messages; send media as message content parts"
        )
    if not isinstance(options, dict):
        raise ValueError("Composition options must be an object")
    else:
        options = dict(options)
    for name in ("seed", "max_new_tokens"):
        value = request.params.get(name)
        if value is not None:
            options[name] = value
        else:
            pass
    parsed_options = UMMRequestOptions.model_validate(options)
    artifacts: dict[str, UMMArtifact] = {}
    media_counts: Counter[MediaKind] = Counter()
    content: list[ContentPart] = []
    lines: list[str] = []

    def add_line(line: str) -> None:
        lines.append(line)
        content.append({"type": "text", "text": line})

    def attach(artifact_id: str, media: InlineMedia, role: str) -> None:
        if len(artifacts) >= composition.max_input_media:
            raise ValueError(
                f"Composition accepts at most {composition.max_input_media} input media items"
            )
        elif media.size_bytes > composition.max_inline_media_bytes:
            raise ValueError("Input media exceeds the composition byte limit")
        else:
            artifacts[artifact_id] = UMMArtifact(value=media, action=None)
        add_line(f"[{role}] attached {artifact_id} ({media.kind})")
        if media.kind in composition.planner_media_kinds:
            content.append(media_part(media))
        else:
            pass

    encoded_limit = 2 * composition.max_inline_media_bytes
    audio_instruction_refs: list[str] = []
    has_user_text = False
    for message in conversation.messages:
        parts = (
            [TextPart(type="text", text=message.content)]
            if isinstance(message.content, str)
            else message.content
        )
        if message.role == "user":
            audio_instruction_refs = []
            has_user_text = any(
                isinstance(part, TextPart) and bool(part.text.strip()) for part in parts
            )
        else:
            pass
        for part in parts:
            if isinstance(part, TextPart):
                add_line(f"[{message.role}] {part.text}")
            else:
                media = part_media(part, encoded_limit)
                media_counts[media.kind] += 1
                attach(
                    f"user_{media.kind}_{media_counts[media.kind]}",
                    media,
                    message.role,
                )
                if message.role == "user" and media.kind == "audio":
                    audio_instruction_refs.append(f"user_audio_{media_counts['audio']}")
                else:
                    pass
    if parsed_options.image_base64 is not None:
        if len(parsed_options.image_base64) > encoded_limit:
            raise ValueError("Input media exceeds the composition byte limit")
        else:
            attach(
                INPUT_IMAGE_ID,
                InlineImage.from_png(parsed_options.image_base64),
                "user",
            )
    else:
        pass
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > MAX_TASK_TEXT_BYTES:
        raise ValueError(f"Composition task text exceeds {MAX_TASK_TEXT_BYTES} bytes")
    elif (
        composition.audio_instruction_capability is not None
        and not has_user_text
        and len(audio_instruction_refs) > 1
    ):
        raise ValueError("A voice instruction must contain exactly one audio clip")
    else:
        return ClientTask(
            options=parsed_options,
            content=content,
            text=text,
            artifacts=artifacts,
            audio_instruction_ref=(
                audio_instruction_refs[0]
                if composition.audio_instruction_capability is not None
                and not has_user_text
                and audio_instruction_refs
                else None
            ),
        )
