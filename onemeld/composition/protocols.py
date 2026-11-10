# SPDX-License-Identifier: Apache-2.0
"""Backend requests and results of each composition capability protocol."""

import base64
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onemeld.composition.media import InlineImage, InlineMedia
from onemeld.composition.plan import (
    ContentPart,
    UMMArtifact,
    UMMCapability,
    UMMRequestOptions,
    UMMStepCall,
    media_part,
)
from onemeld.composition.types import StepRequest

STEP_SYSTEM_PROMPT = (
    "Execute only the assigned step and return its actual result in the user's language. The original "
    "user request supplies context and constraints; other steps are handled separately. Do not repeat "
    "the instruction, describe what you will do, or claim to have executed another step. Preserve all "
    "applicable constraints, including forbidden content, when writing material for a later step. "
    "Inspect supplied images when present and ground comparisons in observable "
    "details. Follow the requested answer format. For scores, use the same explicit numeric scale "
    "for every item and criterion. Recommend an item only when assigned to do so, using the supplied "
    "scores and the user's tie-breaking rule without contradicting or changing the scores."
)
SEED_MODULUS = 2**63


class NativeTextResult(BaseModel):
    text: str = Field(min_length=1)
    finish_reason: str = "stop"


class NativeTranscriptionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str


class Base64ImageResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    modality: Literal["image"]
    image_b64: str


def step_request(
    capability: UMMCapability,
    step: UMMStepCall,
    artifacts: dict[str, UMMArtifact],
    options: UMMRequestOptions,
    task_text: str,
    seed_offset: int,
    max_new_tokens: int,
) -> StepRequest:
    """The backend request that runs step on a backend of capability."""
    texts: list[tuple[str, str]] = []
    media: list[InlineMedia] = []
    for reference in step.input_refs:
        value = artifacts[reference].value
        if isinstance(value, str):
            texts.append((reference, value))
        else:
            media.append(value)
    if capability.protocol == "native_chat":
        content: list[ContentPart] = [
            {
                "type": "text",
                "text": f"User request:\n{task_text}\nTask for this step: {step.instruction}",
            }
        ]
        for reference in step.input_refs:
            value = artifacts[reference].value
            if isinstance(value, str):
                content.append({"type": "text", "text": f"Text {reference}: {value}"})
            else:
                content.extend(
                    [
                        {"type": "text", "text": f"{value.kind.title()} {reference}:"},
                        media_part(value),
                    ]
                )
        return StepRequest(
            inputs=[
                {"role": "system", "content": STEP_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            params={**capability.parameters, "max_new_tokens": max_new_tokens},
        )
    elif capability.protocol == "image_base64":
        prompt = step.instruction + "".join(
            f"\nContext {reference}: {text}" for reference, text in texts
        )
        inputs: dict[str, JsonValue] = dict(capability.parameters)
        for name, value in (
            ("width", options.width),
            ("height", options.height),
            ("num_inference_steps", options.steps),
        ):
            if value is not None:
                inputs[name] = value
            else:
                pass
        seed = options.seed if options.seed is not None else inputs.get("seed")
        if seed is None:
            pass
        elif type(seed) is not int:
            raise ValueError("Generation seed must be an integer")
        else:
            inputs["seed"] = (seed + seed_offset) % SEED_MODULUS
        return StepRequest(inputs=prompt, params={**inputs, "n": 1})
    else:
        (audio,) = media
        return StepRequest(
            inputs=base64.b64decode(audio.encoded_content()),
            params=dict(capability.parameters),
        )


def step_artifact(
    capability: UMMCapability,
    step: UMMStepCall,
    result: object,
    max_inline_media_bytes: int,
) -> UMMArtifact:
    """Validate a backend result as the artifact of step."""
    if capability.protocol == "native_chat":
        response = NativeTextResult.model_validate(result)
        return UMMArtifact(
            value=response.text,
            action=step.action,
            truncated=response.finish_reason == "length",
        )
    elif capability.protocol == "asr":
        transcription = NativeTranscriptionResult.model_validate(result)
        return UMMArtifact(value=transcription.text, action=step.action)
    else:
        image = Base64ImageResult.model_validate(result)
        value = InlineImage.from_png(image.image_b64)
    if value.size_bytes > max_inline_media_bytes:
        raise ValueError(
            f"{step.action} output exceeds the composition byte limit of {max_inline_media_bytes}"
        )
    else:
        value.encoded_content()
    return UMMArtifact(value=value, action=step.action)
