# SPDX-License-Identifier: Apache-2.0
"""Capability bindings and media shared by composition and orchestrator tests."""

import base64
import io
import wave

import numpy as np
import pytest
from PIL import Image

from onemeld.composition.media import InlineImage
from onemeld.composition.plan import UMMCapability, UMMCompositionSpec

SAMPLE_RATE = 16000


@pytest.fixture
def composition() -> UMMCompositionSpec:
    return UMMCompositionSpec(
        planner_stage="planning_worker",
        capabilities={
            "paint": UMMCapability(
                stages=["image_a", "image_b"],
                protocol="image_base64",
                description="Create one image",
                input_kinds=["text"],
                parameters={"width": 512, "height": 512, "num_inference_steps": 20},
            ),
            "inspect": UMMCapability(
                stages=["vision_worker"],
                protocol="native_chat",
                description="Answer questions or inspect media",
                input_kinds=["text", "image"],
            ),
            "transcribe": UMMCapability(
                stages=["asr_worker"],
                protocol="asr",
                description="Transcribe speech",
                input_kinds=["audio"],
            ),
        },
        max_steps=8,
    )


@pytest.fixture
def inline_images() -> list[InlineImage]:
    images: list[InlineImage] = []
    for color in ("red", "blue", "green"):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        images.append(
            InlineImage.from_png(base64.b64encode(buffer.getvalue()).decode())
        )
    return images


@pytest.fixture
def speech_wav() -> bytes:
    times = np.arange(1600, dtype=np.float32) / SAMPLE_RATE
    waveform = 0.5 * np.sin(2 * np.pi * 440 * times)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as recording:
            recording.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            recording.writeframes((waveform * 32767).astype("<i2").tobytes())
        return buffer.getvalue()


def image_result(image: InlineImage) -> dict[str, str]:
    return {"modality": "image", "image_b64": image.encoded_content()}
