# SPDX-License-Identifier: Apache-2.0
"""Capability bindings and media shared by UMM composition tests."""

import base64
import io
import wave

import pytest
from PIL import Image

from sglang_omni.client.image_media import InlineImage
from sglang_omni.pipeline.umm import UMMController
from sglang_omni.pipeline.umm_plan import UMMCapability, UMMCompositionSpec
from tests.unit_test.fixtures.umm_stages import (
    SAMPLE_RATE,
    build_controller,
    sine_waveform,
)


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
def controller(composition: UMMCompositionSpec) -> UMMController:
    return build_controller(composition)


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
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as recording:
            recording.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            recording.writeframes((sine_waveform(1600) * 32767).astype("<i2").tobytes())
        return buffer.getvalue()
