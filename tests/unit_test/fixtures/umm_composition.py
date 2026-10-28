# SPDX-License-Identifier: Apache-2.0
"""Capability bindings and media shared by UMM composition tests."""

import base64
import io

import pytest
from PIL import Image

from sglang_omni.client.audio import encode_wav
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
                protocol="native_image",
                description="Create one image",
                input_kinds=["text"],
                parameters={"width": 512, "height": 512, "num_inference_steps": 20},
            ),
            "retouch": UMMCapability(
                stages=["editor"],
                protocol="native_image",
                description="Edit one image",
                input_kinds=["text", "image"],
            ),
            "animate": UMMCapability(
                stages=["video_worker"],
                protocol="native_video",
                description="Animate one image",
                input_kinds=["text", "image"],
                parameters={"num_frames": 33},
            ),
            "inspect": UMMCapability(
                stages=["vision_worker"],
                protocol="native_chat",
                description="Answer questions or inspect media",
                input_kinds=["text", "image", "video", "audio"],
            ),
            "transcribe": UMMCapability(
                stages=["asr_worker"],
                protocol="asr",
                description="Transcribe speech",
                input_kinds=["audio"],
            ),
            "speak": UMMCapability(
                stages=["speech_worker"],
                protocol="tts",
                description="Read text aloud",
                input_kinds=["text"],
                speech={"voice": "Vivian"},
            ),
        },
        allowed_modes=["plan", "iterative"],
        max_planner_turns=3,
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
    return encode_wav(sine_waveform(1600), SAMPLE_RATE)
