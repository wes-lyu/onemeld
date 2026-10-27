# SPDX-License-Identifier: Apache-2.0
"""Native protocol bindings and artifacts shared by public UMM tests."""

import base64
import io

import pytest
from PIL import Image

from sglang_omni.client.image_media import InlineImage
from sglang_omni.pipeline.umm import UMMController, create_umm_controller
from sglang_omni.pipeline.umm_plan import UMMCapability, UMMCompositionSpec


@pytest.fixture
def composition() -> UMMCompositionSpec:
    return UMMCompositionSpec(
        planner_stage="planning_worker",
        capabilities={
            "paint": UMMCapability(
                stage="image_worker",
                protocol="native_image",
                description="Create one image",
                input_kinds=["text"],
                parameters={"width": 512, "height": 512, "num_inference_steps": 20},
            ),
            "inspect": UMMCapability(
                stage="vision_worker",
                protocol="native_chat",
                description="Answer questions or inspect images",
                input_kinds=["text", "image"],
            ),
        },
    )


@pytest.fixture
def controller(composition: UMMCompositionSpec) -> UMMController:
    return create_umm_controller(
        "unused",
        composition=composition.model_dump(mode="json"),
        phases=composition.phases,
        output_stage="output",
        max_dispatches=7,
        max_outputs=6,
        max_active=1,
        max_retained_bytes=128 * 1024 * 1024,
        timeout_s=60,
    )


@pytest.fixture
def inline_images() -> list[InlineImage]:
    images: list[InlineImage] = []
    for color in ("red", "blue"):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        images.append(
            InlineImage.from_png(base64.b64encode(buffer.getvalue()).decode())
        )
    return images
