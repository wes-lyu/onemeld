# SPDX-License-Identifier: Apache-2.0
"""Backend requests and validated results of each composition protocol."""

import pytest

from onemeld.composition.media import InlineMedia
from onemeld.composition.plan import (
    UMMArtifact,
    UMMCapability,
    UMMCompositionSpec,
    UMMRequestOptions,
    UMMStepCall,
)
from onemeld.composition.protocols import step_artifact, step_request

pytest_plugins = ["tests.unit_test.fixtures.composition"]


@pytest.mark.parametrize("encoded, limit", [("not-base64!", 1024), ("eA==", 0)])
def test_base64_image_result_rejects_invalid_or_oversized_content(
    encoded: str, limit: int
) -> None:
    capability = UMMCapability(
        stages=["generation"],
        protocol="image_base64",
        description="Generate one image",
        input_kinds=["text"],
    )
    with pytest.raises(ValueError):
        step_artifact(
            capability,
            UMMStepCall(id="cat", action="draw", instruction="Cat", input_refs=[]),
            {"modality": "image", "image_b64": encoded},
            limit,
        )


def test_generation_request_carries_text_dependencies_and_options(
    composition: UMMCompositionSpec,
) -> None:
    request = step_request(
        composition.capabilities["paint"],
        UMMStepCall(
            id="out", action="paint", instruction="Make it night", input_refs=["scene"]
        ),
        {"scene": UMMArtifact(value="A fox at dawn", action="inspect")},
        UMMRequestOptions(width=64, height=32, seed=3),
        "[user] task",
        2,
        128,
    )
    assert request.inputs == "Make it night\nContext scene: A fox at dawn"
    assert request.params == {
        "width": 64,
        "height": 32,
        "num_inference_steps": 20,
        "seed": 5,
        "n": 1,
    }


def test_transcription_request_carries_audio_bytes(
    composition: UMMCompositionSpec, speech_wav: bytes
) -> None:
    transcription = step_request(
        composition.capabilities["transcribe"],
        UMMStepCall(
            id="text",
            action="transcribe",
            instruction="Transcribe",
            input_refs=["voice"],
        ),
        {
            "voice": UMMArtifact(
                value=InlineMedia.from_bytes("audio/wav", speech_wav), action=None
            )
        },
        UMMRequestOptions(),
        "",
        0,
        128,
    )
    assert transcription.inputs == speech_wav


def test_empty_chat_result_is_rejected(composition: UMMCompositionSpec) -> None:
    with pytest.raises(ValueError, match="at least 1 character"):
        step_artifact(
            composition.capabilities["inspect"],
            UMMStepCall(id="out", action="inspect", instruction="Go", input_refs=[]),
            {"text": ""},
            1024,
        )
