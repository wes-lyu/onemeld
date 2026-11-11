# SPDX-License-Identifier: Apache-2.0
"""Backend requests and validated results of each composition protocol."""

import pytest

from onemeld.composition.media import InlineImage, InlineMedia
from onemeld.composition.plan import (
    UMMArtifact,
    UMMCapability,
    UMMCompositionSpec,
    UMMRequestOptions,
    UMMStepCall,
)
from onemeld.composition.protocols import (
    labelled_instruction,
    step_artifact,
    step_request,
)

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
    assert request.inputs == "Make it night\nText 1: A fox at dawn"
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


def test_chat_inputs_are_labelled_without_step_ids(
    composition: UMMCompositionSpec, inline_images: list[InlineImage]
) -> None:
    first, second = inline_images[:2]
    request = step_request(
        composition.capabilities["inspect"],
        UMMStepCall(
            id="verdict",
            action="inspect",
            instruction="Score step1 and the second photo (ID: step2) using notes_1.",
            input_refs=["step1", "step2", "notes_1"],
        ),
        {
            "step1": UMMArtifact(value=first, action="paint"),
            "step2": UMMArtifact(value=second, action="paint"),
            "notes_1": UMMArtifact(value="Prefer clarity", action="inspect"),
        },
        UMMRequestOptions(),
        "Compare two cats",
        0,
        128,
    )
    (_, user_message) = request.inputs
    labels = [
        part["text"] for part in user_message["content"] if part["type"] == "text"
    ]
    assert labels == [
        "User request:\nCompare two cats\nTask for this step: "
        "Score Image 1 and the second photo (Image 2) using Text 1.",
        "Image 1:",
        "Image 2:",
        "Text 1: Prefer clarity",
    ]


@pytest.mark.parametrize(
    ("instruction", "labels", "expected"),
    [
        ("根据step1生成图片", {"step1": "Text 1"}, "根据Text 1生成图片"),
        (
            "Keep step10 apart from step1",
            {"step1": "Image 1"},
            "Keep step10 apart from Image 1",
        ),
        (
            "Paint the scene from ID: scene",
            {"scene": "Text 1"},
            "Paint the scene from Text 1",
        ),
        ("Use design_proposal", {"design_proposal": "Text 1"}, "Use Text 1"),
    ],
)
def test_reference_ids_in_an_instruction_become_labels(
    instruction: str, labels: dict[str, str], expected: str
) -> None:
    assert labelled_instruction(instruction, labels) == expected
