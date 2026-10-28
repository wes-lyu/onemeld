# SPDX-License-Identifier: Apache-2.0
"""Native requests and validated results of each composition protocol."""

import base64
import io
import wave

import pytest

from sglang_omni.client.image_media import InlineImage, InlineMedia
from sglang_omni.pipeline.umm_plan import (
    UMMArtifact,
    UMMCompositionSpec,
    UMMRequestOptions,
    UMMStepCall,
)
from sglang_omni.pipeline.umm_protocols import step_artifact, step_request
from tests.unit_test.fixtures.umm_stages import (
    SAMPLE_RATE,
    media_result,
    sine_waveform,
    waveform_result,
)

pytest_plugins = ["tests.unit_test.fixtures.umm_composition"]


def artifacts(
    inline_images: list[InlineImage], speech_wav: bytes
) -> dict[str, UMMArtifact]:
    return {
        "scene": UMMArtifact(value="A fox at dawn", action="inspect"),
        "first": UMMArtifact(value=inline_images[0], action="paint"),
        "second": UMMArtifact(value=inline_images[1], action="paint"),
        "voice": UMMArtifact(
            value=InlineMedia.from_bytes("audio/wav", speech_wav), action=None
        ),
    }


@pytest.mark.parametrize(
    "action, input_refs, expected",
    [
        (
            "retouch",
            ["scene", "first"],
            {"prompt": "Make it night\nContext scene: A fox at dawn", "image_path": 0},
        ),
        ("retouch", ["first", "second"], {"image_path": [0, 1]}),
        ("animate", ["first"], {"image_path": 0, "num_frames": 33, "seed": 5}),
        ("paint", ["scene"], {"width": 64, "height": 32, "seed": 5}),
    ],
    ids=["edit", "multi-image-edit", "image-to-video", "text-to-image"],
)
def test_generation_requests_carry_references_and_sizes(
    composition: UMMCompositionSpec,
    inline_images: list[InlineImage],
    speech_wav: bytes,
    action: str,
    input_refs: list[str],
    expected: dict[str, str | int | list[int]],
) -> None:
    """Referenced images become image_path; only text-to-image takes the client size."""
    request = step_request(
        composition.capabilities[action],
        UMMStepCall(
            id="out", action=action, instruction="Make it night", input_refs=input_refs
        ),
        artifacts(inline_images, speech_wav),
        UMMRequestOptions(width=64, height=32, seed=3),
        "[user] task",
        2,
        128,
    )
    inputs = request.inputs
    for name, value in expected.items():
        if name == "image_path":
            urls = (
                [inline_images[index].url for index in value]
                if isinstance(value, list)
                else inline_images[value].url
            )
            assert inputs[name] == urls
        else:
            assert inputs[name] == value
    assert inputs["num_outputs_per_prompt"] == 1
    assert ("width" in inputs) == (action == "paint")


def test_speech_requests_carry_audio_bytes_and_text(
    composition: UMMCompositionSpec, inline_images: list[InlineImage], speech_wav: bytes
) -> None:
    available = artifacts(inline_images, speech_wav)
    transcription = step_request(
        composition.capabilities["transcribe"],
        UMMStepCall(
            id="text",
            action="transcribe",
            instruction="Transcribe",
            input_refs=["voice"],
        ),
        available,
        UMMRequestOptions(),
        "",
        0,
        128,
    )
    assert transcription.inputs == {"audio_bytes": speech_wav}
    for input_refs, text in ((["scene"], "A fox at dawn"), ([], "Say hello")):
        speech = step_request(
            composition.capabilities["speak"],
            UMMStepCall(
                id="said",
                action="speak",
                instruction="Say hello",
                input_refs=input_refs,
            ),
            available,
            UMMRequestOptions(),
            "",
            0,
            128,
        )
        assert speech.inputs == text
        assert speech.metadata == {"task": "tts", "tts_params": {"voice": "Vivian"}}


def test_tts_waveform_becomes_verified_wav(composition: UMMCompositionSpec) -> None:
    waveform = sine_waveform(800)
    artifact = step_artifact(
        composition.capabilities["speak"],
        UMMStepCall(id="said", action="speak", instruction="Say hello", input_refs=[]),
        waveform_result(waveform),
        1 << 20,
    )
    assert isinstance(artifact.value, InlineMedia)
    assert (artifact.kind, artifact.value.mime_type) == ("audio", "audio/wav")
    with wave.open(
        io.BytesIO(base64.b64decode(artifact.value.encoded_content()))
    ) as audio:
        assert (audio.getframerate(), audio.getnframes()) == (SAMPLE_RATE, 800)


@pytest.mark.parametrize(
    "action, result, limit, message",
    [
        (
            "paint",
            media_result(InlineMedia.from_bytes("video/mp4", b"video")),
            1 << 20,
            "returned video",
        ),
        ("paint", media_result(InlineImage.from_png("eA==")), 0, "byte limit"),
        (
            "paint",
            {
                "media": [
                    {**InlineImage.from_png("eA==").model_dump(), "sha256": "0" * 64}
                ]
            },
            1 << 20,
            "SHA-256",
        ),
        ("speak", waveform_result(sine_waveform(16000)), 1024, "byte limit"),
        ("inspect", {"text": ""}, 1 << 20, "at least 1 character"),
    ],
    ids=["wrong-kind", "oversize-image", "tampered", "oversize-speech", "empty-text"],
)
def test_native_results_are_validated(
    composition: UMMCompositionSpec,
    action: str,
    result: dict[str, object],
    limit: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        step_artifact(
            composition.capabilities[action],
            UMMStepCall(id="out", action=action, instruction="Go", input_refs=[]),
            result,
            limit,
        )
