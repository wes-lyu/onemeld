# SPDX-License-Identifier: Apache-2.0
"""Routing, media and lifecycle of upfront plans."""

import base64
import json

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.image_media import InlineImage
from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.pipeline.umm import UMMController
from sglang_omni.pipeline.umm_plan import (
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMPlanStep,
    UMMStepCall,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.fixtures.umm_stages import (
    FakeStages,
    build_controller,
    image_result,
)

pytest_plugins = ["tests.unit_test.fixtures.umm_composition"]


def plan_reply(*steps: UMMPlanStep) -> dict[str, str]:
    return {"text": UMMExecutionPlan(steps=list(steps)).model_dump_json()}


def call(step_id: str, action: str, *input_refs: str) -> UMMStepCall:
    return UMMStepCall(
        id=step_id,
        action=action,
        instruction=f"Do {step_id}",
        input_refs=list(input_refs),
    )


def planned(step: UMMStepCall, publish: bool) -> UMMPlanStep:
    return UMMPlanStep(**step.model_dump(), publish=publish)


def audio_task(speech_wav: bytes, mode: str) -> OmniRequest:
    return Client.build_omni_request(
        GenerateRequest(
            messages=[
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "Draw what I say, then narrate it"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(speech_wav).decode("ascii"),
                                "format": "wav",
                            },
                        },
                    ],
                )
            ],
            stream=False,
            sampling=SamplingParams(seed=42, max_new_tokens=256),
            output_modalities=["text", "image", "audio"],
            stage_params={"orchestrator": {"mode": mode, "steps": 4}},
        )
    )


@pytest.mark.parametrize("voice_instruction", [True, False])
def test_voice_instruction_is_transcribed_before_the_only_plan(
    composition: UMMCompositionSpec,
    speech_wav: bytes,
    voice_instruction: bool,
) -> None:
    """Only audio-only instructions need transcription before the planner can act."""
    composition = composition.model_copy(
        update={
            "audio_instruction_capability": "transcribe",
            "max_plan_attempts": 1,
        }
    )
    controller = build_controller(composition)
    stages = FakeStages(controller, "voice")
    request = audio_task(speech_wav, "plan")
    if voice_instruction:
        request.inputs[0]["content"] = request.inputs[0]["content"][1:]
    else:
        pass
    stages.submit(request)
    if voice_instruction:
        assert set(stages.held) == {"asr_worker"}
        assert stages.held["asr_worker"].request.inputs["audio_bytes"] == speech_wav
        stages.reply("asr_worker", {"text": "What is two plus three?"})
        planner_content = stages.held["planning_worker"].request.inputs
        assert "What is two plus three?" in json.dumps(planner_content)
        assert "user_audio_1" not in json.dumps(planner_content)
    else:
        assert set(stages.held) == {"planning_worker"}
    stages.reply(
        "planning_worker", plan_reply(planned(call("answer", "inspect"), True))
    )
    answer_request = stages.held["vision_worker"].request.inputs
    if voice_instruction:
        assert "What is two plus three?" in json.dumps(answer_request)
    else:
        assert "Draw what I say" in json.dumps(answer_request)
    stages.reply("vision_worker", {"text": "5"})
    assert [segment["data"] for segment in stages.segments()] == ["5"]
    assert not controller.adapter.executions


@pytest.mark.parametrize("outcome", ["empty", "abort"])
def test_voice_transcription_failure_never_enters_planning(
    composition: UMMCompositionSpec,
    speech_wav: bytes,
    outcome: str,
) -> None:
    composition = composition.model_copy(
        update={"audio_instruction_capability": "transcribe"}
    )
    controller = build_controller(composition)
    stages = FakeStages(controller, "voice-failure")
    request = audio_task(speech_wav, "plan")
    request.inputs[0]["content"] = request.inputs[0]["content"][1:]
    stages.submit(request)
    assert set(stages.held) == {"asr_worker"}
    if outcome == "empty":
        stages.reply("asr_worker", {"text": "  "})
        assert "planning_worker" not in stages.held
        assert "no transcription" in str(controller.outbox.get_nowait().data)
    else:
        controller.abort("voice-failure")
    assert not controller.adapter.executions


def test_serial_composition_waits_for_an_independent_stage(
    composition: UMMCompositionSpec,
    inline_images: list[InlineImage],
) -> None:
    controller = build_controller(
        composition.model_copy(update={"max_concurrent_steps": 1})
    )
    stages = FakeStages(controller, "serial")
    stages.submit(OmniRequest(inputs="Draw and answer"))
    stages.reply(
        "planning_worker",
        plan_reply(
            planned(call("picture", "paint"), True),
            planned(call("answer", "inspect"), True),
        ),
    )
    assert set(stages.held) == {"image_a"}
    stages.reply("image_a", image_result(inline_images[0]))
    assert set(stages.held) == {"vision_worker"}
    stages.reply("vision_worker", {"text": "Finished"})
    assert [segment["kind"] for segment in stages.segments()] == ["image", "text"]


def test_independent_steps_spread_over_replicas_and_publish_in_order(
    controller: UMMController, inline_images: list[InlineImage]
) -> None:
    """Parallel results arrive out of order; segments keep the planned order."""
    stages = FakeStages(controller, "replicas")
    stages.submit(
        OmniRequest(
            "Draw three cats and pick one",
            metadata={"composition": {"mode": "plan", "seed": 7}},
        )
    )
    paints = [call(f"cat_{index}", "paint") for index in range(3)]
    verdict = call("verdict", "inspect", "cat_0", "cat_1", "cat_2")
    stages.reply(
        "planning_worker",
        plan_reply(*(planned(step, True) for step in [*paints, verdict])),
    )
    assert set(stages.held) == {"image_a", "image_b"}
    assert [
        stages.held[stage].request.params["seed"] for stage in ("image_a", "image_b")
    ] == [7, 8]
    context = controller.contexts["replicas"]
    stages.reply("image_b", image_result(inline_images[1]))
    # The free replica takes the queued step; cat_1 waits for cat_0 to publish.
    assert stages.held["image_b"].request.inputs == "Do cat_2"
    assert stages.held["image_b"].request.params["seed"] == 9
    assert context.outputs == []
    stages.reply("image_a", image_result(inline_images[0]))
    assert [segment["data"] for segment in context.outputs] == [
        inline_images[0].model_dump(),
        inline_images[1].model_dump(),
    ]
    assert "vision_worker" not in stages.held
    stages.reply("image_b", image_result(inline_images[2]))
    images = [
        part["image_url"]["url"]
        for part in stages.held["vision_worker"].request.inputs[-1]["content"]
        if part["type"] == "image_url"
    ]
    assert images == [image.url for image in inline_images]
    stages.reply("vision_worker", {"text": "The blue cat"})
    assert [segment["data"] for segment in stages.segments()] == [
        *(image.model_dump() for image in inline_images),
        "The blue cat",
    ]
    assert not controller.adapter.executions


def test_multipart_conversation_becomes_typed_inputs(
    controller: UMMController, inline_images: list[InlineImage]
) -> None:
    stages = FakeStages(controller, "multipart")
    stages.submit(
        OmniRequest(
            [
                {"role": "system", "content": "Answer briefly"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What happens here?"},
                        {"type": "image_url", "image_url": inline_images[0].url},
                    ],
                },
            ]
        )
    )
    task = stages.held["planning_worker"].request.inputs[1]["content"]
    assert [part["text"] for part in task if part["type"] == "text"] == [
        "[system] Answer briefly",
        "[user] What happens here?",
        "[user] attached user_image_1 (image)",
    ]
    assert [part["type"] for part in task if part["type"] != "text"] == ["image_url"]
    stages.reply(
        "planning_worker",
        plan_reply(planned(call("answer", "inspect", "user_image_1"), True)),
    )
    content = stages.held["vision_worker"].request.inputs[-1]["content"]
    assert [part for part in content if part["type"] != "text"] == [
        {"type": "image_url", "image_url": {"url": inline_images[0].url}},
    ]
    assert content[0]["text"].startswith("User request:\n[system] Answer briefly")


def test_legacy_input_image_is_an_input_artifact(
    controller: UMMController, inline_images: list[InlineImage]
) -> None:
    image = inline_images[0]
    stages = FakeStages(controller, "legacy")
    stages.submit(
        OmniRequest(
            "Make it blue",
            metadata={
                "composition": {"image_base64": image.encoded_image(), "width": 64}
            },
        )
    )
    stages.reply(
        "planning_worker",
        plan_reply(planned(call("answer", "inspect", "input_image"), True)),
    )
    content = stages.held["vision_worker"].request.inputs[-1]["content"]
    assert content[-1]["image_url"]["url"] == image.url


@pytest.mark.parametrize(
    "inputs, options",
    [
        ("Draw", {"mode": "compare"}),
        ("Draw", {"steps": 0}),
        ("Draw", {"width": 257}),
        ({"messages": [], "images": ["data:image/png;base64,eA=="]}, {}),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.invalid/cat.png"},
                        }
                    ],
                }
            ],
            {},
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "data:image/png;base64,eA=="},
                        }
                    ],
                }
            ],
            {},
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,eA=="},
                        }
                    ]
                    * 5,
                }
            ],
            {},
        ),
        ([{"role": "tool", "content": "result"}], {}),
    ],
    ids=[
        "unknown-mode",
        "zero-steps",
        "unaligned-width",
        "top-level-media",
        "remote-media",
        "mismatched-kind",
        "too-many-media",
        "tool-role",
    ],
)
def test_invalid_client_task_never_reaches_planning(
    controller: UMMController,
    inputs: str | list[dict[str, object]] | dict[str, object],
    options: dict[str, str | int],
) -> None:
    request = (
        OmniRequest(inputs, metadata={"composition": options})
        if isinstance(inputs, str)
        else OmniRequest(inputs)
    )
    assert controller.advance(StagePayload("invalid", request, None)) is None
    assert controller.outbox.get_nowait().type == "error"
    assert not controller.adapter.executions


def test_mode_outside_the_allowed_set_is_a_client_error(
    composition: UMMCompositionSpec,
) -> None:
    controller = build_controller(composition)
    request = OmniRequest("Draw", metadata={"composition": {"mode": "iterative"}})
    assert controller.advance(StagePayload("mode", request, None)) is None
    assert "Input should be 'plan'" in str(controller.outbox.get_nowait().data)


def test_abort_with_parallel_steps_in_flight_releases_the_execution(
    controller: UMMController, inline_images: list[InlineImage]
) -> None:
    stages = FakeStages(controller, "aborted")
    stages.submit(OmniRequest("Draw two cats"))
    stages.reply(
        "planning_worker",
        plan_reply(
            planned(call("a", "paint"), True), planned(call("b", "paint"), True)
        ),
    )
    assert set(stages.held) == {"image_a", "image_b"}
    controller.abort("aborted")
    assert not controller.adapter.executions
    with pytest.raises(RuntimeError, match="inactive request"):
        stages.reply("image_a", image_result(inline_images[0]))
    assert not controller.contexts
