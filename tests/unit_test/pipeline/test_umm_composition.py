# SPDX-License-Identifier: Apache-2.0
"""One executor for planned and iterative compositions: routing, media and lifecycle."""

import base64
import io
import wave
from dataclasses import replace

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.image_media import InlineImage, InlineMedia
from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.pipeline.umm import UMMController
from sglang_omni.pipeline.umm_plan import (
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMFinal,
    UMMPlannerTurn,
    UMMPlanStep,
    UMMStepCall,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.fixtures.umm_stages import (
    FakeStages,
    build_controller,
    media_result,
    sine_waveform,
    waveform_result,
)

pytest_plugins = ["tests.unit_test.fixtures.umm_composition"]


def plan_reply(*steps: UMMPlanStep) -> dict[str, str]:
    return {"text": UMMExecutionPlan(steps=list(steps)).model_dump_json()}


def turn_reply(steps: list[UMMStepCall], final: UMMFinal | None) -> dict[str, str]:
    return {"text": UMMPlannerTurn(steps=steps, final=final).model_dump_json()}


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


@pytest.mark.parametrize("mode", ["plan", "iterative"])
def test_speech_to_image_to_speech_runs_the_same_in_both_modes(
    controller: UMMController,
    inline_images: list[InlineImage],
    speech_wav: bytes,
    mode: str,
) -> None:
    """Both modes dispatch the same specialist requests and publish the same output."""
    stages = FakeStages(controller, "any")
    stages.submit(audio_task(speech_wav, mode))
    first_planning = stages.held["planning_worker"].request
    response_format = first_planning.params["response_format"]["json_schema"]
    assert response_format["name"] == f"umm_{mode}_reply"
    step_schema = response_format["schema"]["$defs"][
        "UMMPlanStep" if mode == "plan" else "UMMStepCall"
    ]
    assert set(step_schema["properties"]["action"]["enum"]) == set(
        controller.adapter.composition.capabilities
    )
    transcribe = call("transcript", "transcribe", "user_audio_1")
    paint = call("picture", "paint", "transcript")
    inspect = call("caption", "inspect", "picture")
    speak = call("narration", "speak", "caption")
    if mode == "plan":
        stages.reply(
            "planning_worker",
            plan_reply(
                planned(transcribe, False),
                planned(paint, True),
                planned(inspect, False),
                planned(speak, True),
            ),
        )
    else:
        stages.reply("planning_worker", turn_reply([transcribe, paint], None))
    assert base64.b64encode(
        stages.held["asr_worker"].request.inputs["audio_bytes"]
    ) == (base64.b64encode(speech_wav))
    stages.reply("asr_worker", {"text": "a red fox", "language": "English"})
    painting = stages.held["image_a"].request.inputs
    assert "a red fox" in painting["prompt"]
    assert painting["seed"] == 42 and painting["num_inference_steps"] == 4
    stages.reply("image_a", media_result(inline_images[0]))
    if mode == "iterative":
        observation = stages.held["planning_worker"].request.inputs
        assert observation[2] == {
            "role": "assistant",
            "content": turn_reply([transcribe, paint], None)["text"],
        }
        summaries = observation[3]["content"][0]["text"]
        assert '"text": "a red fox"' in summaries and '"id": "picture"' in summaries
        assert [
            part["image_url"]["url"]
            for part in observation[3]["content"]
            if part["type"] == "image_url"
        ] == [inline_images[0].url]
        stages.reply(
            "planning_worker",
            turn_reply(
                [inspect, speak], UMMFinal(publish=["picture", "narration"], answer="")
            ),
        )
    else:
        pass
    inspection = stages.held["vision_worker"].request.inputs[-1]["content"]
    assert inspection[-1] == {
        "type": "image_url",
        "image_url": {"url": inline_images[0].url},
    }
    stages.reply("vision_worker", {"text": "A red fox sleeps in the snow"})
    narration = stages.held["speech_worker"].request
    assert narration.inputs == "A red fox sleeps in the snow"
    assert narration.metadata["tts_params"] == {"voice": "Vivian"}
    stages.reply("speech_worker", waveform_result(sine_waveform(800)))
    segments = stages.segments()
    assert [segment["kind"] for segment in segments] == ["image", "audio"]
    assert segments[0]["data"] == inline_images[0].model_dump()
    audio = InlineMedia.model_validate(segments[1]["data"])
    with wave.open(io.BytesIO(base64.b64decode(audio.encoded_content()))) as speech:
        assert (speech.getframerate(), speech.getnframes()) == (16000, 800)
    assert not controller.adapter.executions


@pytest.mark.parametrize("mode", ["plan", "iterative"])
def test_independent_steps_spread_over_replicas_and_publish_in_order(
    controller: UMMController, inline_images: list[InlineImage], mode: str
) -> None:
    """Parallel results arrive out of order; segments keep the planned order."""
    stages = FakeStages(controller, "replicas")
    stages.submit(
        OmniRequest(
            "Draw three cats and pick one",
            metadata={"composition": {"mode": mode, "seed": 7}},
        )
    )
    paints = [call(f"cat_{index}", "paint") for index in range(3)]
    verdict = call("verdict", "inspect", "cat_0", "cat_1", "cat_2")
    publish = ["cat_0", "cat_1", "cat_2", "verdict"]
    if mode == "plan":
        stages.reply(
            "planning_worker",
            plan_reply(*(planned(step, True) for step in [*paints, verdict])),
        )
    else:
        stages.reply(
            "planning_worker",
            turn_reply([*paints, verdict], UMMFinal(publish=publish, answer="")),
        )
    assert set(stages.held) == {"image_a", "image_b"}
    assert [
        stages.held[stage].request.inputs["seed"] for stage in ("image_a", "image_b")
    ] == [7, 8]
    context = controller.contexts["replicas"]
    stages.reply("image_b", media_result(inline_images[1]))
    # The free replica takes the queued step; cat_1 waits for cat_0 to publish.
    assert stages.held["image_b"].request.inputs["prompt"] == "Do cat_2"
    assert stages.held["image_b"].request.inputs["seed"] == 9
    assert context.outputs == []
    stages.reply("image_a", media_result(inline_images[0]))
    assert [segment["data"] for segment in context.outputs] == [
        inline_images[0].model_dump(),
        inline_images[1].model_dump(),
    ]
    assert "vision_worker" not in stages.held
    stages.reply("image_b", media_result(inline_images[2]))
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


def test_iterative_turns_retry_and_publish_only_the_accepted_result(
    controller: UMMController, inline_images: list[InlineImage]
) -> None:
    """Superseded attempts stay private; the final turn chooses what the user sees."""
    stages = FakeStages(controller, "retry")
    stages.submit(
        OmniRequest(
            "Draw a blue cat",
            metadata={"composition": {"mode": "iterative", "seed": 10}},
        )
    )
    stages.reply(
        "planning_worker",
        turn_reply([call("first", "paint"), call("check", "inspect", "first")], None),
    )
    stages.reply("image_a", media_result(inline_images[0]))
    stages.reply("vision_worker", {"text": "The cat is red"})
    stages.reply("planning_worker", turn_reply([call("second", "paint")], None))
    assert stages.held["image_a"].request.inputs["seed"] == 11
    stages.reply("image_a", media_result(inline_images[1]))
    last_turn = stages.held["planning_worker"].request
    schema = last_turn.params["response_format"]["json_schema"]["schema"]
    # The last turn must finish, and may still run the steps it publishes.
    assert schema["properties"]["final"] == {"$ref": "#/$defs/UMMFinal"}
    assert schema["properties"]["steps"]["maxItems"] == 5
    stages.reply(
        "planning_worker",
        turn_reply(
            [call("caption", "inspect", "second")],
            UMMFinal(publish=["second", "caption"], answer=""),
        ),
    )
    stages.reply("vision_worker", {"text": "A blue cat"})
    assert [segment["data"] for segment in stages.segments()] == [
        inline_images[1].model_dump(),
        "A blue cat",
    ]
    assert not controller.adapter.executions


def test_last_turn_without_final_is_rejected(
    composition: UMMCompositionSpec,
) -> None:
    controller = build_controller(
        composition.model_copy(
            update={"max_planner_turns": 2, "execution_mode": "iterative"}
        )
    )
    stages = FakeStages(controller, "endless")
    stages.submit(OmniRequest("Keep drawing"))
    stages.reply("planning_worker", turn_reply([call("first", "paint")], None))
    stages.reply("image_a", media_result(InlineImage.from_png("eA==")))
    planning = stages.held.pop("planning_worker")
    with pytest.raises(ValueError, match="last planner turn"):
        controller.transition(
            replace(planning, data=turn_reply([call("second", "paint")], None))
        )
    assert not controller.adapter.executions


def test_multipart_conversation_becomes_typed_inputs(
    controller: UMMController, inline_images: list[InlineImage], speech_wav: bytes
) -> None:
    """Every client media part is an input artifact; the planner sees only images."""
    video = InlineMedia.from_bytes("video/mp4", b"\x00\x00\x00\x18ftypmp42")
    audio = InlineMedia.from_bytes("audio/wav", speech_wav)
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
                        {"type": "video_url", "video_url": {"url": video.url}},
                        {"type": "audio_url", "audio_url": {"url": audio.url}},
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
        "[user] attached user_video_1 (video)",
        "[user] attached user_audio_1 (audio)",
    ]
    assert [part["type"] for part in task if part["type"] != "text"] == ["image_url"]
    stages.reply(
        "planning_worker",
        plan_reply(
            planned(call("answer", "inspect", "user_video_1", "user_audio_1"), True)
        ),
    )
    content = stages.held["vision_worker"].request.inputs[-1]["content"]
    assert [part for part in content if part["type"] != "text"] == [
        {"type": "video_url", "video_url": {"url": video.url}},
        {"type": "audio_url", "audio_url": {"url": audio.url}},
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
        plan_reply(planned(call("edited", "retouch", "input_image"), True)),
    )
    edit = stages.held["editor"].request.inputs
    assert edit["image_path"] == image.url
    assert "width" not in edit and edit["prompt"] == "Do edited"


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
    controller = build_controller(
        composition.model_copy(update={"allowed_modes": ["plan"]})
    )
    request = OmniRequest("Draw", metadata={"composition": {"mode": "iterative"}})
    assert controller.advance(StagePayload("mode", request, None)) is None
    assert "not allowed" in str(controller.outbox.get_nowait().data)


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
        stages.reply("image_a", media_result(inline_images[0]))
    assert not controller.contexts
