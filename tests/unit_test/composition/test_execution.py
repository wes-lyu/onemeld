# SPDX-License-Identifier: Apache-2.0
"""Routing, media and lifecycle of upfront plans run by the orchestrator."""

import asyncio
import base64
import json

import pytest

from onemeld.composition.media import InlineImage
from onemeld.composition.plan import (
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMPlanStep,
    UMMStepCall,
)
from onemeld.composition.types import TaskRequest, UMMRequestError
from tests.unit_test.fixtures.composition import image_result
from tests.unit_test.fixtures.orchestrator import Harness, settle, task_of

pytest_plugins = ["tests.unit_test.fixtures.composition"]


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


def audio_task(speech_wav: bytes, *, with_text: bool) -> TaskRequest:
    audio = {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(speech_wav).decode("ascii"),
            "format": "wav",
        },
    }
    text = {"type": "text", "text": "Draw what I say, then narrate it"}
    return TaskRequest(
        messages=[{"role": "user", "content": [text, audio] if with_text else [audio]}],
        options={"mode": "plan", "steps": 4},
        seed=42,
        max_new_tokens=256,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_instruction", [True, False])
async def test_voice_instruction_is_transcribed_before_the_only_plan(
    composition: UMMCompositionSpec,
    speech_wav: bytes,
    voice_instruction: bool,
) -> None:
    """Only audio-only instructions need transcription before the planner can act."""
    harness = Harness(
        composition.model_copy(
            update={
                "audio_instruction_capability": "transcribe",
                "max_plan_attempts": 1,
            }
        )
    )
    await harness.submit(audio_task(speech_wav, with_text=not voice_instruction))
    if voice_instruction:
        assert harness.held() == {"asr_worker"}
        assert harness.request("asr_worker").inputs == speech_wav
        await harness.reply("asr_worker", {"text": "What is two plus three?"})
        planner_content = json.dumps(harness.request("planning_worker").inputs)
        assert "What is two plus three?" in planner_content
        assert "user_audio_1" not in planner_content
    else:
        assert harness.held() == {"planning_worker"}
    await harness.reply(
        "planning_worker", plan_reply(planned(call("answer", "inspect"), True))
    )
    answer_request = json.dumps(harness.request("vision_worker").inputs)
    if voice_instruction:
        assert "What is two plus three?" in answer_request
    else:
        assert "Draw what I say" in answer_request
    await harness.reply("vision_worker", {"text": "5"})
    assert [segment["data"] for segment in await harness.finished()] == ["5"]
    assert not harness.orchestrator.adapter.executions


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["empty", "abort"])
async def test_voice_transcription_failure_never_enters_planning(
    composition: UMMCompositionSpec,
    speech_wav: bytes,
    outcome: str,
) -> None:
    harness = Harness(
        composition.model_copy(update={"audio_instruction_capability": "transcribe"})
    )
    await harness.submit(audio_task(speech_wav, with_text=False))
    assert harness.held() == {"asr_worker"}
    if outcome == "empty":
        await harness.reply("asr_worker", {"text": "  "})
        assert "no transcription" in str(harness.error())
    else:
        assert harness.task is not None
        harness.task.cancel()
        await settle()
        assert harness.backends["asr_worker"].cancelled
    assert harness.held() == set()
    assert not harness.orchestrator.adapter.executions


@pytest.mark.asyncio
async def test_serial_composition_waits_for_an_independent_backend(
    composition: UMMCompositionSpec,
    inline_images: list[InlineImage],
) -> None:
    harness = Harness(composition.model_copy(update={"max_concurrent_steps": 1}))
    await harness.submit(task_of("Draw and answer"))
    await harness.reply(
        "planning_worker",
        plan_reply(
            planned(call("picture", "paint"), True),
            planned(call("answer", "inspect"), True),
        ),
    )
    assert harness.held() == {"image_a"}
    await harness.reply("image_a", image_result(inline_images[0]))
    assert harness.held() == {"vision_worker"}
    await harness.reply("vision_worker", {"text": "Finished"})
    assert [segment["kind"] for segment in await harness.finished()] == [
        "image",
        "text",
    ]


@pytest.mark.asyncio
async def test_independent_steps_spread_over_replicas_and_publish_in_order(
    composition: UMMCompositionSpec, inline_images: list[InlineImage]
) -> None:
    """Parallel results arrive out of order; segments keep the planned order."""
    harness = Harness(composition)
    await harness.submit(task_of("Draw three cats and pick one", mode="plan", seed=7))
    paints = [call(f"cat_{index}", "paint") for index in range(3)]
    verdict = call("verdict", "inspect", "cat_0", "cat_1", "cat_2")
    await harness.reply(
        "planning_worker",
        plan_reply(*(planned(step, True) for step in [*paints, verdict])),
    )
    assert harness.held() == {"image_a", "image_b"}
    assert [
        harness.request(name).params["seed"] for name in ("image_a", "image_b")
    ] == [7, 8]
    await harness.reply("image_b", image_result(inline_images[1]))
    # The free replica takes the queued step; cat_1 waits for cat_0 to publish.
    assert harness.request("image_b").inputs == "Do cat_2"
    assert harness.request("image_b").params["seed"] == 9
    assert harness.segments == []
    await harness.reply("image_a", image_result(inline_images[0]))
    assert [segment["data"] for segment in harness.segments] == [
        inline_images[0].model_dump(),
        inline_images[1].model_dump(),
    ]
    assert "vision_worker" not in harness.held()
    await harness.reply("image_b", image_result(inline_images[2]))
    images = [
        part["image_url"]["url"]
        for part in harness.request("vision_worker").inputs[-1]["content"]
        if part["type"] == "image_url"
    ]
    assert images == [image.url for image in inline_images]
    await harness.reply("vision_worker", {"text": "The blue cat"})
    segments = await harness.finished()
    assert [segment["data"] for segment in segments] == [
        *(image.model_dump() for image in inline_images),
        "The blue cat",
    ]
    assert [segment["segment_index"] for segment in segments] == [0, 1, 2, 3]
    assert len({segment["execution_id"] for segment in segments}) == 1
    assert not harness.orchestrator.adapter.executions


@pytest.mark.asyncio
async def test_multipart_conversation_becomes_typed_inputs(
    composition: UMMCompositionSpec, inline_images: list[InlineImage]
) -> None:
    harness = Harness(composition)
    await harness.submit(
        TaskRequest(
            messages=[
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
    task = harness.request("planning_worker").inputs[1]["content"]
    assert [part["text"] for part in task if part["type"] == "text"] == [
        "[system] Answer briefly",
        "[user] What happens here?",
        "[user] attached user_image_1 (image)",
    ]
    assert [part["type"] for part in task if part["type"] != "text"] == ["image_url"]
    await harness.reply(
        "planning_worker",
        plan_reply(planned(call("answer", "inspect", "user_image_1"), True)),
    )
    content = harness.request("vision_worker").inputs[-1]["content"]
    assert [part for part in content if part["type"] != "text"] == [
        {"type": "image_url", "image_url": {"url": inline_images[0].url}},
    ]
    assert content[0]["text"].startswith("User request:\n[system] Answer briefly")


@pytest.mark.asyncio
async def test_option_input_image_is_an_input_artifact(
    composition: UMMCompositionSpec, inline_images: list[InlineImage]
) -> None:
    image = inline_images[0]
    harness = Harness(composition)
    await harness.submit(
        task_of("Make it blue", image_base64=image.encoded_image(), width=64)
    )
    await harness.reply(
        "planning_worker",
        plan_reply(planned(call("answer", "inspect", "input_image"), True)),
    )
    content = harness.request("vision_worker").inputs[-1]["content"]
    assert content[-1]["image_url"]["url"] == image.url


def media_message(part: dict[str, str | dict[str, str]], copies: int = 1) -> list:
    return [{"role": "user", "content": [part] * copies}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task",
    [
        task_of("Draw", mode="compare"),
        task_of("Draw", steps=0),
        task_of("Draw", width=257),
        TaskRequest(messages=[]),
        TaskRequest(
            messages=media_message(
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.invalid/cat.png"},
                }
            )
        ),
        TaskRequest(
            messages=media_message(
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:image/png;base64,eA=="},
                }
            )
        ),
        TaskRequest(
            messages=media_message(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,eA=="},
                },
                copies=5,
            )
        ),
        TaskRequest(messages=[{"role": "tool", "content": "result"}]),
    ],
    ids=[
        "unknown-mode",
        "zero-steps",
        "unaligned-width",
        "no-messages",
        "remote-media",
        "mismatched-kind",
        "too-many-media",
        "tool-role",
    ],
)
async def test_invalid_client_task_never_reaches_planning(
    composition: UMMCompositionSpec, task: TaskRequest
) -> None:
    harness = Harness(composition)
    await harness.submit(task)
    assert isinstance(harness.error(), UMMRequestError)
    assert harness.held() == set()
    assert not harness.orchestrator.adapter.executions


@pytest.mark.asyncio
async def test_mode_outside_the_allowed_set_is_a_client_error(
    composition: UMMCompositionSpec,
) -> None:
    harness = Harness(composition)
    await harness.submit(task_of("Draw", mode="iterative"))
    assert "Input should be 'plan'" in str(harness.error())


@pytest.mark.asyncio
async def test_abort_with_parallel_steps_in_flight_releases_the_execution(
    composition: UMMCompositionSpec,
) -> None:
    harness = Harness(composition)
    await harness.submit(task_of("Draw two cats"))
    await harness.reply(
        "planning_worker",
        plan_reply(
            planned(call("a", "paint"), True), planned(call("b", "paint"), True)
        ),
    )
    assert harness.held() == {"image_a", "image_b"}
    assert harness.task is not None
    harness.task.cancel()
    await settle()
    assert harness.held() == set()
    assert (
        harness.backends["image_a"].cancelled and harness.backends["image_b"].cancelled
    )
    assert not harness.orchestrator.adapter.executions
    assert not harness.orchestrator.active
    await asyncio.gather(harness.task, return_exceptions=True)
