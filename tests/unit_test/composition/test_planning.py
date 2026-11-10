# SPDX-License-Identifier: Apache-2.0
"""Invalid planner replies, bounded repairs, and capability and budget contracts."""

import json

import pytest

from onemeld.composition.media import InlineImage
from onemeld.composition.plan import UMMCapability, UMMCompositionSpec
from onemeld.composition.types import TaskRequest, UMMPlanningError
from onemeld.orchestrator import Orchestrator
from tests.unit_test.fixtures.composition import image_result
from tests.unit_test.fixtures.orchestrator import (
    Harness,
    ScriptedBackend,
    limits_for,
    task_of,
)

pytest_plugins = ["tests.unit_test.fixtures.composition"]

STEP = {
    "id": "answer",
    "action": "inspect",
    "instruction": "Describe it",
    "input_refs": [],
}


async def rejected(
    composition: UMMCompositionSpec,
    text: str,
    message: str,
    finish_reason: str = "stop",
) -> None:
    """The reply fails validation with message before any step is dispatched."""
    harness = Harness(composition)
    await harness.submit(task_of("Draw a cat", image_base64="eA=="))
    await harness.reply(
        "planning_worker", {"text": text, "finish_reason": finish_reason}
    )
    error = harness.error()
    assert isinstance(error, UMMPlanningError)
    assert message in str(error)
    assert harness.held() == set()
    assert not harness.orchestrator.adapter.executions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "steps, finish_reason, message",
    [
        ([{**STEP, "input_refs": ["later"]}], "stop", "unavailable"),
        (
            [
                {**STEP, "id": "photo", "action": "paint"},
                {**STEP, "action": "paint", "input_refs": ["photo"]},
            ],
            "stop",
            "cannot consume",
        ),
        ([{**STEP, "action": "transcribe"}], "stop", "exactly one audio"),
        ([{**STEP, "input_refs": ["user_image_1"]}], "stop", "unavailable"),
        ([STEP], "length", "did not finish"),
        ([{**STEP, "action": "unknown"}], "stop", "Unknown capability"),
        (
            [{**STEP, "id": f"cat_{index}", "action": "paint"} for index in range(5)],
            "stop",
            "max_images",
        ),
    ],
    ids=[
        "forward-reference",
        "wrong-kind",
        "missing-required-audio",
        "missing-input",
        "truncated",
        "unknown-action",
        "too-many-images",
    ],
)
async def test_invalid_steps_never_dispatch(
    composition: UMMCompositionSpec,
    steps: list[dict[str, str | list[str]]],
    finish_reason: str,
    message: str,
) -> None:
    text = json.dumps({"steps": [{**step, "publish": True} for step in steps]})
    await rejected(composition, text, message, finish_reason)


@pytest.mark.asyncio
async def test_plan_without_published_output_is_rejected(
    composition: UMMCompositionSpec,
) -> None:
    await rejected(
        composition,
        json.dumps({"steps": [{**STEP, "publish": False}]}),
        "at least one output",
    )


@pytest.mark.parametrize(
    "capability, message",
    [
        ({"protocol": "asr", "input_kinds": ["text"]}, "accepts only"),
        (
            {"protocol": "asr", "input_kinds": ["audio"], "stages": ["a", "a"]},
            "distinct",
        ),
        (
            {
                "protocol": "image_base64",
                "input_kinds": ["text"],
                "parameters": {"n": 2},
            },
            "runtime-owned",
        ),
    ],
    ids=[
        "asr-without-audio",
        "duplicate-backend",
        "owned-field",
    ],
)
def test_capabilities_respect_their_protocol_contract(
    capability: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        UMMCapability.model_validate(
            {"stages": ["worker"], "description": "x", **capability}
        )


def test_each_replica_backend_is_its_own_phase(
    composition: UMMCompositionSpec,
) -> None:
    assert composition.phases["plan"] == "planning_worker"
    assert {composition.phases["image_a"], composition.phases["image_b"]} == {
        "image_a",
        "image_b",
    }
    values = composition.model_dump(mode="json")
    values["capabilities"]["paint"]["stages"] = ["plan"]
    with pytest.raises(ValueError, match="reserved"):
        UMMCompositionSpec.model_validate(values)


def test_orchestrator_rejects_a_phase_without_a_backend(
    composition: UMMCompositionSpec,
) -> None:
    backends = {
        name: ScriptedBackend()
        for name in set(composition.phases.values()) - {"image_b"}
    }
    with pytest.raises(ValueError, match=r"no binding: \['image_b'\]"):
        Orchestrator(composition, backends, limits_for(composition))


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_invalid_reply_gets_only_a_bounded_model_repair(
    composition: UMMCompositionSpec, repair_succeeds: bool
) -> None:
    """A rejected dependency must never run; the model can repair it once."""
    harness = Harness(composition.model_copy(update={"max_plan_attempts": 2}))
    await harness.submit(task_of("Answer a question"))

    def reply(input_refs: list[str]) -> str:
        step = {**STEP, "input_refs": input_refs}
        return json.dumps({"steps": [{**step, "publish": True}]})

    first = harness.request("planning_worker")
    await harness.reply("planning_worker", {"text": reply(["answer"])})
    repair = harness.request("planning_worker")
    assert repair.inputs[-2]["content"] == reply(["answer"])
    assert "unavailable" in repair.inputs[-1]["content"]
    # The first reply decodes greedily; the repair samples so it can differ.
    assert (first.params["temperature"], repair.params["temperature"]) == (0.0, 0.7)
    if repair_succeeds:
        await harness.reply("planning_worker", {"text": reply([])})
        await harness.reply("vision_worker", {"text": "A corrected answer"})
        assert [segment["data"] for segment in await harness.finished()] == [
            "A corrected answer"
        ]
    else:
        await harness.reply("planning_worker", {"text": reply(["answer"])})
        assert "after 2 planner attempts" in str(harness.error())
    assert not harness.orchestrator.adapter.executions


@pytest.mark.asyncio
async def test_dispatch_budget_covers_repair_and_all_steps(
    composition: UMMCompositionSpec,
) -> None:
    harness = Harness(
        composition.model_copy(update={"max_plan_attempts": 2, "max_steps": 3})
    )
    await harness.submit(task_of("Give three answers"))
    await harness.reply("planning_worker", {"text": "not json"})
    steps = [{**STEP, "id": f"step_{index}", "publish": True} for index in range(3)]
    await harness.reply("planning_worker", {"text": json.dumps({"steps": steps})})
    for index in range(3):
        await harness.reply("vision_worker", {"text": f"Answer {index}"})
    assert [segment["data"] for segment in await harness.finished()] == [
        "Answer 0",
        "Answer 1",
        "Answer 2",
    ]
    assert not harness.orchestrator.active


@pytest.mark.asyncio
async def test_empty_planner_reply_gets_a_repair(
    composition: UMMCompositionSpec,
) -> None:
    """A planner that stops at once is asked again rather than failing the request."""
    harness = Harness(composition.model_copy(update={"max_plan_attempts": 2}))
    await harness.submit(task_of("Answer a question"))
    await harness.reply("planning_worker", {"text": ""})
    repair = harness.request("planning_worker").inputs
    assert repair[-2] == {"role": "assistant", "content": ""}
    assert "rejected" in repair[-1]["content"]


@pytest.mark.asyncio
async def test_repeated_and_input_named_step_ids_are_renamed(
    composition: UMMCompositionSpec, inline_images: list[InlineImage]
) -> None:
    """IDs a small planner reuses are renamed; references follow the latest step."""
    harness = Harness(composition)
    await harness.submit(
        TaskRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Draw twice, describe my photo, then compare",
                        },
                        {"type": "image_url", "image_url": inline_images[2].url},
                    ],
                }
            ]
        )
    )
    steps = [
        {**STEP, "id": "generate", "action": "paint"},
        {**STEP, "id": "generate", "action": "paint"},
        {
            **STEP,
            "id": "user_image_1",
            "action": "inspect",
            "input_refs": ["user_image_1"],
        },
        {**STEP, "id": "check", "input_refs": ["generate", "user_image_1"]},
    ]
    text = json.dumps({"steps": [{**step, "publish": True} for step in steps]})
    await harness.reply("planning_worker", {"text": text})
    first = harness.request("vision_worker").inputs[-1]["content"]
    assert first[-1]["image_url"]["url"] == inline_images[2].url
    await harness.reply("image_a", image_result(inline_images[0]))
    await harness.reply("image_b", image_result(inline_images[1]))
    await harness.reply("vision_worker", {"text": "Green cat"})
    check = harness.request("vision_worker").inputs[-1]["content"]
    assert [part["text"] for part in check if part["type"] == "text"][1:] == [
        "Image generate_2:",
        "Text image_1: Green cat",
    ]
    assert [
        part["image_url"]["url"] for part in check if part["type"] == "image_url"
    ] == [inline_images[1].url]


@pytest.mark.asyncio
async def test_exhausted_planning_names_its_trace(
    composition: UMMCompositionSpec,
) -> None:
    """A planner that never produces a valid reply fails with its trace execution ID."""
    harness = Harness(composition)
    await harness.submit(task_of("Draw a cat"))
    (request_id,) = harness.backends["planning_worker"].held
    await harness.reply("planning_worker", {"text": "not json"})
    error = harness.error()
    assert isinstance(error, UMMPlanningError)
    execution_id = request_id.rpartition("-")[0]
    assert f"execution_id={execution_id}" in str(error)
    assert not harness.orchestrator.adapter.executions
