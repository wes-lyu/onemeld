# SPDX-License-Identifier: Apache-2.0
"""Invalid planner replies, bounded repairs, and capability and budget contracts."""

import json
from dataclasses import replace

import pytest

from sglang_omni.client.image_media import InlineImage
from sglang_omni.pipeline.umm import (
    UMMController,
    UMMPlanningError,
    create_umm_controller,
)
from sglang_omni.pipeline.umm_plan import UMMCapability, UMMCompositionSpec
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_errors import generation_error_status_code
from tests.unit_test.fixtures.umm_stages import (
    FakeStages,
    build_controller,
    image_result,
)

pytest_plugins = ["tests.unit_test.fixtures.umm_composition"]

STEP = {
    "id": "answer",
    "action": "inspect",
    "instruction": "Describe it",
    "input_refs": [],
}


def rejected(
    controller: UMMController,
    text: str,
    message: str,
    finish_reason: str = "stop",
) -> None:
    """The reply fails validation with message before any step is dispatched."""
    stages = FakeStages(controller, "rejected")
    stages.submit(
        OmniRequest(
            "Draw a cat",
            metadata={"composition": {"image_base64": "eA=="}},
        )
    )
    planning = stages.held.pop("planning_worker")
    with pytest.raises(UMMPlanningError, match=message):
        controller.transition(
            replace(planning, data={"text": text, "finish_reason": finish_reason})
        )
    assert not controller.adapter.executions


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
def test_invalid_steps_never_dispatch(
    controller: UMMController,
    steps: list[dict[str, str | list[str]]],
    finish_reason: str,
    message: str,
) -> None:
    text = json.dumps({"steps": [{**step, "publish": True} for step in steps]})
    rejected(controller, text, message, finish_reason)


def test_plan_without_published_output_is_rejected(controller: UMMController) -> None:
    rejected(
        controller,
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
        "duplicate-stage",
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


def test_each_replica_stage_is_its_own_phase(composition: UMMCompositionSpec) -> None:
    assert composition.phases["plan"] == "planning_worker"
    assert {composition.phases["image_a"], composition.phases["image_b"]} == {
        "image_a",
        "image_b",
    }
    values = composition.model_dump(mode="json")
    values["capabilities"]["paint"]["stages"] = ["plan"]
    with pytest.raises(ValueError, match="reserved"):
        UMMCompositionSpec.model_validate(values)


def test_composition_rejects_inconsistent_stage_bindings(
    composition: UMMCompositionSpec,
) -> None:
    with pytest.raises(ValueError, match="bindings"):
        create_umm_controller(
            "unused",
            composition=composition,
            phases={"plan": "wrong"},
            output_stage="output",
            max_dispatches=composition.required_dispatches(),
            max_outputs=composition.required_outputs(),
            max_active=1,
            max_retained_bytes=128 * 1024 * 1024,
            timeout_s=60,
        )


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_invalid_reply_gets_only_a_bounded_model_repair(
    composition: UMMCompositionSpec, repair_succeeds: bool
) -> None:
    """A rejected dependency must never run; the model can repair it once."""
    controller = build_controller(
        composition.model_copy(update={"max_plan_attempts": 2})
    )
    stages = FakeStages(controller, "repair")
    stages.submit(OmniRequest("Answer a question"))

    def reply(input_refs: list[str]) -> str:
        step = {**STEP, "input_refs": input_refs}
        return json.dumps({"steps": [{**step, "publish": True}]})

    first = stages.held["planning_worker"].request
    stages.reply("planning_worker", {"text": reply(["answer"])})
    repair = stages.held["planning_worker"].request
    assert repair.inputs[-2]["content"] == reply(["answer"])
    assert "unavailable" in repair.inputs[-1]["content"]
    # The first reply decodes greedily; the repair samples so it can differ.
    assert (first.params["temperature"], repair.params["temperature"]) == (0.0, 0.7)
    if repair_succeeds:
        stages.reply("planning_worker", {"text": reply([])})
        stages.reply("vision_worker", {"text": "A corrected answer"})
        assert [segment["data"] for segment in stages.segments()] == [
            "A corrected answer"
        ]
    else:
        planning = stages.held.pop("planning_worker")
        with pytest.raises(UMMPlanningError, match="after 2 planner attempts"):
            controller.transition(replace(planning, data={"text": reply(["answer"])}))
    assert not controller.adapter.executions


def test_dispatch_budget_covers_repair_and_all_steps(
    composition: UMMCompositionSpec,
) -> None:
    composition = composition.model_copy(
        update={"max_plan_attempts": 2, "max_steps": 3}
    )
    controller = build_controller(composition)
    stages = FakeStages(controller, "budget")
    stages.submit(OmniRequest("Give three answers"))
    stages.reply("planning_worker", {"text": "not json"})
    steps = [{**STEP, "id": f"step_{index}", "publish": True} for index in range(3)]
    stages.reply("planning_worker", {"text": json.dumps({"steps": steps})})
    for index in range(3):
        stages.reply("vision_worker", {"text": f"Answer {index}"})
    assert [segment["data"] for segment in stages.segments()] == [
        "Answer 0",
        "Answer 1",
        "Answer 2",
    ]
    assert not controller.contexts


def test_empty_planner_reply_gets_a_repair(composition: UMMCompositionSpec) -> None:
    """A planner that stops at once is asked again rather than failing the request."""
    controller = build_controller(
        composition.model_copy(update={"max_plan_attempts": 2})
    )
    stages = FakeStages(controller, "empty")
    stages.submit(OmniRequest("Answer a question"))
    stages.reply("planning_worker", {"text": ""})
    repair = stages.held["planning_worker"].request.inputs
    assert repair[-2] == {"role": "assistant", "content": ""}
    assert "rejected" in repair[-1]["content"]


def test_repeated_and_input_named_step_ids_are_renamed(
    controller: UMMController, inline_images: list[InlineImage]
) -> None:
    """IDs a small planner reuses are renamed; references follow the latest step."""
    stages = FakeStages(controller, "renamed")
    stages.submit(
        OmniRequest(
            [
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
            ],
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
    stages.reply("planning_worker", {"text": text})
    first = stages.held["vision_worker"].request.inputs[-1]["content"]
    assert first[-1]["image_url"]["url"] == inline_images[2].url
    stages.reply("image_a", image_result(inline_images[0]))
    stages.reply("image_b", image_result(inline_images[1]))
    stages.reply("vision_worker", {"text": "Green cat"})
    check = stages.held["vision_worker"].request.inputs[-1]["content"]
    assert [part["text"] for part in check if part["type"] == "text"][1:] == [
        "Image generate_2:",
        "Text image_1: Green cat",
    ]
    assert [
        part["image_url"]["url"] for part in check if part["type"] == "image_url"
    ] == [inline_images[1].url]


def test_exhausted_planning_is_reported_as_unprocessable(
    controller: UMMController,
) -> None:
    """A planner that never produces a valid reply is HTTP 422, not a server fault."""
    stages = FakeStages(controller, "unplannable")
    stages.submit(OmniRequest("Draw a cat"))
    planning = stages.held.pop("planning_worker")
    context = controller.contexts["unplannable"]
    assert controller.advance(replace(planning, data={"text": "not json"})) is None
    error = controller.outbox.get_nowait().data
    # The error crosses stage IPC as text before serving classifies it.
    assert generation_error_status_code(RuntimeError(str(error))) == 422
    assert f"execution_id={context.execution_id}" in str(error)
    assert not controller.adapter.executions
