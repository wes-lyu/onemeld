# SPDX-License-Identifier: Apache-2.0
"""Invalid planner replies, bounded repairs, and capability and budget contracts."""

import json
from dataclasses import replace

import pytest

from sglang_omni.pipeline.umm import UMMController, create_umm_controller
from sglang_omni.pipeline.umm_plan import (
    UMMCapability,
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMPlanExample,
    UMMPlanStep,
)
from sglang_omni.proto.request import OmniRequest
from tests.unit_test.fixtures.umm_stages import FakeStages, build_controller

pytest_plugins = ["tests.unit_test.fixtures.umm_composition"]

STEP = {
    "id": "answer",
    "action": "inspect",
    "instruction": "Describe it",
    "input_refs": [],
}


def rejected(
    controller: UMMController,
    mode: str,
    text: str,
    message: str,
    finish_reason: str = "stop",
) -> None:
    """The reply fails validation with message before any step is dispatched."""
    stages = FakeStages(controller, "rejected")
    stages.submit(
        OmniRequest(
            "Draw a cat",
            metadata={"composition": {"mode": mode, "image_base64": "eA=="}},
        )
    )
    planning = stages.held.pop("planning_worker")
    with pytest.raises(ValueError, match=message):
        controller.transition(
            replace(planning, data={"text": text, "finish_reason": finish_reason})
        )
    assert not controller.adapter.executions


@pytest.mark.parametrize(
    "steps, finish_reason, message",
    [
        ([{**STEP, "input_refs": ["later"]}], "stop", "unavailable"),
        ([STEP, STEP], "stop", "Duplicate"),
        ([{**STEP, "id": "user_answer"}], "stop", "reserved"),
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
        "duplicate-id",
        "reserved-id",
        "wrong-kind",
        "missing-required-audio",
        "missing-input",
        "truncated",
        "unknown-action",
        "too-many-images",
    ],
)
@pytest.mark.parametrize("mode", ["plan", "iterative"])
def test_invalid_steps_never_dispatch_in_either_mode(
    controller: UMMController,
    steps: list[dict[str, str | list[str]]],
    finish_reason: str,
    message: str,
    mode: str,
) -> None:
    if mode == "plan":
        text = json.dumps({"steps": [{**step, "publish": True} for step in steps]})
    else:
        text = json.dumps({"steps": steps, "final": None})
    rejected(controller, mode, text, message, finish_reason)


@pytest.mark.parametrize(
    "reply, message",
    [
        (
            {"steps": [], "final": {"publish": ["input_image"], "answer": ""}},
            "only step outputs",
        ),
        (
            {"steps": [], "final": {"publish": ["missing"], "answer": ""}},
            "only step outputs",
        ),
        (
            {"steps": [STEP], "final": {"publish": ["answer", "answer"], "answer": ""}},
            "twice",
        ),
        (
            {"steps": [STEP], "final": {"publish": ["answer"], "answer": "Too early"}},
            "observed results",
        ),
        ({"steps": [], "final": {"publish": [], "answer": " "}}, "at least one output"),
        ({"steps": [], "final": None}, "run steps or return final"),
    ],
    ids=["input", "unknown", "duplicate", "answer-before-results", "empty", "no-work"],
)
def test_final_output_publishes_only_step_outputs(
    controller: UMMController, reply: dict[str, object], message: str
) -> None:
    rejected(controller, "iterative", json.dumps(reply), message)


def test_plan_without_published_output_is_rejected(controller: UMMController) -> None:
    rejected(
        controller,
        "plan",
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
        ({"protocol": "tts", "input_kinds": ["image"]}, "accepts only"),
        (
            {
                "protocol": "native_image",
                "input_kinds": ["text"],
                "parameters": {"image_path": "x"},
            },
            "runtime-owned",
        ),
        (
            {
                "protocol": "native_chat",
                "input_kinds": ["text"],
                "speech": {"voice": "a"},
            },
            "speech",
        ),
    ],
    ids=[
        "asr-without-audio",
        "duplicate-stage",
        "tts-image",
        "owned-field",
        "speech-on-chat",
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
    values = composition.model_dump(mode="json")
    values["execution_mode"] = "iterative"
    values["allowed_modes"] = ["plan"]
    with pytest.raises(ValueError, match="allowed_modes"):
        UMMCompositionSpec.model_validate(values)


@pytest.mark.parametrize("mode", ["plan", "iterative"])
@pytest.mark.parametrize("valid_example", [True, False])
def test_plan_examples_guide_both_modes_without_running(
    composition: UMMCompositionSpec, valid_example: bool, mode: str
) -> None:
    """Prompt examples must be validated and never dispatched as user work."""
    example = UMMPlanExample(
        prompt="Generate one image",
        plan=UMMExecutionPlan(
            steps=[
                UMMPlanStep(
                    id="example_photo",
                    action="paint",
                    instruction="A bicycle",
                    input_refs=[] if valid_example else ["unavailable"],
                    publish=True,
                )
            ]
        ),
    )
    values = composition.model_dump(mode="json")
    values["plan_examples"] = [example.model_dump(mode="json")]
    if not valid_example:
        with pytest.raises(ValueError, match="unavailable"):
            UMMCompositionSpec.model_validate(values)
    else:
        stages = FakeStages(
            build_controller(UMMCompositionSpec.model_validate(values)), "example"
        )
        stages.submit(
            OmniRequest("Answer a question", metadata={"composition": {"mode": mode}})
        )
        planning = stages.held["planning_worker"].request.inputs
        assert [message["role"] for message in planning] == ["system", "user"]
        assert "A bicycle" in planning[0]["content"]
        assert ('"final": {"publish": ["example_photo"]' in planning[0]["content"]) == (
            mode == "iterative"
        )
        assert planning[1]["content"] == [
            {"type": "text", "text": "[user] Answer a question"}
        ]


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


@pytest.mark.parametrize("mode", ["plan", "iterative"])
@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_invalid_reply_gets_only_a_bounded_model_repair(
    composition: UMMCompositionSpec, repair_succeeds: bool, mode: str
) -> None:
    """A rejected dependency must never run; the model can repair it once."""
    controller = build_controller(
        composition.model_copy(update={"max_plan_attempts": 2})
    )
    stages = FakeStages(controller, "repair")
    stages.submit(
        OmniRequest("Answer a question", metadata={"composition": {"mode": mode}})
    )

    def reply(input_refs: list[str]) -> str:
        step = {**STEP, "input_refs": input_refs}
        if mode == "plan":
            return json.dumps({"steps": [{**step, "publish": True}]})
        else:
            return json.dumps(
                {"steps": [step], "final": {"publish": ["answer"], "answer": ""}}
            )

    stages.reply("planning_worker", {"text": reply(["answer"])})
    repair = stages.held["planning_worker"].request.inputs
    assert repair[-2]["content"] == reply(["answer"])
    assert "unavailable" in repair[-1]["content"]
    if repair_succeeds:
        stages.reply("planning_worker", {"text": reply([])})
        stages.reply("vision_worker", {"text": "A corrected answer"})
        assert [segment["data"] for segment in stages.segments()] == [
            "A corrected answer"
        ]
    else:
        planning = stages.held.pop("planning_worker")
        with pytest.raises(ValueError, match="unavailable"):
            controller.transition(replace(planning, data={"text": reply(["answer"])}))
    assert not controller.adapter.executions


@pytest.mark.parametrize(
    "allowed_modes, max_planner_turns, max_plan_attempts",
    [(["plan"], 4, 2), (["plan", "iterative"], 3, 2)],
)
def test_controller_budget_covers_the_costliest_allowed_mode(
    composition: UMMCompositionSpec,
    allowed_modes: list[str],
    max_planner_turns: int,
    max_plan_attempts: int,
) -> None:
    """An iterative request that uses every turn and repair still fits the budget."""
    composition = composition.model_copy(
        update={
            "allowed_modes": allowed_modes,
            "max_planner_turns": max_planner_turns,
            "max_plan_attempts": max_plan_attempts,
            "max_steps": max_planner_turns - 1,
        }
    )
    controller = build_controller(composition)
    mode = allowed_modes[-1]
    stages = FakeStages(controller, "budget")
    stages.submit(OmniRequest("Think hard", metadata={"composition": {"mode": mode}}))
    turns = max_planner_turns if mode == "iterative" else 1
    for turn in range(turns):
        stages.reply("planning_worker", {"text": "not json"})
        is_last = turn == turns - 1
        if mode == "plan":
            text = json.dumps(
                {
                    "steps": [
                        {**STEP, "id": f"step_{index}", "publish": True}
                        for index in range(composition.max_steps)
                    ]
                }
            )
        elif is_last:
            text = json.dumps({"steps": [], "final": {"publish": [], "answer": "Done"}})
        else:
            text = json.dumps(
                {"steps": [{**STEP, "id": f"step_{turn}"}], "final": None}
            )
        stages.reply("planning_worker", {"text": text})
        while "vision_worker" in stages.held:
            stages.reply("vision_worker", {"text": "Observed"})
    assert stages.segments()
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
