# SPDX-License-Identifier: Apache-2.0
"""Invalid plans, bounded model repairs, and composition binding contracts."""

import json
from dataclasses import replace

import pytest

from sglang_omni.pipeline.umm import UMMController, create_umm_controller, route_umm
from sglang_omni.pipeline.umm_plan import (
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMPlanExample,
    UMMPlanStep,
)
from sglang_omni.proto.request import OmniRequest, StagePayload

pytest_plugins = ["tests.unit_test.fixtures.umm_composition"]


@pytest.mark.parametrize(
    "case",
    [
        "forward_reference",
        "duplicate_id",
        "wrong_kind",
        "missing_input",
        "truncated",
        "unknown_action",
        "no_public_output",
    ],
)
def test_invalid_model_plan_never_dispatches_execution(
    controller: UMMController, case: str
) -> None:
    dispatched = controller.advance(
        StagePayload("invalid-plan", OmniRequest("Draw a cat"), None)
    )
    step = {
        "id": "answer",
        "action": "inspect",
        "instruction": "Describe it",
        "input_refs": [],
        "publish": True,
    }
    steps = [step]
    finish_reason = "stop"
    if case == "forward_reference":
        step["input_refs"] = ["later"]
    elif case == "duplicate_id":
        steps.append(dict(step))
    elif case == "wrong_kind":
        steps = [
            {
                "id": "photo",
                "action": "paint",
                "instruction": "A cat",
                "input_refs": [],
                "publish": True,
            },
            {**step, "action": "paint", "input_refs": ["photo"]},
        ]
    elif case == "missing_input":
        step["input_refs"] = ["input_image"]
    elif case == "truncated":
        finish_reason = "length"
    elif case == "unknown_action":
        step["action"] = "unknown"
    else:
        step["publish"] = False
    with pytest.raises(ValueError):
        controller.transition(
            replace(
                dispatched,
                data={
                    "text": json.dumps({"steps": steps}),
                    "finish_reason": finish_reason,
                },
            )
        )
    assert not controller.adapter.executions


@pytest.mark.parametrize("valid_example", [True, False])
def test_configured_plan_examples_guide_planning_without_running(
    composition: UMMCompositionSpec, valid_example: bool
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
        controller = create_umm_controller(
            "unused",
            composition=values,
            phases=composition.phases,
            output_stage="output",
            max_dispatches=7,
            max_outputs=6,
            max_active=1,
            max_retained_bytes=128 * 1024 * 1024,
            timeout_s=60,
        )
        dispatched = controller.advance(
            StagePayload("example", OmniRequest("Answer a question"), None)
        )
        assert [message["role"] for message in dispatched.request.inputs] == [
            "system",
            "user",
        ]
        assert example.prompt in dispatched.request.inputs[0]["content"]
        assert "A bicycle" in dispatched.request.inputs[0]["content"]
        assert (
            dispatched.request.inputs[-1]["content"][0]["text"] == "Answer a question"
        )
        plan = UMMExecutionPlan(
            steps=[
                UMMPlanStep(
                    id="answer",
                    action="inspect",
                    instruction="Answer the actual question",
                    input_refs=[],
                    publish=True,
                )
            ]
        )
        dispatched = controller.advance(
            replace(dispatched, data={"text": plan.model_dump_json()})
        )
        assert route_umm("example", dispatched) == "vision_worker"
        completed = controller.advance(replace(dispatched, data={"text": "Answer"}))
        assert [segment["kind"] for segment in completed.data["segments"]] == ["text"]


def test_composition_rejects_inconsistent_stage_bindings(
    composition: UMMCompositionSpec,
) -> None:
    with pytest.raises(ValueError, match="bindings"):
        create_umm_controller(
            "unused",
            composition=composition,
            phases={"plan": "wrong"},
            output_stage="output",
            max_dispatches=7,
            max_outputs=6,
            max_active=1,
            max_retained_bytes=128 * 1024 * 1024,
            timeout_s=60,
        )


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_invalid_self_reference_gets_only_a_bounded_model_repair(
    composition: UMMCompositionSpec, repair_succeeds: bool
) -> None:
    """A rejected dependency must never run; the model can repair it once."""
    composition = composition.model_copy(
        update={"max_steps": 1, "max_plan_attempts": 2}
    )
    controller = create_umm_controller(
        "unused",
        composition=composition,
        phases=composition.phases,
        output_stage="output",
        max_dispatches=3,
        max_outputs=1,
        max_active=1,
        max_retained_bytes=128 * 1024 * 1024,
        timeout_s=60,
    )
    dispatched = controller.advance(
        StagePayload("repair", OmniRequest("Answer a question"), None)
    )
    invalid = UMMExecutionPlan(
        steps=[
            UMMPlanStep(
                id="answer",
                action="inspect",
                instruction="Answer",
                input_refs=["answer"],
                publish=True,
            )
        ]
    )
    repair = controller.advance(
        replace(dispatched, data={"text": invalid.model_dump_json()})
    )
    assert route_umm("repair", repair) == "planning_worker"
    assert repair.request.inputs[-2]["content"] == invalid.model_dump_json()
    assert "unavailable" in repair.request.inputs[-1]["content"]
    if repair_succeeds:
        valid = UMMExecutionPlan(
            steps=[
                UMMPlanStep(
                    id="answer",
                    action="inspect",
                    instruction="Answer",
                    input_refs=[],
                    publish=True,
                )
            ]
        )
        dispatched = controller.advance(
            replace(repair, data={"text": valid.model_dump_json()})
        )
        assert route_umm("repair", dispatched) == "vision_worker"
        completed = controller.advance(
            replace(dispatched, data={"text": "A corrected answer"})
        )
        assert [segment["data"] for segment in completed.data["segments"]] == [
            "A corrected answer"
        ]
    else:
        with pytest.raises(ValueError, match="unavailable"):
            controller.transition(
                replace(repair, data={"text": invalid.model_dump_json()})
            )
    assert not controller.adapter.executions
