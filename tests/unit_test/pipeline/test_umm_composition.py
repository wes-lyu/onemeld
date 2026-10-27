# SPDX-License-Identifier: Apache-2.0
"""Public capability routing, typed artifacts, and request lifecycle through UMM."""

from dataclasses import replace

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.image_media import InlineImage
from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.pipeline.umm import UMMController, route_umm
from sglang_omni.pipeline.umm_plan import UMMExecutionPlan, UMMPlanStep
from sglang_omni.proto.request import OmniRequest, StagePayload

pytest_plugins = ["tests.unit_test.fixtures.umm_composition"]


@pytest.mark.parametrize(
    "actions", [("inspect",), ("paint",), ("paint", "paint", "inspect")]
)
def test_model_plan_selects_configured_capabilities(
    controller: UMMController,
    inline_images: list[InlineImage],
    actions: tuple[str, ...],
) -> None:
    request = Client.build_omni_request(
        GenerateRequest(
            messages=[Message(role="user", content="Complete my task")],
            stream=False,
            sampling=SamplingParams(seed=42, max_new_tokens=512),
            output_modalities=["text", "image"],
            stage_params={"orchestrator": {"steps": 4}},
        )
    )
    dispatched = controller.advance(StagePayload("dynamic", request, None))
    assert route_umm("dynamic", dispatched) == "planning_worker"
    schema = dispatched.request.params["response_format"]["json_schema"]["schema"]
    assert set(schema["$defs"]["UMMPlanStep"]["properties"]["action"]["enum"]) == {
        "paint",
        "inspect",
    }
    image_ids: list[str] = []
    steps: list[UMMPlanStep] = []
    for index, action in enumerate(actions):
        step = UMMPlanStep(
            id=f"step_{index}",
            action=action,
            instruction=f"Instruction {index}",
            input_refs=list(image_ids) if action == "inspect" else [],
            publish=True,
        )
        steps.append(step)
        if action == "paint":
            image_ids.append(step.id)
        else:
            pass
    dispatched = controller.advance(
        replace(
            dispatched, data={"text": UMMExecutionPlan(steps=steps).model_dump_json()}
        )
    )
    expected_outputs: list[dict[str, str | int] | str] = []
    generated_images = 0
    for index, action in enumerate(actions):
        if action == "paint":
            assert route_umm("dynamic", dispatched) == "image_worker"
            assert dispatched.request.inputs["prompt"] == f"Instruction {index}"
            assert dispatched.request.inputs["seed"] == 42 + generated_images
            assert dispatched.request.inputs["num_inference_steps"] == 4
            image = inline_images[generated_images]
            generated_images += 1
            expected_outputs.append(image.model_dump())
            dispatched = controller.advance(
                replace(
                    dispatched,
                    data={"media": [{**image.model_dump(), "generation_time": 1.0}]},
                )
            )
        else:
            assert route_umm("dynamic", dispatched) == "vision_worker"
            content = dispatched.request.inputs[-1]["content"]
            assert [
                part["image_url"]["url"]
                for part in content
                if part["type"] == "image_url"
            ] == [image.url for image in inline_images[:generated_images]]
            expected_outputs.append("Completed answer")
            dispatched = controller.advance(
                replace(dispatched, data={"text": "Completed answer"})
            )
    assert route_umm("dynamic", dispatched) == "output"
    assert [
        segment["data"] for segment in dispatched.data["segments"]
    ] == expected_outputs
    assert not controller.adapter.executions


def test_private_text_artifact_conditions_another_model(
    controller: UMMController, inline_images: list[InlineImage]
) -> None:
    """Private text must reach generation without leaking as a user output."""
    dispatched = controller.advance(
        StagePayload("composed", OmniRequest("Describe then draw and inspect"), None)
    )
    plan = UMMExecutionPlan(
        steps=[
            UMMPlanStep(
                id="description",
                action="inspect",
                instruction="Write a visual prompt",
                input_refs=[],
                publish=False,
            ),
            UMMPlanStep(
                id="photo",
                action="paint",
                instruction="Illustrate the supplied description",
                input_refs=["description"],
                publish=True,
            ),
            UMMPlanStep(
                id="answer",
                action="inspect",
                instruction="Compare the image with its description",
                input_refs=["description", "photo"],
                publish=True,
            ),
        ]
    )
    dispatched = controller.advance(
        replace(dispatched, data={"text": plan.model_dump_json()})
    )
    dispatched = controller.advance(
        replace(dispatched, data={"text": "A blue cat with green eyes"})
    )
    assert route_umm("composed", dispatched) == "image_worker"
    assert "A blue cat with green eyes" in dispatched.request.inputs["prompt"]
    image = inline_images[0]
    dispatched = controller.advance(
        replace(dispatched, data={"media": [image.model_dump()]})
    )
    content = dispatched.request.inputs[-1]["content"]
    assert any(
        part.get("text") == "Text description: A blue cat with green eyes"
        for part in content
    )
    assert content[-1]["image_url"]["url"] == image.url
    completed = controller.advance(
        replace(dispatched, data={"text": "The image matches"})
    )
    assert [segment["kind"] for segment in completed.data["segments"]] == [
        "image",
        "text",
    ]
    assert completed.data["segments"][-1]["data"] == "The image matches"
    assert not controller.adapter.executions


def test_abort_releases_artifacts(
    controller: UMMController, inline_images: list[InlineImage]
) -> None:
    dispatched = controller.advance(
        StagePayload("aborted", OmniRequest("Draw and inspect"), None)
    )
    plan = UMMExecutionPlan(
        steps=[
            UMMPlanStep(
                id="photo",
                action="paint",
                instruction="A cat",
                input_refs=[],
                publish=True,
            ),
            UMMPlanStep(
                id="answer",
                action="inspect",
                instruction="Describe it",
                input_refs=["photo"],
                publish=True,
            ),
        ]
    )
    dispatched = controller.advance(
        replace(dispatched, data={"text": plan.model_dump_json()})
    )
    dispatched = controller.advance(
        replace(dispatched, data={"media": [inline_images[0].model_dump()]})
    )
    controller.abort("aborted")
    assert not controller.adapter.executions
    with pytest.raises(RuntimeError, match="inactive request"):
        controller.transition(replace(dispatched, data={"text": "Late answer"}))


def test_source_image_is_a_typed_input_artifact(
    controller: UMMController, inline_images: list[InlineImage]
) -> None:
    image = inline_images[0]
    dispatched = controller.advance(
        StagePayload(
            "input",
            OmniRequest(
                "Name the color",
                metadata={"composition": {"image_base64": image.encoded_image()}},
            ),
            None,
        )
    )
    assert dispatched.request.inputs[-1]["content"][1]["image_url"]["url"] == image.url
    plan = UMMExecutionPlan(
        steps=[
            UMMPlanStep(
                id="answer",
                action="inspect",
                instruction="Name the color",
                input_refs=["input_image"],
                publish=True,
            )
        ]
    )
    dispatched = controller.advance(
        replace(dispatched, data={"text": plan.model_dump_json()})
    )
    assert dispatched.request.inputs[-1]["content"][-1]["image_url"]["url"] == image.url


@pytest.mark.parametrize("options", [{"mode": "compare"}, {"steps": 0}, {"width": 257}])
def test_invalid_client_request_never_reaches_planning(
    controller: UMMController, options: dict[str, str | int]
) -> None:
    assert (
        controller.advance(
            StagePayload(
                "invalid", OmniRequest("Draw", metadata={"composition": options}), None
            )
        )
        is None
    )
    assert controller.outbox.get_nowait().type == "error"
    assert not controller.adapter.executions
