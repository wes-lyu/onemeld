# SPDX-License-Identifier: Apache-2.0
"""Public capability bindings and execution plans for composed UMM services."""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sglang_omni.client.image_media import InlineImage
from sglang_omni.proto.request import OmniRequest

COMPOSITION_ADAPTER_PATH = "sglang_omni.pipeline.umm_composition.UMMCompositionAdapter"
PLAN_PHASE = "plan"
INPUT_IMAGE_ID = "input_image"
ArtifactKind = Literal["text", "image"]


class UMMCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    stage: str = Field(min_length=1)
    protocol: Literal["native_chat", "native_image"]
    description: str = Field(min_length=1)
    input_kinds: list[ArtifactKind] = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def output_kind(self) -> ArtifactKind:
        return "image" if self.protocol == "native_image" else "text"

    @model_validator(mode="after")
    def validate_protocol(self) -> "UMMCapability":
        protected = {"prompt", "messages", "stream", "num_outputs_per_prompt"}
        if protected.intersection(self.parameters):
            raise ValueError(
                "Capability parameters cannot replace runtime-owned inputs"
            )
        elif self.protocol == "native_image" and "image" in self.input_kinds:
            raise ValueError(
                "native_image currently supports text inputs, not image editing"
            )
        else:
            return self


class UMMPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    action: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=2048)
    input_refs: list[str]
    publish: bool


class UMMExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    steps: list[UMMPlanStep] = Field(min_length=1)


class UMMPlanExample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1, max_length=4096)
    plan: UMMExecutionPlan


class UMMCompositionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    planner_stage: str = Field(min_length=1)
    capabilities: dict[str, UMMCapability] = Field(min_length=1)
    plan_examples: list[UMMPlanExample] = Field(default_factory=list, max_length=4)
    max_steps: int = Field(default=6, ge=1, le=64)
    max_plan_attempts: int = Field(default=1, ge=1, le=3)
    max_images: int = Field(default=4, ge=1, le=16)
    max_inline_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    planning_max_new_tokens: int = Field(default=1536, ge=256)
    response_max_new_tokens: int = Field(default=1024, ge=1)
    trace: bool = False

    @property
    def phases(self) -> dict[str, str]:
        return {
            PLAN_PHASE: self.planner_stage,
            **{
                name: capability.stage for name, capability in self.capabilities.items()
            },
        }

    @model_validator(mode="after")
    def validate_capabilities(self) -> "UMMCompositionSpec":
        if PLAN_PHASE in self.capabilities:
            raise ValueError("The plan phase name is reserved")
        elif any(
            not name.isidentifier() or name.startswith("_")
            for name in self.capabilities
        ):
            raise ValueError("Capability names must be public identifiers")
        else:
            pass
        for example in self.plan_examples:
            self.validate_plan(example.plan, {})
        return self

    def validate_plan(
        self, plan: UMMExecutionPlan, inputs: dict[str, ArtifactKind]
    ) -> None:
        kinds = dict(inputs)
        step_ids = set(inputs) | {INPUT_IMAGE_ID}
        image_count = 0
        if len(plan.steps) > self.max_steps:
            raise ValueError("Composition plan exceeds max_steps")
        elif not any(step.publish for step in plan.steps):
            raise ValueError("Composition plan must publish at least one output")
        else:
            pass
        for step in plan.steps:
            if step.id in step_ids:
                raise ValueError(f"Duplicate or reserved step ID: {step.id}")
            elif step.action not in self.capabilities:
                raise ValueError(f"Unknown capability: {step.action}")
            elif not step.instruction.strip():
                raise ValueError(f"Step {step.id} has an empty instruction")
            elif not set(step.input_refs).issubset(kinds):
                raise ValueError(
                    f"Step {step.id} references unavailable artifacts {sorted(set(step.input_refs) - kinds.keys())}; available IDs: {sorted(kinds)}. Never reference the current step."
                )
            else:
                capability = self.capabilities[step.action]
            if any(
                kinds[reference] not in capability.input_kinds
                for reference in step.input_refs
            ):
                raise ValueError(
                    f"Step {step.id} cannot consume the referenced artifact kind"
                )
            else:
                kinds[step.id] = capability.output_kind
                step_ids.add(step.id)
                image_count += capability.output_kind == "image"
        if image_count > self.max_images:
            raise ValueError("Composition plan exceeds max_images")
        else:
            pass


@dataclass(frozen=True, kw_only=True)
class UMMPlanRepair:
    response: str
    error: str


def build_planning_request(
    composition: UMMCompositionSpec,
    prompt: str,
    images: list[InlineImage],
    max_new_tokens: int,
    repair: UMMPlanRepair | None,
) -> OmniRequest:
    """Describe configured capabilities and optionally request a bounded plan repair."""
    capabilities = [
        {
            "action": name,
            "description": capability.description,
            "input_kinds": capability.input_kinds,
            "output_kind": capability.output_kind,
        }
        for name, capability in composition.capabilities.items()
    ]
    instruction = (
        "Plan the user's task using the available independent model capabilities. "
        "Return only JSON matching the schema, not an answer. Each step selects an "
        "available action and supplies an instruction describing what that model "
        "must do. Choose the actions and their number from the user's request. "
        "Preserve every explicitly requested computation and its dependencies, "
        "including intermediate text that the user does not want displayed. "
        "The planner must not perform these computations itself. If the user "
        "asks a text model to write a description before generating an image, "
        "include a text-producing step and make image generation reference that "
        "step. Do not replace its output with a description invented in the plan. "
        "Each image-producing step generates exactly one image. To create two "
        "images, include two such steps, with a visual prompt for each image. "
        "For generation alone, omit text answers. For a pure text question, use "
        "only a text-producing action. For generation and comparison, generate "
        "the images, then use a text-producing capability that accepts images; "
        "reference every image it needs to inspect. Do not invent scores or "
        "observations before those steps run. input_refs contains only outputs of "
        "strictly earlier steps, never the current step. Use input_refs=[] when "
        "there are no dependencies, including the first generation step. Text outputs "
        "can condition later image prompts. If an input image is attached its ID is "
        "input_image; otherwise never reference it. Use unique step IDs. Set "
        "publish=true for images and final answers requested by the user, false "
        "for intermediate work. Referenced kinds must be accepted by the capability. "
        f"Use at most {composition.max_steps} steps and {composition.max_images} images. "
        f"Capabilities: {json.dumps(capabilities, ensure_ascii=False)}"
    )
    schema = UMMExecutionPlan.model_json_schema()
    schema["properties"]["steps"]["maxItems"] = composition.max_steps
    schema["$defs"]["UMMPlanStep"]["properties"]["action"]["enum"] = list(
        composition.capabilities
    )
    content: list[dict[str, str | dict[str, str]]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image.url}})
    if composition.plan_examples:
        examples = [
            {"task": example.prompt, "plan": example.plan.model_dump(mode="json")}
            for example in composition.plan_examples
        ]
        instruction += (
            "\nExamples of plan syntax and dependency references, not tasks to execute:\n"
            + json.dumps(examples)
            + "\nReturn a new plan only for the actual user request below. Do not repeat example content."
        )
    else:
        pass
    messages: list[dict[str, JsonValue]] = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": content},
    ]
    if repair is not None:
        messages.extend(
            [
                {"role": "assistant", "content": repair.response},
                {
                    "role": "user",
                    "content": f"The plan was rejected before execution: {repair.error}\nReturn a corrected complete plan for the original task. Preserve the user's requested outputs. Do not repeat invalid references.",
                },
            ]
        )
    else:
        pass
    return OmniRequest(
        inputs=messages,
        params={
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "umm_execution_plan",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    )
