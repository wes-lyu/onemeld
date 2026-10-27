# SPDX-License-Identifier: Apache-2.0
"""Shared planning and native stage execution for independent model compositions."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from sglang_omni.client.image_media import InlineImage
from sglang_omni.pipeline.umm import (
    CloseReason,
    UMMContext,
    UMMOutput,
    UMMRequestError,
    UMMStep,
)
from sglang_omni.pipeline.umm_plan import (
    INPUT_IMAGE_ID,
    PLAN_PHASE,
    ArtifactKind,
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMPlanRepair,
    build_planning_request,
)
from sglang_omni.proto.request import OmniRequest

logger = logging.getLogger(__name__)


class CompositionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1, max_length=4096)
    image_base64: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)
    planning_max_new_tokens: int | None = Field(default=None, ge=256)
    width: int | None = Field(default=None, ge=16, multiple_of=16)
    height: int | None = Field(default=None, ge=16, multiple_of=16)
    steps: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


class CompositionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["user"]
    content: str


class CompositionChatInput(BaseModel):
    messages: list[CompositionMessage] = Field(min_length=1, max_length=1)


class NativeTextResult(BaseModel):
    text: str = Field(min_length=1)
    finish_reason: str = "stop"


class NativeImage(InlineImage):
    model_config = ConfigDict(extra="ignore", strict=True)


class NativeImageResult(BaseModel):
    media: list[NativeImage] = Field(min_length=1, max_length=1)


@dataclass(kw_only=True)
class UMMArtifact:
    value: str | InlineImage

    @property
    def kind(self) -> ArtifactKind:
        return "text" if isinstance(self.value, str) else "image"

    @property
    def sha256(self) -> str:
        return (
            hashlib.sha256(self.value.encode("utf-8")).hexdigest()
            if isinstance(self.value, str)
            else self.value.sha256
        )


@dataclass(kw_only=True)
class CompositionExecution:
    request: CompositionRequest
    plan: UMMExecutionPlan | None = None
    step_index: int = 0
    generated_images: int = 0
    plan_attempts: int = 0
    repair: UMMPlanRepair | None = None
    artifacts: dict[str, UMMArtifact] = field(default_factory=dict)


class UMMCompositionAdapter:
    def __init__(
        self, model_path: str, *, composition: UMMCompositionSpec, stage_name: str
    ) -> None:
        """Native stages own model paths; this adapter owns the composition contract."""
        self.composition = composition
        self.stage_name = stage_name
        self.executions: dict[str, CompositionExecution] = {}

    def emit_trace(
        self,
        context: UMMContext,
        event: Literal["plan", "plan_rejected", "dispatch", "artifact"],
        details: dict[str, JsonValue],
    ) -> None:
        if self.composition.trace:
            trace = {"event": event, "execution_id": context.execution_id, **details}
            logger.info(
                f"UMM composition trace {json.dumps(trace, ensure_ascii=False)}"
            )
        else:
            pass

    def start(self, context: UMMContext) -> UMMStep:
        try:
            if isinstance(context.request.inputs, str):
                prompt = context.request.inputs
                options = context.request.metadata.get("composition", {})
            else:
                chat = CompositionChatInput.model_validate(
                    {"messages": context.request.inputs}
                )
                prompt = chat.messages[0].content
                parameters = context.request.params.get("stage_params", {})
                if not isinstance(parameters, dict):
                    raise ValueError("stage_params must be an object")
                else:
                    options = parameters.get(self.stage_name, {})
            if not isinstance(options, dict):
                raise ValueError("Composition options must be an object")
            else:
                options = dict(options)
            for name in ("seed", "max_new_tokens"):
                value = context.request.params.get(name)
                if value is not None:
                    options[name] = value
                else:
                    pass
            request = CompositionRequest.model_validate({**options, "prompt": prompt})
            execution = CompositionExecution(request=request)
            if request.image_base64 is not None:
                if (
                    len(request.image_base64)
                    > 2 * self.composition.max_inline_image_bytes
                ):
                    raise ValueError("Input image exceeds the composition byte limit")
                else:
                    image = InlineImage.from_png(request.image_base64)
                if image.size_bytes > self.composition.max_inline_image_bytes:
                    raise ValueError("Input image exceeds the composition byte limit")
                else:
                    execution.artifacts[INPUT_IMAGE_ID] = UMMArtifact(value=image)
            else:
                pass
        except ValueError as error:
            raise UMMRequestError(str(error)) from error
        self.executions[context.execution_id] = execution
        return self.planning_step(context)

    def planning_step(self, context: UMMContext) -> UMMStep:
        execution = self.executions[context.execution_id]
        images = [
            artifact.value
            for artifact in execution.artifacts.values()
            if isinstance(artifact.value, InlineImage)
        ]
        return UMMStep(
            phase=PLAN_PHASE,
            request=build_planning_request(
                self.composition,
                execution.request.prompt,
                images,
                execution.request.planning_max_new_tokens
                or self.composition.planning_max_new_tokens,
                execution.repair,
            ),
        )

    def next_step(self, context: UMMContext, outputs: tuple[UMMOutput, ...]) -> UMMStep:
        execution = self.executions[context.execution_id]
        assert execution.plan is not None
        if execution.step_index == len(execution.plan.steps):
            return UMMStep(outputs=outputs)
        else:
            step = execution.plan.steps[execution.step_index]
        capability = self.composition.capabilities[step.action]
        request = execution.request
        if capability.protocol == "native_image":
            prompt = step.instruction
            for reference in step.input_refs:
                value = execution.artifacts[reference].value
                assert isinstance(value, str)
                prompt += f"\nContext {reference}: {value}"
            inputs = dict(capability.parameters)
            for name, value in (
                ("width", request.width),
                ("height", request.height),
                ("num_inference_steps", request.steps),
            ):
                if value is not None:
                    inputs[name] = value
                else:
                    pass
            seed = request.seed if request.seed is not None else inputs.get("seed")
            if seed is None:
                pass
            elif type(seed) is not int:
                raise ValueError("Image seed must be an integer")
            else:
                inputs["seed"] = (seed + execution.generated_images) % (2**63)
            inputs.update(prompt=prompt, num_outputs_per_prompt=1)
            native_request = OmniRequest(inputs=inputs)
        else:
            content: list[dict[str, str | dict[str, str]]] = [
                {
                    "type": "text",
                    "text": f"User request: {request.prompt}\nTask for this step: {step.instruction}",
                }
            ]
            for reference in step.input_refs:
                value = execution.artifacts[reference].value
                if isinstance(value, str):
                    content.append(
                        {"type": "text", "text": f"Text {reference}: {value}"}
                    )
                else:
                    content.extend(
                        [
                            {"type": "text", "text": f"Image {reference}:"},
                            {"type": "image_url", "image_url": {"url": value.url}},
                        ]
                    )
            native_request = OmniRequest(
                inputs=[
                    {
                        "role": "system",
                        "content": "Complete the assigned step in the user's language. Inspect supplied images when present and ground comparisons in visible details. Follow the requested answer format.",
                    },
                    {"role": "user", "content": content},
                ],
                params={
                    **capability.parameters,
                    "max_new_tokens": request.max_new_tokens
                    or self.composition.response_max_new_tokens,
                },
            )
        self.emit_trace(
            context,
            "dispatch",
            {
                "step_id": step.id,
                "action": step.action,
                "stage": capability.stage,
                "input_refs": step.input_refs,
                "publish": step.publish,
                "input_artifacts": [
                    {
                        "id": reference,
                        "kind": execution.artifacts[reference].kind,
                        "sha256": execution.artifacts[reference].sha256,
                    }
                    for reference in step.input_refs
                ],
                "image_sha256": [
                    execution.artifacts[reference].value.sha256
                    for reference in step.input_refs
                    if isinstance(execution.artifacts[reference].value, InlineImage)
                ],
            },
        )
        return UMMStep(phase=step.action, request=native_request, outputs=outputs)

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        execution = self.executions[context.execution_id]
        if phase == PLAN_PHASE:
            response = NativeTextResult.model_validate(result)
            execution.plan_attempts += 1
            try:
                if response.finish_reason != "stop":
                    raise ValueError("Planning did not finish; refusing a partial plan")
                else:
                    plan = UMMExecutionPlan.model_validate_json(response.text)
                self.composition.validate_plan(
                    plan,
                    {
                        name: artifact.kind
                        for name, artifact in execution.artifacts.items()
                    },
                )
            except ValueError as error:
                self.emit_trace(
                    context,
                    "plan_rejected",
                    {
                        "response": response.text,
                        "error": str(error),
                        "attempt": execution.plan_attempts,
                    },
                )
                if (
                    execution.plan_attempts < self.composition.max_plan_attempts
                    and context.remaining_dispatches >= 2
                ):
                    execution.repair = UMMPlanRepair(
                        response=response.text, error=str(error)
                    )
                    return self.planning_step(context)
                else:
                    raise
            if len(plan.steps) > context.remaining_dispatches:
                raise ValueError("Composition plan exceeds the UMM dispatch budget")
            else:
                execution.plan = plan
                execution.repair = None
            self.emit_trace(
                context,
                "plan",
                {
                    "task_sha256": hashlib.sha256(
                        execution.request.prompt.encode("utf-8")
                    ).hexdigest(),
                    "plan": plan.model_dump(mode="json"),
                },
            )
            return self.next_step(context, ())
        else:
            assert execution.plan is not None
            step = execution.plan.steps[execution.step_index]
        if phase != step.action:
            raise ValueError(
                f"Unexpected phase {phase!r} for planned action {step.action!r}"
            )
        else:
            capability = self.composition.capabilities[step.action]
        if capability.protocol == "native_image":
            image = NativeImageResult.model_validate(result).media[0]
            if image.size_bytes > self.composition.max_inline_image_bytes:
                raise ValueError("Generated image exceeds the composition byte limit")
            else:
                image.encoded_image()
            artifact = UMMArtifact(value=image)
            execution.generated_images += 1
            output = UMMOutput("image", image.model_dump())
            self.emit_trace(
                context,
                "artifact",
                {"step_id": step.id, "kind": "image", "sha256": image.sha256},
            )
        else:
            response = NativeTextResult.model_validate(result)
            artifact = UMMArtifact(value=response.text)
            output = UMMOutput("text", response.text)
            self.emit_trace(
                context,
                "artifact",
                {"step_id": step.id, "kind": "text", "sha256": artifact.sha256},
            )
        execution.artifacts[step.id] = artifact
        execution.step_index += 1
        return self.next_step(context, (output,) if step.publish else ())

    def usage(self, context: UMMContext) -> int:
        execution = self.executions[context.execution_id]
        retained = len(execution.request.model_dump_json().encode("utf-8"))
        if execution.plan is not None:
            retained += len(execution.plan.model_dump_json().encode("utf-8"))
        else:
            pass
        if execution.repair is not None:
            retained += len(execution.repair.response.encode("utf-8")) + len(
                execution.repair.error.encode("utf-8")
            )
        else:
            pass
        for artifact in execution.artifacts.values():
            retained += (
                len(artifact.value.encode("utf-8"))
                if isinstance(artifact.value, str)
                else len(artifact.value.model_dump_json().encode("utf-8"))
            )
        return retained + 2 * self.composition.max_inline_image_bytes

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        self.executions.pop(context.execution_id, None)
