# SPDX-License-Identifier: Apache-2.0
"""Compile source-grounded task operations into existing UMM capability calls."""

import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from sglang_omni.pipeline.umm_plan import (
    ArtifactKind,
    UMMCompositionSpec,
    UMMFinal,
    UMMPlannerTurn,
    UMMStepCall,
    UMMTaskBinding,
)
from sglang_omni.pipeline.umm_planner import UMMPlanRepair
from sglang_omni.proto.request import OmniRequest

CONTRACT_PROMPT = (
    "将用户请求解析成任务契约，只规划，不执行任务、不构思具体画面、不添加用户没有要求的条件。"
    "操作从提供的operation定义中选择，具体模型和stage由框架绑定。"
    "interpretation先说明请求的操作、数量、依赖、限制和要返回的内容。"
    "source_text从schema列出的原文片段中选择对应操作。每个操作的constraints只选择适用于该操作的原文限制，"
    "不要把后续操作的要求传给前面的操作；比较规则属于比较操作，不属于媒体生成。不要把限制条件当成额外操作。"
    "depends_on引用前面操作的序号，从0开始，不能引用自己。input_refs只引用用户已有的输入ID；"
    "未来生成结果通过depends_on引用。count是该操作产生的新媒体数量；所有文字操作都填1，"
    "一次文字操作可以描述或评价多张图片。返回用户要求的文字和媒体；明确私下处理或仅返回其他结果时才隐藏。"
    "所有步骤一次规划完，输出符合schema的JSON。"
)
AUDIT_PROMPT = (
    "对照原始请求审核任务契约。检查是否遗漏或反转要求、添加无依据的条件、目标或数量错误、"
    "缺少必要依赖、漏掉要求返回的结果。用户输入列表不是可用能力列表；生成操作可以不需要上传图片，"
    "其结果由后续操作通过depends_on读取。能力以提供的operations定义为准。"
    "不要执行任务，不要评价尚未生成的内容质量。只有请求与契约相符才accepted=true，"
    "否则列出具体violations。"
)


class UMMTaskOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operation: str
    source_text: str = Field(min_length=1)
    constraints: list[str]
    count: int = Field(ge=1, le=16)
    depends_on: list[int]
    input_refs: list[str]


class UMMTaskContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    interpretation: str = Field(min_length=1)
    operations: list[UMMTaskOperation] = Field(min_length=1)
    publish: list[int] = Field(min_length=1)


class UMMContractReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    accepted: bool
    violations: list[str]


@dataclass(frozen=True, kw_only=True)
class UMMContractCandidate:
    response: str
    contract: UMMTaskContract
    reply: UMMPlannerTurn


def text_image_task_bindings(
    text_capability: str, image_capability: str
) -> dict[str, UMMTaskBinding]:
    """Reusable task vocabulary for text, image understanding and generation."""
    return {
        "write_text": UMMTaskBinding(
            capability=text_capability,
            description="Write requested text or a visual design, without generating media.",
            instruction="Complete only the requested writing task. Preserve all user constraints in the written result.",
            input_types=["text", "visual_description"],
            output_type="text",
        ),
        "describe_image": UMMTaskBinding(
            capability=text_capability,
            description="Read supplied images and describe the requested visual aspects.",
            instruction="Describe the requested visual aspects of the supplied images. When describing style, separate style from the depicted subject.",
            input_types=["image"],
            output_type="visual_description",
            requires_input=True,
        ),
        "generate_image": UMMTaskBinding(
            capability=image_capability,
            description="Generate new images from text. No uploaded image is needed; referenced descriptions can supply style or design.",
            instruction="Generate the requested subject, using referenced text for the requested style or design. Preserve the user's constraints.",
            input_types=["text", "visual_description"],
            output_type="image",
        ),
        "inspect_images": UMMTaskBinding(
            capability=text_capability,
            description="Inspect actual images for the attributes or conditions requested by the user.",
            instruction="Inspect the supplied actual images for every requested condition and report the observations.",
            input_types=["image"],
            output_type="text",
            requires_input=True,
        ),
        "evaluate_images": UMMTaskBinding(
            capability=text_capability,
            description="Score supplied images using the user criteria. One text result can score multiple images; count is 1.",
            instruction="Score each supplied image artifact once on each requested criterion using the same explicit numeric scale. Identify results by their input artifact IDs, even when an image contains several panels or subjects. Explain the scores. Do not recommend unless this operation requests it.",
            input_types=["image"],
            output_type="evaluation",
            requires_input=True,
        ),
        "recommend": UMMTaskBinding(
            capability=text_capability,
            description="Recommend a result from preceding evaluations, following the user criteria and tie-breaking rule.",
            instruction="Recommend a result using the supplied evaluations and the user's comparison rules. Give the reason without changing or contradicting the scores.",
            input_types=["evaluation"],
            output_type="text",
            requires_input=True,
        ),
    }


def compile_task_contract(
    composition: UMMCompositionSpec,
    response: str,
    task_text: str,
    artifacts: dict[str, ArtifactKind],
) -> UMMContractCandidate:
    """Reject unsupported operations and invented quotes before building calls."""
    contract = UMMTaskContract.model_validate_json(response)
    if contract.publish != sorted(set(contract.publish)) or any(
        index < 0 or index >= len(contract.operations) for index in contract.publish
    ):
        raise ValueError(
            "publish must select existing operations once in operation order"
        )
    else:
        pass
    semantic_types: dict[str, str] = dict(artifacts)
    produced: dict[int, list[str]] = {}
    steps: list[UMMStepCall] = []
    publish: list[str] = []
    for index, operation in enumerate(contract.operations):
        if operation.operation not in composition.task_operations:
            raise ValueError(f"Unknown task operation {operation.operation}")
        elif (
            not operation.source_text.strip() or operation.source_text not in task_text
        ):
            raise ValueError(f"Operation {index} source_text is not an exact quote")
        elif any(
            not text.strip() or text not in task_text for text in operation.constraints
        ):
            raise ValueError(
                f"Operation {index} constraints must be exact request quotes"
            )
        elif any(
            previous < 0 or previous >= index for previous in operation.depends_on
        ):
            raise ValueError(f"Operation {index} must reference earlier operations")
        elif not set(operation.input_refs) <= artifacts.keys():
            raise ValueError(f"Operation {index} references unknown user inputs")
        else:
            binding = composition.task_operations[operation.operation]
        capability = composition.capabilities[binding.capability]
        references = list(dict.fromkeys(operation.input_refs))
        for previous in dict.fromkeys(operation.depends_on):
            references.extend(produced[previous])
        if binding.requires_input and not references:
            raise ValueError(f"Operation {index} requires referenced inputs")
        elif any(
            semantic_types[name] not in binding.input_types for name in references
        ):
            raise ValueError(
                f"Operation {index} accepts semantic inputs {binding.input_types}"
            )
        elif capability.output_kind == "text" and operation.count != 1:
            raise ValueError(f"Operation {index} is a text operation; count must be 1")
        else:
            pass
        instruction = (
            f"{binding.instruction}\nRequested operation: {operation.source_text}"
        )
        if operation.constraints:
            instruction += "\nUser constraints for this operation:\n" + "\n".join(
                operation.constraints
            )
        else:
            pass
        produced[index] = []
        for number in range(operation.count):
            step_id = f"operation_{index}_{number}"
            if capability.output_kind != "text":
                step_instruction = (
                    f"{instruction}\nThis call produces output {number + 1} of {operation.count}. "
                    f"Produce exactly one {capability.output_kind}. The runtime makes separate calls for multiple requested outputs; "
                    "do not combine those outputs into a grid or comparison sheet. Do not perform later evaluation or recommendation tasks in the generated content."
                )
            else:
                step_instruction = instruction
            steps.append(
                UMMStepCall(
                    id=step_id,
                    action=binding.capability,
                    instruction=step_instruction,
                    input_refs=references,
                )
            )
            semantic_types[step_id] = binding.output_type
            produced[index].append(step_id)
            if index in contract.publish:
                publish.append(step_id)
            else:
                pass
    final = UMMFinal(publish=publish, answer="")
    composition.validate_steps(steps, artifacts, Counter(), composition.max_steps)
    composition.validate_final(final, {step.id for step in steps})
    return UMMContractCandidate(
        response=response,
        contract=contract,
        reply=UMMPlannerTurn(steps=steps, final=final),
    )


def build_contract_request(
    composition: UMMCompositionSpec,
    task_text: str,
    artifacts: dict[str, ArtifactKind],
    max_new_tokens: int,
    repair: UMMPlanRepair | None,
    candidate: UMMContractCandidate | None,
    seed: int | None = None,
) -> OmniRequest:
    """Submit a task proposal or its optional pre-execution review."""
    operations = {
        name: {
            "description": binding.description,
            "input_types": binding.input_types,
            "output_type": binding.output_type,
            "requires_input": binding.requires_input,
        }
        for name, binding in composition.task_operations.items()
    }
    content: dict[str, JsonValue] = {
        "original_request": task_text,
        "user_inputs": artifacts,
        "available_operations": operations,
    }
    if candidate is None:
        schema = UMMTaskContract.model_json_schema()
        schema["properties"]["operations"]["maxItems"] = composition.max_steps
        operation_schema = schema["$defs"]["UMMTaskOperation"]["properties"]
        operation_schema["operation"]["enum"] = list(operations)
        quotes = list(
            dict.fromkeys(
                [
                    part.strip()
                    for part in re.split(r"[，。；！？,;!?\n]|\.(?:\s+|$)", task_text)
                    if part.strip()
                ]
            )
        )
        operation_schema["source_text"]["enum"] = quotes
        operation_schema["constraints"]["items"]["enum"] = quotes
        if artifacts:
            operation_schema["input_refs"]["items"]["enum"] = list(artifacts)
        else:
            operation_schema["input_refs"]["maxItems"] = 0
        schema["properties"]["publish"].update(
            items={
                "type": "integer",
                "minimum": 0,
                "maximum": composition.max_steps - 1,
            },
            maxItems=composition.max_steps,
        )
        prompt = (
            CONTRACT_PROMPT
            + "计划末尾的publish列出要返回给用户的操作序号（从0开始），按操作顺序升序排列且不重复。只供内部使用的操作不列入publish，但仍正常执行并供后续步骤引用。明确不要执行的操作不能加入operations。"
        )
    else:
        schema = UMMContractReview.model_json_schema()
        prompt = AUDIT_PROMPT
        content["contract"] = candidate.contract.model_dump(mode="json")
        content["compiled_execution"] = candidate.reply.model_dump(mode="json")
        content["execution_semantics"] = (
            "Each operation runs its bound model to produce a new output. source_text is an instruction quote, not the output itself. "
            "publish only controls visibility to the user: hidden outputs are still available to dependencies. "
            "A text operation with count=1 can assess many images in one response."
        )
    messages: list[dict[str, JsonValue]] = [
        {"role": "system", "content": prompt + "\nSchema:\n" + json.dumps(schema)},
        {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
    ]
    if candidate is None:
        slots = []
        for index in range(composition.max_steps):
            slot = deepcopy(schema["$defs"]["UMMTaskOperation"])
            dependencies = slot["properties"]["depends_on"]
            if index == 0:
                dependencies["maxItems"] = 0
            else:
                dependencies["items"]["enum"] = list(range(index))
            slots.append(slot)
        schema["properties"]["operations"]["prefixItems"] = slots
        schema["properties"]["operations"]["items"] = False
    else:
        pass
    if repair is not None and candidate is None:
        messages.extend(
            [
                {"role": "assistant", "content": repair.response},
                {
                    "role": "user",
                    "content": f"Correct the contract against the unchanged request. Errors: {repair.error}",
                },
            ]
        )
    else:
        pass
    return OmniRequest(
        inputs=messages,
        params={
            **(
                {"chat_template_kwargs": composition.planning_chat_template_kwargs}
                if composition.planning_chat_template_kwargs
                else {}
            ),
            **({"seed": seed} if seed is not None else {}),
            "temperature": composition.planning_temperature,
            "max_new_tokens": max_new_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "umm_task_contract",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    )
