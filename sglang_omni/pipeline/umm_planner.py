# SPDX-License-Identifier: Apache-2.0
"""Prompt and JSON schema for an upfront plan."""

import json
from dataclasses import dataclass

from pydantic import JsonValue

from sglang_omni.pipeline.umm_plan import (
    ContentPart,
    UMMArtifact,
    UMMCompositionSpec,
    UMMExecutionPlan,
)
from sglang_omni.proto.request import OmniRequest

REPAIR_TEMPERATURE = 0.7

SHARED_RULES = (
    "Each step selects an available action and gives an instruction describing what that model must do. "
    "Choose the actions and their number from the user's request. Preserve every requested computation and "
    "its dependencies, including intermediate text the user does not want displayed. Never perform a "
    "step's computation yourself or invent its result. Each media-producing step makes exactly one item; "
    "to make two images, plan two steps with a prompt each. A step may reference client inputs, earlier "
    "results and earlier steps of the same reply in input_refs, never itself; use input_refs=[] when it has "
    "no dependencies. Independent steps may run in parallel when the execution limit allows it. Referenced kinds must be "
    "accepted by the capability. Use unique lowercase step IDs that do not start with user_."
)
PLAN_RULES = (
    "Plan the whole task now, before any step runs. Return only JSON matching the schema. "
    "Follow these rules in order:\n"
    "1. Identify every requested operation. If the user asks to first write or describe something and "
    "then generate media, create a text step followed by a media step referencing that text. Do not "
    "write the text yourself inside the media instruction or replace a requested step with a check. "
    "Do not add checks or other operations the user did not request.\n"
    "2. Each instruction must preserve the user's target subject, quantity, attributes and prohibitions "
    "that apply to that step. Media generators receive only their step instruction and referenced "
    "results, not the original request. State the target subject and constraints explicitly in each "
    "media instruction even when it references a description. Never add content the user forbids.\n"
    "3. Reference the actual artifacts needed for each operation. When transferring style, ask the "
    "description step to describe visual style separately from depicted subject matter; instruct the "
    "generation step to apply that style to the user's new subject, not copy the reference subject. "
    "When evaluating outputs, reference the generated media. When choosing from evaluations, reference "
    "the evaluations and preserve the user's criteria and tie-breaking rule.\n"
    "4. Set publish=true for each requested written answer, description, design, evaluation and media "
    "output, including earlier steps in a sequence. A result can be both published and used as input "
    "to another step. Set publish=false only for implementation work the user did not ask to see or "
    "work explicitly requested as private. Follow explicit 'only return' restrictions.\n"
    "5. Before returning JSON, check that every requested operation has a step, every target and "
    "prohibition is preserved, dependencies point to the needed results, and all requested outputs "
    "are published. Examples show syntax; their steps and publication choices are not defaults."
)


@dataclass(frozen=True, kw_only=True)
class UMMPlanRepair:
    response: str
    error: str


def build_planner_request(
    composition: UMMCompositionSpec,
    task: list[ContentPart],
    artifacts: dict[str, UMMArtifact],
    max_new_tokens: int,
    repair: UMMPlanRepair | None,
) -> OmniRequest:
    """Give the planner the task, input references and available capabilities."""
    capabilities = [
        {
            "action": name,
            "description": capability.description,
            "input_kinds": capability.input_kinds,
            "output_kind": capability.output_kind,
            **(
                {"requires": f"exactly one {capability.contract.required_kind} input"}
                if capability.contract.required_kind is not None
                else {}
            ),
        }
        for name, capability in composition.capabilities.items()
    ]
    instruction = (
        f"You plan a task over independent model capabilities. {PLAN_RULES} "
        f"{SHARED_RULES} Use at most {composition.max_steps} steps and at most these new media items: "
        f"{json.dumps(composition.media_limits)}. Capabilities: {json.dumps(capabilities, ensure_ascii=False)}"
        f" Available input artifact IDs and kinds: {json.dumps({name: artifact.kind for name, artifact in artifacts.items()})}. "
        "Copy reference IDs exactly; use [] for no dependencies, never an empty string."
    )
    messages: list[dict[str, JsonValue]] = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": task},
    ]
    if repair is not None:
        messages.extend(
            [
                {"role": "assistant", "content": repair.response},
                {
                    "role": "user",
                    "content": f"The reply was rejected before execution: {repair.error}\nReturn a corrected complete reply for this turn. Preserve the user's requested outputs. Do not repeat invalid references.",
                },
            ]
        )
    else:
        pass
    schema = UMMExecutionPlan.model_json_schema()
    schema["properties"]["steps"]["maxItems"] = composition.max_steps
    schema["$defs"]["UMMPlanStep"]["properties"]["action"]["enum"] = list(
        composition.capabilities
    )
    return OmniRequest(
        inputs=messages,
        params={
            **(
                {"chat_template_kwargs": composition.planning_chat_template_kwargs}
                if composition.planning_chat_template_kwargs
                else {}
            ),
            "max_new_tokens": max_new_tokens,
            # A repair samples, since a greedy repair tends to repeat the rejected reply.
            "temperature": (
                composition.planning_temperature
                if repair is None
                else REPAIR_TEMPERATURE
            ),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "umm_plan_reply",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    )
