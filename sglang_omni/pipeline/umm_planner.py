# SPDX-License-Identifier: Apache-2.0
"""Planner prompts, schemas and observations for both composition modes."""

import json
from dataclasses import dataclass, field

from pydantic import JsonValue

from sglang_omni.pipeline.umm_plan import (
    ContentPart,
    ExecutionMode,
    UMMArtifact,
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMPlannerTurn,
    media_part,
    turn_from_plan,
)
from sglang_omni.proto.request import OmniRequest

SHARED_RULES = (
    "Each step selects an available action and gives an instruction describing what that model must do. "
    "Choose the actions and their number from the user's request. Preserve every requested computation and "
    "its dependencies, including intermediate text the user does not want displayed. Never perform a "
    "step's computation yourself or invent its result. Each media-producing step makes exactly one item; "
    "to make two images, plan two steps with a prompt each. A step may reference client inputs, earlier "
    "results and earlier steps of the same reply in input_refs, never itself; use input_refs=[] when it has "
    "no dependencies. Steps that do not reference each other run in parallel. Referenced kinds must be "
    "accepted by the capability. Use unique lowercase step IDs that do not start with user_."
)
PLAN_RULES = (
    "Plan the whole task now, before any step runs. Return only JSON matching the schema. Set publish=true "
    "for the outputs the user asked for and false for intermediate work."
)
ITERATIVE_RULES = (
    "You plan in turns and see each turn's results before the next. Return only JSON matching the schema. "
    "Set final=null to run steps and observe their results. When the task is done, return final with the "
    "step outputs to show the user in publish, in display order, and an optional short answer. Inspect "
    "results before you rely on them, retry a step with a better instruction when its result is wrong, and "
    "publish only the results that satisfy the request. A turn that runs steps cannot give a final answer, "
    "but it can return final with the steps whose outputs to publish."
)


@dataclass(kw_only=True)
class PlannerTurnRecord:
    """An accepted planner reply and the artifact IDs its steps produced."""

    response: str
    observed: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class PlannerBudget:
    remaining_steps: int
    remaining_turns: int
    remaining_media: dict[str, int]

    @property
    def is_last_turn(self) -> bool:
        return self.remaining_turns <= 1 or self.remaining_steps == 0


@dataclass(frozen=True, kw_only=True)
class UMMPlanRepair:
    response: str
    error: str


def observation_content(
    composition: UMMCompositionSpec,
    turn_number: int,
    observed: list[tuple[str, UMMArtifact]],
    budget: PlannerBudget | None,
) -> list[ContentPart]:
    """Results of one turn: summaries, then each viewable media item once."""
    summaries = [
        artifact.summary(artifact_id, composition.observation_text_bytes)
        for artifact_id, artifact in observed
    ]
    text = (
        f"Results of turn {turn_number}:\n{json.dumps(summaries, ensure_ascii=False)}"
    )
    if budget is not None:
        text += (
            f"\nRemaining: {budget.remaining_steps} steps, {budget.remaining_turns} turns, media "
            f"{json.dumps(budget.remaining_media)}. Reply with more steps, or final."
        )
    else:
        pass
    content: list[ContentPart] = [{"type": "text", "text": text}]
    for artifact_id, artifact in observed:
        if (
            not isinstance(artifact.value, str)
            and artifact.value.kind in composition.planner_media_kinds
        ):
            content.extend(
                [
                    {"type": "text", "text": f"Artifact {artifact_id}:"},
                    media_part(artifact.value),
                ]
            )
        else:
            pass
    return content


def planner_schema(
    composition: UMMCompositionSpec, mode: ExecutionMode, budget: PlannerBudget
) -> dict[str, JsonValue]:
    if mode == "plan":
        schema = UMMExecutionPlan.model_json_schema()
        step_definition = "UMMPlanStep"
    else:
        schema = UMMPlannerTurn.model_json_schema()
        step_definition = "UMMStepCall"
    schema["properties"]["steps"]["maxItems"] = budget.remaining_steps
    schema["$defs"][step_definition]["properties"]["action"]["enum"] = list(
        composition.capabilities
    )
    if mode == "iterative" and budget.is_last_turn:
        # The grammar, not only validation, ends the request on its last turn.
        schema["properties"]["final"] = {"$ref": "#/$defs/UMMFinal"}
    else:
        pass
    return schema


def build_planner_request(
    composition: UMMCompositionSpec,
    mode: ExecutionMode,
    task: list[ContentPart],
    turns: list[PlannerTurnRecord],
    artifacts: dict[str, UMMArtifact],
    budget: PlannerBudget,
    max_new_tokens: int,
    repair: UMMPlanRepair | None,
) -> OmniRequest:
    """Describe the capabilities, the task and every observed turn to the planner."""
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
        f"You plan a task over independent model capabilities. {PLAN_RULES if mode == 'plan' else ITERATIVE_RULES} "
        f"{SHARED_RULES} Use at most {budget.remaining_steps} steps and at most these new media items: "
        f"{json.dumps(budget.remaining_media)}. Capabilities: {json.dumps(capabilities, ensure_ascii=False)}"
    )
    if composition.plan_examples:
        examples = [
            {
                "task": example.prompt,
                "reply": (
                    example.plan.model_dump(mode="json")
                    if mode == "plan"
                    else turn_from_plan(example.plan).model_dump(mode="json")
                ),
            }
            for example in composition.plan_examples
        ]
        instruction += (
            "\nExamples of reply syntax and dependency references, not tasks to execute:\n"
            + json.dumps(examples)
            + "\nReply only for the actual user request below. Do not repeat example content."
        )
    else:
        pass
    messages: list[dict[str, JsonValue]] = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": task},
    ]
    for index, turn in enumerate(turns):
        messages.append({"role": "assistant", "content": turn.response})
        is_latest = index == len(turns) - 1
        messages.append(
            {
                "role": "user",
                "content": observation_content(
                    composition,
                    index + 1,
                    [
                        (artifact_id, artifacts[artifact_id])
                        for artifact_id in turn.observed
                    ],
                    budget if is_latest else None,
                ),
            }
        )
    if budget.is_last_turn and mode == "iterative":
        messages.append(
            {
                "role": "user",
                "content": "This is the last turn. Return final; steps you add still run before it is published.",
            }
        )
    else:
        pass
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
    return OmniRequest(
        inputs=messages,
        params={
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"umm_{mode}_reply",
                    "strict": True,
                    "schema": planner_schema(composition, mode, budget),
                },
            },
        },
    )
