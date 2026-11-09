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
    plan_calls,
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
ITERATIVE_RULES = (
    "You plan in turns. Return only JSON matching the schema. Each reply either runs steps with final=null, "
    "or finishes with final and no steps. After steps run you see their actual results, including images, "
    "in the next turn. Never publish or describe a result you have not seen. When the observed results "
    "satisfy the request, finish: list the IDs of observed results to show the user in final.publish, in "
    "display order, with an optional short answer. When a result is wrong, run a corrected step instead "
    "and check it again."
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
        f" Available input artifact IDs and kinds: {json.dumps({name: artifact.kind for name, artifact in artifacts.items()})}. "
        "Copy reference IDs exactly; use [] for no dependencies, never an empty string."
    )
    if composition.plan_examples:
        examples = [
            (
                {"task": example.prompt, "reply": example.plan.model_dump(mode="json")}
                if mode == "plan"
                else {
                    "task": example.prompt,
                    "replies": [
                        {
                            "steps": [
                                step.model_dump(mode="json")
                                for step in plan_calls(example.plan)
                            ],
                            "final": None,
                        },
                        "(the results of these steps are observed here)",
                        {
                            "steps": [],
                            "final": {
                                "publish": [
                                    step.id
                                    for step in example.plan.steps
                                    if step.publish
                                ],
                                "answer": "",
                            },
                        },
                    ],
                }
            )
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
                "content": "This is the last turn. Return final. Steps you add still run, and their outputs are published without being observed.",
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
                    "name": f"umm_{mode}_reply",
                    "strict": True,
                    "schema": planner_schema(composition, mode, budget),
                },
            },
        },
    )
