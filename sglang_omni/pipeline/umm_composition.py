# SPDX-License-Identifier: Apache-2.0
"""Planned and iterative execution of independent model compositions on UMM."""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic import JsonValue

from sglang_omni.pipeline.umm import (
    CloseReason,
    UMMContext,
    UMMDispatch,
    UMMOutput,
    UMMPlanningError,
    UMMRequestError,
    UMMStep,
)
from sglang_omni.pipeline.umm_contract import (
    UMMContractCandidate,
    UMMContractReview,
    build_contract_request,
    compile_task_contract,
)
from sglang_omni.pipeline.umm_plan import (
    PLAN_PHASE,
    ArtifactKind,
    UMMArtifact,
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMFinal,
    UMMPlannerTurn,
    UMMStepCall,
    media_part,
    normalize_steps,
    plan_calls,
)
from sglang_omni.pipeline.umm_planner import (
    PlannerBudget,
    PlannerTurnRecord,
    UMMPlanRepair,
    build_planner_request,
)
from sglang_omni.pipeline.umm_protocols import (
    NativeTextResult,
    step_artifact,
    step_request,
)
from sglang_omni.pipeline.umm_task import MAX_TASK_TEXT_BYTES, ClientTask, parse_task

logger = logging.getLogger(__name__)

SEEDED_PROTOCOLS = ("native_image", "native_video", "image_base64")
TraceEvent = Literal[
    "plan",
    "plan_rejected",
    "normalized",
    "dispatch",
    "artifact",
    "observe",
    "input_transcription",
    "contract",
    "contract_review",
]


@dataclass(frozen=True, kw_only=True)
class ScheduledStep:
    step: UMMStepCall
    seed_offset: int


@dataclass(kw_only=True)
class CompositionExecution:
    task: ClientTask
    artifacts: dict[str, UMMArtifact]
    turns: list[PlannerTurnRecord] = field(default_factory=list)
    pending: list[ScheduledStep] = field(default_factory=list)
    # The step each capability stage runs; a stage runs one phase per request.
    in_flight: dict[str, ScheduledStep] = field(default_factory=dict)
    final: UMMFinal | None = None
    published: int = 0
    accepted_steps: int = 0
    seeded_steps: int = 0
    plan_attempts: int = 0
    repair: UMMPlanRepair | None = None
    input_transcription: UMMStepCall | None = None
    candidate: UMMContractCandidate | None = None


def generated_kinds(execution: CompositionExecution) -> Counter[ArtifactKind]:
    return Counter(
        artifact.kind
        for artifact in execution.artifacts.values()
        if artifact.action is not None
    )


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
        event: TraceEvent,
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
            task = parse_task(context.request, self.composition, self.stage_name)
        except ValueError as error:
            raise UMMRequestError(str(error)) from error
        execution = CompositionExecution(task=task, artifacts=dict(task.artifacts))
        self.executions[context.execution_id] = execution
        if task.audio_instruction_ref is None:
            return self.planner_step(context, ())
        else:
            action = self.composition.audio_instruction_capability
            assert action is not None
            capability = self.composition.capabilities[action]
            transcription = UMMStepCall(
                id="user_transcript",
                action=action,
                instruction="Transcribe the user's spoken instruction.",
                input_refs=[task.audio_instruction_ref],
            )
            execution.input_transcription = transcription
            self.emit_trace(
                context,
                "input_transcription",
                {
                    "stage": capability.stages[0],
                    "input_ref": task.audio_instruction_ref,
                },
            )
            return UMMStep(
                phase=capability.stages[0],
                request=step_request(
                    capability,
                    transcription,
                    execution.artifacts,
                    task.options,
                    task.text,
                    0,
                    self.composition.response_max_new_tokens,
                ),
            )

    def budget(self, execution: CompositionExecution) -> PlannerBudget:
        composition = self.composition
        generated = generated_kinds(execution)
        turns = (
            composition.max_planner_turns if execution.task.mode == "iterative" else 1
        )
        return PlannerBudget(
            remaining_steps=composition.max_steps - execution.accepted_steps,
            remaining_turns=turns - len(execution.turns),
            remaining_media={
                kind: limit - generated[kind]
                for kind, limit in composition.media_limits.items()
            },
        )

    def planner_step(
        self, context: UMMContext, outputs: tuple[UMMOutput, ...]
    ) -> UMMStep:
        execution = self.executions[context.execution_id]
        if self.composition.task_operations:
            return UMMStep(
                phase=PLAN_PHASE,
                request=build_contract_request(
                    self.composition,
                    execution.task.text,
                    {
                        name: artifact.kind
                        for name, artifact in execution.artifacts.items()
                    },
                    execution.task.options.planning_max_new_tokens
                    or self.composition.planning_max_new_tokens,
                    execution.repair,
                    execution.candidate,
                    seed=execution.task.options.seed,
                ),
                outputs=outputs,
            )
        else:
            pass
        return UMMStep(
            phase=PLAN_PHASE,
            request=build_planner_request(
                self.composition,
                execution.task.mode,
                execution.task.content,
                execution.turns,
                execution.artifacts,
                self.budget(execution),
                execution.task.options.planning_max_new_tokens
                or self.composition.planning_max_new_tokens,
                execution.repair,
            ),
            outputs=outputs,
        )

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        execution = self.executions[context.execution_id]
        transcription = execution.input_transcription
        if transcription is not None:
            capability = self.composition.capabilities[transcription.action]
            if phase != capability.stages[0]:
                raise ValueError(f"Unexpected input transcription phase {phase!r}")
            else:
                artifact = step_artifact(
                    capability,
                    transcription,
                    result,
                    self.composition.max_inline_media_bytes,
                )
            assert isinstance(artifact.value, str)
            instruction = artifact.value.strip()
            audio_reference = execution.task.audio_instruction_ref
            assert audio_reference is not None
            audio_artifact = execution.artifacts[audio_reference]
            assert not isinstance(audio_artifact.value, str)
            audio_marker = {
                "type": "text",
                "text": f"[user] attached {audio_reference} (audio)",
            }
            content = [
                part
                for part in execution.task.content
                if part not in (audio_marker, media_part(audio_artifact.value))
            ]
            content.append({"type": "text", "text": f"[user] {instruction}"})
            task_text = "\n".join(
                part["text"] for part in content if isinstance(part.get("text"), str)
            )
            if not instruction:
                raise UMMRequestError("The voice instruction has no transcription")
            elif len(task_text.encode("utf-8")) > MAX_TASK_TEXT_BYTES:
                raise UMMRequestError(
                    "The transcribed instruction exceeds the task text limit"
                )
            else:
                execution.artifacts[transcription.id] = UMMArtifact(
                    value=instruction, action=None
                )
                execution.artifacts.pop(audio_reference)
                execution.task = replace(
                    execution.task,
                    text=task_text,
                    content=content,
                    artifacts=dict(execution.artifacts),
                    audio_instruction_ref=None,
                )
                execution.input_transcription = None
                self.emit_trace(
                    context,
                    "artifact",
                    {
                        "step_id": transcription.id,
                        "stage": phase,
                        "kind": "text",
                        "sha256": execution.artifacts[transcription.id].sha256,
                        "text": instruction,
                    },
                )
                return self.planner_step(context, ())
        elif phase == PLAN_PHASE:
            return self.accept_reply(context, result)
        else:
            scheduled = execution.in_flight.pop(phase, None)
        if scheduled is None:
            raise ValueError(f"Unexpected composition phase {phase!r}")
        else:
            step = scheduled.step
        artifact = step_artifact(
            self.composition.capabilities[step.action],
            step,
            result,
            self.composition.max_inline_media_bytes,
        )
        execution.artifacts[step.id] = artifact
        execution.turns[-1].observed.append(step.id)
        self.emit_trace(
            context,
            "artifact",
            {
                "step_id": step.id,
                "stage": phase,
                "kind": artifact.kind,
                "sha256": artifact.sha256,
            },
        )
        return self.schedule(context)

    def accept_reply(self, context: UMMContext, result: object) -> UMMStep:
        """Validate a planner reply, queue its steps and record its final output."""
        if self.composition.task_operations:
            return self.accept_contract_reply(context, result)
        else:
            pass
        execution = self.executions[context.execution_id]
        composition = self.composition
        reply_text = (
            result["text"]
            if isinstance(result, dict) and isinstance(result.get("text"), str)
            else ""
        )
        budget = self.budget(execution)
        execution.plan_attempts += 1
        try:
            response = NativeTextResult.model_validate(result)
            if response.finish_reason != "stop":
                raise ValueError("Planning did not finish; refusing a partial reply")
            else:
                reply = self.normalize_reply(context, response.text, budget)
            composition.validate_steps(
                reply.steps,
                {name: artifact.kind for name, artifact in execution.artifacts.items()},
                generated_kinds(execution),
                budget.remaining_steps,
            )
            if reply.final is not None:
                composition.validate_final(
                    reply.final,
                    {
                        name
                        for name, artifact in execution.artifacts.items()
                        if artifact.action is not None
                    }
                    | {step.id for step in reply.steps},
                )
            elif budget.is_last_turn:
                raise ValueError("The last planner turn must return final")
            else:
                pass
            dispatches = len(reply.steps) + (reply.final is None)
            if dispatches > context.remaining_dispatches:
                raise ValueError("Composition steps exceed the UMM dispatch budget")
            else:
                pass
        except ValueError as error:
            return self.reject_reply(context, reply_text, error)
        return self.enqueue_reply(context, reply)

    def accept_contract_reply(self, context: UMMContext, result: object) -> UMMStep:
        """Compile a task proposal and optionally review it before dispatch."""
        execution = self.executions[context.execution_id]
        candidate = execution.candidate
        reply_text = (
            candidate.response
            if candidate is not None
            else result.get("text", "") if isinstance(result, dict) else ""
        )
        if not isinstance(reply_text, str):
            reply_text = ""
        else:
            pass
        if candidate is None:
            execution.plan_attempts += 1
        else:
            pass
        try:
            response = NativeTextResult.model_validate(result)
            if response.finish_reason != "stop":
                raise ValueError("Planning did not finish; refusing a partial reply")
            elif candidate is None:
                candidate = compile_task_contract(
                    self.composition,
                    response.text,
                    execution.task.text,
                    {
                        name: artifact.kind
                        for name, artifact in execution.artifacts.items()
                    },
                )
                review_dispatches = int(self.composition.review_plan)
                if (
                    len(candidate.reply.steps) + review_dispatches
                    > context.remaining_dispatches
                ):
                    raise ValueError(
                        "Task contract exceeds the remaining dispatch budget"
                    )
                else:
                    execution.candidate = candidate
                self.emit_trace(
                    context,
                    "contract",
                    {
                        "attempt": execution.plan_attempts,
                        "contract": candidate.contract.model_dump(mode="json"),
                    },
                )
                if self.composition.review_plan:
                    return self.planner_step(context, ())
                else:
                    execution.candidate = None
                    return self.enqueue_reply(context, candidate.reply)
            else:
                review = UMMContractReview.model_validate_json(response.text)
                self.emit_trace(
                    context, "contract_review", review.model_dump(mode="json")
                )
                if not review.accepted or review.violations:
                    raise ValueError(
                        "; ".join(review.violations)
                        or "Task contract review rejected the proposal"
                    )
                else:
                    execution.candidate = None
        except ValueError as error:
            return self.reject_reply(context, reply_text, error)
        return self.enqueue_reply(context, candidate.reply)

    def reject_reply(
        self, context: UMMContext, response: str, error: ValueError
    ) -> UMMStep:
        execution = self.executions[context.execution_id]
        execution.candidate = None
        attempt = execution.plan_attempts
        self.emit_trace(
            context,
            "plan_rejected",
            {"response": response, "error": str(error), "attempt": attempt},
        )
        if (
            attempt < self.composition.max_plan_attempts
            and context.remaining_dispatches >= 1
        ):
            execution.repair = UMMPlanRepair(response=response, error=str(error))
            return self.planner_step(context, ())
        else:
            raise UMMPlanningError(
                f"{error} (after {attempt} planner attempts; trace execution_id={context.execution_id})"
            ) from error

    def enqueue_reply(self, context: UMMContext, reply: UMMPlannerTurn) -> UMMStep:
        execution = self.executions[context.execution_id]
        composition = self.composition
        for step in reply.steps:
            seeded = composition.capabilities[step.action].protocol in SEEDED_PROTOCOLS
            execution.pending.append(
                ScheduledStep(step=step, seed_offset=execution.seeded_steps)
            )
            execution.seeded_steps += seeded
        execution.accepted_steps += len(reply.steps)
        # The history shows the reply as it runs, with any renamed IDs.
        execution.turns.append(PlannerTurnRecord(response=reply.model_dump_json()))
        execution.final = reply.final
        execution.plan_attempts = 0
        execution.repair = None
        self.emit_trace(
            context,
            "plan",
            {
                "mode": execution.task.mode,
                "turn": len(execution.turns),
                "reply": reply.model_dump(mode="json"),
            },
        )
        return self.schedule(context)

    def normalize_reply(
        self, context: UMMContext, text: str, budget: PlannerBudget
    ) -> UMMPlannerTurn:
        """Parse a reply and fix what the server can fix without the planner.

        Repeated or input-named step IDs are renamed. In iterative mode a turn
        that runs steps observes them before publishing, so its final output is
        dropped; on the last turn, which must finish, only an answer written
        before the results is dropped.
        """
        execution = self.executions[context.execution_id]
        existing = set(execution.artifacts)
        dropped: list[str] = []
        if execution.task.mode == "plan":
            plan = UMMExecutionPlan.model_validate_json(text)
            normalized = normalize_steps(plan_calls(plan), existing)
            final = UMMFinal(
                publish=[
                    step.id
                    for step, planned in zip(normalized.steps, plan.steps, strict=True)
                    if planned.publish
                ],
                answer="",
            )
        else:
            turn = UMMPlannerTurn.model_validate_json(text)
            normalized = normalize_steps(turn.steps, existing)
            final = (
                None
                if turn.final is None
                else UMMFinal(
                    publish=[
                        normalized.latest.get(name, name) for name in turn.final.publish
                    ],
                    answer=turn.final.answer,
                )
            )
            if final is None or not turn.steps:
                pass
            elif not budget.is_last_turn:
                final = None
                dropped.append("final")
            elif final.answer:
                final = UMMFinal(
                    publish=final.publish or [step.id for step in normalized.steps],
                    answer="",
                )
                dropped.append("answer")
            else:
                pass
        if normalized.renames or dropped:
            self.emit_trace(
                context,
                "normalized",
                {
                    "renamed": [list(rename) for rename in normalized.renames],
                    "dropped": dropped,
                },
            )
        else:
            pass
        return UMMPlannerTurn(steps=normalized.steps, final=final)

    def schedule(self, context: UMMContext) -> UMMStep:
        """Dispatch ready steps to free stages, then publish, wait, finish or replan."""
        execution = self.executions[context.execution_id]
        composition = self.composition
        dispatches: list[UMMDispatch] = []
        waiting: list[ScheduledStep] = []
        for scheduled in execution.pending:
            step = scheduled.step
            capability = composition.capabilities[step.action]
            stage = (
                next(
                    (
                        name
                        for name in capability.stages
                        if name not in execution.in_flight
                    ),
                    None,
                )
                if set(step.input_refs) <= execution.artifacts.keys()
                and (
                    composition.max_concurrent_steps is None
                    or len(execution.in_flight) < composition.max_concurrent_steps
                )
                else None
            )
            if stage is None:
                waiting.append(scheduled)
            else:
                dispatches.append(
                    UMMDispatch(
                        phase=stage,
                        request=step_request(
                            capability,
                            step,
                            execution.artifacts,
                            execution.task.options,
                            execution.task.text,
                            scheduled.seed_offset,
                            execution.task.options.max_new_tokens
                            or composition.response_max_new_tokens,
                        ),
                    )
                )
                execution.in_flight[stage] = scheduled
                self.emit_trace(
                    context,
                    "dispatch",
                    {
                        "step_id": step.id,
                        "action": step.action,
                        "stage": stage,
                        "input_artifacts": [
                            {
                                "id": reference,
                                "kind": execution.artifacts[reference].kind,
                                "sha256": execution.artifacts[reference].sha256,
                            }
                            for reference in step.input_refs
                        ],
                    },
                )
        execution.pending = waiting
        outputs = self.release(execution)
        if dispatches:
            return UMMStep(outputs=outputs, dispatches=tuple(dispatches))
        elif execution.in_flight:
            return UMMStep(outputs=outputs)
        elif execution.final is not None:
            assert not execution.pending, "an idle request has only ready steps"
            assert execution.published == len(execution.final.publish)
            answer = (
                (UMMOutput("text", execution.final.answer),)
                if execution.final.answer
                else ()
            )
            return UMMStep(outputs=outputs + answer)
        else:
            assert execution.task.mode == "iterative" and not execution.pending
            self.emit_trace(
                context,
                "observe",
                {
                    "turn": len(execution.turns),
                    "observed": list(execution.turns[-1].observed),
                },
            )
            return self.planner_step(context, outputs)

    def release(self, execution: CompositionExecution) -> tuple[UMMOutput, ...]:
        """Publish the completed prefix of the final output list, in its order."""
        if execution.final is None:
            return ()
        else:
            publish = execution.final.publish
        outputs: list[UMMOutput] = []
        while (
            execution.published < len(publish)
            and publish[execution.published] in execution.artifacts
        ):
            value = execution.artifacts[publish[execution.published]].value
            outputs.append(
                UMMOutput("text", value)
                if isinstance(value, str)
                else UMMOutput(value.kind, value.model_dump())
            )
            execution.published += 1
        return tuple(outputs)

    def usage(self, context: UMMContext) -> int:
        execution = self.executions[context.execution_id]
        retained = len(execution.task.options.model_dump_json().encode("utf-8")) + len(
            execution.task.text.encode("utf-8")
        )
        retained += sum(len(turn.response.encode("utf-8")) for turn in execution.turns)
        if execution.repair is not None:
            retained += len(execution.repair.response.encode("utf-8")) + len(
                execution.repair.error.encode("utf-8")
            )
        else:
            pass
        for artifact in execution.artifacts.values():
            # A media artifact is held as its data URL, which dominates its size.
            retained += (
                len(artifact.value.encode("utf-8"))
                if isinstance(artifact.value, str)
                else len(artifact.value.url)
            )
        # Room for the largest result each running step, or the planner, can return.
        running = max(1, len(execution.in_flight))
        return retained + 2 * self.composition.max_inline_media_bytes * running

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        self.executions.pop(context.execution_id, None)
