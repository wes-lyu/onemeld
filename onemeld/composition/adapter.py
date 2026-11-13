# SPDX-License-Identifier: Apache-2.0
"""Plan a request, then run its validated steps across independent model backends."""

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Literal

from pydantic import JsonValue

from onemeld.composition.plan import (
    PLAN_PHASE,
    UMMAgentTurn,
    UMMArtifact,
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMStepCall,
    media_part,
    normalize_steps,
    plan_calls,
)
from onemeld.composition.planner import UMMPlanRepair, build_planner_request
from onemeld.composition.protocols import NativeTextResult, step_artifact, step_request
from onemeld.composition.task import MAX_TASK_TEXT_BYTES, ClientTask, parse_task
from onemeld.composition.trace import TraceRecorder
from onemeld.composition.types import (
    CloseReason,
    UMMContext,
    UMMDispatch,
    UMMOutput,
    UMMPlanningError,
    UMMRequestError,
    UMMStep,
)

logger = logging.getLogger(__name__)

TraceEvent = Literal[
    "plan",
    "plan_rejected",
    "normalized",
    "dispatch",
    "artifact",
    "agent_turn",
    "input_transcription",
]


@dataclass(frozen=True, kw_only=True)
class ScheduledStep:
    step: UMMStepCall
    seed_offset: int


@dataclass(kw_only=True)
class CompositionExecution:
    task: ClientTask
    artifacts: dict[str, UMMArtifact]
    pending: list[ScheduledStep] = field(default_factory=list)
    # The step each capability backend runs; a backend runs one phase per request.
    in_flight: dict[str, ScheduledStep] = field(default_factory=dict)
    publish_ids: list[str] = field(default_factory=list)
    plan_bytes: int = 0
    published: int = 0
    seeded_steps: int = 0
    plan_attempts: int = 0
    repair: UMMPlanRepair | None = None
    input_transcription: UMMStepCall | None = None
    agent_steps: list[UMMStepCall] | None = None


class UMMCompositionAdapter:
    def __init__(self, composition: UMMCompositionSpec) -> None:
        self.composition = composition
        self.executions: dict[str, CompositionExecution] = {}
        self.recorder = (
            TraceRecorder(composition.trace_dir)
            if composition.trace_dir is not None
            else None
        )

    def emit_trace(
        self,
        context: UMMContext,
        event: TraceEvent,
        details: dict[str, JsonValue],
    ) -> None:
        if self.recorder is not None:
            self.recorder.record(
                context.execution_id,
                event,
                details,
                (
                    self.executions[context.execution_id].artifacts[details["step_id"]]
                    if event == "artifact"
                    else None
                ),
            )
        else:
            pass
        if self.composition.trace:
            trace = {"event": event, "execution_id": context.execution_id, **details}
            logger.info(
                f"UMM composition trace {json.dumps(trace, ensure_ascii=False)}"
            )
        else:
            pass

    def start(self, context: UMMContext) -> UMMStep:
        try:
            task = parse_task(context.task, self.composition)
        except ValueError as error:
            raise UMMRequestError(str(error)) from error
        execution = CompositionExecution(
            task=task,
            artifacts=dict(task.artifacts),
            agent_steps=[] if task.options.mode == "agent" else None,
        )
        self.executions[context.execution_id] = execution
        if task.audio_instruction_ref is None:
            return self.planner_step(context)
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
            request = step_request(
                capability,
                transcription,
                execution.artifacts,
                task.options,
                task.text,
                0,
                self.composition.response_max_new_tokens,
            )
            return UMMStep(dispatches=(UMMDispatch(capability.stages[0], request),))

    def planner_step(self, context: UMMContext) -> UMMStep:
        execution = self.executions[context.execution_id]
        if self.recorder is not None:
            self.recorder.record(context.execution_id, "planner_start", {})
        else:
            pass
        options = execution.task.options
        if (options.mode == "oracle") != (options.oracle_plan is not None):
            raise UMMRequestError("Mode oracle and oracle_plan must be set together")
        elif options.oracle_plan is not None:
            # The plan goes through the same validation as a planner reply, with no model call.
            return self.accept_reply(
                context,
                {"text": json.dumps(options.oracle_plan), "finish_reason": "stop"},
            )
        else:
            pass
        request = build_planner_request(
            self.composition,
            execution.task.content,
            execution.artifacts,
            execution.task.options.planning_max_new_tokens
            or self.composition.planning_max_new_tokens,
            execution.repair,
            execution.agent_steps,
        )
        return UMMStep(dispatches=(UMMDispatch(PLAN_PHASE, request),))

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        execution = self.executions[context.execution_id]
        if self.recorder is not None and phase == PLAN_PHASE:
            self.recorder.planner_result(context.execution_id, result)
        else:
            pass
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
                return self.planner_step(context)
        elif phase == PLAN_PHASE:
            if execution.agent_steps is None:
                return self.accept_reply(context, result)
            else:
                return self.accept_agent_turn(context, result)
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
        """Validate all steps before dispatching any work."""
        execution = self.executions[context.execution_id]
        composition = self.composition
        reply_text = (
            result["text"]
            if isinstance(result, dict) and isinstance(result.get("text"), str)
            else ""
        )
        execution.plan_attempts += 1
        try:
            response = NativeTextResult.model_validate(result)
            if response.finish_reason != "stop":
                raise ValueError("Planning did not finish; refusing a partial reply")
            else:
                # Without constrained decoding a model may wrap the JSON in a code fence.
                reply = (
                    response.text
                    if composition.planning_constrained_decoding
                    else response.text[
                        response.text.find("{") : response.text.rfind("}") + 1
                    ]
                )
                plan = UMMExecutionPlan.model_validate_json(reply)
            normalized = normalize_steps(plan_calls(plan), set(execution.artifacts))
            composition.validate_steps(
                normalized.steps,
                {name: artifact.kind for name, artifact in execution.artifacts.items()},
            )
            publish_ids = [
                step.id
                for step, planned in zip(normalized.steps, plan.steps, strict=True)
                if planned.publish
            ]
            if not publish_ids:
                raise ValueError("Composition must publish at least one output")
            elif len(normalized.steps) > context.remaining_dispatches:
                raise ValueError("Composition steps exceed the UMM dispatch budget")
            else:
                pass
        except ValueError as error:
            return self.reject_reply(context, reply_text, error)
        if normalized.renames:
            self.emit_trace(
                context,
                "normalized",
                {"renamed": [list(rename) for rename in normalized.renames]},
            )
        else:
            pass
        return self.accept_steps(context, normalized.steps, publish_ids, response.text)

    def accept_agent_turn(self, context: UMMContext, result: object) -> UMMStep:
        execution = self.executions[context.execution_id]
        agent_steps = execution.agent_steps
        assert agent_steps is not None
        execution.plan_attempts += 1
        reply_text = ""
        try:
            response = NativeTextResult.model_validate(result)
            reply_text = response.text
            if response.finish_reason != "stop":
                raise ValueError("The agent did not finish its tool decision")
            else:
                turn = UMMAgentTurn.model_validate_json(response.text)
            if not turn.steps and not turn.publish:
                raise ValueError("Choose ready calls or publish at least one result")
            elif any(
                not set(step.input_refs) <= execution.artifacts.keys()
                for step in turn.steps
            ):
                raise ValueError("Agent calls can only read already completed results")
            else:
                all_steps = [*agent_steps, *turn.steps]
            self.composition.validate_steps(
                all_steps,
                {
                    name: artifact.kind
                    for name, artifact in execution.task.artifacts.items()
                },
            )
            if turn.publish and (
                len(set(turn.publish)) != len(turn.publish)
                or not set(turn.publish) <= {step.id for step in all_steps}
            ):
                raise ValueError("Publish distinct tool result IDs")
            elif len(turn.steps) + (not turn.publish) > context.remaining_dispatches:
                raise ValueError("Agent calls exceed the remaining dispatch budget")
            else:
                pass
        except ValueError as error:
            return self.reject_reply(context, reply_text, error)
        execution.agent_steps = all_steps
        self.emit_trace(
            context,
            "agent_turn",
            {"turn": turn.model_dump(mode="json")},
        )
        return self.accept_steps(
            context, turn.steps, turn.publish or None, response.text
        )

    def accept_steps(
        self,
        context: UMMContext,
        steps: list[UMMStepCall],
        publish_ids: list[str] | None,
        reply_text: str,
    ) -> UMMStep:
        execution = self.executions[context.execution_id]
        composition = self.composition
        for step in steps:
            execution.pending.append(
                ScheduledStep(step=step, seed_offset=execution.seeded_steps)
            )
            execution.seeded_steps += (
                composition.capabilities[step.action].protocol == "image_base64"
            )
        execution.plan_bytes += len(reply_text.encode("utf-8"))
        execution.repair = None
        execution.plan_attempts = 0
        if publish_ids is None:
            return self.schedule(context)
        else:
            execution.publish_ids = publish_ids
        all_steps = steps if execution.agent_steps is None else execution.agent_steps
        self.emit_trace(
            context,
            "plan",
            {
                "mode": "plan" if execution.agent_steps is None else "agent",
                "turn": 1,
                "reply": {
                    "steps": [step.model_dump(mode="json") for step in all_steps],
                    "final": {"publish": publish_ids, "answer": ""},
                },
            },
        )
        return self.schedule(context)

    def reject_reply(
        self, context: UMMContext, response: str, error: ValueError
    ) -> UMMStep:
        execution = self.executions[context.execution_id]
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
            return self.planner_step(context)
        else:
            raise UMMPlanningError(
                f"{error} (after {attempt} planner attempts; trace execution_id={context.execution_id})"
            ) from error

    def schedule(self, context: UMMContext) -> UMMStep:
        """Dispatch ready steps, then publish completed results in plan order."""
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
        elif execution.agent_steps is not None and not execution.publish_ids:
            return self.planner_step(context)
        else:
            assert not execution.pending, "an idle request has only ready steps"
            assert execution.published == len(execution.publish_ids)
            return UMMStep(outputs=outputs)

    def release(self, execution: CompositionExecution) -> tuple[UMMOutput, ...]:
        """Publish the completed prefix of the final output list, in its order."""
        publish = execution.publish_ids
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
        retained += execution.plan_bytes
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
