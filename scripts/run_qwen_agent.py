# SPDX-License-Identifier: Apache-2.0
"""Serve the paired benchmark with Qwen-Agent and the same OneMeld workers.

Install the SGLang and test extras and qwen-agent==0.0.34. For the measured
profile, set the understanding backend's sampling_backend to pytorch, set
composition.trace to false, and set limits.max_active, limits.max_dispatches
and limits.timeout_s to 1, 21 and 180.0, respectively. The trace directory
still records structured per-request traces. Run from the repository root:

    PYTHONPATH=. QWEN_AGENT_MAX_LLM_CALL_PER_RUN=14 python scripts/run_qwen_agent.py \\
        --config examples/onemeld/profile-sglang-48gb.yaml \\
        --port 8897 --trace-dir benchmark-trace
    python scripts/run_cases.py --trace-dir benchmark-trace --output benchmark-results \\
        --only F1,F2,F3,F4,F6,F7 --repeat 3 --modes plan,agent_rules
    python scripts/summarize_runs.py benchmark-results
"""

import argparse
import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Coroutine, Iterator
from contextlib import aclosing, asynccontextmanager
from threading import Event

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.oai import TextChatAtOAI
from qwen_agent.llm.schema import Message
from qwen_agent.tools import BaseTool

from onemeld.backends.base import StepResult
from onemeld.composition.adapter import ScheduledStep, UMMCompositionAdapter
from onemeld.composition.plan import PLAN_PHASE, UMMArtifact, UMMStepCall
from onemeld.composition.types import (
    QueueFullError,
    StepRequest,
    TaskRequest,
    UMMContext,
    UMMDeadlineError,
    UMMDispatch,
    UMMPlanningError,
    UMMStep,
)
from onemeld.config import load_config
from onemeld.orchestrator import Orchestrator, Segment
from onemeld.server import create_app

SYSTEM_MESSAGE = (
    "Use the available tools to fulfill the user's request. Call only the tools and the "
    "number of times needed. A tool that generates media returns exactly one item. "
    "Run each requested operation, including intermediate work. If the user asks for "
    "text before another operation that uses it, write it with the text tool first. "
    "Then pass the completed text result ID in input_refs to the tool that uses it, "
    "even when its instruction already includes some of the same details. Use only IDs "
    "listed in the input or returned by tools. Set publish to true when the user wants that "
    "result returned. Keep intermediates private when the user only wants later results. "
    "A published result can also be used as a tool input. Use select_results if later "
    "observations change which results should be returned. Preserve the user's subject, "
    "constraints, evaluation criteria and tie-breakers in tool instructions. "
    "You see only text. For any requested work that requires seeing or hearing media, "
    "call a tool with the actual media IDs in input_refs before answering. When asked "
    "to compare or choose among results, finish the comparison and state the choice "
    "and reason. Compare all relevant items together in one tool call when possible, "
    "or combine the individual tool observations before answering. Use the same "
    "user-facing labels throughout tool instructions and returned text, such as Image 1 "
    "and Image 2. When asked to score items, label each item and give each requested "
    "criterion its own numeric score before stating the recommendation. If the user "
    "requires one choice and all criteria and specified tie-breakers still tie, choose "
    "the first item and state that the scores were tied. After the "
    "requested results are ready, write your final "
    "reply. The service returns the selected tool results and then your final reply, "
    "so write any requested answer that no selected result already holds there, "
    "without repeating the selected results."
)
# The numbered rules the OneMeld planner gets, reworded for a tool loop: a plan step
# becomes a tool call and a reference becomes input_refs. Nothing else is added.
PLANNING_RULES = (
    " Follow these rules in order:\n"
    "1. Identify every requested operation. If the user asks to first write or describe something and "
    "then generate media, call a text tool first, then call the media tool with that text result in "
    "input_refs. Do not write the text yourself inside the media instruction or replace a requested "
    "step with a check. Do not add checks or other operations the user did not request.\n"
    "2. Each instruction must preserve the user's target subject, quantity, attributes and prohibitions "
    "that apply to that call. Media generators receive only their instruction and referenced "
    "results, not the original request. State the target subject and constraints explicitly in each "
    "media instruction even when it references a description. Never add content the user forbids.\n"
    "3. Reference the actual artifacts needed for each operation. When transferring style, ask the "
    "description call to describe visual style separately from depicted subject matter; instruct the "
    "generation call to apply that style to the user's new subject, not copy the reference subject. "
    "When evaluating outputs, reference the generated media. When choosing from evaluations, reference "
    "the evaluations and preserve the user's criteria and tie-breaking rule.\n"
    "4. Set publish=true for each requested written answer, description, design, evaluation and media "
    "output, including earlier calls in a sequence. A result can be both published and used as input "
    "to another call. Set publish=false only for implementation work the user did not ask to see or "
    "work explicitly requested as private. Follow explicit 'only return' restrictions.\n"
    "5. Before ending, check that every requested operation has a tool call, every target and "
    "prohibition is preserved, input_refs point to the needed results, and all requested outputs "
    "are published."
)
SELECT_RESULTS = "select_results"
# The agent's last message, returned as text after the selected tool results.
FINAL_REPLY = "final_reply"
AGENT_MODES = ("agent", "agent_constrained", "agent_rules")


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    instruction: str = Field(min_length=1, max_length=2048)
    input_refs: list[str] = Field(
        default_factory=list,
        description="IDs of all inputs and completed results this call uses. Include each dependency even if its details are repeated in instruction. Omit or use [] for independent calls.",
    )
    publish: bool


class ResultSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    include: list[str]
    exclude: list[str]


class QwenWorkerModel(TextChatAtOAI):
    """Use Qwen-Agent's function calling and send model inputs to the shared VLM."""

    def __init__(self, run: "QwenAgentRun", model_name: str) -> None:
        super().__init__(
            {
                "model": model_name,
                "model_type": "oai",
                "generate_cfg": {
                    "parallel_function_calls": True,
                    "max_input_tokens": run.context_length,
                    "use_raw_api": False,
                },
            }
        )
        self.run = run

    def _chat_no_stream(
        self, messages: list[Message], generate_cfg: dict[str, JsonValue]
    ) -> list[Message]:
        request = self.convert_messages_to_dicts(messages)
        result = self.run.submit(self.run.control(request, generate_cfg))
        return [Message(role="assistant", content=result["text"])]

    def _chat_stream(
        self,
        messages: list[Message],
        delta_stream: bool,
        generate_cfg: dict[str, JsonValue],
    ) -> Iterator[list[Message]]:
        yield self._chat_no_stream(messages, generate_cfg)


class CompositionTool(BaseTool):
    def __init__(self, run: "QwenAgentRun", name: str, description: str) -> None:
        self.run = run
        self.name = name
        self.description = description
        schema = (
            ResultSelection.model_json_schema()
            if name == SELECT_RESULTS
            else ToolArguments.model_json_schema()
        )
        self.parameters = {
            key: schema[key] for key in ("type", "properties", "required")
        }
        super().__init__()

    def call(
        self, params: str | dict[str, JsonValue], **kwargs: JsonValue
    ) -> StepResult:
        return self.run.submit(self.run.tool(self.name, params))


class QwenAgentAdapter(UMMCompositionAdapter):
    def planner_step(self, context: UMMContext) -> UMMStep:
        # The official tool loop owns control after the shared input preprocessing.
        return UMMStep()


class QwenAgentRun:
    def __init__(
        self,
        orchestrator: Orchestrator,
        context: UMMContext,
        model_name: str,
        context_length: int,
        constrain_tool_calls: bool,
        planning_rules: bool,
    ) -> None:
        self.orchestrator = orchestrator
        self.context = context
        self.constrain_tool_calls = constrain_tool_calls
        self.planning_rules = planning_rules
        self.model_name = model_name
        self.context_length = context_length
        self.adapter = orchestrator.adapter
        self.composition = self.adapter.composition
        self.loop = asyncio.get_running_loop()
        self.closed = Event()
        self.steps: list[UMMStepCall] = []
        self.selected: list[str] = []
        self.control_calls = 0
        self.pending: asyncio.Task[StepResult] | None = None

    def submit(self, coroutine: Coroutine[None, None, StepResult]) -> StepResult:
        if self.closed.is_set():
            coroutine.close()
            raise UMMDeadlineError("The benchmark request has closed")
        else:
            return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    async def control(
        self, messages: list[dict[str, JsonValue]], generate_cfg: dict[str, JsonValue]
    ) -> StepResult:
        if self.closed.is_set():
            raise UMMDeadlineError("The benchmark request has closed")
        else:
            pass
        max_control_calls = (
            self.composition.max_steps + 1
        ) * self.composition.max_plan_attempts
        if self.control_calls >= max_control_calls:
            raise UMMPlanningError("Qwen-Agent exceeded the control call budget")
        else:
            self.control_calls += 1
        options = self.adapter.executions[self.context.execution_id].task.options
        request = StepRequest(
            inputs=messages,
            params={
                "temperature": self.composition.planning_temperature,
                "max_new_tokens": options.planning_max_new_tokens
                or self.composition.planning_max_new_tokens,
                "chat_template_kwargs": self.composition.planning_chat_template_kwargs,
                **({"stop": generate_cfg["stop"]} if generate_cfg.get("stop") else {}),
                **(
                    {"response_format": self.tool_call_format()}
                    if self.constrain_tool_calls
                    else {}
                ),
            },
        )
        dispatch = UMMDispatch(PLAN_PHASE, request)
        result = await self.dispatch(UMMStep(dispatches=(dispatch,)))
        if self.adapter.recorder is not None:
            self.adapter.recorder.planner_result(self.context.execution_id, result)
        else:
            pass
        if result["finish_reason"] != "stop":
            raise UMMPlanningError("Qwen-Agent controller output did not finish")
        else:
            return result

    def tool_call_format(self) -> dict[str, JsonValue]:
        """Constrain each tool call's arguments to its schema; other text stays free."""
        schemas = {
            name: ToolArguments.model_json_schema()
            for name in self.composition.capabilities
        }
        schemas[SELECT_RESULTS] = ResultSelection.model_json_schema()
        return {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": f'<tool_call>\n{{"name": "{name}", "arguments": ',
                    "schema": schema,
                    "end": "}\n</tool_call>",
                }
                for name, schema in schemas.items()
            ],
            "triggers": ["<tool_call>"],
        }

    async def dispatch(self, step: UMMStep) -> StepResult:
        if self.closed.is_set():
            raise UMMDeadlineError("The benchmark request has closed")
        else:
            pass
        running: dict[asyncio.Task[StepResult], str] = {}
        self.orchestrator.accept(self.context, step, running, client=False)
        (self.pending,) = running
        try:
            return await self.pending
        finally:
            self.pending = None

    async def tool(self, name: str, params: str | dict[str, JsonValue]) -> StepResult:
        execution = self.adapter.executions[self.context.execution_id]
        try:
            decoded = json.loads(params) if isinstance(params, str) else params
            if name == SELECT_RESULTS:
                selection = ResultSelection.model_validate(decoded)
                if not (set(selection.include) | set(selection.exclude)) <= {
                    step.id for step in self.steps
                }:
                    raise ValueError(
                        "Select completed tool results by their returned IDs"
                    )
                else:
                    self.selected = [
                        reference
                        for reference in self.selected
                        if reference not in selection.exclude
                    ]
                    for reference in selection.include:
                        if reference not in self.selected:
                            self.selected.append(reference)
                        else:
                            pass
                return {"selected": self.selected}
            else:
                arguments = ToolArguments.model_validate(decoded)
            call = UMMStepCall(
                id=f"step_{len(self.steps) + 1}",
                action=name,
                instruction=arguments.instruction,
                input_refs=arguments.input_refs,
            )
            self.composition.validate_steps(
                [*self.steps, call],
                {
                    name: artifact.kind
                    for name, artifact in execution.task.artifacts.items()
                },
            )
        except ValueError as error:
            if self.adapter.recorder is not None:
                self.adapter.recorder.record(
                    self.context.execution_id,
                    "plan_rejected",
                    {"action": name, "error": str(error), "response": params},
                )
            else:
                pass
            return {"error": str(error), "available_refs": list(execution.artifacts)}
        seed_offset = execution.seeded_steps
        execution.seeded_steps += (
            self.composition.capabilities[name].protocol == "image_base64"
        )
        execution.pending.append(ScheduledStep(step=call, seed_offset=seed_offset))
        step = self.adapter.schedule(self.context)
        (dispatch,) = step.dispatches
        result = await self.dispatch(step)
        self.adapter.advance(self.context, dispatch.phase, result)
        self.steps.append(call)
        if arguments.publish:
            self.selected.append(call.id)
        else:
            pass
        artifact = execution.artifacts[call.id]
        return {
            "id": call.id,
            "kind": artifact.kind,
            "used_input_refs": call.input_refs,
            **({"text": artifact.value} if isinstance(artifact.value, str) else {}),
            "selected": self.selected,
        }

    def run_controller(self, task_text: str) -> str:
        """Run the tool loop and return the agent's final reply."""
        tools = [
            CompositionTool(
                self,
                name,
                f"{capability.description} Allowed reference kinds: {', '.join(capability.input_kinds)}.",
            )
            for name, capability in self.composition.capabilities.items()
        ]
        tools.append(
            CompositionTool(
                self,
                SELECT_RESULTS,
                "Include or exclude completed tool results by ID. Keep earlier selections unless explicitly changed.",
            )
        )
        agent = FnCallAgent(
            function_list=tools,
            llm=QwenWorkerModel(self, self.model_name),
            system_message=(
                SYSTEM_MESSAGE + PLANNING_RULES
                if self.planning_rules
                else SYSTEM_MESSAGE
            ),
        )
        last = []
        for last in agent.run(messages=[Message(role="user", content=task_text)]):
            if self.closed.is_set():
                raise UMMDeadlineError("The benchmark request has closed")
            else:
                pass
        if not last or last[-1].role != "assistant" or last[-1].function_call:
            raise UMMPlanningError(
                "Qwen-Agent did not finish after the last tool result"
            )
        else:
            pass
        content = last[-1].content
        reply = (
            content
            if isinstance(content, str)
            else "".join(item.text or "" for item in content)
        ).strip()
        if not self.selected and not reply:
            raise UMMPlanningError("Qwen-Agent returned no tool result and no reply")
        else:
            return reply


class QwenAgentOrchestrator(Orchestrator):
    def __init__(
        self, shared: Orchestrator, model_name: str, context_length: int
    ) -> None:
        super().__init__(shared.adapter.composition, shared.backends, shared.limits)
        self.adapter = QwenAgentAdapter(shared.adapter.composition)
        self.active = shared.active
        self.model_name = model_name
        self.context_length = context_length

    async def stream(self, task: TaskRequest) -> AsyncIterator[Segment]:
        if len(self.active) >= self.limits.max_active:
            raise QueueFullError()
        else:
            pass
        context = UMMContext(
            execution_id=uuid.uuid4().hex,
            task=task,
            deadline=self.clock() + self.limits.timeout_s,
            remaining_dispatches=self.limits.max_dispatches,
        )
        self.active.add(context.execution_id)
        run = QwenAgentRun(
            self,
            context,
            self.model_name,
            self.context_length,
            task.options.get("mode") == "agent_constrained",
            task.options.get("mode") == "agent_rules",
        )
        try:
            async with asyncio.timeout(self.limits.timeout_s):
                start = self.adapter.start(context)
                if start.dispatches:
                    (dispatch,) = start.dispatches
                    result = await run.dispatch(start)
                    self.adapter.advance(context, dispatch.phase, result)
                else:
                    pass
                execution = self.adapter.executions[context.execution_id]
                # The shared adapter also serves the older in-tree Agent mode.
                execution.agent_steps = None
                reply = await asyncio.to_thread(run.run_controller, execution.task.text)
                if reply:
                    execution.artifacts[FINAL_REPLY] = UMMArtifact(
                        value=reply, action=None
                    )
                    self.adapter.emit_trace(
                        context,
                        "artifact",
                        {
                            "step_id": FINAL_REPLY,
                            "stage": PLAN_PHASE,
                            "kind": "text",
                            "sha256": execution.artifacts[FINAL_REPLY].sha256,
                        },
                    )
                    execution.publish_ids = [*run.selected, FINAL_REPLY]
                else:
                    execution.publish_ids = run.selected
                self.adapter.emit_trace(
                    context,
                    "plan",
                    {
                        "mode": "qwen-agent",
                        "reply": {
                            "steps": [
                                step.model_dump(mode="json") for step in run.steps
                            ],
                            "final": {
                                "publish": execution.publish_ids,
                                "answer": reply,
                            },
                        },
                    },
                )
                outputs = UMMStep(outputs=self.adapter.release(execution))
                for segment in self.accept(context, outputs, {}, client=False):
                    yield segment
        except TimeoutError as error:
            raise UMMDeadlineError("Qwen-Agent reached the request deadline") from error
        finally:
            run.closed.set()
            if run.pending is not None:
                run.pending.cancel()
                await asyncio.gather(run.pending, return_exceptions=True)
            else:
                pass
            self.active.discard(context.execution_id)
            self.adapter.close(context, "shutdown")


class PairedOrchestrator:
    def __init__(self, plan: Orchestrator, agent: QwenAgentOrchestrator) -> None:
        self.plan = plan
        self.agent = agent

    def health(self) -> dict[str, bool]:
        return self.plan.health()

    async def stream(self, task: TaskRequest) -> AsyncIterator[Segment]:
        controller = (
            self.agent if task.options.get("mode") in AGENT_MODES else self.plan
        )
        async with aclosing(controller.stream(task)) as stream:
            async for segment in stream:
                yield segment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    options = parser.parse_args()
    config = load_config(options.config)
    config.composition.trace_dir = options.trace_dir
    app = create_app(config)
    original_lifespan = app.router.lifespan_context
    planner = config.backends[config.composition.planner_stage]
    context_length = planner.args.get("context_length", 8192)
    if type(context_length) is not int or context_length < 1:
        parser.error("The planner context_length must be a positive integer")
    else:
        pass

    @asynccontextmanager
    async def paired_lifespan(application: FastAPI) -> AsyncIterator[None]:
        async with original_lifespan(application):
            shared = application.state.orchestrator
            application.state.orchestrator = PairedOrchestrator(
                shared, QwenAgentOrchestrator(shared, planner.model, context_length)
            )
            yield

    app.router.lifespan_context = paired_lifespan
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=options.port)


if __name__ == "__main__":
    main()
