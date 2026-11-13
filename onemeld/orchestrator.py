# SPDX-License-Identifier: Apache-2.0
"""Drive each request through the composition adapter and the model backends."""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import Protocol

from pydantic import JsonValue

from onemeld.backends.base import StepResult, StepRunner
from onemeld.composition.adapter import UMMCompositionAdapter
from onemeld.composition.plan import PLAN_PHASE, UMMCompositionSpec
from onemeld.composition.types import (
    CloseReason,
    QueueFullError,
    TaskRequest,
    UMMContext,
    UMMDeadlineError,
    UMMDispatch,
    UMMLimits,
    UMMOutput,
    UMMRequestError,
    UMMStep,
)

logger = logging.getLogger(__name__)

Segment = dict[str, JsonValue]


class MonotonicClock(Protocol):
    def __call__(self) -> float: ...


def encoded_bytes(output: UMMOutput) -> int:
    if isinstance(output.data, str):
        return len(output.data.encode("utf-8"))
    else:
        encoded = json.dumps(output.data, ensure_ascii=False, allow_nan=False)
        return len(encoded.encode("utf-8"))


class Orchestrator:
    """Run each request's planned steps on the backends its phases name.

    Backends keep compute, batching and caches. The orchestrator keeps only
    per-request identity, accepted outputs and bounds.
    """

    def __init__(
        self,
        composition: UMMCompositionSpec,
        backends: Mapping[str, StepRunner],
        limits: UMMLimits,
        clock: MonotonicClock = time.monotonic,
    ) -> None:
        unbound = sorted(set(composition.phases.values()) - backends.keys())
        if unbound:
            raise ValueError(f"Composition names backends with no binding: {unbound}")
        else:
            pass
        self.adapter = UMMCompositionAdapter(composition)
        self.phases = composition.phases
        self.backends = dict(backends)
        self.limits = limits
        self.clock = clock
        self.active: set[str] = set()

    def health(self) -> dict[str, bool]:
        return {name: backend.healthy for name, backend in self.backends.items()}

    def budget_error(self, name: str, limit: int, *, client: bool) -> Exception:
        # Before the first dispatch only the client's input exists, so a bound it
        # crosses is a client error. Later breaches come from model output.
        detail = (
            f"UMM request exceeds {name}={limit}. Raise limits.{name} to allow more"
        )
        return UMMRequestError(detail) if client else RuntimeError(detail)

    def check_deadline(self, context: UMMContext) -> None:
        if self.clock() >= context.deadline:
            raise UMMDeadlineError(
                f"UMM request deadline exceeded after timeout_s={self.limits.timeout_s}. "
                "Raise limits.timeout_s for longer requests"
            )
        else:
            pass

    async def stream(self, task: TaskRequest) -> AsyncIterator[Segment]:
        """Yield each accepted segment of one request, in publication order."""
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
        running: dict[asyncio.Task[StepResult], str] = {}
        reason: CloseReason = "failed"
        try:
            step = self.adapter.start(context)
            client = True
            while True:
                for segment in self.accept(context, step, running, client=client):
                    yield segment
                client = False
                if not running:
                    reason = "completed"
                    return
                else:
                    pass
                done, _ = await asyncio.wait(
                    running,
                    timeout=max(0.0, context.deadline - self.clock()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                self.check_deadline(context)
                # One result advances the request at a time; the rest stay done
                # and return at once from the next wait.
                finished = next(iter(done))
                phase = running.pop(finished)
                step = self.adapter.advance(context, phase, finished.result())
        except UMMDeadlineError:
            reason = "expired"
            raise
        except (asyncio.CancelledError, GeneratorExit):
            reason = "aborted"
            raise
        finally:
            for pending in running:
                pending.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            self.active.discard(context.execution_id)
            try:
                self.adapter.close(context, reason)
            except Exception:
                logger.exception(f"UMM adapter close failed for {context.execution_id}")

    def accept(
        self,
        context: UMMContext,
        step: UMMStep,
        running: dict[asyncio.Task[StepResult], str],
        *,
        client: bool,
    ) -> list[Segment]:
        """Check step against the request bounds, start its phases, return its segments."""
        limits = self.limits
        if len(context.outputs) + len(step.outputs) > limits.max_outputs:
            raise self.budget_error("max_outputs", limits.max_outputs, client=client)
        else:
            pass
        retained = context.output_bytes + sum(
            encoded_bytes(output) for output in step.outputs
        )
        if retained + self.adapter.usage(context) > limits.max_retained_bytes:
            raise self.budget_error(
                "max_retained_bytes", limits.max_retained_bytes, client=client
            )
        elif len(step.dispatches) > context.remaining_dispatches:
            raise self.budget_error(
                "max_dispatches", limits.max_dispatches, client=False
            )
        else:
            pass
        busy = set(running.values())
        for dispatch in step.dispatches:
            if dispatch.phase not in self.phases:
                raise ValueError(f"UMM adapter chose unmapped phase {dispatch.phase!r}")
            elif dispatch.phase in busy:
                raise ValueError(
                    f"UMM adapter dispatched phase {dispatch.phase!r} while it already "
                    "runs a step of this request"
                )
            else:
                busy.add(dispatch.phase)
        segments: list[Segment] = [
            {
                "type": "segment",
                "execution_id": context.execution_id,
                "segment_index": len(context.outputs) + offset,
                "kind": output.kind,
                "data": output.data,
            }
            for offset, output in enumerate(step.outputs)
        ]
        context.outputs.extend(segments)
        context.output_bytes = retained
        for dispatch in step.dispatches:
            index = limits.max_dispatches - context.remaining_dispatches
            context.remaining_dispatches -= 1
            started = asyncio.create_task(self.run_backend(context, dispatch, index))
            running[started] = dispatch.phase
        return segments

    async def run_backend(
        self, context: UMMContext, dispatch: UMMDispatch, dispatch_number: int
    ) -> StepResult:
        backend_name = self.phases[dispatch.phase]
        backend = self.backends[backend_name]
        request_id = f"{context.execution_id}-{dispatch_number}"
        recorder = self.adapter.recorder
        if recorder is None:
            return await backend.run(request_id, dispatch.request)
        else:
            pass
        started_ns = time.monotonic_ns()
        result: StepResult | None = None
        try:
            result = await backend.run(
                request_id, replace(dispatch.request, measure=True)
            )
            return result
        finally:
            recorder.record(
                context.execution_id,
                "model_call",
                {
                    "request_id": request_id,
                    "role": "control" if dispatch.phase == PLAN_PHASE else "tool",
                    "backend": backend_name,
                    "elapsed_ns": time.monotonic_ns() - started_ns,
                    "backend_elapsed_ns": (result or {}).get("backend_elapsed_ns"),
                    "usage": (result or {}).get("usage"),
                    "returned": result is not None,
                },
            )
