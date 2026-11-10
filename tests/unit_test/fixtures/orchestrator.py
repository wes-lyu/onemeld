# SPDX-License-Identifier: Apache-2.0
"""Scripted backends that let a test reply to each step the orchestrator starts."""

import asyncio
from contextlib import aclosing

from onemeld.backends.base import StepResult
from onemeld.composition.plan import UMMCompositionSpec
from onemeld.composition.types import StepRequest, TaskRequest, UMMLimits
from onemeld.orchestrator import Orchestrator, Segment

# Scheduler turns that let every ready coroutine run to its next real wait.
SETTLE_TURNS = 50


class ScriptedBackend:
    """Holds each step request until the test replies to it."""

    def __init__(self) -> None:
        self.held: dict[str, tuple[StepRequest, asyncio.Future[StepResult]]] = {}
        self.cancelled: list[str] = []
        self.healthy = True

    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        future: asyncio.Future[StepResult] = asyncio.get_running_loop().create_future()
        self.held[request_id] = (request, future)
        try:
            return await future
        except asyncio.CancelledError:
            self.cancelled.append(request_id)
            raise
        finally:
            self.held.pop(request_id, None)


def limits_for(composition: UMMCompositionSpec) -> UMMLimits:
    return UMMLimits(
        max_dispatches=composition.required_dispatches(),
        max_outputs=composition.required_outputs(),
        max_active=1,
        max_retained_bytes=128 * 1024 * 1024,
        timeout_s=60,
    )


class Harness:
    """Runs one request and exposes the steps waiting at each backend."""

    def __init__(
        self, composition: UMMCompositionSpec, limits: UMMLimits | None = None
    ) -> None:
        self.backends = {
            name: ScriptedBackend() for name in set(composition.phases.values())
        }
        self.orchestrator = Orchestrator(
            composition, self.backends, limits or limits_for(composition)
        )
        self.segments: list[Segment] = []
        self.task: asyncio.Task[None] | None = None

    async def consume(self, task: TaskRequest) -> None:
        async with aclosing(self.orchestrator.stream(task)) as stream:
            async for segment in stream:
                self.segments.append(segment)

    async def submit(self, task: TaskRequest) -> None:
        self.task = asyncio.create_task(self.consume(task))
        await settle()

    def held(self) -> set[str]:
        return {name for name, backend in self.backends.items() if backend.held}

    def request(self, name: str) -> StepRequest:
        ((request, _),) = self.backends[name].held.values()
        return request

    async def reply(self, name: str, result: StepResult) -> None:
        ((_, future),) = self.backends[name].held.values()
        future.set_result(result)
        await settle()

    async def finished(self) -> list[Segment]:
        assert self.task is not None and self.task.done(), f"still held: {self.held()}"
        self.task.result()
        return self.segments

    def error(self) -> BaseException:
        assert self.task is not None and self.task.done()
        error = self.task.exception()
        assert error is not None
        return error


async def settle() -> None:
    for _ in range(SETTLE_TURNS):
        await asyncio.sleep(0)


def task_of(text: str, **options: str | int) -> TaskRequest:
    return TaskRequest(messages=[{"role": "user", "content": text}], options=options)
