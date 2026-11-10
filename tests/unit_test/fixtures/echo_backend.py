# SPDX-License-Identifier: Apache-2.0
"""A CPU backend that worker tests load through BackendSpec.adapter."""

import asyncio
import os
import time

from onemeld.backends.base import BackendSpec, StepResult
from onemeld.composition.types import StepRequest

CRASH_EXIT_CODE = 3


class EchoBackend:
    """Echoes its step; params choose to wait, fail, report or exit instead."""

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        self.cancelled: list[str] = []

    def load(self) -> None:
        while self.spec.args.get("block_load") is True:
            # Stands in for a model load that outlives the orchestrator.
            time.sleep(1)

    async def start(self) -> None:
        pass

    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        action = request.params.get("action")
        if action == "wait":
            await asyncio.Event().wait()
            raise AssertionError("an endless wait returned")
        elif action == "fail":
            raise ValueError("scripted failure")
        elif action == "exit":
            os._exit(  # noqa: leading-underscore  # stdlib name; exits without cleanup
                CRASH_EXIT_CODE
            )
        elif action == "report":
            return {
                "cancelled": self.cancelled,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        else:
            inputs = request.inputs
            return {
                "text": inputs.decode() if isinstance(inputs, bytes) else str(inputs),
                "model": self.spec.model,
            }

    async def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)

    async def close(self) -> None:
        pass


class FailingBackend(EchoBackend):
    async def start(self) -> None:
        raise RuntimeError("scripted startup failure")


class FailingLoadBackend(EchoBackend):
    def load(self) -> None:
        raise RuntimeError("scripted load failure")
