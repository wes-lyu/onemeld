# SPDX-License-Identifier: Apache-2.0
"""Requests, steps, limits and errors shared by the composer and the orchestrator."""

import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import JsonValue

CloseReason = Literal["completed", "failed", "aborted", "expired", "shutdown"]
ChatMessages = list[dict[str, JsonValue]]
# Chat messages for a chat model, a prompt for an image model, audio for ASR.
StepInputs = ChatMessages | str | bytes


@dataclass(frozen=True, kw_only=True)
class TaskRequest:
    """One client request: its conversation and the composition controls it sets."""

    messages: ChatMessages
    options: dict[str, JsonValue] = field(default_factory=dict)
    seed: int | None = None
    max_new_tokens: int | None = None
    stream: bool = False


@dataclass(frozen=True, kw_only=True)
class StepRequest:
    """The request one model backend runs for one step."""

    inputs: StepInputs
    params: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class UMMOutput:
    kind: str
    data: JsonValue


@dataclass(frozen=True)
class UMMDispatch:
    """One phase a step starts on the backend that phase names."""

    phase: str
    request: StepRequest


@dataclass(frozen=True)
class UMMStep:
    """The adapter's choice after start and after every phase result.

    A step starts every phase in dispatches, at most one per backend of the
    request. A step that starts none publishes its outputs and completes the
    request, or waits while phases of it run.
    """

    outputs: tuple[UMMOutput, ...] = ()
    dispatches: tuple[UMMDispatch, ...] = ()


@dataclass(eq=False, kw_only=True)
class UMMContext:
    """Per-request view passed to every adapter callback.

    Adapters read execution_id, task and remaining_dispatches without changing
    them. The other fields belong to the orchestrator.
    """

    execution_id: str
    task: TaskRequest
    deadline: float
    remaining_dispatches: int
    outputs: list[dict[str, JsonValue]] = field(default_factory=list)
    output_bytes: int = 0


@dataclass(frozen=True, kw_only=True)
class UMMLimits:
    """Bounds on one deployment's requests."""

    max_dispatches: int
    max_outputs: int
    max_active: int
    max_retained_bytes: int
    timeout_s: float

    def __post_init__(self) -> None:
        problems = [
            f"{name} must be a positive integer"
            for name in (
                "max_dispatches",
                "max_outputs",
                "max_active",
                "max_retained_bytes",
            )
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0
        ]
        if type(self.timeout_s) not in (int, float) or not (
            0 < self.timeout_s < math.inf
        ):
            problems.append("timeout_s must be a finite positive number")
        else:
            pass
        if problems:
            raise ValueError(". ".join(problems))
        else:
            pass


class UMMRequestError(ValueError):
    """A client-caused rejection, reported to the client as a bad request."""

    PREFIX = "Invalid UMM request: "

    def __init__(self, detail: str) -> None:
        super().__init__(self.PREFIX + detail)


class UMMPlanningError(RuntimeError):
    """A planner whose replies stayed invalid after every allowed repair (HTTP 422)."""

    PREFIX = "UMM planning failed: "

    def __init__(self, detail: str) -> None:
        super().__init__(self.PREFIX + detail)


class UMMDeadlineError(TimeoutError):
    """A request reached its timeout_s deadline. Its close reason is expired."""


class QueueFullError(RuntimeError):
    """The deployment is at max_active requests (HTTP 503)."""

    MESSAGE = "The request queue is full."

    def __init__(self, message: str = MESSAGE) -> None:
        super().__init__(message)
