# SPDX-License-Identifier: Apache-2.0
"""The contract between the orchestrator and the framework that runs one model."""

import importlib
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onemeld.composition.plan import CapabilityProtocol
from onemeld.composition.types import StepRequest

BackendTask = Literal["chat", "transcription", "image"]
StepResult = dict[str, JsonValue]

# The backend task each composition protocol needs.
PROTOCOL_TASKS: dict[CapabilityProtocol, BackendTask] = {
    "native_chat": "chat",
    "asr": "transcription",
    "image_base64": "image",
}

# Adapter class of each framework and task, as "module:Class". A spec's adapter
# field names any other class, so a new framework needs no change here.
ADAPTERS: dict[tuple[str, BackendTask], str] = {
    ("sglang", "chat"): "onemeld.backends.sglang:SGLangChatBackend",
    ("sglang", "transcription"): "onemeld.backends.sglang:SGLangTranscriptionBackend",
    ("sglang", "image"): "onemeld.backends.sglang:SGLangImageBackend",
    ("sglang_omni", "chat"): "onemeld.backends.sglang_omni:SGLangOmniBackend",
    ("sglang_omni", "transcription"): "onemeld.backends.sglang_omni:SGLangOmniBackend",
    ("sglang_omni", "image"): "onemeld.backends.sglang_omni:SGLangOmniBackend",
}


class BackendSpec(BaseModel):
    """One model, the framework that hosts it, and where its worker runs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    framework: str = Field(min_length=1)
    task: BackendTask
    model: str = Field(min_length=1)
    adapter: str | None = Field(default=None, pattern=r"^[\w.]+:\w+$")
    # Interpreter of the environment that has this framework installed.
    python: str | None = None
    gpu: int | None = Field(default=None, ge=0)
    env: dict[str, str] = Field(default_factory=dict)
    # Framework arguments only the adapter interprets.
    args: dict[str, JsonValue] = Field(default_factory=dict)
    startup_timeout_s: float = Field(default=1800.0, gt=0)

    @property
    def adapter_path(self) -> str:
        if self.adapter is not None:
            return self.adapter
        elif (self.framework, self.task) in ADAPTERS:
            return ADAPTERS[(self.framework, self.task)]
        else:
            raise ValueError(
                f"Backend {self.name!r}: framework {self.framework!r} has no {self.task} "
                f"adapter; known adapters: {sorted(ADAPTERS)}. Set adapter to a "
                "'module:Class' path to use another one"
            )


class Backend(Protocol):
    """A model hosted in a worker process through one framework's library API.

    The worker calls load once on its main thread before its event loop
    starts, since frameworks such as SGLang install signal handlers and own
    their event loop. It then calls start once inside the loop, run
    concurrently for independent requests, cancel for a request the
    orchestrator gave up on, and close at shutdown. run returns the result
    shape of the backend's task: chat {text, finish_reason}, transcription
    {text}, image {modality, image_b64}.
    """

    def load(self) -> None: ...

    async def start(self) -> None: ...

    async def run(self, request_id: str, request: StepRequest) -> StepResult: ...

    async def cancel(self, request_id: str) -> None: ...

    async def close(self) -> None: ...


class BackendFactory(Protocol):
    def __call__(self, spec: BackendSpec) -> Backend: ...


class StepRunner(Protocol):
    """How the orchestrator reaches a backend: a worker process, or a test fake."""

    @property
    def healthy(self) -> bool: ...

    async def run(self, request_id: str, request: StepRequest) -> StepResult: ...


def load_backend(spec: BackendSpec) -> Backend:
    """Import the adapter class of spec and build its backend."""
    module_name, _, class_name = spec.adapter_path.partition(":")
    factory: BackendFactory = getattr(importlib.import_module(module_name), class_name)
    return factory(spec)
