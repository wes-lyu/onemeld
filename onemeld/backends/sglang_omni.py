# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni backends: one SGLang-Omni pipeline per backend, for any task."""

import logging

from pydantic import JsonValue
from sglang_omni.client import Client
from sglang_omni.client.types import CompletionResult, GenerateRequest, Message
from sglang_omni.client.types import SamplingParams as OmniSamplingParams
from sglang_omni.config.manager import ConfigManager
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

from onemeld.backends.base import BackendSpec, StepResult
from onemeld.composition.types import StepRequest

logger = logging.getLogger(__name__)

# Step parameters that map onto SGLang-Omni sampling fields.
SAMPLING_PARAMS = frozenset({"temperature", "top_p", "top_k", "seed", "max_new_tokens"})


def omni_sampling(params: dict[str, JsonValue]) -> OmniSamplingParams:
    return OmniSamplingParams(
        **{name: value for name, value in params.items() if name in SAMPLING_PARAMS}
    )


def extra_params(params: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Step parameters the pipeline reads as request params, such as image size."""
    return {
        name: value for name, value in params.items() if name not in SAMPLING_PARAMS
    }


class SGLangOmniBackend:
    """Runs one SGLang-Omni pipeline and maps steps of the spec's task onto it.

    args.config names a pipeline YAML or JSON file; otherwise the pipeline
    comes from the model path, with args.variant when the model has several.
    """

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        self.runner: MultiProcessPipelineRunner | None = None
        self.client: Client | None = None

    def load(self) -> None:
        config_file = self.spec.args.get("config")
        variant = self.spec.args.get("variant")
        if isinstance(config_file, str):
            manager = ConfigManager.from_file(config_file)
            if manager.config.model_path != self.spec.model:
                raise ValueError(
                    f"Backend {self.spec.name!r}: pipeline file {config_file} loads "
                    f"{manager.config.model_path}, but the backend's model is {self.spec.model}"
                )
            else:
                pass
        elif isinstance(variant, str):
            manager = ConfigManager.from_model_path(self.spec.model, variant=variant)
        else:
            manager = ConfigManager.from_model_path(self.spec.model)
        self.runner = MultiProcessPipelineRunner(manager.config)

    async def start(self) -> None:
        assert self.runner is not None
        await self.runner.start(timeout=self.spec.startup_timeout_s)
        self.client = Client(self.runner.coordinator)

    def generate_request(self, request: StepRequest) -> GenerateRequest:
        params = dict(request.params)
        task = self.spec.task
        if task == "chat" and isinstance(request.inputs, list):
            return GenerateRequest(
                messages=[
                    Message(role=str(message["role"]), content=message["content"])
                    for message in request.inputs
                ],
                sampling=omni_sampling(params),
                extra_params=extra_params(params),
                stream=False,
                output_modalities=["text"],
            )
        elif task == "transcription" and isinstance(request.inputs, bytes):
            return GenerateRequest(
                prompt={"audio_bytes": request.inputs},
                sampling=omni_sampling(params),
                extra_params=extra_params(params),
                stream=False,
                output_modalities=["text"],
            )
        elif task == "image" and isinstance(request.inputs, str):
            if params.pop("n", 1) != 1:
                raise ValueError("An image step generates exactly one image")
            else:
                pass
            return GenerateRequest(
                prompt=request.inputs,
                sampling=omni_sampling(params),
                extra_params=extra_params(params),
                stream=False,
                output_modalities=["image"],
            )
        else:
            raise ValueError(
                f"Backend {self.spec.name!r} cannot run a {task} step with "
                f"{type(request.inputs).__name__} inputs"
            )

    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        assert self.client is not None
        result: CompletionResult = await self.client.completion(
            self.generate_request(request), request_id=request_id
        )
        if self.spec.task == "image":
            if result.image is None:
                raise RuntimeError(f"Backend {self.spec.name!r} returned no image")
            else:
                return {"modality": "image", "image_b64": result.image}
        elif self.spec.task == "transcription":
            return {"text": result.text}
        else:
            return {"text": result.text, "finish_reason": result.finish_reason}

    async def cancel(self, request_id: str) -> None:
        if self.client is not None:
            await self.client.abort(request_id)
        else:
            pass

    async def close(self) -> None:
        if self.runner is not None:
            await self.runner.stop()
        else:
            pass
