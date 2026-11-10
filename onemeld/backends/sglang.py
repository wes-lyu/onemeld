# SPDX-License-Identifier: Apache-2.0
"""SGLang backends: chat and transcription on sglang.Engine, images on DiffGenerator."""

import asyncio
import json
import logging
import threading
from collections.abc import Coroutine

from pydantic import JsonValue
from sglang import Engine
from sglang.multimodal_gen import DiffGenerator
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    TranscriptionResponse,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from starlette.responses import Response

from onemeld.backends.base import BackendSpec, StepResult
from onemeld.backends.chat_request import chat_request_fields, chat_result
from onemeld.backends.images import image_result
from onemeld.composition.types import StepRequest

logger = logging.getLogger(__name__)

ServingResponse = (
    ChatCompletionResponse | TranscriptionResponse | ErrorResponse | Response
)
LOOP_START_TIMEOUT_S = 10.0
LOOP_STOP_TIMEOUT_S = 10.0
# Engine arguments the backend spec owns.
SPEC_OWNED_ARGS = frozenset({"model_path", "base_gpu_id", "gpu_id_step"})
WARMUP_PROMPT = "The capital city of France is"
# Image step parameters SGLang's diffusion sampling takes under the same name.
IMAGE_SAMPLING_PARAMS = frozenset(
    {"width", "height", "num_inference_steps", "guidance_scale", "seed"}
)


def engine_kwargs(spec: BackendSpec) -> dict[str, JsonValue]:
    owned = sorted(SPEC_OWNED_ARGS.intersection(spec.args))
    if owned:
        raise ValueError(
            f"Backend {spec.name!r}: set {owned} through the backend's model and gpu"
        )
    else:
        pass
    # The worker sees only its own GPU, as device 0.
    kwargs: dict[str, JsonValue] = {
        **spec.args,
        "model_path": spec.model,
        "base_gpu_id": 0,
    }
    # A worker shares its host with other workers, so the scheduler waits on its
    # request sockets instead of spinning while idle.
    kwargs.setdefault("sleep_on_idle", True)
    return kwargs


def response_body(response: ServingResponse) -> dict[str, JsonValue]:
    if isinstance(response, Response):
        return json.loads(response.body)
    else:
        return response.model_dump()


async def cancel_engine_tasks(engine: Engine) -> None:
    """Abort native requests and end the engine loop's own tasks before it stops."""
    engine.tokenizer_manager.abort_request(abort_all=True)
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


class EngineHost:
    """One sglang.Engine and the thread that runs its event loop."""

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        self.engine: Engine | None = None
        self.thread: threading.Thread | None = None

    def load(self) -> None:
        # note (Wes): Engine installs signal handlers and creates its own loop, so it
        # is built on the main thread before the worker's loop starts, and its loop
        # then runs on a thread of its own, as sglang serve does.
        self.engine = Engine(**engine_kwargs(self.spec))
        running = threading.Event()
        self.thread = threading.Thread(
            target=self.run_loop, args=(running,), name=f"{self.spec.name}-engine-loop"
        )
        self.thread.start()
        if not running.wait(LOOP_START_TIMEOUT_S):
            raise RuntimeError("The SGLang engine event loop did not start")
        else:
            pass

    def run_loop(self, running: threading.Event) -> None:
        assert self.engine is not None
        asyncio.set_event_loop(self.engine.loop)
        self.engine.loop.call_soon(running.set)
        self.engine.loop.run_forever()

    @property
    def served_model_name(self) -> str:
        assert self.engine is not None
        return self.engine.tokenizer_manager.served_model_name

    async def call(
        self, coroutine: Coroutine[None, None, ServingResponse]
    ) -> ServingResponse:
        """Run a serving coroutine on the engine loop; cancelling this cancels it there."""
        assert self.engine is not None
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coroutine, self.engine.loop)
        )

    def abort(self, request_id: str) -> None:
        assert self.engine is not None
        self.engine.loop.call_soon_threadsafe(
            self.engine.tokenizer_manager.abort_request, request_id
        )

    async def close(self) -> None:
        engine = self.engine
        if engine is None:
            return
        else:
            pass
        try:
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    cancel_engine_tasks(engine), engine.loop
                )
            )
            engine.shutdown()
        finally:
            engine.loop.call_soon_threadsafe(engine.loop.stop)
            if self.thread is not None:
                await asyncio.to_thread(self.thread.join, LOOP_STOP_TIMEOUT_S)
            else:
                pass


class SGLangChatBackend:
    """Chat completions through SGLang's own OpenAI chat serving layer."""

    def __init__(self, spec: BackendSpec) -> None:
        self.host = EngineHost(spec)
        self.serving: OpenAIServingChat | None = None
        self.fields = set(ChatCompletionRequest.model_fields)

    def load(self) -> None:
        self.host.load()
        assert self.host.engine is not None
        self.serving = OpenAIServingChat(
            self.host.engine.tokenizer_manager, self.host.engine.template_manager
        )

    async def start(self) -> None:
        """Send the request sglang serve sends before it reports ready."""
        assert self.host.engine is not None
        if self.host.engine.server_args.skip_server_warmup:
            return
        else:
            pass
        result = await self.run(
            "warmup",
            StepRequest(
                inputs=[{"role": "user", "content": WARMUP_PROMPT}],
                params={"max_new_tokens": 8, "temperature": 0.0},
            ),
        )
        logger.info(f"Chat warmup returned {len(result['text'])} characters")

    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        assert self.serving is not None
        fields = chat_request_fields(request, self.host.served_model_name, self.fields)
        chat_request = ChatCompletionRequest(**fields, rid=request_id)
        settled = False
        try:
            response = await self.host.call(
                self.serving.handle_request(chat_request, None)
            )
            settled = True
        finally:
            if not settled:
                self.host.abort(request_id)
            else:
                pass
        return chat_result(response_body(response))

    async def cancel(self, request_id: str) -> None:
        self.host.abort(request_id)

    async def close(self) -> None:
        await self.host.close()


class SGLangTranscriptionBackend:
    """Speech transcription through SGLang's OpenAI transcription serving layer."""

    def __init__(self, spec: BackendSpec) -> None:
        self.host = EngineHost(spec)
        self.serving: OpenAIServingTranscription | None = None

    def load(self) -> None:
        self.host.load()
        assert self.host.engine is not None
        self.serving = OpenAIServingTranscription(self.host.engine.tokenizer_manager)

    async def start(self) -> None:
        pass

    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        assert self.serving is not None
        if not isinstance(request.inputs, bytes):
            raise ValueError("A transcription step needs audio bytes")
        else:
            pass
        language = request.params.get("language")
        temperature = request.params.get("temperature", 0.0)
        response = await self.host.call(
            self.serving.create_transcription(
                audio_data=request.inputs,
                model=self.host.served_model_name,
                language=language if isinstance(language, str) else None,
                response_format="json",
                temperature=(
                    float(temperature) if isinstance(temperature, (int, float)) else 0.0
                ),
                stream=False,
                raw_request=None,
            )
        )
        body = response_body(response)
        text = body.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"Transcription backend error: {body.get('message', body)}"
            )
        else:
            return {"text": text}

    async def cancel(self, request_id: str) -> None:
        # note (Wes): create_transcription assigns its own request id, so a cancelled
        # step stops waiting but the native request runs to completion.
        pass

    async def close(self) -> None:
        await self.host.close()


class SGLangImageBackend:
    """Image generation with SGLang's diffusion runtime.

    Out-of-tree pipelines, such as SenseNova from SGLang-Omni, load when the
    worker's env sets SGLANG_EXTERNAL_MODEL_PACKAGE.
    """

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        self.generator: DiffGenerator | None = None
        # The generator runs one request at a time.
        self.lock = asyncio.Lock()

    def load(self) -> None:
        self.generator = DiffGenerator.from_pretrained(
            model_path=self.spec.model, **self.spec.args
        )

    async def start(self) -> None:
        pass

    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        assert self.generator is not None
        if not isinstance(request.inputs, str):
            raise ValueError("An image step needs a text prompt")
        elif request.params.get("n", 1) != 1:
            raise ValueError("An image step generates exactly one image")
        else:
            pass
        unknown = sorted(set(request.params) - IMAGE_SAMPLING_PARAMS - {"n"})
        if unknown:
            raise ValueError(
                f"The image backend does not accept step parameters {unknown}"
            )
        else:
            pass
        sampling = {
            "prompt": request.inputs,
            **{name: value for name, value in request.params.items() if name != "n"},
            # Keep the image in memory: by default the generator writes it to
            # outputs/ and returns only the file path.
            "save_output": False,
            "return_file_paths_only": False,
        }
        async with self.lock:
            result = await asyncio.to_thread(
                self.generator.generate, sampling_params_kwargs=sampling
            )
        if result is None:
            raise RuntimeError(f"Image generation produced no output for {request_id}")
        else:
            first = result[0] if isinstance(result, list) else result
        return image_result(
            first.samples[0] if isinstance(first.samples, list) else first.samples
        )

    async def cancel(self, request_id: str) -> None:
        # note (Wes): a diffusion run cannot be interrupted between steps here, so a
        # cancelled step finishes in its thread and its image is discarded.
        pass

    async def close(self) -> None:
        if self.generator is not None:
            await asyncio.to_thread(self.generator.shutdown)
        else:
            pass
