# SPDX-License-Identifier: Apache-2.0
"""OneMeld's Chat Completions endpoint, health and model discovery."""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing, asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from onemeld import __version__
from onemeld.backends.client import WorkerClient
from onemeld.composition.types import TaskRequest
from onemeld.config import ServeConfig
from onemeld.errors import error_status_code
from onemeld.orchestrator import Orchestrator, Segment
from onemeld.protocol import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamResponse,
    ModelCard,
    ModelList,
)

logger = logging.getLogger(__name__)

# The stage_params entry that carries composition options.
COMPOSITION_PARAMS_KEY = "orchestrator"
DISCONNECT_POLL_INTERVAL_S = 0.05
STREAM_DONE = "data: [DONE]\n\n"


def task_request(request: ChatCompletionRequest) -> TaskRequest:
    options = (request.stage_params or {}).get(COMPOSITION_PARAMS_KEY, {})
    return TaskRequest(
        messages=[
            {"role": message.role, "content": message.content}
            for message in request.messages
        ],
        options=dict(options),
        seed=request.seed,
        max_new_tokens=request.effective_max_tokens,
        stream=request.stream,
    )


def text_of(segments: list[Segment]) -> str:
    return "".join(
        segment["data"]
        for segment in segments
        if segment["kind"] == "text" and isinstance(segment["data"], str)
    )


async def start_workers(config: ServeConfig) -> dict[str, WorkerClient]:
    workers = {name: WorkerClient(spec) for name, spec in config.backends.items()}
    started = await asyncio.gather(
        *(worker.start() for worker in workers.values()), return_exceptions=True
    )
    failures = [error for error in started if isinstance(error, BaseException)]
    if failures:
        await asyncio.gather(*(worker.close() for worker in workers.values()))
        raise failures[0]
    else:
        return workers


def create_app(
    config: ServeConfig, orchestrator: Orchestrator | None = None
) -> FastAPI:
    """Serve config; a given orchestrator replaces the backend workers, as in tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if orchestrator is not None:
            app.state.orchestrator = orchestrator
            yield
            return
        else:
            pass
        workers = await start_workers(config)
        app.state.orchestrator = Orchestrator(
            config.composition, workers, config.request_limits
        )
        try:
            yield
        finally:
            await asyncio.gather(*(worker.close() for worker in workers.values()))

    app = FastAPI(title="OneMeld", version=__version__, lifespan=lifespan)
    app.state.model_name = config.model_name

    @app.get("/health")
    async def health() -> JSONResponse:
        backends: Mapping[str, bool] = app.state.orchestrator.health()
        healthy = all(backends.values())
        return JSONResponse(
            content={
                "status": "healthy" if healthy else "unhealthy",
                "backends": backends,
            },
            status_code=200 if healthy else 503,
        )

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        model_name: str = app.state.model_name
        cards = ModelList(data=[ModelCard(id=model_name, root=model_name)])
        return JSONResponse(content=cards.model_dump())

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest, request: Request
    ) -> Response:
        orchestrator: Orchestrator = app.state.orchestrator
        response_id = f"chatcmpl-{body.request_id or uuid.uuid4()}"
        model = body.model or app.state.model_name
        task = task_request(body)
        if body.stream:
            return StreamingResponse(
                chat_stream(orchestrator, task, response_id, model, body),
                media_type="text/event-stream",
            )
        else:
            return await chat_response(orchestrator, task, response_id, model, request)

    return app


async def collect(orchestrator: Orchestrator, task: TaskRequest) -> list[Segment]:
    async with aclosing(orchestrator.stream(task)) as stream:
        return [segment async for segment in stream]


async def wait_for_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(DISCONNECT_POLL_INTERVAL_S)


async def chat_response(
    orchestrator: Orchestrator,
    task: TaskRequest,
    response_id: str,
    model: str,
    request: Request,
) -> JSONResponse:
    """Answer once every segment is ready, or abandon the request if the client leaves."""
    completion = asyncio.create_task(collect(orchestrator, task))
    disconnect = asyncio.create_task(wait_for_disconnect(request))
    try:
        await asyncio.wait(
            {completion, disconnect}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        disconnect.cancel()
        if not completion.done():
            # Closing the request's stream cancels its running steps.
            completion.cancel()
            await asyncio.gather(completion, return_exceptions=True)
        else:
            pass
    if completion.cancelled():
        raise HTTPException(status_code=499, detail="Client disconnected")
    else:
        pass
    error = completion.exception()
    if error is not None:
        status = error_status_code(error)
        if status == 500:
            logger.error(f"Request {response_id} failed", exc_info=error)
        else:
            pass
        raise HTTPException(status_code=status, detail=str(error))
    else:
        segments = completion.result()
    response = ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                message={
                    "role": "assistant",
                    "content": text_of(segments),
                    "segments": segments,
                },
                finish_reason="stop",
            )
        ],
    )
    return JSONResponse(content=response.model_dump())


def stream_event(
    response_id: str,
    created: int,
    model: str,
    delta: ChatCompletionStreamDelta,
    finish_reason: str | None,
    segment: Segment | None = None,
) -> str:
    chunk = ChatCompletionStreamResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[ChatCompletionStreamChoice(delta=delta, finish_reason=finish_reason)],
    ).model_dump(exclude_none=True)
    chunk["choices"][0].setdefault("finish_reason", None)
    if segment is not None:
        chunk["choices"][0]["delta"]["segment"] = segment
    else:
        pass
    return f"data: {json.dumps(chunk)}\n\n"


async def chat_stream(
    orchestrator: Orchestrator,
    task: TaskRequest,
    response_id: str,
    model: str,
    body: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """One event per segment, then the finish event; failures become an error event."""
    created = int(time.time())
    text_requested = "text" in (body.modalities or ["text"])
    first = True
    try:
        async with aclosing(orchestrator.stream(task)) as stream:
            async for segment in stream:
                delta = ChatCompletionStreamDelta(role="assistant" if first else None)
                first = False
                if segment["kind"] == "text" and text_requested:
                    delta.content = segment["data"]
                else:
                    pass
                yield stream_event(response_id, created, model, delta, None, segment)
        yield stream_event(
            response_id, created, model, ChatCompletionStreamDelta(), "stop"
        )
    except Exception as error:
        status = error_status_code(error)
        if status == 500:
            logger.error(f"Stream {response_id} failed", exc_info=error)
        else:
            pass
        kind = "invalid_request_error" if status < 500 else "server_error"
        failure = {"error": {"message": str(error), "type": kind, "code": status}}
        yield f"data: {json.dumps(failure)}\n\n"
    yield STREAM_DONE
