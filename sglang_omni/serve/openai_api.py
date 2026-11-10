# SPDX-License-Identifier: Apache-2.0
"""OneMeld's Chat Completions endpoint and service discovery."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Coroutine
from contextlib import aclosing

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from sglang_omni import __version__
from sglang_omni.client.client import Client
from sglang_omni.client.types import (
    ClientError,
    CompletionResult,
    GenerateRequest,
    Message,
    SamplingParams,
)
from sglang_omni.serve.openai_errors import generation_error_status_code
from sglang_omni.serve.protocol import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamResponse,
    ModelCard,
    ModelList,
)
from sglang_omni.serve.streaming import STREAM_DONE_SENTINEL, ClosableStreamingResponse

logger = logging.getLogger(__name__)
HTTP_DISCONNECT_POLL_INTERVAL_S = 0.05
HTTP_DISCONNECT_CANCEL_TIMEOUT_S = 0.1


def create_app(client: Client, *, model_name: str = "onemeld") -> FastAPI:
    app = FastAPI(title="OneMeld", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.client = client
    app.state.model_name = model_name
    register_health(app)
    register_models(app)
    register_chat_completions(app)
    return app


def register_health(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> JSONResponse:
        """Report whether the native worker group is running."""
        client: Client = app.state.client
        info = client.health()
        if info.get("running", False):
            status = "healthy"
        else:
            status = "unhealthy"
        return JSONResponse(
            content={"status": status, **info},
            status_code=200 if status == "healthy" else 503,
        )


def register_models(app: FastAPI) -> None:
    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """List available models."""
        model_name: str = app.state.model_name
        model_list = ModelList(
            data=[
                ModelCard(
                    id=model_name,
                    root=model_name,
                    created=0,
                )
            ]
        )
        return JSONResponse(content=model_list.model_dump())


def register_chat_completions(app: FastAPI) -> None:
    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest, request: Request
    ) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name

        request_id = req.request_id or str(uuid.uuid4())
        response_id = f"chatcmpl-{request_id}"
        created = int(time.time())
        model = req.model or default_model

        gen_req = build_chat_generate_request(req)

        if req.stream:
            return ClosableStreamingResponse(
                chat_stream(
                    client,
                    gen_req,
                    request_id,
                    response_id,
                    created,
                    model,
                    req,
                ),
                media_type="text/event-stream",
            )
        else:
            pass

        return await chat_non_stream(
            request,
            client,
            gen_req,
            request_id,
            response_id,
            created,
            model,
        )


async def chat_non_stream(
    request: Request,
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
) -> JSONResponse:
    """Handle non-streaming chat completions."""
    try:
        result = await complete_unless_disconnected(
            request,
            client,
            client.completion(gen_req, request_id=request_id),
            request_id=request_id,
        )
    except ClientError as exc:
        raise HTTPException(
            status_code=generation_error_status_code(exc), detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception(f"Error generating response for request {request_id}")
        raise HTTPException(
            status_code=generation_error_status_code(exc), detail=str(exc)
        ) from exc

    message = {
        "role": "assistant",
        "content": result.text,
        "segments": [segment.to_dict() for segment in result.segments],
    }
    response = ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=result.finish_reason,
            )
        ],
    )

    return JSONResponse(content=response.model_dump())


async def chat_stream(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """Report failures inside an already-started event stream."""
    try:
        async with aclosing(
            chat_stream_events(
                client=client,
                gen_req=gen_req,
                request_id=request_id,
                response_id=response_id,
                created=created,
                model=model,
                req=req,
            )
        ) as events:
            async for event in events:
                yield event
    except Exception as exc:
        status_code = generation_error_status_code(exc)
        if status_code == 500:
            logger.exception(f"Chat stream failed for request {request_id}")
        else:
            logger.warning(
                f"Chat stream for request {request_id} failed with "
                f"status {status_code}. {exc}"
            )
        error = {
            "error": {
                "message": str(exc),
                "type": (
                    "invalid_request_error" if status_code < 500 else "server_error"
                ),
                "code": status_code,
            }
        }
        yield f"data: {json.dumps(error)}\n\n"
        yield f"data: {STREAM_DONE_SENTINEL}\n\n"


async def chat_stream_events(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """Streaming chat completion generator (yields SSE events)."""
    role_sent = False
    requested_modalities = req.modalities or ["text"]
    finish_reason: str | None = None
    async with aclosing(
        client.completion_stream(gen_req, request_id=request_id)
    ) as stream:
        async for chunk in stream:
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
                continue
            else:
                pass
            delta = ChatCompletionStreamDelta()
            segment = chunk.segment

            # Send role on first chunk
            if not role_sent:
                delta.role = "assistant"
                role_sent = True
            else:
                pass

            if (
                segment is not None
                and segment.kind == "text"
                and "text" in requested_modalities
            ):
                delta.content = segment.data
            else:
                pass
            stream_resp = ChatCompletionStreamResponse(
                id=response_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=delta,
                        finish_reason=None,
                    )
                ],
            )

            data = stream_resp.model_dump(exclude_none=True)
            for choice in data.get("choices", []):
                choice.setdefault("finish_reason", None)
            if segment is not None:
                data["choices"][0]["delta"]["segment"] = segment.to_dict()
            else:
                pass
            yield f"data: {json.dumps(data)}\n\n"

    # Finish chunk: empty delta + finish_reason.
    finish_resp = ChatCompletionStreamResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChatCompletionStreamDelta(),
                finish_reason=finish_reason or "stop",
            )
        ],
    )
    data = finish_resp.model_dump(exclude_none=True)
    for choice in data.get("choices", []):
        choice.setdefault("finish_reason", None)
    yield f"data: {json.dumps(data)}\n\n"

    yield f"data: {STREAM_DONE_SENTINEL}\n\n"


def build_chat_generate_request(req: ChatCompletionRequest) -> GenerateRequest:
    """Convert a ChatCompletionRequest into a client GenerateRequest."""
    # Parse stop sequences
    stop: list[str] = []
    if isinstance(req.stop, str):
        stop = [req.stop]
    elif isinstance(req.stop, list):
        stop = list(req.stop)
    else:
        pass

    # Build sampling params
    sampling = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        min_p=req.min_p,
        repetition_penalty=req.repetition_penalty,
        stop=stop,
        seed=req.seed,
        max_new_tokens=req.effective_max_tokens,
    )

    # Convert messages
    messages = [Message(role=m.role, content=m.content) for m in req.messages]

    # Determine output modalities
    output_modalities = req.modalities or ["text"]  # e.g. ["text", "audio"]

    # Build per-stage sampling overrides
    stage_sampling: dict[str, SamplingParams] | None = None
    if req.stage_sampling:
        stage_sampling = {}
        for stage_name, params_dict in req.stage_sampling.items():
            stage_sampling[stage_name] = SamplingParams(**params_dict)
    else:
        pass

    return GenerateRequest(
        model=req.model,
        messages=messages,
        sampling=sampling,
        stage_sampling=stage_sampling,
        stage_params=req.stage_params,
        stream=req.stream,
        max_tokens=req.effective_max_tokens,
        output_modalities=output_modalities,
    )


async def complete_unless_disconnected(
    request: Request,
    client: Client,
    completion: Coroutine[None, None, CompletionResult],
    *,
    request_id: str,
) -> CompletionResult:
    """Await a buffered completion; if the HTTP client disconnects first, abort its
    pipeline request and raise ClientError, which the route maps to an error response.
    """
    completion_task = asyncio.create_task(completion)
    disconnect_task = asyncio.create_task(wait_for_request_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            {completion_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if completion_task in done:
            return completion_task.result()
        else:
            pass
        await client.abort(request_id)
        raise ClientError(f"Client disconnected; aborted request {request_id}")
    except asyncio.CancelledError:
        await client.abort(request_id)
        raise
    finally:
        if not completion_task.done():
            await cancel_task_bounded(completion_task)
        else:
            pass
        if not disconnect_task.done():
            await cancel_task_bounded(disconnect_task)
        else:
            pass


async def cancel_task_bounded(
    task: asyncio.Task[CompletionResult] | asyncio.Task[None],
) -> None:
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=HTTP_DISCONNECT_CANCEL_TIMEOUT_S)
    if done:
        await asyncio.gather(*done, return_exceptions=True)
    else:
        task.add_done_callback(discard_cancelled_task_result)


def discard_cancelled_task_result(
    task: asyncio.Task[CompletionResult] | asyncio.Task[None],
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("Cancelled request task finished with an error", exc_info=True)


async def wait_for_request_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(HTTP_DISCONNECT_POLL_INTERVAL_S)
