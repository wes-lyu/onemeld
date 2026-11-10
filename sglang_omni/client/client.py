# SPDX-License-Identifier: Apache-2.0
"""Adapt the coordinator's UMM outputs for Chat Completions."""

from collections.abc import AsyncIterator
from contextlib import aclosing

from sglang_omni.client.types import (
    ClientError,
    CompletionResult,
    CompletionStreamChunk,
    GenerateRequest,
    UMMSegment,
)
from sglang_omni.pipeline.coordinator import Coordinator, CoordinatorHealth
from sglang_omni.proto.messages import StreamMessage
from sglang_omni.proto.request import OmniRequest


class Client:
    def __init__(self, coordinator: Coordinator) -> None:
        self.coordinator = coordinator

    async def completion(
        self, request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        if request.stream:
            streamed: list[UMMSegment] = []
            finish_reason = None
            async with aclosing(
                self.completion_stream(request, request_id=request_id)
            ) as stream:
                async for chunk in stream:
                    if chunk.segment is not None:
                        streamed.append(chunk.segment)
                    else:
                        finish_reason = chunk.finish_reason
            if finish_reason is None:
                raise ClientError("No final response from pipeline")
            else:
                segments = streamed
        else:
            result = await self.coordinator.submit(
                request_id, self.build_omni_request(request)
            )
            segments, finish_reason = final_segments(result)
            if any(segment.data is None for segment in segments):
                raise ValueError("UMM final result is missing segment data")
            else:
                pass

        return CompletionResult(
            text="".join(
                segment.data
                for segment in segments
                if segment.kind == "text" and isinstance(segment.data, str)
            ),
            finish_reason=finish_reason,
            segments=segments,
        )

    async def completion_stream(
        self, request: GenerateRequest, *, request_id: str
    ) -> AsyncIterator[CompletionStreamChunk]:
        streamed: list[tuple[str, int, str]] = []
        coordinator_stream = self.coordinator.stream(
            request_id, self.build_omni_request(request)
        )
        async with aclosing(coordinator_stream):
            async for message in coordinator_stream:
                if isinstance(message, StreamMessage):
                    segment = UMMSegment.from_dict(message.chunk)
                    streamed.append(
                        (segment.execution_id, segment.segment_index, segment.kind)
                    )
                    yield CompletionStreamChunk(segment=segment)
                else:
                    segments, finish_reason = final_segments(message.result)
                    if streamed != [
                        (segment.execution_id, segment.segment_index, segment.kind)
                        for segment in segments
                    ]:
                        raise ValueError(
                            "Streamed UMM segments do not match the final result"
                        )
                    else:
                        yield CompletionStreamChunk(finish_reason=finish_reason)
                        return
        raise ClientError("No final response from pipeline")

    async def abort(self, request_id: str) -> bool:
        return await self.coordinator.abort(request_id)

    def health(self) -> CoordinatorHealth:
        return self.coordinator.health()

    @staticmethod
    def build_omni_request(request: GenerateRequest) -> OmniRequest:
        inputs = extract_inputs(request)
        params = build_params(request)
        metadata = dict(request.metadata)
        if request.model:
            metadata.setdefault("model", request.model)
        else:
            pass
        if request.output_modalities is not None:
            metadata["output_modalities"] = request.output_modalities
        else:
            pass
        return OmniRequest(inputs=inputs, params=params, metadata=metadata)


def final_segments(result: object) -> tuple[list[UMMSegment], str]:
    """Decode the terminal manifest for one ordered execution."""
    if not isinstance(result, dict) or result.get("type") != "umm_result":
        raise ValueError("Invalid UMM final result")
    else:
        pass
    raw_segments = result.get("segments")
    execution_id = result.get("execution_id")
    if not isinstance(raw_segments, list) or not isinstance(execution_id, str):
        raise ValueError("Invalid UMM final result")
    else:
        pass
    segments = [UMMSegment.from_dict(item) for item in raw_segments]
    if any(
        segment.execution_id != execution_id or segment.segment_index != index
        for index, segment in enumerate(segments)
    ):
        raise ValueError("UMM final segments must be one ordered execution")
    else:
        finish_reason = result.get("finish_reason")
        return segments, finish_reason if isinstance(finish_reason, str) else "stop"


def extract_inputs(request: GenerateRequest) -> object:
    choices = [
        request.prompt is not None,
        request.prompt_token_ids is not None,
        request.messages is not None,
    ]
    if sum(choices) != 1:
        raise ValueError(
            "GenerateRequest requires exactly one input: "
            "prompt, prompt_token_ids, or messages."
        )
    else:
        pass
    if request.prompt is not None:
        return request.prompt
    else:
        pass
    if request.prompt_token_ids is not None:
        return list(request.prompt_token_ids)
    else:
        pass

    return [message.to_dict() for message in request.messages or []]


def build_params(request: GenerateRequest) -> dict[str, object]:
    params = request.sampling.to_dict()
    if request.max_tokens is not None:
        params["max_new_tokens"] = request.max_tokens
    else:
        pass
    params["stream"] = request.stream
    if request.stage_sampling:
        params["stage_sampling"] = {
            key: value.to_dict() for key, value in request.stage_sampling.items()
        }
    else:
        pass
    if request.stage_params:
        params["stage_params"] = request.stage_params
    else:
        pass
    if request.extra_params:
        params.update(request.extra_params)
    else:
        pass
    return params
