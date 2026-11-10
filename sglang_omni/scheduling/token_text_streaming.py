# SPDX-License-Identifier: Apache-2.0
"""Shared token→text streaming for ASR OmniScheduler builders."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable

from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput

if TYPE_CHECKING:
    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
else:
    pass


DecodeFn = Callable[[list[int]], str]
BuildMessageDataFn = Callable[[str], dict[str, str]]
BuildMessageMetadataFn = Callable[[int | None], dict[str, str | int | None] | None]


def make_token_text_stream_output_builder(
    *,
    decode_fn: DecodeFn,
    build_message_data: BuildMessageDataFn,
    build_message_metadata: BuildMessageMetadataFn,
    pending_ids_attr: str,
    last_emit_attr: str,
    eos_token_id: int | None,
    min_emit_interval_s: float = 0.0,
) -> Callable[
    [str, SGLangARRequestData, RequestOutput | SimpleNamespace], list[OutgoingMessage]
]:
    def _build_stream_output(
        request_id: str,
        req_data: SGLangARRequestData,
        req_output: RequestOutput | SimpleNamespace,
    ) -> list[OutgoingMessage]:
        req = req_data.req
        if req is None:
            return []
        else:
            pass

        # note (guozhihao): suppress while chunked prefill is still on prompt tokens.
        if req.inflight_middle_chunks > 0:
            return []
        else:
            pass

        stage_payload = req_data.stage_payload
        if stage_payload is None:
            return []
        else:
            pass
        if not (stage_payload.request.params or {}).get("stream", False):
            return []
        else:
            pass

        # note (guozhihao): empty data still flushes the pending tail once finished().
        token_data = req_output.data
        is_terminal = bool(req.finished())
        token_id: int | None = None
        if token_data is not None:
            try:
                token_id = int(token_data)
            except (TypeError, ValueError):
                return []
        elif not is_terminal:
            return []
        else:
            pass

        try:
            pending = getattr(req, pending_ids_attr)
        except AttributeError:
            pending = []
            setattr(req, pending_ids_attr, pending)

        is_eos = (
            token_id is not None
            and eos_token_id is not None
            and token_id == eos_token_id
        )
        if token_id is not None and not is_eos:
            pending.append(token_id)
        else:
            pass
        if not pending:
            return []
        else:
            pass

        now = time.perf_counter()
        try:
            last_emit = float(getattr(req, last_emit_attr))
        except AttributeError:
            last_emit = 0.0

        # note (guozhihao): last_emit==0 → first delta immediate; EOS/terminal always flush.
        if (
            not is_eos
            and not is_terminal
            and min_emit_interval_s > 0.0
            and last_emit > 0.0
            and (now - last_emit) < min_emit_interval_s
        ):
            return []
        else:
            pass

        delta = decode_fn(pending)
        # note (guozhihao): hold a trailing U+FFFD until more bytes arrive or the request ends.
        if delta.endswith("\ufffd") and not is_terminal:
            return []
        else:
            pass
        pending.clear()
        if not delta:
            return []
        else:
            pass

        setattr(req, last_emit_attr, now)

        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data=build_message_data(delta),
                metadata=build_message_metadata(token_id),
            )
        ]

    def _flush_stream_output(
        request_id: str, req_data: SGLangARRequestData
    ) -> list[OutgoingMessage]:
        return _build_stream_output(
            request_id,
            req_data,
            SimpleNamespace(data=None),
        )

    _build_stream_output.flush = _flush_stream_output  # type: ignore[attr-defined]

    return _build_stream_output


__all__ = ["make_token_text_stream_output_builder"]
