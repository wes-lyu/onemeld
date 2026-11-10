# SPDX-License-Identifier: Apache-2.0
"""Shared OpenAI-compatible API error classification helpers."""

from __future__ import annotations

import re

from sglang_omni.admission import QueueFullError

_BAD_REQUEST_MARKERS = (
    "Unsupported language:",
    "longer than the model's context length",
    "Requested token count exceeds the model's maximum context length",
    "Request has no prompt tokens",
    "Request requires more tokens than the thinker KV cache can hold",
    "out-of-vocabulary token id",
    "accepts audio up to",
    "could not decode the uploaded audio",
    "chat content part",
    "max_new_tokens must be",
    "exceeds the maximum allowed length",
    "sequence exceeds max_length",
    "Native generation request ",
    "Native chat request rejected: ",
    "Invalid UMM request: ",
)
# The prefix of UMMPlanningError, which reaches serving as text.
UNPROCESSABLE_MARKER = "UMM planning failed: "
_BAD_REQUEST_PATTERNS = (
    re.compile(r"^Could not decode .+ audio input$"),
    re.compile(r"^Request\s+\S+\s+exceeds the maximum number of tokens:"),
    re.compile(r"^Request\s+\S+\s+requires too many SWA KV tokens for"),
    re.compile(r"^Request .+ already exists$", re.DOTALL),
    re.compile(r"^stop_regex is \d+ bytes, over the \d+-byte limit"),
    re.compile(r"^Media URL returned HTTP 404: "),
)


def is_bad_request_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(marker in message for marker in _BAD_REQUEST_MARKERS) or any(
        pattern.search(message) is not None for pattern in _BAD_REQUEST_PATTERNS
    )


def generation_error_status_code(exc: BaseException) -> int:
    """Map a failure to 503 for a full queue, 400 for a caller error, 422 for a
    task the UMM planner could not plan, else 500."""
    if QueueFullError.matches(exc):
        return 503
    elif is_bad_request_error(exc):
        return 400
    elif UNPROCESSABLE_MARKER in str(exc):
        return 422
    else:
        return 500
