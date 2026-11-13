# SPDX-License-Identifier: Apache-2.0
"""OpenAI-format chat requests and results, for frameworks whose serving layer takes them."""

from collections.abc import Mapping
from copy import deepcopy

from pydantic import JsonValue

from onemeld.backends.base import StepResult
from onemeld.composition.types import StepRequest

# Step parameters named after the composer's vocabulary, and their OpenAI name.
RENAMED_PARAMS = {"max_new_tokens": "max_completion_tokens"}


def chat_request_fields(
    request: StepRequest, model: str, accepted: set[str]
) -> dict[str, JsonValue]:
    """The fields of one non-streaming, single-choice chat completion request.

    accepted is the field set of the framework's request type. A step parameter
    outside it is a configuration error rather than an option to drop.
    """
    if not isinstance(request.inputs, list) or not request.inputs:
        raise ValueError("A chat step needs at least one chat message")
    else:
        pass
    params = {
        RENAMED_PARAMS.get(name, name): value for name, value in request.params.items()
    }
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise ValueError(f"The chat backend does not accept step parameters {unknown}")
    elif params.get("n", 1) != 1:
        raise ValueError("A chat step returns exactly one completion")
    else:
        pass
    return {
        **params,
        "model": model,
        "messages": deepcopy(request.inputs),
        "n": 1,
        "stream": False,
    }


def chat_result(response: Mapping[str, object]) -> StepResult:
    """The step result of a decoded chat completion response or error body."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError(f"Chat backend error: {response.get('error', response)}")
    else:
        choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    finish_reason = choice.get("finish_reason")
    usage = response.get("usage")
    return {
        "text": content if isinstance(content, str) else "",
        "finish_reason": finish_reason if isinstance(finish_reason, str) else "stop",
        **({"usage": usage} if isinstance(usage, dict) else {}),
    }
