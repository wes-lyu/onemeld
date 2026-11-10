# SPDX-License-Identifier: Apache-2.0
"""OpenAI-format chat requests built from steps, and results read back."""

import pytest

from onemeld.backends.chat_request import chat_request_fields, chat_result
from onemeld.composition.plan import UMMCompositionSpec
from onemeld.composition.planner import build_planner_request
from onemeld.composition.types import StepRequest

pytest_plugins = ["tests.unit_test.fixtures.composition"]

OPENAI_FIELDS = {
    "model",
    "messages",
    "n",
    "stream",
    "temperature",
    "seed",
    "max_completion_tokens",
    "response_format",
    "chat_template_kwargs",
}


def test_planner_request_maps_onto_openai_fields(
    composition: UMMCompositionSpec,
) -> None:
    """The planner's structured-output request survives the mapping intact."""
    composition = composition.model_copy(
        update={"planning_chat_template_kwargs": {"enable_thinking": False}}
    )
    request = build_planner_request(composition, [], {}, 512, None)
    fields = chat_request_fields(request, "planner", OPENAI_FIELDS)
    assert fields["max_completion_tokens"] == 512
    assert "max_new_tokens" not in fields
    assert fields["response_format"]["type"] == "json_schema"
    assert fields["chat_template_kwargs"] == {"enable_thinking": False}
    assert (fields["model"], fields["n"], fields["stream"]) == ("planner", 1, False)
    # The step keeps its own messages if the framework mutates the request.
    fields["messages"][0]["content"] = "changed"
    assert request.inputs[0]["content"] != "changed"


@pytest.mark.parametrize(
    "request_, message",
    [
        (StepRequest(inputs="plain text"), "chat message"),
        (StepRequest(inputs=[]), "chat message"),
        (
            StepRequest(inputs=[{"role": "user", "content": "x"}], params={"top_x": 1}),
            r"\['top_x'\]",
        ),
        (
            StepRequest(inputs=[{"role": "user", "content": "x"}], params={"n": 2}),
            "exactly one completion",
        ),
    ],
    ids=["not-messages", "no-messages", "unknown-param", "several-choices"],
)
def test_invalid_chat_steps_are_rejected(request_: StepRequest, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        chat_request_fields(request_, "model", OPENAI_FIELDS)


def test_chat_result_reads_text_and_finish_reason() -> None:
    response = {
        "choices": [
            {"message": {"content": "Paris"}, "finish_reason": "length"},
        ]
    }
    assert chat_result(response) == {"text": "Paris", "finish_reason": "length"}


def test_chat_error_body_becomes_an_error() -> None:
    with pytest.raises(ValueError, match="context length"):
        chat_result({"object": "error", "error": "Exceeds the context length"})
