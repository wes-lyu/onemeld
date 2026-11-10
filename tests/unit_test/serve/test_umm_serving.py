# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.client import Client, final_segments
from sglang_omni.client.types import GenerateRequest, UMMSegment
from sglang_omni.proto import CompleteMessage, StreamMessage
from sglang_omni.serve import create_app

SEGMENTS = [
    UMMSegment("execution", 0, "text", "Drawing."),
    UMMSegment(
        "execution",
        1,
        "image",
        {
            "kind": "image",
            "url": "data:image/png;base64,eA==",
            "mime_type": "image/png",
            "size_bytes": 1,
        },
    ),
    UMMSegment("execution", 2, "text", "Finished."),
]


def final_result(streamed: bool) -> dict[str, object]:
    # A streamed request already received each segment with its data.
    return {
        "type": "umm_result",
        "execution_id": "execution",
        "segments": [
            {
                key: value
                for key, value in segment.to_dict().items()
                if not (streamed and key == "data")
            }
            for segment in SEGMENTS
        ],
        "finish_reason": "stop",
    }


class Coordinator:
    def __init__(self):
        self.closed = False
        self.submitted = []

    def health(self):
        return {"running": True}

    async def submit(self, request_id, request):
        self.submitted.append(request_id)
        return final_result(streamed=False)

    async def stream(self, request_id, request):
        self.submitted.append(request_id)
        try:
            for segment in SEGMENTS:
                yield StreamMessage(
                    request_id,
                    "output",
                    segment.to_dict(),
                    modality=segment.kind,
                    chunk_id=segment.segment_index,
                )
            yield CompleteMessage(
                request_id, "output", True, result=final_result(streamed=True)
            )
        finally:
            self.closed = True


def umm_app(coordinator=None):
    app = create_app(
        Client(coordinator or Coordinator()),
        model_name="onemeld",
    )
    return TestClient(app)


# Stop strings never apply to an interleaved request, so its text matches
# its segments whether or not the client sets one.
@pytest.mark.parametrize("stop", [None, ["Fin"]])
def test_chat_keeps_ordered_terminal_manifest(stop):
    client = umm_app()
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "onemeld",
            "messages": [{"role": "user", "content": "Draw then describe"}],
            "stop": stop,
        },
    )
    assert chat.status_code == 200
    message = chat.json()["choices"][0]["message"]
    assert message["segments"] == [s.to_dict() for s in SEGMENTS]
    assert message["content"] == "Drawing.Finished."


@pytest.mark.parametrize("stop", [None, ["Fin"]])
def test_chat_sse_exposes_media_segment_without_duplicate_text(stop):
    client = umm_app()
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "onemeld",
            "messages": [{"role": "user", "content": "Draw then describe"}],
            "stream": True,
            "stop": stop,
        },
    )
    messages = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [msg["choices"][0]["delta"] for msg in messages]
    assert [d["segment"] for d in deltas if "segment" in d] == [
        s.to_dict() for s in SEGMENTS
    ]
    assert "".join(d.get("content", "") for d in deltas) == "Drawing.Finished."
    assert messages[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_completion_keeps_ordered_segments_with_their_data(stream):
    result = await Client(Coordinator()).completion(
        GenerateRequest(prompt="hello", stream=stream), request_id="r"
    )
    assert result.segments == SEGMENTS
    assert result.text == "Drawing.Finished."


@pytest.mark.parametrize("invalid", ["not_list", "wrong_execution", "reordered"])
def test_corrupt_final_result_is_rejected(invalid):
    result = final_result(streamed=False)
    if invalid == "not_list":
        result["segments"] = {"unexpected": 1}
    elif invalid == "wrong_execution":
        result["execution_id"] = "other"
    else:
        result["segments"].reverse()
    with pytest.raises(ValueError, match="UMM final"):
        final_segments(result)
