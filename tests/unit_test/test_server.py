# SPDX-License-Identifier: Apache-2.0
"""The Chat Completions surface: ordered segments, streaming, errors and health."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from onemeld import server
from onemeld.backends.base import BackendSpec
from onemeld.composition.types import (
    QueueFullError,
    TaskRequest,
    UMMPlanningError,
    UMMRequestError,
)
from onemeld.config import ServeConfig, load_config
from onemeld.orchestrator import Segment
from onemeld.server import create_app
from tests.unit_test.fixtures.orchestrator import settle

CONFIG = load_config(Path(__file__).parents[2] / "examples" / "onemeld" / "serve.yaml")
IMAGE = {
    "kind": "image",
    "url": "data:image/png;base64,eA==",
    "mime_type": "image/png",
    "size_bytes": 1,
}
SEGMENTS: list[Segment] = [
    {
        "type": "segment",
        "execution_id": "e",
        "segment_index": index,
        "kind": kind,
        "data": data,
    }
    for index, (kind, data) in enumerate(
        [("text", "Drawing."), ("image", IMAGE), ("text", "Finished.")]
    )
]


class FakeOrchestrator:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.tasks: list[TaskRequest] = []
        self.closed = False
        self.healthy = True

    def health(self) -> dict[str, bool]:
        return {"planning": True, "generation": self.healthy}

    async def stream(self, task: TaskRequest) -> AsyncIterator[Segment]:
        self.tasks.append(task)
        try:
            for segment in SEGMENTS:
                if self.failure is not None and segment["segment_index"] == 1:
                    raise self.failure
                else:
                    pass
                yield segment
        finally:
            self.closed = True


def client_for(orchestrator: FakeOrchestrator) -> TestClient:
    return TestClient(create_app(CONFIG, orchestrator=orchestrator))


def test_example_request_becomes_the_composition_task() -> None:
    """The HTTP request shape is unchanged: options sit under stage_params.orchestrator."""
    orchestrator = FakeOrchestrator()
    example = json.loads(
        (
            Path(__file__).parents[2] / "examples" / "onemeld" / "compare-cats.json"
        ).read_text()
    )
    with client_for(orchestrator) as client:
        assert client.post("/v1/chat/completions", json=example).status_code == 200
    (task,) = orchestrator.tasks
    assert task.options == {"width": 512, "height": 512, "steps": 20}
    assert (task.seed, task.max_new_tokens, task.stream) == (42, 1024, False)
    assert task.messages == example["messages"]


# Stop strings never apply to a composed request, so its text matches its
# segments whether or not the client sets one.
@pytest.mark.parametrize("stop", [None, ["Fin"]])
def test_chat_returns_ordered_segments(stop: list[str] | None) -> None:
    with client_for(FakeOrchestrator()) as client:
        chat = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Draw then describe"}],
                "stop": stop,
            },
        )
    assert chat.status_code == 200
    body = chat.json()
    assert body["model"] == "onemeld"
    message = body["choices"][0]["message"]
    assert message["segments"] == SEGMENTS
    assert message["content"] == "Drawing.Finished."


def test_sse_carries_each_segment_once_and_text_as_content() -> None:
    with client_for(FakeOrchestrator()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Draw then describe"}],
                "stream": True,
            },
        )
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line[6:]) for line in lines[:-1]]
    deltas = [event["choices"][0]["delta"] for event in events]
    assert [delta["segment"] for delta in deltas if "segment" in delta] == SEGMENTS
    assert "".join(delta.get("content", "") for delta in deltas) == "Drawing.Finished."
    assert deltas[0]["role"] == "assistant"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "failure, status",
    [
        (UMMRequestError("bad media"), 400),
        (UMMPlanningError("no valid plan"), 422),
        (QueueFullError(), 503),
        (RuntimeError("backend exited"), 500),
    ],
    ids=["client-error", "unplannable", "full", "server-error"],
)
def test_failures_map_to_their_status(failure: Exception, status: int) -> None:
    orchestrator = FakeOrchestrator(failure)
    with client_for(orchestrator) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}]},
        )
        streamed = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}], "stream": True},
        )
    assert response.status_code == status
    assert str(failure) in response.json()["detail"]
    error_events = [
        json.loads(line[6:])
        for line in streamed.text.splitlines()
        if line.startswith("data: {") and '"error"' in line
    ]
    assert [event["error"]["code"] for event in error_events] == [status]
    assert streamed.text.rstrip().endswith("data: [DONE]")
    assert orchestrator.closed


def test_health_reflects_every_backend() -> None:
    orchestrator = FakeOrchestrator()
    with client_for(orchestrator) as client:
        assert client.get("/health").status_code == 200
        orchestrator.healthy = False
        unhealthy = client.get("/health")
    assert unhealthy.status_code == 503
    assert unhealthy.json()["backends"] == {"planning": True, "generation": False}


def test_models_lists_the_profile_name() -> None:
    with client_for(FakeOrchestrator()) as client:
        models = client.get("/v1/models").json()
    assert [card["id"] for card in models["data"]] == ["onemeld"]


class RecordingWorker:
    """Records when each backend starts loading and becomes ready."""

    events: list[str] = []

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec

    async def start(self) -> None:
        self.events.append(f"load {self.spec.name}")
        await settle()
        self.events.append(f"ready {self.spec.name}")

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_backends_sharing_a_gpu_start_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine sizes its cache from free memory, so it must not load beside another."""
    values = CONFIG.model_dump(mode="json")
    values["backends"]["generation"]["gpu"] = 1
    config = ServeConfig.model_validate(values)
    RecordingWorker.events = []
    monkeypatch.setattr(server, "WorkerClient", RecordingWorker)
    await server.start_workers(config)
    events = RecordingWorker.events
    gpu0 = [name for name in config.backends if config.backends[name].gpu == 0]
    gpu0_events = [event for event in events if event.split()[1] in gpu0]
    assert gpu0_events == [
        event for name in gpu0 for event in (f"load {name}", f"ready {name}")
    ]
    # The backend on its own GPU loads alongside the first one on GPU 0.
    assert events.index("load generation") < events.index(f"ready {gpu0[0]}")
