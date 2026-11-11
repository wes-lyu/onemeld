# SPDX-License-Identifier: Apache-2.0
"""The example profile's shape served end to end through real worker processes."""

import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from onemeld.config import ServeConfig
from onemeld.server import create_app

EXAMPLES = Path(__file__).parents[2] / "examples" / "onemeld"
SCRIPTED = "tests.unit_test.fixtures.scripted_backends"
ADAPTERS = {
    "understanding": "VisionBackend",
    "generation": "PainterBackend",
    "asr": "ListenerBackend",
}


def scripted_config() -> ServeConfig:
    """serve.yaml with every model replaced by a scripted CPU backend."""
    values = yaml.safe_load((EXAMPLES / "serve.yaml").read_text())
    for name, adapter in ADAPTERS.items():
        backend = values["backends"][name]
        backend.update(adapter=f"{SCRIPTED}:{adapter}", args={}, env={}, gpu=None)
        backend["startup_timeout_s"] = 60.0
    return ServeConfig.model_validate(values)


def test_compare_cats_request_runs_through_worker_processes() -> None:
    request = json.loads((EXAMPLES / "compare-cats.json").read_text())
    with TestClient(create_app(scripted_config())) as client:
        assert client.get("/health").json()["status"] == "healthy"
        response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    assert [segment["kind"] for segment in message["segments"]] == [
        "image",
        "image",
        "text",
    ]
    assert message["content"] == "Compared 2 images"
    assert len({segment["data"]["sha256"] for segment in message["segments"][:2]}) == 2
