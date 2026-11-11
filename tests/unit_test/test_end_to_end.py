# SPDX-License-Identifier: Apache-2.0
"""The example profile's shape served end to end through real worker processes."""

import json

from fastapi.testclient import TestClient

from onemeld.server import create_app
from tests.unit_test.fixtures.scripted_backends import EXAMPLES, scripted_config


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
    scored_images = [
        line for line in message["content"].splitlines() if line.startswith("Photo ")
    ]
    assert len(scored_images) == 2
    assert len({segment["data"]["sha256"] for segment in message["segments"][:2]}) == 2
