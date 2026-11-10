# SPDX-License-Identifier: Apache-2.0
"""The checked-in request runs through OneMeld's HTTP and composition layers."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.client.image_media import InlineImage
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.sources import sources_from_config_file
from sglang_omni.models.qwen_umm.config import OneMeldPipelineConfig
from sglang_omni.pipeline.umm import UMMController, create_umm_controller
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_api import create_app
from tests.unit_test.fixtures.umm_stages import FakeStages

pytest_plugins = ["tests.unit_test.fixtures.umm_composition"]


class ExampleCoordinator:
    def __init__(self, controller: UMMController, images: list[InlineImage]) -> None:
        self.controller = controller
        self.images = images
        self.dispatches: list[list[str]] = []

    async def submit(self, request_id: str, request: OmniRequest) -> dict[str, object]:
        stages = FakeStages(self.controller, request_id)
        stages.submit(request)
        dispatches = ["planning"]
        self.dispatches.append(dispatches)
        assert set(stages.held) == {"planning"}
        planner_request = stages.held["planning"].request
        task = planner_request.inputs[-1]["content"][0]["text"]
        image_count = 2 if "两张" in task else 1
        image_ids = [f"picture_{index}" for index in range(image_count)]
        steps = [
            {
                "id": name,
                "action": "generate_image",
                "instruction": "Draw a cat",
                "input_refs": [],
                "publish": True,
            }
            for name in image_ids
        ]
        if image_count == 2:
            steps.append(
                {
                    "id": "pick",
                    "action": "respond",
                    "instruction": "Compare both photos",
                    "input_refs": image_ids,
                    "publish": True,
                }
            )
        stages.reply("planning", {"text": json.dumps({"steps": steps})})
        for index in range(image_count):
            dispatches.append("generation")
            assert set(stages.held) == {"generation"}
            generation = stages.held["generation"].request
            assert generation.params["seed"] == 42 + index
            stages.reply(
                "generation",
                {
                    "modality": "image",
                    "image_b64": self.images[index].encoded_content(),
                },
            )
        if image_count == 2:
            dispatches.append("understanding")
            assert set(stages.held) == {"understanding"}
            content = stages.held["understanding"].request.inputs[-1]["content"]
            assert [
                part["image_url"]["url"]
                for part in content
                if part["type"] == "image_url"
            ] == [image.url for image in self.images[:2]]
            stages.reply("understanding", {"text": "I recommend the first image."})
        return stages.held["output"].data


def test_checked_in_profile_routes_two_requests_differently(
    inline_images: list[InlineImage],
) -> None:
    base, patches = sources_from_config_file(
        "examples/onemeld/serve.yaml", OneMeldPipelineConfig
    )
    config = ConfigResolver(base).resolve(patches).config
    controller = create_umm_controller(
        config.model_path,
        output_stage="output",
        **resolve_stage_typed_kwargs(config.stage_named("orchestrator")),
    )
    coordinator = ExampleCoordinator(controller, inline_images)
    request = json.loads(Path("examples/onemeld/compare-cats.json").read_text())
    with TestClient(create_app(Client(coordinator), model_name=config.name)) as client:
        first = client.post("/v1/chat/completions", json=request)
        assert first.status_code == 200
        segments = first.json()["choices"][0]["message"]["segments"]
        assert [segment["kind"] for segment in segments] == ["image", "image", "text"]
        assert [segment["data"]["sha256"] for segment in segments[:2]] == [
            image.sha256 for image in inline_images[:2]
        ]
        assert segments[2]["data"] == "I recommend the first image."
        request["messages"] = [{"role": "user", "content": "只生成一张猫咪照片"}]
        second = client.post("/v1/chat/completions", json=request)
        assert second.status_code == 200
        assert [
            segment["kind"]
            for segment in second.json()["choices"][0]["message"]["segments"]
        ] == ["image"]
        assert client.get("/v1/models").json()["data"][0]["id"] == "onemeld"
        assert client.post("/v1/audio/speech", json={}).status_code == 404
    assert coordinator.dispatches == [
        ["planning", "generation", "generation", "understanding"],
        ["planning", "generation"],
    ]
