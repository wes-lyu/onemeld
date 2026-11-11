# SPDX-License-Identifier: Apache-2.0
"""CPU backends that play each model of the example profile in real worker processes."""

import json
from pathlib import Path

import yaml
from PIL import Image

from onemeld.backends.base import BackendSpec, StepResult
from onemeld.backends.images import image_result
from onemeld.composition.types import StepRequest
from onemeld.config import ServeConfig

COLORS = ("red", "blue")
EXAMPLES = Path(__file__).parents[3] / "examples" / "onemeld"
ADAPTERS = {
    "understanding": "VisionBackend",
    "generation": "PainterBackend",
    "asr": "ListenerBackend",
}


def scripted_config(trace_dir: Path | None = None) -> ServeConfig:
    """serve.yaml with every model replaced by a scripted CPU backend."""
    values = yaml.safe_load((EXAMPLES / "serve.yaml").read_text())
    for name, adapter in ADAPTERS.items():
        backend = values["backends"][name]
        backend.update(adapter=f"{__name__}:{adapter}", args={}, env={}, gpu=None)
        backend["startup_timeout_s"] = 60.0
    if trace_dir is not None:
        values["composition"]["trace_dir"] = str(trace_dir)
    else:
        pass
    return ServeConfig.model_validate(values)


class ScriptedBackendBase:
    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        self.runs = 0

    def load(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def cancel(self, request_id: str) -> None:
        pass

    async def close(self) -> None:
        pass


class VisionBackend(ScriptedBackendBase):
    """Plans two images and their comparison, then compares the images it is shown."""

    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        if "response_format" in request.params:
            return self.plan()
        else:
            return self.compare(request)

    def plan(self) -> StepResult:
        steps = [
            {
                "id": f"cat_{index}",
                "action": "generate_image",
                "instruction": f"A {color} cat",
                "input_refs": [],
                "publish": True,
            }
            for index, color in enumerate(COLORS)
        ]
        steps.append(
            {
                "id": "verdict",
                "action": "respond",
                "instruction": "Compare the cats",
                "input_refs": ["cat_0", "cat_1"],
                "publish": True,
            }
        )
        return {"text": json.dumps({"steps": steps}), "finish_reason": "stop"}

    def compare(self, request: StepRequest) -> StepResult:
        assert isinstance(request.inputs, list)
        images = sum(
            part.get("type") == "image_url"
            for message in request.inputs
            if isinstance(message.get("content"), list)
            for part in message["content"]
        )
        scores = "\n".join(
            f"Photo {index}: clarity {9 - index}/10, subject completeness 8/10, "
            "composition 8/10"
            for index in range(1, images + 1)
        )
        return {"text": f"{scores}\nRecommendation: Photo 1", "finish_reason": "stop"}


class PainterBackend(ScriptedBackendBase):
    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        color = COLORS[self.runs % len(COLORS)]
        self.runs += 1
        return image_result(Image.new("RGB", (16, 16), color))


class ListenerBackend(ScriptedBackendBase):
    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        return {"text": "unused"}
