# SPDX-License-Identifier: Apache-2.0
"""Fake native stages and stage results for UMM composition tests."""

from dataclasses import replace

import numpy as np

from sglang_omni.client.image_media import InlineMedia
from sglang_omni.pipeline.umm import UMMController, create_umm_controller, route_umm
from sglang_omni.pipeline.umm_plan import UMMCompositionSpec
from sglang_omni.proto.request import OmniRequest, RoutedPayloads, StagePayload

SAMPLE_RATE = 16000


def build_controller(composition: UMMCompositionSpec) -> UMMController:
    return create_umm_controller(
        "unused",
        composition=composition.model_dump(mode="json"),
        phases=composition.phases,
        output_stage="output",
        max_dispatches=composition.required_dispatches(),
        max_outputs=composition.required_outputs(),
        max_active=1,
        max_retained_bytes=128 * 1024 * 1024,
        timeout_s=60,
    )


def sine_waveform(sample_count: int) -> np.ndarray:
    times = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)


def waveform_result(waveform: np.ndarray) -> dict[str, bytes | list[int] | str | int]:
    """The final payload a native vocoder returns for a non-streamed request."""
    return {
        "audio_waveform": waveform.tobytes(),
        "audio_waveform_shape": list(waveform.shape),
        "audio_waveform_dtype": "float32",
        "sample_rate": SAMPLE_RATE,
        "modality": "audio",
    }


def media_result(media: InlineMedia) -> dict[str, list[dict[str, str | int | float]]]:
    """The inline result of native generation, with fields the executor ignores."""
    return {"media": [{**media.model_dump(), "generation_time": 1.0}]}


class FakeStages:
    """Holds each payload the controller routes and replies to it as that stage."""

    def __init__(self, controller: UMMController, request_id: str) -> None:
        self.controller = controller
        self.request_id = request_id
        self.held: dict[str, StagePayload] = {}

    def submit(self, request: OmniRequest) -> None:
        self.collect(
            self.controller.advance(StagePayload(self.request_id, request, None))
        )

    def collect(self, dispatched: StagePayload | RoutedPayloads | None) -> None:
        if isinstance(dispatched, RoutedPayloads):
            self.held.update(dispatched.payloads)
        elif dispatched is not None:
            self.held[route_umm(self.request_id, dispatched)] = dispatched
        else:
            pass

    def reply(self, stage: str, data: object) -> None:
        payload = self.held.pop(stage)
        self.collect(self.controller.advance(replace(payload, data=data)))

    def segments(self) -> list[dict[str, object]]:
        """Segments of the completed request, which only the output stage holds."""
        assert set(self.held) == {"output"}, f"stages still hold {sorted(self.held)}"
        return self.held["output"].data["segments"]
