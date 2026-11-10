# SPDX-License-Identifier: Apache-2.0
"""Native stages for the OneMeld example profile."""

from collections.abc import Mapping
from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import EngineStageConfig, FactoryArgs, StageConfig
from sglang_omni.pipeline.umm import (
    CONTROLLER_FACTORY_PATH,
    OUTPUT_FACTORY_PATH,
    ROUTE_FN_PATH,
    STREAM_DONE_FN_PATH,
    UMMPipelineConfig,
)
from sglang_omni.pipeline.umm_plan import UMMCapability, UMMCompositionSpec

CHAT_FACTORY_PATH = (
    "sglang_omni.scheduling.native_chat_scheduler.create_native_chat_scheduler"
)
GENERATION_FACTORY_PATH = (
    "sglang_omni.models.sensenova_u1.stages.create_generation_executor"
)
ASR_FACTORY_PATH = (
    "sglang_omni.models.qwen3_asr.stages.create_sglang_qwen3_asr_executor"
)

RESPOND = UMMCapability(
    stages=["understanding"],
    protocol="native_chat",
    description="Answer text questions, or inspect and compare referenced images, including scores and a visible reason for a winner when requested.",
    input_kinds=["text", "image"],
    parameters={"temperature": 0.0},
)
GENERATE_IMAGE = UMMCapability(
    stages=["generation"],
    protocol="image_base64",
    description="Generate exactly one image from a visual prompt. Use separate calls for separate requested images.",
    input_kinds=["text"],
    parameters={
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 4.0,
    },
)
TRANSCRIBE = UMMCapability(
    stages=["asr"],
    protocol="asr",
    description="Transcribe one referenced speech recording of at most 30 seconds into text.",
    input_kinds=["audio"],
)


def onemeld_stages() -> list[StageConfig]:
    composition = UMMCompositionSpec(
        planner_stage="planning",
        capabilities={
            "respond": RESPOND,
            "generate_image": GENERATE_IMAGE,
            "transcribe": TRANSCRIBE,
        },
        max_steps=6,
        max_plan_attempts=1,
        max_images=2,
        audio_instruction_capability="transcribe",
        max_concurrent_steps=1,
        trace=True,
    )
    return [
        StageConfig(
            name="orchestrator",
            process="orchestrator",
            factory_path=CONTROLLER_FACTORY_PATH,
            factory=FactoryArgs(
                composition=composition.model_dump(mode="json"),
                phases=composition.phases,
                max_dispatches=composition.required_dispatches(),
                max_outputs=composition.required_outputs(),
                max_active=1,
                max_retained_bytes=256 * 1024 * 1024,
                timeout_s=1800.0,
            ),
            next=["understanding", "generation", "asr", "planning", "output"],
            route_fn=ROUTE_FN_PATH,
            stream_to=["output"],
            stream_done_to_fn=STREAM_DONE_FN_PATH,
        ),
        StageConfig(
            name="output",
            process="orchestrator",
            factory_path=OUTPUT_FACTORY_PATH,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        EngineStageConfig(
            name="understanding",
            process="understanding",
            factory_path=CHAT_FACTORY_PATH,
            allow_child_processes=True,
            factory=FactoryArgs(stage_name="understanding"),
            gpu=0,
            next="orchestrator",
        ),
        StageConfig(
            name="generation",
            process="generation",
            factory_path=GENERATION_FACTORY_PATH,
            allow_child_processes=True,
            gpu=0,
            next="orchestrator",
        ),
        StageConfig(
            name="asr",
            process="asr",
            factory_path=ASR_FACTORY_PATH,
            factory=FactoryArgs(
                max_running_requests=4,
                enable_torch_compile=False,
                max_audio_clip_s=30.0,
            ),
            gpu=0,
            next="orchestrator",
        ),
        EngineStageConfig(
            name="planning",
            process="planning",
            factory_path=CHAT_FACTORY_PATH,
            allow_child_processes=True,
            factory=FactoryArgs(stage_name="planning"),
            gpu=0,
            next="orchestrator",
        ),
    ]


class OneMeldPipelineConfig(UMMPipelineConfig):
    """Serve text, image and spoken requests through an upfront plan."""

    architecture: ClassVar[str] = "OneMeld"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "understanding": EngineStageConfig,
        "planning": EngineStageConfig,
    }
    name: str = "onemeld"
    entry_stage: str = "orchestrator"
    model_path: str
    planner_model_path: str = "Qwen/Qwen3.5-4B"
    text_model_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    asr_model_path: str = "Qwen/Qwen3-ASR-1.7B"
    stages: list[StageConfig] = Field(default_factory=onemeld_stages)

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "planning":
            return {"model_path": self.planner_model_path}
        elif stage_name == "understanding":
            return {"model_path": self.text_model_path}
        elif stage_name == "asr":
            return {"model_path": self.asr_model_path}
        else:
            return super().stage_factory_kwargs(stage_name)
