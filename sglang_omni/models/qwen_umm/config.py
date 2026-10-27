# SPDX-License-Identifier: Apache-2.0
"""Native SGLang text and image stages under one UMM controller."""

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
from sglang_omni.pipeline.umm_plan import (
    UMMCapability,
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMPlanExample,
    UMMPlanStep,
)

QWEN_COMPOSITION = UMMCompositionSpec(
    planner_stage="understanding",
    capabilities={
        "generate_image": UMMCapability(
            stage="generation",
            protocol="native_image",
            description="Generate exactly one image from a visual prompt. Use separate calls for separate requested images.",
            input_kinds=["text"],
            parameters={
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,
                "guidance_scale": 4.0,
            },
        ),
        "respond": UMMCapability(
            stage="understanding",
            protocol="native_chat",
            description="Answer text questions or inspect and compare actual referenced images, including scores and a visible reason for a winner when requested.",
            input_kinds=["text", "image"],
            parameters={"temperature": 0.0},
        ),
    },
    plan_examples=[
        UMMPlanExample(
            prompt="First have a text model privately describe a red bicycle, then generate its image, then inspect the result against the description. Return only the image and final check.",
            plan=UMMExecutionPlan(
                steps=[
                    UMMPlanStep(
                        id="draft",
                        action="respond",
                        instruction="Write a short visual description of a red bicycle.",
                        input_refs=[],
                        publish=False,
                    ),
                    UMMPlanStep(
                        id="picture",
                        action="generate_image",
                        instruction="Generate an image matching the supplied description.",
                        input_refs=["draft"],
                        publish=True,
                    ),
                    UMMPlanStep(
                        id="check",
                        action="respond",
                        instruction="Inspect whether the actual image matches the original description.",
                        input_refs=["draft", "picture"],
                        publish=True,
                    ),
                ]
            ),
        )
    ],
    max_plan_attempts=2,
    trace=True,
)


class QwenUMMPipelineConfig(UMMPipelineConfig):
    architecture: ClassVar[str] = "QwenUMMDemo"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "understanding": EngineStageConfig
    }
    name: str = "qwen_umm_demo"
    model_path: str
    text_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    entry_stage: str = "orchestrator"
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(
                name="orchestrator",
                process="orchestrator",
                factory_path=CONTROLLER_FACTORY_PATH,
                factory=FactoryArgs(
                    composition=QWEN_COMPOSITION.model_dump(mode="json"),
                    phases=QWEN_COMPOSITION.phases,
                    max_dispatches=QWEN_COMPOSITION.max_steps
                    + QWEN_COMPOSITION.max_plan_attempts,
                    max_outputs=QWEN_COMPOSITION.max_steps,
                    max_active=1,
                    max_retained_bytes=128 * 1024 * 1024,
                    timeout_s=1200.0,
                ),
                next=["understanding", "generation", "output"],
                route_fn=ROUTE_FN_PATH,
                stream_to=["output"],
                stream_done_to_fn=STREAM_DONE_FN_PATH,
            ),
            EngineStageConfig(
                name="understanding",
                process="understanding",
                factory_path="sglang_omni.scheduling.native_chat_scheduler.create_native_chat_scheduler",
                allow_child_processes=True,
                factory=FactoryArgs(stage_name="understanding"),
                gpu=0,
                next="orchestrator",
            ),
            StageConfig(
                name="generation",
                process="generation",
                factory_path="sglang_omni.native_generation.stage.create_generation_scheduler",
                allow_child_processes=True,
                gpu=0,
                next="orchestrator",
            ),
            StageConfig(
                name="output",
                process="orchestrator",
                factory_path=OUTPUT_FACTORY_PATH,
                terminal=True,
                can_accept_stream_before_payload=True,
            ),
        ]
    )

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "understanding":
            return {"model_path": self.text_model_path}
        elif stage_name == "generation":
            return {"inline_media": True}
        else:
            return super().stage_factory_kwargs(stage_name)


EntryClass = QwenUMMPipelineConfig
