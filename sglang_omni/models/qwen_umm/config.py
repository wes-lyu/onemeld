# SPDX-License-Identifier: Apache-2.0
"""Independent Qwen checkpoints composed under one UMM controller.

Stage factories are bound by import path, as every pipeline binds its stages,
so these configurations import nothing from the ASR and TTS model packages.
"""

from collections.abc import Mapping
from typing import ClassVar

from pydantic import Field

from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.schema import EngineStageConfig, FactoryArgs, StageConfig
from sglang_omni.pipeline.umm import (
    CONTROLLER_FACTORY_PATH,
    OUTPUT_FACTORY_PATH,
    ROUTE_FN_PATH,
    STREAM_DONE_FN_PATH,
    UMMPipelineConfig,
)
from sglang_omni.pipeline.umm_plan import (
    ExecutionMode,
    UMMCapability,
    UMMCompositionSpec,
    UMMExecutionPlan,
    UMMPlanExample,
    UMMPlanStep,
)
from sglang_omni.platforms import current_platform

CHAT_FACTORY_PATH = (
    "sglang_omni.scheduling.native_chat_scheduler.create_native_chat_scheduler"
)
GENERATION_FACTORY_PATH = (
    "sglang_omni.native_generation.stage.create_generation_scheduler"
)
ASR_FACTORY_PATH = (
    "sglang_omni.models.qwen3_asr.stages.create_sglang_qwen3_asr_executor"
)
TTS_FACTORY_PREFIX = "sglang_omni.models.qwen3_tts.stages"
SPEECH_STAGES = ("tts_preprocessing", "tts_engine", "vocoder")

RESPOND = UMMCapability(
    stages=["understanding"],
    protocol="native_chat",
    description="Answer text questions, or inspect and compare referenced images and videos, including scores and a visible reason for a winner when requested.",
    input_kinds=["text", "image", "video"],
    parameters={"temperature": 0.0},
)
GENERATE_IMAGE = UMMCapability(
    stages=["generation"],
    protocol="native_image",
    description="Generate exactly one image from a visual prompt. Use separate calls for separate requested images.",
    input_kinds=["text"],
    parameters={
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 4.0,
    },
)
EDIT_IMAGE = UMMCapability(
    stages=["edit"],
    protocol="native_image",
    description="Edit one referenced image as the instruction describes and return the edited image.",
    input_kinds=["text", "image"],
    parameters={"width": 512, "height": 512, "num_inference_steps": 20},
)
TRANSCRIBE = UMMCapability(
    stages=["asr"],
    protocol="asr",
    description="Transcribe one referenced speech recording of at most 30 seconds into text.",
    input_kinds=["audio"],
)
SPEAK = UMMCapability(
    stages=["tts_preprocessing"],
    protocol="tts",
    description="Read the referenced text aloud, or the instruction when nothing is referenced, and return the speech audio.",
    input_kinds=["text"],
)
BICYCLE_EXAMPLE = UMMPlanExample(
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

CAPABILITIES = {
    "respond": RESPOND,
    "generate_image": GENERATE_IMAGE,
    "edit_image": EDIT_IMAGE,
    "transcribe": TRANSCRIBE,
    "speak": SPEAK,
}


def capability_stages(name: str) -> list[StageConfig]:
    """The native stages that serve one capability."""
    if name == "respond":
        return [
            EngineStageConfig(
                name="understanding",
                process="understanding",
                factory_path=CHAT_FACTORY_PATH,
                allow_child_processes=True,
                factory=FactoryArgs(stage_name="understanding"),
                gpu=0,
                next="orchestrator",
            )
        ]
    elif name in ("generate_image", "edit_image"):
        stage = "generation" if name == "generate_image" else "edit"
        return [
            StageConfig(
                name=stage,
                process=stage,
                factory_path=GENERATION_FACTORY_PATH,
                allow_child_processes=True,
                gpu=0,
                next="orchestrator",
            )
        ]
    elif name == "transcribe":
        return [
            StageConfig(
                name="asr",
                process="asr",
                factory_path=ASR_FACTORY_PATH,
                factory=FactoryArgs(max_running_requests=4, enable_torch_compile=False),
                gpu=0,
                next="orchestrator",
            )
        ]
    else:
        # One speech chain is one phase: the vocoder returns the waveform.
        # note (Wes): stages are built in list order; the vocoder comes before
        # the engine so its weights are resident when the KV pool is sized.
        return [
            StageConfig(
                name="tts_preprocessing",
                process="speech",
                factory_path=f"{TTS_FACTORY_PREFIX}.create_preprocessing_executor",
                next="tts_engine",
            ),
            StageConfig(
                name="vocoder",
                process="speech",
                factory_path=f"{TTS_FACTORY_PREFIX}.create_vocoder_executor",
                factory=FactoryArgs(dtype="bfloat16"),
                gpu=0,
                next="orchestrator",
                can_accept_stream_before_payload=True,
            ),
            EngineStageConfig(
                name="tts_engine",
                process="speech",
                factory_path=f"{TTS_FACTORY_PREFIX}.create_sglang_tts_engine_executor",
                factory=FactoryArgs(dtype="bfloat16"),
                gpu=0,
                next="vocoder",
                stream_to=["vocoder"],
            ),
        ]


def qwen_stages(
    capability_names: tuple[str, ...], execution_mode: ExecutionMode
) -> list[StageConfig]:
    """The controller, output relay and native stages of the named capabilities.

    Qwen-VL plans on the understanding stage, so respond is always included.
    """
    names = ("respond", *capability_names)
    composition = UMMCompositionSpec(
        planner_stage="understanding",
        capabilities={name: CAPABILITIES[name] for name in names},
        plan_examples=[BICYCLE_EXAMPLE] if "generate_image" in names else [],
        execution_mode=execution_mode,
        allowed_modes=["plan", "iterative"],
        max_planner_turns=4,
        max_steps=8,
        max_plan_attempts=2,
        max_videos=0,
        trace=True,
    )
    stages = [stage for name in names for stage in capability_stages(name)]
    phase_entries = [
        stage.name for stage in stages if stage.name not in ("tts_engine", "vocoder")
    ]
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
            next=[*phase_entries, "output"],
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
        *stages,
    ]


class QwenUMMPipelineConfig(UMMPipelineConfig):
    """Qwen-VL plans and inspects; Qwen-Image generates."""

    architecture: ClassVar[str] = "QwenUMMDemo"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "understanding": EngineStageConfig
    }
    name: str = "qwen_umm_demo"
    model_path: str
    text_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    entry_stage: str = "orchestrator"
    stages: list[StageConfig] = Field(
        default_factory=lambda: qwen_stages(("generate_image",), "plan")
    )

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "understanding":
            return {"model_path": self.text_model_path}
        elif stage_name == "generation":
            return {"inline_media": True}
        else:
            return super().stage_factory_kwargs(stage_name)


class QwenSpeechInPipelineConfig(QwenUMMPipelineConfig):
    """Adds Qwen3-ASR speech input; replies are text and images."""

    architecture: ClassVar[str] = "QwenUMMSpeechIn"
    name: str = "qwen_umm_speech_in"
    asr_model_path: str = "Qwen/Qwen3-ASR-1.7B"
    stages: list[StageConfig] = Field(
        default_factory=lambda: qwen_stages(
            ("generate_image", "transcribe"), "iterative"
        )
    )

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "asr":
            return {"model_path": self.asr_model_path}
        else:
            return super().stage_factory_kwargs(stage_name)


class QwenEditPipelineConfig(QwenUMMPipelineConfig):
    """Qwen-VL plans and inspects; Qwen-Image-Edit, at model_path, edits."""

    architecture: ClassVar[str] = "QwenUMMEdit"
    name: str = "qwen_umm_edit"
    stages: list[StageConfig] = Field(
        default_factory=lambda: qwen_stages(("edit_image",), "iterative")
    )

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "edit":
            return {"inline_media": True}
        else:
            return super().stage_factory_kwargs(stage_name)


class QwenAnyToAnyPipelineConfig(QwenSpeechInPipelineConfig):
    """Adds Qwen3-TTS speech output."""

    architecture: ClassVar[str] = "QwenUMMAnyToAny"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "understanding": EngineStageConfig,
        "tts_engine": EngineStageConfig,
    }
    name: str = "qwen_umm_any_to_any"
    tts_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    stages: list[StageConfig] = Field(
        default_factory=lambda: qwen_stages(
            ("generate_image", "transcribe", "speak"), "iterative"
        )
    )

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "vocoder":
            # Each request runs at most one speech phase at a time.
            max_active = resolve_stage_typed_kwargs(self.stage_named("orchestrator"))[
                "max_active"
            ]
            # A platform that has not measured the vocoder's decode graph
            # captures starts it without async decode, as the speech pipeline does.
            fast_path = current_platform.enable_tts_vocoder_fast_path()
            return {
                "model_path": self.tts_model_path,
                "running_stream_count": max_active,
                **({} if fast_path else {"async_decode": False}),
            }
        elif stage_name in SPEECH_STAGES:
            return {"model_path": self.tts_model_path}
        else:
            return super().stage_factory_kwargs(stage_name)


class QwenAnyToAnyEditPipelineConfig(QwenAnyToAnyPipelineConfig):
    """Adds a Qwen-Image-Edit stage; holds two diffusion models in host memory."""

    architecture: ClassVar[str] = "QwenUMMAnyToAnyEdit"
    name: str = "qwen_umm_any_to_any_edit"
    edit_model_path: str = "Qwen/Qwen-Image-Edit-2509"
    stages: list[StageConfig] = Field(
        default_factory=lambda: qwen_stages(
            ("generate_image", "edit_image", "transcribe", "speak"), "iterative"
        )
    )

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "edit":
            return {"model_path": self.edit_model_path, "inline_media": True}
        else:
            return super().stage_factory_kwargs(stage_name)


EntryClass = QwenUMMPipelineConfig
Variants = {
    "speech_in": QwenSpeechInPipelineConfig,
    "edit": QwenEditPipelineConfig,
    "any_to_any": QwenAnyToAnyPipelineConfig,
    "any_to_any_edit": QwenAnyToAnyEditPipelineConfig,
}
