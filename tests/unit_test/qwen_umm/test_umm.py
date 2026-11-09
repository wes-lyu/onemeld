# SPDX-License-Identifier: Apache-2.0
"""The Qwen examples bind independent checkpoints to the public UMM executor."""

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.schema import FactoryArgs
from sglang_omni.models.qwen_umm.config import (
    QwenAnyToAnyEditPipelineConfig,
    QwenAnyToAnyPipelineConfig,
    QwenSpeechInPipelineConfig,
)
from sglang_omni.pipeline.umm import create_umm_controller
from sglang_omni.pipeline.umm_composition import UMMCompositionAdapter
from sglang_omni.platforms import current_platform


@pytest.mark.parametrize(
    "profile, edits",
    [("launch_any_to_any.yaml", False), ("launch_any_to_any_edit.yaml", True)],
)
def test_any_to_any_config_binds_each_checkpoint_to_its_native_stage(
    profile: str, edits: bool
) -> None:
    configuration = ConfigManager.from_file(f"examples/qwen_umm/{profile}").config
    assert isinstance(configuration, QwenAnyToAnyPipelineConfig)
    assert isinstance(configuration, QwenAnyToAnyEditPipelineConfig) == edits
    controller = create_umm_controller(
        configuration.model_path,
        output_stage="output",
        **resolve_stage_typed_kwargs(configuration.stage_named("orchestrator")),
    )
    assert isinstance(controller.adapter, UMMCompositionAdapter)
    model_paths = {
        "understanding": configuration.text_model_path,
        "asr": configuration.asr_model_path,
        "tts_preprocessing": configuration.tts_model_path,
        "tts_engine": configuration.tts_model_path,
        "vocoder": configuration.tts_model_path,
    }
    for stage_name, model_path in model_paths.items():
        assert (
            configuration.stage_factory_kwargs(stage_name)["model_path"] == model_path
        )
    assert configuration.stage_factory_kwargs("generation")["inline_media"] is True
    vocoder = configuration.stage_factory_kwargs("vocoder")
    assert vocoder["running_stream_count"] == 1
    assert vocoder.get("async_decode", True) == (
        current_platform.enable_tts_vocoder_fast_path()
    )
    if edits:
        assert configuration.stage_factory_kwargs("edit") == {
            "model_path": configuration.edit_model_path,
            "inline_media": True,
        }
    else:
        pass
    factory_paths = {
        "understanding": "sglang_omni.scheduling.native_chat_scheduler.create_native_chat_scheduler",
        "generation": "sglang_omni.native_generation.stage.create_generation_scheduler",
        "asr": "sglang_omni.models.qwen3_asr.stages.create_sglang_qwen3_asr_executor",
        "tts_preprocessing": "sglang_omni.models.qwen3_tts.stages.create_preprocessing_executor",
        "tts_engine": "sglang_omni.models.qwen3_tts.stages.create_sglang_tts_engine_executor",
        "vocoder": "sglang_omni.models.qwen3_tts.stages.create_vocoder_executor",
    }
    for stage_name, factory_path in factory_paths.items():
        assert configuration.stage_named(stage_name).factory_path == factory_path


@pytest.mark.parametrize(
    "profile, capabilities, default_mode, model_stages",
    [
        (
            "launch.yaml",
            {"generate_image", "respond"},
            "plan",
            {"understanding", "generation"},
        ),
        (
            "launch_speech_in.yaml",
            {"generate_image", "respond", "transcribe"},
            "iterative",
            {"understanding", "generation", "asr"},
        ),
        (
            "launch_edit.yaml",
            {"edit_image", "respond"},
            "iterative",
            {"understanding", "edit"},
        ),
    ],
)
def test_profiles_without_speech_output_run_both_modes(
    profile: str, capabilities: set[str], default_mode: str, model_stages: set[str]
) -> None:
    """These 4090 profiles declare no TTS stage; loading checks the iterative budget."""
    configuration = ConfigManager.from_file(f"examples/qwen_umm/{profile}").config
    adapter = create_umm_controller(
        configuration.model_path,
        output_stage="output",
        **resolve_stage_typed_kwargs(configuration.stage_named("orchestrator")),
    ).adapter
    assert isinstance(adapter, UMMCompositionAdapter)
    assert set(adapter.composition.capabilities) == capabilities
    assert adapter.composition.execution_mode == default_mode
    assert set(adapter.composition.allowed_modes) == {"plan", "iterative"}
    assert {stage.name for stage in configuration.stages} == {
        "orchestrator",
        "output",
        *model_stages,
    }
    if isinstance(configuration, QwenSpeechInPipelineConfig):
        assert configuration.stage_factory_kwargs("asr") == {
            "model_path": configuration.asr_model_path
        }
    else:
        pass
    if "edit" in model_stages:
        # The edit-only profile loads Qwen-Image-Edit from model_path.
        assert configuration.stage_factory_kwargs("edit") == {"inline_media": True}
    else:
        pass


def test_reasoning_profile_routes_planning_separately_from_visual_tasks() -> None:
    configuration = ConfigManager.from_file(
        "examples/qwen_umm/launch_sensenova_speech_reasoning.yaml"
    ).config
    controller = create_umm_controller(
        configuration.model_path,
        output_stage="output",
        **resolve_stage_typed_kwargs(configuration.stage_named("orchestrator")),
    )
    composition = controller.adapter.composition
    assert composition.phases["plan"] == "planning"
    assert composition.capabilities["respond"].stages == ["understanding"]
    assert configuration.stage_factory_kwargs("planning")["model_path"] == (
        "Qwen/Qwen3-4B"
    )
    assert configuration.stage_factory_kwargs("understanding")["model_path"] == (
        "Qwen/Qwen3-VL-4B-Instruct"
    )
    assert (
        resolve_stage_typed_kwargs(configuration.stage_named("planning"))[
            "server_args_overrides"
        ]["reasoning_parser"]
        == "qwen3"
    )
    assert composition.planning_temperature == 0.0
    assert composition.planning_max_new_tokens == 1536
    assert composition.audio_instruction_capability == "transcribe"


def test_controller_budget_below_the_iterative_requirement_is_rejected() -> None:
    configuration = QwenAnyToAnyPipelineConfig(model_path="model")
    controller = configuration.stage_named("orchestrator")
    factory = resolve_stage_typed_kwargs(controller)
    starved = controller.model_copy(
        update={
            "factory": FactoryArgs(
                **{**factory, "max_dispatches": factory["max_dispatches"] - 1}
            )
        }
    )
    stages = [
        starved if stage.name == "orchestrator" else stage
        for stage in configuration.stages
    ]
    with pytest.raises(ValueError, match="max_dispatches must cover"):
        QwenAnyToAnyPipelineConfig(model_path="model", stages=stages)


@pytest.mark.parametrize(
    "profile", ["launch_speech_plan.yaml", "launch_sensenova_speech_plan.yaml"]
)
def test_speech_plan_profile_reserves_transcription_before_one_plan(
    profile: str,
) -> None:
    configuration = ConfigManager.from_file(f"examples/qwen_umm/{profile}").config
    controller = create_umm_controller(
        configuration.model_path,
        output_stage="output",
        **resolve_stage_typed_kwargs(configuration.stage_named("orchestrator")),
    )
    composition = controller.adapter.composition
    assert composition.allowed_modes == ["plan"]
    assert composition.execution_mode == "plan"
    assert composition.max_plan_attempts == 1
    assert composition.max_concurrent_steps == 1
    assert (
        composition.capabilities[composition.audio_instruction_capability].protocol
        == "asr"
    )
    assert composition.required_dispatches() == composition.max_steps + 2
    assert (
        configuration.stage_factory_kwargs("asr")["model_path"] == "Qwen/Qwen3-ASR-0.6B"
    )
    if profile == "launch_sensenova_speech_plan.yaml":
        generation = configuration.stage_named("generation")
        assert generation.factory_path == (
            "sglang_omni.models.sensenova_u1.stages.create_generation_executor"
        )
        assert configuration.model_path == "SenseNova/SenseNova-U1.5-8B-MoT"
        assert "inline_media" not in configuration.stage_factory_kwargs("generation")
        assert resolve_stage_typed_kwargs(generation)["enable_cache_dit"] is True
        assert composition.capabilities["generate_image"].protocol == "image_base64"
    else:
        assert configuration.stage_factory_kwargs("generation")["inline_media"] is True
