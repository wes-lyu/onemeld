# SPDX-License-Identifier: Apache-2.0
"""The Qwen examples bind independent checkpoints to the public UMM executor."""

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.schema import FactoryArgs
from sglang_omni.models.qwen_umm.config import (
    QwenAnyToAnyEditPipelineConfig,
    QwenAnyToAnyPipelineConfig,
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


def test_text_to_image_config_keeps_its_two_capabilities() -> None:
    configuration = ConfigManager.from_file("examples/qwen_umm/launch.yaml").config
    adapter = create_umm_controller(
        configuration.model_path,
        output_stage="output",
        **resolve_stage_typed_kwargs(configuration.stage_named("orchestrator")),
    ).adapter
    assert isinstance(adapter, UMMCompositionAdapter)
    assert set(adapter.composition.capabilities) == {"generate_image", "respond"}


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
