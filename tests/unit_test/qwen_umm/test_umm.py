# SPDX-License-Identifier: Apache-2.0
"""The Qwen example binds independent checkpoints to the public UMM executor."""

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.pipeline.umm import create_umm_controller
from sglang_omni.pipeline.umm_composition import UMMCompositionAdapter


def test_qwen_config_uses_public_composition_with_native_stages() -> None:
    configuration = ConfigManager.from_file("examples/qwen_umm/launch.yaml").config
    controller = create_umm_controller(
        configuration.model_path,
        output_stage="output",
        **resolve_stage_typed_kwargs(configuration.stage_named("orchestrator")),
    )
    assert isinstance(controller.adapter, UMMCompositionAdapter)
    assert (
        configuration.stage_factory_kwargs("understanding")["model_path"]
        == configuration.text_model_path
    )
    assert configuration.stage_factory_kwargs("generation")["inline_media"] is True
    assert (
        configuration.stage_named("generation").factory_path
        == "sglang_omni.native_generation.stage.create_generation_scheduler"
    )
    assert (
        configuration.stage_named("understanding").factory_path
        == "sglang_omni.scheduling.native_chat_scheduler.create_native_chat_scheduler"
    )
