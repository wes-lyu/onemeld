# SPDX-License-Identifier: Apache-2.0
"""Serving profiles: backend bindings, task contracts and framework extensibility."""

from pathlib import Path
from typing import Protocol

import pytest
import yaml
from pydantic import ValidationError

from onemeld.config import ServeConfig, load_config

EXAMPLES = Path(__file__).parents[2] / "examples" / "onemeld"
Profile = dict[str, dict]


class ProfileChange(Protocol):
    def __call__(self, values: Profile) -> None: ...


def profile() -> Profile:
    return yaml.safe_load((EXAMPLES / "profile-sglang-48gb.yaml").read_text())


@pytest.mark.parametrize(
    "name", ["profile-sglang-48gb.yaml", "profile-sglang-omni-asr-48gb.yaml"]
)
def test_example_profiles_load(name: str) -> None:
    config = load_config(EXAMPLES / name)
    assert list(config.backends) == ["understanding", "asr", "generation"]
    assert all(name == spec.name for name, spec in config.backends.items())
    assert config.request_limits.max_dispatches == 9


def test_a_new_framework_plugs_in_through_its_adapter_path() -> None:
    """A framework without a built-in adapter is served by naming its class."""
    values = profile()
    values["backends"]["understanding"] = {
        "framework": "vllm",
        "task": "chat",
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "adapter": "example_vllm_backend:VLLMChatBackend",
        "python": "/opt/vllm/bin/python",
    }
    config = ServeConfig.model_validate(values)
    spec = config.backends["understanding"]
    assert spec.adapter_path == "example_vllm_backend:VLLMChatBackend"
    assert spec.python == "/opt/vllm/bin/python"


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda values: values["backends"].pop("asr"), "unconfigured backend 'asr'"),
        (
            lambda values: values["backends"]["generation"].update(task="chat"),
            "needs a image backend, but 'generation' is a chat backend",
        ),
        (
            lambda values: values["backends"]["understanding"].update(
                task="transcription"
            ),
            "must be a chat backend",
        ),
        (
            lambda values: values["backends"]["asr"].update(framework="vllm"),
            "'vllm' has no transcription adapter",
        ),
    ],
    ids=[
        "missing-backend",
        "task-mismatch",
        "planner-not-chat",
        "no-adapter",
    ],
)
def test_inconsistent_profiles_fail_validation(
    change: ProfileChange, message: str
) -> None:
    values = profile()
    change(values)
    with pytest.raises(ValidationError, match=message):
        ServeConfig.model_validate(values)
