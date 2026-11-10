# SPDX-License-Identifier: Apache-2.0
"""Hugging Face helper utilities."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import torch.nn as nn
from transformers import AutoConfig, PretrainedConfig
from transformers.initialization import no_init_weights
from transformers.utils.hub import cached_file

from sglang_omni.utils.json import JsonValue

MROPE_ONLY_KEYS = frozenset({"interleaved", "mrope_interleaved", "mrope_section"})


def drop_mrope_keys(
    rope_parameters: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """The rope parameters without the keys only MRoPE reads."""
    return {
        key: value
        for key, value in rope_parameters.items()
        if key not in MROPE_ONLY_KEYS
    }


def use_text_rope(config: PretrainedConfig) -> None:
    """Drop the MRoPE-only rope keys so the runtime builds text RoPE."""
    if config.rope_parameters is None:
        pass
    else:
        config.rope_parameters = drop_mrope_keys(config.rope_parameters)


@lru_cache(maxsize=8)
def load_hf_config(
    model_path: str,
    *,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
) -> PretrainedConfig:
    """Load the HF config, preferring the local cache."""
    try:
        config_path = cached_file(
            model_path, "config.json", local_files_only=local_files_only
        )
        cfg = AutoConfig.from_pretrained(
            str(Path(config_path).parent),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception:
        cfg = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
    return cfg


def instantiate_module(
    module_cls: type[nn.Module], config: PretrainedConfig
) -> nn.Module:
    """Instantiate a module without allocating its parameters."""
    with no_init_weights():
        if hasattr(module_cls, "_from_config"):
            return module_cls._from_config(
                config
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        else:
            pass
        return module_cls(config)
