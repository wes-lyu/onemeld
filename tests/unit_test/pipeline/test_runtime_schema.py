# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.config.runtime import (
    apply_typed_stage_kwargs,
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.config.schema import FactoryDefault
from tests.unit_test.fixtures.pipeline_fakes import dummy_factory

FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def make_stage(**kwargs) -> StageConfig:
    data = {
        "name": "stage",
        "process": "pipeline",
        "factory_path": FACTORY,
        "terminal": True,
    }
    data.update(kwargs)
    cls = kwargs.pop("cls", None)
    if cls is not None:
        data.pop("cls", None)
        return cls(**data)
    return StageConfig(**data)


def engine_stage(**kwargs) -> EngineStageConfig:
    data = {
        "name": "stage",
        "process": "pipeline",
        "factory_path": FACTORY,
        "terminal": True,
    }
    data.update(kwargs)
    return EngineStageConfig(**data)


def test_invalid_gpu_memory_fraction_raises() -> None:
    with pytest.raises(ValueError, match="gpu_memory_fraction"):
        make_stage(gpu_memory_fraction=0.0)


def test_invalid_engine_mem_fraction_static_raises() -> None:
    with pytest.raises(ValueError, match="mem_fraction_static"):
        EngineArgs(mem_fraction_static=1.0)


def test_invalid_model_group_values_raise() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        FactoryArgs(max_seq_len=0)
    with pytest.raises(ValueError, match="video_fps"):
        FactoryArgs(video_fps=-1.0)


def test_prefill_coalesce_range_is_enforced_at_validation() -> None:
    assert FactoryArgs(prefill_coalesce_requests=32).prefill_coalesce_requests == 32
    assert FactoryArgs(prefill_coalesce_wait_ms=300).prefill_coalesce_wait_ms == 300.0

    with pytest.raises(ValueError, match="prefill_coalesce_requests"):
        FactoryArgs(prefill_coalesce_requests=-1)
    with pytest.raises(ValueError, match="prefill_coalesce_wait_ms"):
        FactoryArgs(prefill_coalesce_wait_ms=0.0)


def test_prefill_coalesce_requests_of_one_warns(caplog) -> None:
    with caplog.at_level("WARNING", logger="sglang_omni.config.schema"):
        FactoryArgs(prefill_coalesce_requests=1)
    assert "disables coalescing" in caplog.text


def test_the_engine_block_is_refused_off_engine_stages() -> None:
    with pytest.raises(ValueError, match="not an engine stage"):
        make_stage(engine={"mem_fraction_static": 0.7})


def test_undeclared_group_keys_pass_through_to_the_consumer() -> None:
    """The group vocabularies belong to the modules that read them; the
    schema keeps unknown keys instead of guessing at their legality."""
    stage = engine_stage(
        engine={"disable_radix_cache": True},
        factory={"made_up_knob": 7, "lookahead": 9},
    )

    assert stage.engine.overrides()["disable_radix_cache"] is True
    assert stage.factory.model_extra["made_up_knob"] == 7
    assert stage.factory.model_extra["lookahead"] == 9


def test_stage_rejects_terminal_with_next() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                make_stage(name="source", next="sink", terminal=True),
                make_stage(name="sink"),
            ],
        )


def test_tp_size_below_one_raises() -> None:
    with pytest.raises(ValueError, match="tp_size"):
        make_stage(tp_size=0)


def test_invalid_placement_limit_raises() -> None:
    with pytest.raises(ValueError, match="max_total_gpu_memory_fraction_per_gpu"):
        PlacementConfig(max_total_gpu_memory_fraction_per_gpu=1.1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2 * 1024**3, 2 * 1024**3),
        ("512KiB", 512 * 1024),
        ("128MiB", 128 * 1024**2),
        ("2GiB", 2 * 1024**3),
        ("1TiB", 1024**4),
    ],
)
def test_engine_kv_cache_bytes_parses_sizes(value, expected) -> None:
    from sglang_omni.config import EngineArgs

    assert EngineArgs(kv_cache_bytes=value).kv_cache_bytes == expected


@pytest.mark.parametrize("value", [0, True, "2GB", "1.5GiB", "GiB", " 2GiB"])
def test_engine_kv_cache_bytes_rejects_bad_values(value) -> None:
    from sglang_omni.config import EngineArgs

    with pytest.raises(ValueError):
        EngineArgs(kv_cache_bytes=value)


def test_engine_kv_cache_bytes_rejects_mem_fraction_static() -> None:
    from sglang_omni.config import EngineArgs

    with pytest.raises(ValueError, match="keep exactly one|cannot be set together"):
        EngineArgs(kv_cache_bytes="2GiB", mem_fraction_static=0.7)


def test_stage_total_reserve_rejects_gpu_memory_fraction() -> None:
    from sglang_omni.config import StageConfig

    with pytest.raises(ValueError, match="same budget in two units"):
        StageConfig(
            name="a",
            factory_path="x.y",
            terminal=True,
            gpu_memory_fraction=0.5,
            total_reserve_bytes="8GiB",
        )


def test_stage_rejects_kv_above_total_reserve() -> None:
    from sglang_omni.config import EngineArgs, EngineStageConfig

    with pytest.raises(ValueError, match="must not exceed"):
        EngineStageConfig(
            name="a",
            factory_path="x.y",
            terminal=True,
            total_reserve_bytes="2GiB",
            engine=EngineArgs(kv_cache_bytes="4GiB"),
        )


def test_engine_kv_cache_bytes_rejects_max_total_tokens() -> None:
    from sglang_omni.config import EngineArgs

    with pytest.raises(ValueError, match="cannot be set together"):
        EngineArgs(kv_cache_bytes="2GiB", max_total_tokens=4096)


class DecodeFactoryArgs(FactoryArgs):
    dtype: Annotated[str | None, FactoryDefault(value="bfloat16")] = None
    enable_async_decode: Annotated[bool | None, FactoryDefault(value=False)] = None
    max_batch_wait_ms: Annotated[float | None, FactoryDefault(value=10.0)] = None


class DecodeStageConfig(StageConfig):
    factory: DecodeFactoryArgs = Field(default_factory=DecodeFactoryArgs)


class DecodePipelineConfig(PipelineConfig):
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "decode": DecodeStageConfig,
    }

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, str | float | bool]:
        assert stage_name == "decode"
        return {
            "dtype": "float16",
            "enable_async_decode": True,
            "max_batch_wait_ms": 5.0,
        }


@pytest.mark.parametrize("round_trip", [False, True])
@pytest.mark.parametrize("replica", [False, True])
@pytest.mark.parametrize("selection", ["omitted", "none", "explicit"])
def test_factory_override_precedence(
    selection: Literal["omitted", "none", "explicit"],
    replica: bool,
    round_trip: bool,
) -> None:
    explicit_values = {
        "dtype": "float32",
        "enable_async_decode": False,
        "max_batch_wait_ms": 0.0,
    }
    if selection == "omitted":
        factory = DecodeFactoryArgs()
    elif selection == "none":
        factory = DecodeFactoryArgs.model_validate(dict.fromkeys(explicit_values))
    else:
        factory = DecodeFactoryArgs.model_validate(explicit_values)
    config = DecodePipelineConfig(
        model_path="unused",
        entry_stage="decode",
        stages=[
            DecodeStageConfig(
                name="decode",
                process="decode",
                factory_path=FACTORY,
                factory=factory,
                terminal=True,
            )
        ],
    )
    if round_trip:
        config = DecodePipelineConfig.model_validate(config.model_dump())
    else:
        pass
    stage = config.stage_named("decode")
    if replica:
        stage = stage.model_copy(update={"name": "decode@r0"})
    else:
        pass
    resolved = apply_typed_stage_kwargs(
        dummy_factory,
        resolve_stage_factory_kwargs(stage, config),
        resolve_stage_typed_kwargs(stage),
        stage_name=stage.name,
    )
    expected = (
        explicit_values
        if selection == "explicit"
        else config.stage_factory_kwargs("decode")
    )
    assert resolved == expected


@pytest.mark.parametrize("use_none", [False, True])
def test_factory_fallbacks_without_author_overrides(use_none: bool) -> None:
    expected = {
        "dtype": "bfloat16",
        "enable_async_decode": False,
        "max_batch_wait_ms": 10.0,
    }
    factory = (
        DecodeFactoryArgs.model_validate(dict.fromkeys(expected))
        if use_none
        else DecodeFactoryArgs()
    )
    stage = DecodeStageConfig(
        name="decode",
        process="decode",
        factory_path=FACTORY,
        factory=factory,
        terminal=True,
    )
    config = PipelineConfig(model_path="unused", entry_stage="decode", stages=[stage])
    resolved = apply_typed_stage_kwargs(
        dummy_factory,
        resolve_stage_factory_kwargs(stage, config),
        resolve_stage_typed_kwargs(stage),
        stage_name=stage.name,
    )
    assert resolved == expected
