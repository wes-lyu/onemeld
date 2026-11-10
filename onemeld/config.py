# SPDX-License-Identifier: Apache-2.0
"""One serving profile: the model backends, the capabilities they offer, and limits."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onemeld.backends.base import PROTOCOL_TASKS, BackendSpec
from onemeld.composition.plan import UMMCompositionSpec
from onemeld.composition.types import UMMLimits


class LimitsConfig(BaseModel):
    """Request bounds. Dispatch and output bounds default to what the composition needs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    max_active: int = Field(default=4, ge=1)
    timeout_s: float = Field(default=1800.0, gt=0)
    max_retained_bytes: int = Field(default=256 * 1024 * 1024, ge=1)
    max_dispatches: int | None = Field(default=None, ge=1)
    max_outputs: int | None = Field(default=None, ge=1)


class ServeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_name: str = Field(default="onemeld", min_length=1)
    backends: dict[str, BackendSpec] = Field(min_length=1)
    composition: UMMCompositionSpec
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @model_validator(mode="before")
    @classmethod
    def name_backends(cls, values: JsonValue) -> JsonValue:
        """Each backend is keyed by its name, which its spec also carries."""
        if isinstance(values, dict) and isinstance(values.get("backends"), dict):
            backends = values["backends"]
            values = {
                **values,
                "backends": {
                    name: {**spec, "name": name} if isinstance(spec, dict) else spec
                    for name, spec in backends.items()
                },
            }
        else:
            pass
        return values

    @model_validator(mode="after")
    def validate_bindings(self) -> "ServeConfig":
        planner = self.composition.planner_stage
        if planner not in self.backends:
            raise ValueError(f"The planner backend {planner!r} is not configured")
        elif self.backends[planner].task != "chat":
            raise ValueError(f"The planner backend {planner!r} must be a chat backend")
        else:
            pass
        for action, capability in self.composition.capabilities.items():
            task = PROTOCOL_TASKS[capability.protocol]
            for name in capability.stages:
                if name not in self.backends:
                    raise ValueError(
                        f"Capability {action!r} names unconfigured backend {name!r}"
                    )
                elif self.backends[name].task != task:
                    raise ValueError(
                        f"Capability {action!r} needs a {task} backend, but "
                        f"{name!r} is a {self.backends[name].task} backend"
                    )
                else:
                    pass
        for spec in self.backends.values():
            # Fails here, not at startup, when no adapter serves the spec.
            spec.adapter_path
        return self

    @property
    def request_limits(self) -> UMMLimits:
        limits = self.limits
        return UMMLimits(
            max_dispatches=limits.max_dispatches
            or self.composition.required_dispatches(),
            max_outputs=limits.max_outputs or self.composition.required_outputs(),
            max_active=limits.max_active,
            max_retained_bytes=limits.max_retained_bytes,
            timeout_s=limits.timeout_s,
        )


def load_config(path: str | Path) -> ServeConfig:
    with open(path, encoding="utf-8") as file:
        return ServeConfig.model_validate(yaml.safe_load(file))
