# SPDX-License-Identifier: Apache-2.0
"""Public capability bindings, plans and planner turns for composed UMM services."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onemeld.composition.media import InlineMedia, MediaKind

PLAN_PHASE = "plan"
INPUT_IMAGE_ID = "input_image"
USER_ARTIFACT_PREFIX = "user_"
# Leaves room for a numbered suffix within the 32-character step ID limit.
MAX_RENAMED_BASE = 28
ArtifactKind = Literal["text", "image", "audio"]
CapabilityProtocol = Literal["native_chat", "image_base64", "asr"]
ContentPart = dict[str, str | dict[str, str]]


@dataclass(frozen=True, kw_only=True)
class ProtocolContract:
    """What a backend protocol accepts and produces."""

    input_kinds: frozenset[ArtifactKind]
    output_kind: ArtifactKind
    runtime_fields: frozenset[str]
    max_media_refs: int
    required_kind: ArtifactKind | None


PROTOCOLS: dict[CapabilityProtocol, ProtocolContract] = {
    "native_chat": ProtocolContract(
        input_kinds=frozenset({"text", "image"}),
        output_kind="text",
        runtime_fields=frozenset(
            {"messages", "stream", "response_format", "n", "max_new_tokens"}
        ),
        max_media_refs=8,
        required_kind=None,
    ),
    "image_base64": ProtocolContract(
        input_kinds=frozenset({"text"}),
        output_kind="image",
        runtime_fields=frozenset({"prompt", "n", "stream"}),
        max_media_refs=0,
        required_kind=None,
    ),
    "asr": ProtocolContract(
        input_kinds=frozenset({"audio"}),
        output_kind="text",
        runtime_fields=frozenset({"audio_path", "stream"}),
        max_media_refs=1,
        required_kind="audio",
    ),
}


def media_part(media: InlineMedia) -> ContentPart:
    return {"type": f"{media.kind}_url", f"{media.kind}_url": {"url": media.url}}


@dataclass(kw_only=True)
class UMMArtifact:
    """One named result of a request: client input (action None) or a step output."""

    value: str | InlineMedia
    action: str | None
    truncated: bool = False

    @property
    def kind(self) -> ArtifactKind:
        return "text" if isinstance(self.value, str) else self.value.kind

    @property
    def sha256(self) -> str:
        return (
            hashlib.sha256(self.value.encode("utf-8")).hexdigest()
            if isinstance(self.value, str)
            else self.value.sha256
        )

    @property
    def size_bytes(self) -> int:
        return (
            len(self.value.encode("utf-8"))
            if isinstance(self.value, str)
            else self.value.size_bytes
        )


class UMMCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    stages: list[str] = Field(min_length=1)
    protocol: CapabilityProtocol
    description: str = Field(min_length=1)
    input_kinds: list[ArtifactKind] = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def contract(self) -> ProtocolContract:
        return PROTOCOLS[self.protocol]

    @property
    def output_kind(self) -> ArtifactKind:
        return self.contract.output_kind

    @model_validator(mode="after")
    def validate_protocol(self) -> "UMMCapability":
        contract = self.contract
        if len(set(self.stages)) != len(self.stages) or not all(self.stages):
            raise ValueError("Capability stages must be distinct, non-empty names")
        elif not set(self.input_kinds) <= contract.input_kinds:
            raise ValueError(
                f"{self.protocol} accepts only {sorted(contract.input_kinds)} inputs"
            )
        elif (
            contract.required_kind is not None
            and contract.required_kind not in self.input_kinds
        ):
            raise ValueError(f"{self.protocol} requires {contract.required_kind} input")
        elif contract.runtime_fields.intersection(self.parameters):
            raise ValueError(
                "Capability parameters cannot replace runtime-owned inputs "
                f"{sorted(contract.runtime_fields.intersection(self.parameters))}"
            )
        else:
            return self


class UMMStepCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    action: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=2048)
    input_refs: list[str]


class UMMPlanStep(UMMStepCall):
    publish: bool


class UMMExecutionPlan(BaseModel):
    """The plan-mode planner output: every step, decided before any runs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    steps: list[UMMPlanStep] = Field(min_length=1)


@dataclass(frozen=True, kw_only=True)
class NormalizedSteps:
    """A plan's steps and their corrected IDs."""

    steps: list[UMMStepCall]
    renames: list[tuple[str, str]]


def normalize_steps(steps: list[UMMStepCall], existing: set[str]) -> NormalizedSteps:
    """Rename step IDs that repeat or take an input's name; references follow.

    A reference names the latest earlier step of that name in the reply, or
    else an existing artifact, so "generate" twice becomes generate, generate_2.
    """
    taken = existing | {INPUT_IMAGE_ID}
    latest: dict[str, str] = {}
    renames: list[tuple[str, str]] = []
    normalized: list[UMMStepCall] = []
    for step in steps:
        input_refs = [latest.get(reference, reference) for reference in step.input_refs]
        step_id = step.id
        if step_id in taken or step_id.startswith(USER_ARTIFACT_PREFIX):
            base = step_id.removeprefix(USER_ARTIFACT_PREFIX)
            base = (base if base[:1].isalpha() else f"step_{base}")[:MAX_RENAMED_BASE]
            step_id = base
            number = 2
            while step_id in taken or step_id.startswith(USER_ARTIFACT_PREFIX):
                step_id = f"{base}_{number}"
                number += 1
            renames.append((step.id, step_id))
        else:
            pass
        taken.add(step_id)
        latest[step.id] = step_id
        normalized.append(
            step.model_copy(update={"id": step_id, "input_refs": input_refs})
        )
    return NormalizedSteps(steps=normalized, renames=renames)


def plan_calls(plan: UMMExecutionPlan) -> list[UMMStepCall]:
    return [
        UMMStepCall.model_validate(step.model_dump(exclude={"publish"}))
        for step in plan.steps
    ]


class UMMRequestOptions(BaseModel):
    """Client controls of one composition request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mode: Literal["plan"] | None = None
    image_base64: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)
    planning_max_new_tokens: int | None = Field(default=None, ge=256)
    width: int | None = Field(default=None, ge=16, multiple_of=16)
    height: int | None = Field(default=None, ge=16, multiple_of=16)
    steps: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


class UMMCompositionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    planner_stage: str = Field(min_length=1)
    capabilities: dict[str, UMMCapability] = Field(min_length=1)
    max_steps: int = Field(default=6, ge=1, le=64)
    max_plan_attempts: int = Field(default=1, ge=1, le=3)
    max_images: int = Field(default=4, ge=0, le=16)
    max_audios: int = Field(default=2, ge=0, le=8)
    max_input_media: int = Field(default=4, ge=0, le=16)
    max_inline_media_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    planner_media_kinds: list[MediaKind] = Field(default_factory=lambda: ["image"])
    planning_max_new_tokens: int = Field(default=1536, ge=256)
    planning_temperature: float = Field(default=0.0, ge=0.0)
    planning_chat_template_kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    response_max_new_tokens: int = Field(default=1024, ge=1)
    trace: bool = False
    # Directory for per-request trace files; see onemeld.composition.trace.
    trace_dir: str | None = None
    audio_instruction_capability: str | None = None
    max_concurrent_steps: int | None = Field(default=None, ge=1)

    @property
    def phases(self) -> dict[str, str]:
        """The plan phase, and one phase per capability backend, named after it."""
        return {
            PLAN_PHASE: self.planner_stage,
            **{
                stage: stage
                for capability in self.capabilities.values()
                for stage in capability.stages
            },
        }

    @property
    def media_limits(self) -> dict[ArtifactKind, int]:
        return {
            "image": self.max_images,
            "audio": self.max_audios,
        }

    @model_validator(mode="after")
    def validate_capabilities(self) -> "UMMCompositionSpec":
        stages = {
            stage
            for capability in self.capabilities.values()
            for stage in capability.stages
        }
        if PLAN_PHASE in self.capabilities or PLAN_PHASE in stages:
            raise ValueError("The plan phase name is reserved")
        elif any(
            not name.isidentifier() or name.startswith("_")
            for name in self.capabilities
        ):
            raise ValueError("Capability names must be public identifiers")
        elif self.audio_instruction_capability is not None and (
            self.audio_instruction_capability not in self.capabilities
            or self.capabilities[self.audio_instruction_capability].protocol != "asr"
        ):
            raise ValueError("audio_instruction_capability must name an asr capability")
        else:
            pass
        return self

    def required_dispatches(self) -> int:
        return (
            self.max_plan_attempts
            + self.max_steps
            + (self.audio_instruction_capability is not None)
        )

    def required_outputs(self) -> int:
        return self.max_steps

    def validate_steps(
        self,
        steps: list[UMMStepCall],
        artifacts: dict[str, ArtifactKind],
    ) -> None:
        """Check a plan against its inputs and the request budgets."""
        kinds = dict(artifacts)
        counts: Counter[ArtifactKind] = Counter()
        if len(steps) > self.max_steps:
            raise ValueError(f"Composition steps exceed max_steps={self.max_steps}")
        else:
            pass
        for step in steps:
            if (
                step.id in kinds
                or step.id == INPUT_IMAGE_ID
                or step.id.startswith(USER_ARTIFACT_PREFIX)
            ):
                raise ValueError(f"Duplicate or reserved step ID: {step.id}")
            elif step.action not in self.capabilities:
                raise ValueError(f"Unknown capability: {step.action}")
            elif not step.instruction.strip():
                raise ValueError(f"Step {step.id} has an empty instruction")
            elif not set(step.input_refs).issubset(kinds):
                raise ValueError(
                    f"Step {step.id} references unavailable artifacts {sorted(set(step.input_refs) - kinds.keys())}; available IDs: {sorted(kinds)}. Never reference the current step."
                )
            else:
                capability = self.capabilities[step.action]
            referenced = [kinds[reference] for reference in step.input_refs]
            media_count = sum(kind != "text" for kind in referenced)
            required_kind = capability.contract.required_kind
            if any(kind not in capability.input_kinds for kind in referenced):
                raise ValueError(
                    f"Step {step.id} cannot consume the referenced artifact kind"
                )
            elif media_count > capability.contract.max_media_refs:
                raise ValueError(
                    f"Step {step.id} references more than {capability.contract.max_media_refs} media artifacts"
                )
            elif required_kind is not None and referenced.count(required_kind) != 1:
                raise ValueError(
                    f"Step {step.id} needs exactly one {required_kind} input"
                )
            else:
                kinds[step.id] = capability.output_kind
                counts[capability.output_kind] += 1
        for kind, limit in self.media_limits.items():
            if counts[kind] > limit:
                raise ValueError(f"Composition exceeds max_{kind}s={limit}")
            else:
                pass
