# SPDX-License-Identifier: Apache-2.0
"""Public capability bindings, plans and planner turns for composed UMM services."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sglang_omni.client.image_media import InlineMedia, MediaKind

COMPOSITION_ADAPTER_PATH = "sglang_omni.pipeline.umm_composition.UMMCompositionAdapter"
PLAN_PHASE = "plan"
INPUT_IMAGE_ID = "input_image"
USER_ARTIFACT_PREFIX = "user_"
ArtifactKind = Literal["text", "image", "video", "audio"]
CapabilityProtocol = Literal[
    "native_chat", "native_image", "native_video", "asr", "tts"
]
ExecutionMode = Literal["plan", "iterative"]
ContentPart = dict[str, str | dict[str, str]]


@dataclass(frozen=True, kw_only=True)
class ProtocolContract:
    """What a native stage protocol accepts and produces."""

    input_kinds: frozenset[ArtifactKind]
    output_kind: ArtifactKind
    runtime_fields: frozenset[str]
    max_media_refs: int
    required_kind: ArtifactKind | None


PROTOCOLS: dict[CapabilityProtocol, ProtocolContract] = {
    "native_chat": ProtocolContract(
        input_kinds=frozenset({"text", "image", "video", "audio"}),
        output_kind="text",
        runtime_fields=frozenset(
            {"messages", "stream", "response_format", "n", "max_new_tokens"}
        ),
        max_media_refs=8,
        required_kind=None,
    ),
    "native_image": ProtocolContract(
        input_kinds=frozenset({"text", "image"}),
        output_kind="image",
        runtime_fields=frozenset(
            {"prompt", "image_path", "video_path", "num_outputs_per_prompt", "stream"}
        ),
        max_media_refs=3,
        required_kind=None,
    ),
    "native_video": ProtocolContract(
        input_kinds=frozenset({"text", "image"}),
        output_kind="video",
        runtime_fields=frozenset(
            {"prompt", "image_path", "video_path", "num_outputs_per_prompt", "stream"}
        ),
        max_media_refs=1,
        required_kind=None,
    ),
    "asr": ProtocolContract(
        input_kinds=frozenset({"audio"}),
        output_kind="text",
        runtime_fields=frozenset({"audio_path", "stream"}),
        max_media_refs=1,
        required_kind="audio",
    ),
    "tts": ProtocolContract(
        input_kinds=frozenset({"text"}),
        output_kind="audio",
        runtime_fields=frozenset({"text", "stream"}),
        max_media_refs=0,
        required_kind=None,
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

    def summary(self, artifact_id: str, text_bytes: int) -> dict[str, JsonValue]:
        """Planner-facing description, with text cut to text_bytes."""
        summary: dict[str, JsonValue] = {
            "id": artifact_id,
            "kind": self.kind,
            "action": self.action or "client input",
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if isinstance(self.value, str):
            encoded_text = self.value.encode("utf-8")
            summary["text"] = encoded_text[:text_bytes].decode("utf-8", errors="ignore")
            summary["truncated"] = self.truncated or len(encoded_text) > text_bytes
        else:
            pass
        return summary


class UMMCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    stages: list[str] = Field(min_length=1)
    protocol: CapabilityProtocol
    description: str = Field(min_length=1)
    input_kinds: list[ArtifactKind] = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    speech: dict[str, JsonValue] = Field(default_factory=dict)

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
        elif self.speech and self.protocol != "tts":
            raise ValueError("Only tts capabilities take speech parameters")
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


class UMMFinal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    publish: list[str] = Field(max_length=16)
    answer: str = Field(max_length=4096)


class UMMPlannerTurn(BaseModel):
    """One iterative planner turn; final None asks to observe the steps' results."""

    model_config = ConfigDict(extra="forbid", strict=True)

    steps: list[UMMStepCall] = Field(max_length=64)
    final: UMMFinal | None

    @model_validator(mode="after")
    def validate_turn(self) -> "UMMPlannerTurn":
        if not self.steps and self.final is None:
            raise ValueError("A planner turn must run steps or return final")
        elif self.steps and self.final is not None and self.final.answer:
            raise ValueError(
                "A final answer can only follow observed results; return it in a later turn"
            )
        else:
            return self


def turn_from_plan(plan: UMMExecutionPlan) -> UMMPlannerTurn:
    """The equivalent single turn of an upfront plan."""
    return UMMPlannerTurn(
        steps=[
            UMMStepCall.model_validate(step.model_dump(exclude={"publish"}))
            for step in plan.steps
        ],
        final=UMMFinal(
            publish=[step.id for step in plan.steps if step.publish], answer=""
        ),
    )


class UMMPlanExample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1, max_length=4096)
    plan: UMMExecutionPlan


class UMMRequestOptions(BaseModel):
    """Client controls of one composition request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mode: ExecutionMode | None = None
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
    plan_examples: list[UMMPlanExample] = Field(default_factory=list, max_length=4)
    execution_mode: ExecutionMode = "plan"
    allowed_modes: list[ExecutionMode] = Field(
        default_factory=lambda: ["plan"], min_length=1
    )
    max_planner_turns: int = Field(default=4, ge=1, le=16)
    max_steps: int = Field(default=6, ge=1, le=64)
    max_plan_attempts: int = Field(default=1, ge=1, le=3)
    max_images: int = Field(default=4, ge=0, le=16)
    max_videos: int = Field(default=1, ge=0, le=4)
    max_audios: int = Field(default=2, ge=0, le=8)
    max_input_media: int = Field(default=4, ge=0, le=16)
    max_inline_media_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    observation_text_bytes: int = Field(default=2048, ge=64)
    planner_media_kinds: list[MediaKind] = Field(default_factory=lambda: ["image"])
    planning_max_new_tokens: int = Field(default=1536, ge=256)
    response_max_new_tokens: int = Field(default=1024, ge=1)
    trace: bool = False

    @property
    def phases(self) -> dict[str, str]:
        """The plan phase, and one phase per capability stage, named after it."""
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
            "video": self.max_videos,
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
        elif self.execution_mode not in self.allowed_modes:
            raise ValueError("execution_mode must be one of allowed_modes")
        else:
            pass
        for example in self.plan_examples:
            turn = turn_from_plan(example.plan)
            self.validate_steps(turn.steps, {}, Counter(), self.max_steps)
            assert turn.final is not None
            self.validate_final(turn.final, {step.id for step in turn.steps})
        return self

    def required_dispatches(self) -> int:
        """Controller dispatches the most expensive allowed mode can use."""
        planner_dispatches = max(
            self.max_plan_attempts
            * (self.max_planner_turns if mode == "iterative" else 1)
            for mode in self.allowed_modes
        )
        return planner_dispatches + self.max_steps

    def required_outputs(self) -> int:
        """Published artifacts, plus an iterative final answer."""
        return self.max_steps + ("iterative" in self.allowed_modes)

    def validate_steps(
        self,
        steps: list[UMMStepCall],
        artifacts: dict[str, ArtifactKind],
        generated: Counter[ArtifactKind],
        remaining_steps: int,
    ) -> None:
        """Check one batch against the artifacts that already exist."""
        kinds = dict(artifacts)
        counts = Counter(generated)
        if len(steps) > remaining_steps:
            raise ValueError(
                f"Composition steps exceed max_steps; {remaining_steps} remain"
            )
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

    def validate_final(self, final: UMMFinal, producible: set[str]) -> None:
        """Final output may publish each generated artifact once, never an input."""
        if len(set(final.publish)) != len(final.publish):
            raise ValueError("Final output publishes an artifact twice")
        elif not set(final.publish) <= producible:
            raise ValueError(
                f"Final output can publish only step outputs; not {sorted(set(final.publish) - producible)}"
            )
        elif not final.publish and not final.answer.strip():
            raise ValueError("Composition must publish at least one output")
        else:
            pass
