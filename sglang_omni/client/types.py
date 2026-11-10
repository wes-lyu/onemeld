# SPDX-License-Identifier: Apache-2.0
"""Client request, streaming, and result types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    """Chat-style message."""

    role: str
    content: object

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "content": self.content}


@dataclass
class SamplingParams:
    """Sampling configuration."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None
    max_new_tokens: int | None = None

    def to_dict(self) -> dict[str, object]:
        # note (ratish): a field the caller left unset stays out, so the stage
        # request builder applies its own model default.
        values: dict[str, object] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
        }
        return {
            **{name: value for name, value in values.items() if value is not None},
            "stop": list(self.stop),
            "stop_token_ids": list(self.stop_token_ids),
        }


@dataclass
class GenerateRequest:
    """Client-level request (API-agnostic)."""

    model: str | None = None

    prompt: str | dict[str, object] | None = None
    prompt_token_ids: list[int] | None = None
    messages: list[Message] | None = None

    sampling: SamplingParams = field(default_factory=SamplingParams)
    stage_sampling: dict[str, SamplingParams] | None = None
    stage_params: dict[str, dict[str, object]] | None = None
    extra_params: dict[str, object] = field(default_factory=dict)
    stream: bool = True
    max_tokens: int | None = None

    output_modalities: list[str] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class UMMSegment:
    """One accepted output of an interleaved (UMM) request, in order.

    data is None only in the final result of a streamed request, which already
    delivered each segment with its data.
    """

    execution_id: str
    segment_index: int
    kind: str
    data: object = None

    @classmethod
    def from_dict(cls, value: object) -> UMMSegment:
        fields = {"type", "execution_id", "segment_index", "kind", "data"}
        if (
            not isinstance(value, dict)
            or value.get("type") != "segment"
            or not set(value) <= fields
            or not isinstance(value.get("execution_id"), str)
            or type(value.get("segment_index")) is not int
            or value["segment_index"] < 0
            or not isinstance(value.get("kind"), str)
        ):
            raise ValueError(f"Invalid UMM segment {value!r:.200}")
        else:
            return cls(
                value["execution_id"],
                value["segment_index"],
                value["kind"],
                value.get("data"),
            )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": "segment",
            "execution_id": self.execution_id,
            "segment_index": self.segment_index,
            "kind": self.kind,
        }
        if self.data is not None:
            result["data"] = self.data
        else:
            pass
        return result


@dataclass
class CompletionResult:
    """The ordered outputs of one completed request."""

    text: str
    finish_reason: str
    segments: list[UMMSegment]


@dataclass
class CompletionStreamChunk:
    """A published segment or the final finish reason."""

    segment: UMMSegment | None = None
    finish_reason: str | None = None


class ClientError(Exception):
    """The client received no completion from the pipeline."""
