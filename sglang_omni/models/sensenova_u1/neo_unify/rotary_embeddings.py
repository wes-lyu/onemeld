"""Rotary embeddings."""

import copy
from typing import Protocol

import torch
import torch._dynamo
from torch import nn
from transformers import Qwen3Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

# Modified for SGLang; see this directory's README.md for upstream source.

# note (Wes): The bit-exact RoPE kernel ships only with CUDA builds of sglang.
try:
    from sglang.kernels.ops.diffusion.rope.rope_rotate_half_bitexact import (
        fused_rope_rotate_half_bitexact,
    )
except ImportError:
    fused_rope_rotate_half_bitexact = None

CUDA_FUSED_ROPE_MIN_SEQUENCE_LENGTH = 128


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    if (
        fused_rope_rotate_half_bitexact is not None
        and q.is_cuda
        and q.dtype is torch.bfloat16
        and k.dtype is torch.bfloat16
        and cos.dtype is torch.bfloat16
        and sin.dtype is torch.bfloat16
        and q.device == k.device == cos.device == sin.device
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and unsqueeze_dim == 1
        and q.ndim == k.ndim == 4
        and q.numel() > 0
        and k.numel() > 0
        and q.shape[2] >= CUDA_FUSED_ROPE_MIN_SEQUENCE_LENGTH
        and q.shape[0] == k.shape[0]
        and q.shape[2:] == k.shape[2:]
        and q.shape[-1] in (32, 64)
        and cos.shape == sin.shape == (q.shape[0], q.shape[2], q.shape[3])
    ):
        cos_rows = cos.reshape(-1, cos.shape[-1]).contiguous()
        sin_rows = sin.reshape(-1, sin.shape[-1]).contiguous()
        q_embed = fused_rope_rotate_half_bitexact(
            q.transpose(1, 2).contiguous(), cos_rows, sin_rows
        )
        k_embed = fused_rope_rotate_half_bitexact(
            k.transpose(1, 2).contiguous(), cos_rows, sin_rows
        )
        return q_embed.transpose(1, 2), k_embed.transpose(1, 2)
    else:
        pass

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_default_rope_parameters(
    config: Qwen3Config,
    device: torch.device | None = None,
    **_kwargs: int | float | torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    """Default RoPE frequencies, inlined to avoid breakage across transformers versions."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor


class RopeInitializer(Protocol):
    def __call__(
        self, cfg: Qwen3Config, dev: torch.device | None = None
    ) -> tuple[torch.Tensor, float]:
        """Return rotary frequencies and their attention scaling factor."""


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    compute_default_rope_parameters = staticmethod(compute_default_rope_parameters)

    def __init__(self, config: Qwen3Config, device: torch.device | None = None) -> None:
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type: str | None = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached: int = config.max_position_embeddings
        self.original_max_seq_len: int = config.max_position_embeddings

        self.config: Qwen3Config = config
        if self.rope_type == "default" or self.rope_type is None:
            base_rope_init_fn = compute_default_rope_parameters
        else:
            base_rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        def _rope_init_fn_keep_freq_range(
            cfg: Qwen3Config, dev: torch.device | None = None
        ) -> tuple[torch.Tensor, float]:
            inv_freq, attention_scaling = base_rope_init_fn(cfg, dev)

            cfg2 = copy.deepcopy(cfg)
            head_dim = getattr(cfg2, "head_dim", None)
            if head_dim is None:
                head_dim = cfg2.hidden_size // cfg2.num_attention_heads
                cfg2.head_dim = head_dim
            else:
                pass
            cfg2.head_dim = int(head_dim) * 2

            inv_freq_full, _ = base_rope_init_fn(cfg2, dev)
            inv_freq = inv_freq_full[::2]

            return inv_freq, attention_scaling

        self.rope_init_fn: RopeInitializer = _rope_init_fn_keep_freq_range

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq: torch.Tensor = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


RopeTable = tuple[torch.Tensor, torch.Tensor]


PositionEmbeddings = tuple[RopeTable, RopeTable, RopeTable]
