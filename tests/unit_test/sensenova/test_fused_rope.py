# SPDX-License-Identifier: Apache-2.0
"""CUDA RoPE dispatch preserves eager results."""


import pytest
import torch

from sglang_omni.models.sensenova_u1.neo_unify import rotary_embeddings


def eager(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (
        q * cos + rotary_embeddings.rotate_half(q) * sin,
        k * cos + rotary_embeddings.rotate_half(k) * sin,
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("axis", ["t", "h", "w"])
@torch.no_grad()
def test_cuda_bf16_rope_matches_eager(axis, monkeypatch):
    fused = rotary_embeddings.fused_rope_rotate_half_bitexact
    calls = []

    def count_fused_calls(*args, **kwargs):
        calls.append(None)
        return fused(*args, **kwargs)

    monkeypatch.setattr(
        rotary_embeddings, "fused_rope_rotate_half_bitexact", count_fused_calls
    )
    generator = torch.Generator(device="cuda").manual_seed(2026)
    q = torch.randn(
        1, 129, 32, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    k = torch.randn(
        1, 129, 8, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    if axis != "t":
        start = 0 if axis == "h" else 32
        q, k = q[..., start : start + 32], k[..., start : start + 32]
    angles = torch.randn(1, 129, q.shape[-1], device="cuda", generator=generator)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    q, k = q.transpose(1, 2), k.transpose(1, 2)

    actual = rotary_embeddings.apply_rotary_pos_emb(q, k, cos, sin)
    for result, expected in zip(actual, eager(q, k, cos, sin)):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert len(calls) == 2
