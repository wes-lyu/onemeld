# SPDX-License-Identifier: Apache-2.0
"""Native CPU normalization and accelerator framework dispatch."""

from typing import Literal
from unittest.mock import Mock, patch

import pytest
import sglang.srt.layers.layernorm as layernorm_module
import torch
from sglang.kernels.fused_op import KernelBackend
from sglang.srt.layers.layernorm import RMSNorm

from sglang_omni.models.sensenova_u1.neo_unify.qwen_feed_forward import (
    make_qwen3_rms_norm,
)
from sglang_omni.platforms import current_platform as model_platform


def test_shared_rmsnorm_uses_framework_dispatch():
    norm = make_qwen3_rms_norm(64, eps=1e-6)

    assert isinstance(norm, RMSNorm)
    assert norm.cast_x_before_out_mul
    assert (
        norm._forward_method
        != norm.forward_native  # noqa: leading-underscore -- SGLang CustomOp dispatch field
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [32, 4])
@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize(
    "backend_method", ["forward_cuda", "forward_npu", "forward_aiter", "forward_hip"]
)
def test_shared_rmsnorm_cpu_input_uses_native_path_on_amx(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    hidden_size: int,
    ndim: int,
    has_residual: bool,
    backend_method: Literal[
        "forward_cuda", "forward_npu", "forward_aiter", "forward_hip"
    ],
) -> None:
    """AMX dispatch rejected FP32 and the model's 4D query/key norm inputs."""
    norm = make_qwen3_rms_norm(hidden_size, eps=1e-6).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, hidden_size, dtype=dtype))
    shape = (2, 3, hidden_size * 2) if ndim == 3 else (2, 3, 2, hidden_size * 2)
    hidden_states = torch.randn(shape, dtype=dtype)[..., hidden_size:]
    if has_residual:
        residual = torch.randn_like(hidden_states)
        post_residual_addition = torch.randn_like(hidden_states)
    else:
        residual = None
        post_residual_addition = None
    expected = norm.forward_native(hidden_states, residual, post_residual_addition)
    accelerator_forward = Mock(
        side_effect=AssertionError("Accelerator RMSNorm must not receive CPU inputs")
    )
    monkeypatch.setattr(RMSNorm, backend_method, accelerator_forward)
    monkeypatch.setattr(layernorm_module, "_is_cpu_amx_available", True)
    cpu_kernel = Mock(
        side_effect=AssertionError("AMX RMSNorm must not receive these CPU inputs")
    )
    monkeypatch.setattr(torch.ops.sgl_kernel, "rmsnorm_cpu", cpu_kernel, raising=False)
    monkeypatch.setattr(
        torch.ops.sgl_kernel, "fused_add_rmsnorm_cpu", cpu_kernel, raising=False
    )
    norm._forward_method = (
        getattr(  # noqa: leading-underscore -- SGLang CustomOp dispatch field
            norm, backend_method
        )
    )

    actual = norm(hidden_states, residual, post_residual_addition)

    torch.testing.assert_close(actual, expected)
    accelerator_forward.assert_not_called()
    cpu_kernel.assert_not_called()


@pytest.mark.parametrize("backend", [None, KernelBackend.TORCH])
def test_shared_rmsnorm_non_cpu_preserves_framework_dispatch(
    backend: KernelBackend | None,
) -> None:
    """The CPU guard must preserve framework dispatch and explicit backend selection."""
    norm = make_qwen3_rms_norm(32, eps=1e-6)
    hidden_states = torch.empty(2, 32, device="meta")
    residual = torch.empty_like(hidden_states)
    post_residual_addition = torch.empty_like(hidden_states)
    quant_linear = torch.nn.Identity()
    with patch.object(
        RMSNorm, "forward", autospec=True, return_value=hidden_states
    ) as forward:
        actual = norm(
            hidden_states,
            residual,
            post_residual_addition,
            quant_linear,
            backend=backend,
        )
    assert actual is hidden_states
    forward.assert_called_once_with(
        norm,
        hidden_states,
        residual,
        post_residual_addition,
        quant_linear,
        backend=backend,
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ndim", [3, 4])
def test_shared_rmsnorm_accelerator_input_preserves_framework_kernel(
    dtype: torch.dtype,
    ndim: int,
) -> None:
    """The CPU guard must retain CUDA dispatch for the model's 3D and 4D norms."""
    device_type = model_platform.device_type
    if device_type != "cuda":
        pytest.skip("requires CUDA")
    else:
        pass
    norm = make_qwen3_rms_norm(32, eps=1e-6).to(device=device_type, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 32, device=device_type, dtype=dtype))
    shape = (2, 3, 64) if ndim == 3 else (2, 3, 2, 64)
    hidden_states = torch.randn(shape, device=device_type, dtype=dtype)[..., 32:]
    expected = norm.forward_native(hidden_states)
    backend_method = "forward_cuda"

    with patch.object(
        RMSNorm,
        backend_method,
        autospec=True,
        side_effect=getattr(RMSNorm, backend_method),
    ) as accelerator_forward:
        actual = norm(hidden_states)

    torch.testing.assert_close(actual, expected)
    accelerator_forward.assert_called_once_with(norm, hidden_states, None, None, None)
