# SPDX-License-Identifier: Apache-2.0
"""Attest that SGLang-backed stages captured the prefill CUDA graphs they declared."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
else:
    pass

logger = logging.getLogger(__name__)


def attest_prefill_cuda_graphs(
    model_runner: ModelRunner, *, operator_selected: bool
) -> None:
    """Assert captured prefill graphs match an explicit or realized policy.

    An operator-selected backend must materialize or fail startup. A model
    default keeps SGLang's auto-selection semantics and may fall back to eager
    at capture-time safety gates such as insufficient free memory.
    """
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    prefill_cfg = get_exec().graph.cuda_graph_config.prefill
    # init_cuda_graphs always assigns this before attestation runs.
    runner = model_runner.prefill_cuda_graph_runner
    if not isinstance(runner, PrefillCudaGraphRunner):
        if not operator_selected:
            logger.warning(
                "auto-selected prefill CUDA graph backend %r did not capture; "
                "using SGLang's eager prefill fallback",
                prefill_cfg.backend,
            )
            return
        else:
            pass
        raise RuntimeError(
            f"prefill CUDA graph backend {prefill_cfg.backend!r} was declared "
            "but SGLang did not construct PrefillCudaGraphRunner "
            f"(got {type(runner).__name__}); an eligibility gate disabled "
            "prefill graphs during capture"
        )
    else:
        pass
    if runner.prefill_backend_name != prefill_cfg.backend:
        raise RuntimeError(
            "prefill CUDA graph backend mismatch: declared "
            f"{prefill_cfg.backend!r}, captured {runner.prefill_backend_name!r}"
        )
    else:
        pass
    expected_buckets = sorted(int(b) for b in prefill_cfg.bs)
    if list(runner.capture_num_tokens) != expected_buckets:
        raise RuntimeError(
            "prefill CUDA graph capture shapes differ from the declared "
            f"policy: declared={expected_buckets}, "
            f"captured={list(runner.capture_num_tokens)}"
        )
    else:
        pass
    if not runner.buffer_registry.has_slot("input_embeds"):
        raise RuntimeError(
            "prefill CUDA graph runner has no input_embeds slot; "
            "enable_prefill_input_embeds did not reach the model config "
            "before SGLang ModelRunner construction"
        )
    else:
        pass
    logger.info(
        "prefill CUDA graphs attested: backend=%s buckets=%s",
        runner.prefill_backend_name,
        expected_buckets,
    )
