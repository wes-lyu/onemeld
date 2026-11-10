from __future__ import annotations

import logging
import os
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

from sglang_omni.platforms import current_platform
from sglang_omni.quantization import (
    needs_quant_config_normalization,
    normalize_quant_config,
    resolve_quant_config,
)
from sglang_omni.utils.misc import model_config_has_moe

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs
else:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ModelWorkerConfig:
    model_arch_override: str | None = None
    weight_prefix: str | None = None
    nccl_port: int | None = None
    total_gpu_memory_fraction: float | None = None
    kv_cache_bytes: int | None = None
    enable_prefill_input_embeds: bool = False


@dataclass(slots=True)
class PrefillCudaGraphUsage:
    replay_count: int = 0
    standard_eager_count: int = 0
    custom_eager_count: int = 0
    replay_buckets: Counter[int] = field(default_factory=Counter)


class PrefillCudaGraphInfo(TypedDict):
    backend: str
    runner: str | None
    backend_runner: str | None
    capture_num_tokens: list[int] | None
    input_embeds_slot: bool
    replay_count: int
    standard_eager_count: int
    custom_eager_count: int
    replay_buckets: dict[str, int]


_ARCH_CONFIG_MAP: dict[str, tuple[str, str | None]] = {
    "Qwen3ASRForConditionalGeneration": ("thinker_config", "text_config"),
}


class ModelWorker:
    def __init__(
        self,
        config: ModelWorkerConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int = 0,
    ):
        self.server_args = server_args
        self.model_arch_override = config.model_arch_override
        self.weight_prefix = config.weight_prefix
        self.nccl_port = config.nccl_port
        self.total_gpu_memory_fraction = config.total_gpu_memory_fraction
        self.kv_cache_bytes = config.kv_cache_bytes
        self.enable_prefill_input_embeds = config.enable_prefill_input_embeds

        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.init_model_config()
        effective_quantization = self.configure_backend_policy()
        from sglang.srt.distributed import bootstrap
        from sglang.srt.runtime_context import (
            SpawnRanks,
            get_device,
            get_parallel,
            publish,
            spawn_world_rank,
        )
        from sglang.srt.utils import broadcast_pyobj, set_random_seed

        publish(
            self.server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(
                    self.server_args, tp_rank=tp_rank, pp_rank=0
                ),
                gpu_id=gpu_id,
            ),
        )
        if self.nccl_port is None:
            self.nccl_port = resolve_nccl_port()
        else:
            pass
        bootstrap.init_parallel_runtime(
            server_args=self.server_args,
            device=get_device().device,
            dist_port=self.nccl_port,
        )
        bootstrap.init_layer_runtime(model_config=self.model_config)
        initialize_model_worker_backend_globals(
            self.model_config, effective_quantization
        )
        self.init_model_runner()
        self.prefill_cuda_graph_usage = PrefillCudaGraphUsage()

        self.device = self.model_runner.device

        self.random_seed = broadcast_pyobj(
            [get_device().random_seed],
            self.tp_rank,
            get_parallel().tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)

    def init_model_config(self):
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            server_args=self.server_args,
            is_draft_model=False,
        )

        if self.model_arch_override is not None:
            self.apply_arch_override(self.model_config, self.model_arch_override)
        else:
            pass

    @staticmethod
    def apply_arch_override(model_config: ModelConfig, arch: str) -> None:
        """Override model config for a sub-model architecture."""
        model_config.hf_config.architectures = [arch]
        entry = _ARCH_CONFIG_MAP.get(arch)
        if entry is None:
            return
        else:
            pass
        sub_config_attr, text_config_attr = entry
        sub_cfg = getattr(model_config.hf_config, sub_config_attr, None)
        if sub_cfg is None:
            return
        else:
            pass
        text_cfg = getattr(sub_cfg, text_config_attr) if text_config_attr else sub_cfg
        model_config.hf_text_config = text_cfg
        model_config.num_attention_heads = text_cfg.num_attention_heads
        model_config.num_key_value_heads = text_cfg.num_key_value_heads
        model_config.hidden_size = text_cfg.hidden_size
        model_config.num_hidden_layers = text_cfg.num_hidden_layers
        # note(ratish): SGLang sizes the KV pool from the larger of these two
        # and set the second from the root text config at construction.
        model_config.num_attention_layers = text_cfg.num_hidden_layers

    def configure_backend_policy(self) -> str | None:
        # Apply Omni-specific quantization adapters (stage-local checkpoint name
        # normalization) before SGLang builds its quant config, then run the
        # model_worker backend policy.
        apply_omni_quantization_adapters(self.model_config)

        return current_platform.apply_model_worker_backend_policy(
            self.server_args,
            self.model_config,
            self.model_arch_override,
        )

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def get_worker_info(self):
        max_total_num_tokens = self.model_runner.max_total_num_tokens
        effective_max_total_num_tokens = (
            self.model_runner.effective_max_total_num_tokens
        )
        max_req_len = min(
            self.model_config.context_len - 1,
            effective_max_total_num_tokens - 1,
        )
        max_req_input_len = max_req_len - 1
        req_pool = self.model_runner.req_to_token_pool
        kv_pool = self.model_runner.token_to_kv_pool_allocator
        max_running_requests = self.model_runner.max_running_requests
        return (
            max_total_num_tokens,
            self.server_args.max_prefill_tokens,
            max_running_requests,
            self.server_args.max_queued_requests,
            max_req_len,
            max_req_input_len,
            self.random_seed,
            self.device,
            req_pool.size,
            req_pool.max_context_len,
            kv_pool.size,
        )

    def get_tp_group(self):
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().tp_group

    def get_attention_tp_group(self):
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().attn_tp_group

    def get_attention_tp_cpu_group(self):
        return self.get_attention_tp_group().cpu_group

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def init_model_runner(self):
        from .sglang_model_runner import SGLModelRunner

        self.model_runner = SGLModelRunner(
            model_config=self.model_config,
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            nccl_port=self.nccl_port,
            model_arch_override=self.model_arch_override,
            weight_prefix=self.weight_prefix,
            total_gpu_memory_fraction=self.total_gpu_memory_fraction,
            kv_cache_bytes=self.kv_cache_bytes,
        )

    def forward_batch_generation(self, forward_batch):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        out = self.model_runner.forward(forward_batch=forward_batch)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        self.record_prefill_cuda_graph_usage(
            forward_batch,
            can_run_graph=bool(can_run_cuda_graph),
        )
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        return batch_result

    def record_prefill_cuda_graph_usage(
        self,
        forward_batch: ForwardBatch,
        *,
        can_run_graph: bool,
    ) -> None:
        mode = forward_batch.forward_mode
        if not mode.is_extend() or mode.is_cuda_graph():
            return
        else:
            pass

        if not can_run_graph:
            # Note (wenyao): custom eager forwards (visual/deepstack) return
            # before ModelWorker is called; intentionally absent here.
            self.prefill_cuda_graph_usage.standard_eager_count += 1
            return
        else:
            pass

        runner = self.model_runner.prefill_cuda_graph_runner
        buckets = runner.capture_num_tokens
        actual_bucket = buckets[bisect_left(buckets, len(forward_batch.input_ids))]
        self.prefill_cuda_graph_usage.replay_count += 1
        self.prefill_cuda_graph_usage.replay_buckets[int(actual_bucket)] += 1

    def record_custom_prefill_eager(self) -> None:
        """Record a custom prefill forward that bypasses SGLang graph dispatch."""
        self.prefill_cuda_graph_usage.custom_eager_count += 1

    def prefill_cuda_graph_info(self) -> PrefillCudaGraphInfo:
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        runner = self.model_runner.prefill_cuda_graph_runner
        if isinstance(runner, PrefillCudaGraphRunner):
            capture_num_tokens = [int(value) for value in runner.capture_num_tokens]
            backend_runner = type(runner.backend).__name__
            input_embeds_slot = runner.buffer_registry.has_slot("input_embeds")
        else:
            capture_num_tokens, backend_runner, input_embeds_slot = None, None, False
        from sglang.srt.runtime_context import get_exec

        backend = get_exec().graph.cuda_graph_config.prefill.backend
        usage = self.prefill_cuda_graph_usage
        return {
            "backend": backend,
            "runner": type(runner).__name__ if runner is not None else None,
            "backend_runner": backend_runner,
            "capture_num_tokens": capture_num_tokens,
            "input_embeds_slot": input_embeds_slot,
            "replay_count": int(usage.replay_count),
            "standard_eager_count": int(usage.standard_eager_count),
            "custom_eager_count": int(usage.custom_eager_count),
            "replay_buckets": {
                str(bucket): int(count)
                for bucket, count in sorted(usage.replay_buckets.items())
            },
        }


def resolve_nccl_port() -> int:
    master_port = os.environ.get("MASTER_PORT")
    if master_port:
        return int(master_port)
    else:
        pass

    from sglang_omni.utils.port_claim import NCCL_PORT_BASE, claim_tcp_port

    try:
        port = claim_tcp_port()
    except PermissionError:
        # Some restricted CI / sandbox environments do not allow socket binding
        # during test-time configuration. Fall back to a stable default so
        # callers still receive a valid NCCL port choice.
        port = NCCL_PORT_BASE

    os.environ["MASTER_PORT"] = str(port)
    return port


def apply_omni_quantization_adapters(model_config: ModelConfig) -> None:
    """Apply Omni-specific quantization adapters before SGLang builds its config.

    SGLang owns detection, config parsing, layer construction, and post-load
    hooks. The only Omni-specific step needed here is stage-local checkpoint
    name normalization for methods whose per-block quant names are matched
    against runtime module names, currently AutoRound.
    """
    quant_dict = resolve_quant_config(model_config.hf_config)
    if quant_dict is None:
        return
    else:
        pass

    if needs_quant_config_normalization(quant_dict):
        normalize_quant_config(model_config)
    else:
        pass


def initialize_model_worker_backend_globals(
    model_config: ModelConfig,
    effective_quantization: str | None,
) -> None:
    """Initialize backend globals needed by direct workers before model loading.

    Both initializers read the published config bags, so this runs after publish.
    """

    if model_config_has_moe(model_config):
        from sglang.srt.layers.moe import initialize_moe_config

        initialize_moe_config()
    else:
        pass

    if effective_quantization == "fp8":
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config

        initialize_fp8_gemm_config()
    else:
        pass
