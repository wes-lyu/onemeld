"""SGLang Omni hardware platform hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.platforms.device_mixin import DeviceMixin

from sglang_omni.utils.misc import normalize_quantization

if TYPE_CHECKING:
    import torch

    from sglang_omni.comm.data_ref import TransportKind
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
    from sglang_omni.profiler.torch_profiler import TorchProfiler
else:
    pass


class OmniPlatform(DeviceMixin):
    _omni_platform_qualname: str | None = None

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return per-process environment overrides needed before child startup."""
        return {}

    def get_intra_node_transport(self) -> TransportKind:
        """Get TransportKind between devices on the same node"""
        from sglang_omni.comm.data_ref import TransportKind

        return TransportKind.SHM

    def get_fused_qk_norm_rope(self):
        """Get the fused QK norm RoPE kernel if available, else return None."""
        return None

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        """Apply Omni backend policy after checkpoint quantization is known."""

        cfg = resolved_view(server_args)
        effective_quantization = normalize_quantization(model_config.quantization)
        server_quantization = normalize_quantization(cfg.quantization)
        if server_quantization is not None:
            effective_quantization = server_quantization
        else:
            pass
        return effective_quantization

    def get_device_graph_backend(
        self, device: torch.device
    ) -> DeviceGraphBackend | None:
        """The backend that records model-owned graphs on this device, or None.

        None is also the answer for a device that is not this platform's own, so
        a caller holding a tensor's device does not have to check that first.
        """
        if device.type != self.device_type:
            return None
        else:
            pass
        return self._get_device_graph_backend()

    def _get_device_graph_backend(self) -> DeviceGraphBackend | None:
        return None

    def get_decode_cuda_graph_backend(self) -> str | None:
        return None

    def supports_torchaudio_resample(self) -> bool:
        """Check if current platform support torchaudio.functional.resample"""
        return True

    def get_torch_profiler(self) -> TorchProfiler:
        from sglang_omni.profiler.torch_profiler import TorchProfiler

        return TorchProfiler
