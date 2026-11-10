from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.platforms.cuda import CudaDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang_omni.comm.data_ref import TransportKind
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
else:
    pass


class CUDAOmniPlatform(CudaDeviceMixin, OmniPlatform):
    def _get_device_graph_backend(self) -> DeviceGraphBackend:
        from sglang_omni.platforms.device_graph import CudaDeviceGraphBackend

        return CudaDeviceGraphBackend()

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if spec.allow_child_processes and spec.owned_gpu_ids:
            # Native children inherit this, and host threads that touch an
            # unpinned device stay on the stage's own GPUs.
            from sglang_omni.utils.gpu_memory import narrow_visible_devices

            source_env = env if env is not None else os.environ
            return {
                "CUDA_VISIBLE_DEVICES": narrow_visible_devices(
                    source_env.get("CUDA_VISIBLE_DEVICES"),
                    spec.owned_gpu_ids,
                    "CUDA_VISIBLE_DEVICES",
                )
            }
        elif spec.tp_size <= 1:
            return {}
        else:
            pass

        source_env = env if env is not None else os.environ
        original_visible = source_env.get("CUDA_VISIBLE_DEVICES")
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")
        else:
            pass
        if original_visible:
            visible_devices = [item.strip() for item in original_visible.split(",")]
            if spec.gpu_id >= len(visible_devices):
                raise ValueError(
                    f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
                    f"but CUDA_VISIBLE_DEVICES only exposes {visible_devices}"
                )
            else:
                pass
            mapped_gpu = visible_devices[spec.gpu_id]
        else:
            mapped_gpu = str(spec.gpu_id)

        env_updates = {
            "CUDA_VISIBLE_DEVICES": mapped_gpu,
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        }
        # note (ratish): NVLS multicast binding is not available on every host,
        # and NCCL 2.29 fails communicator init instead of falling back. A
        # value from the shell or the stage configuration stands.
        if "NCCL_NVLS_ENABLE" not in source_env and (
            "NCCL_NVLS_ENABLE" not in spec.env_defaults
        ):
            env_updates["NCCL_NVLS_ENABLE"] = "0"
        else:
            pass
        return env_updates

    def get_intra_node_transport(self) -> TransportKind:
        from sglang_omni.comm.data_ref import TransportKind

        return TransportKind.CUDA_IPC

    def get_fused_qk_norm_rope(self):
        from sgl_kernel import fused_qk_norm_rope

        return fused_qk_norm_rope
