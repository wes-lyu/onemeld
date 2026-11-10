# SPDX-License-Identifier: Apache-2.0
"""Graph capture for the model-owned graph paths, one backend per accelerator.

A model that captures its own graphs asks its platform for the backend instead of
naming torch.cuda. The graph object and the capture context's keywords differ per
accelerator, and SGLang resolves them explicitly for the same reason, so the
choice belongs on the platform rather than in a per-model branch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Literal, Protocol, TypedDict

import torch
from torch.cuda import _POOL_HANDLE as CudaGraphPoolHandle


class CudaCaptureKwargs(TypedDict, total=False):
    pool: CudaGraphPoolHandle
    stream: torch.cuda.Stream
    capture_error_mode: Literal["thread_local"]


class ReplayableGraph(Protocol):
    def replay(self) -> None: ...

    def reset(self) -> None: ...


DeviceGraphPool = CudaGraphPoolHandle


class DeviceGraphBackend(Protocol):
    """Records a model-owned graph on one accelerator."""

    def capture(
        self,
        *,
        pool: DeviceGraphPool | None = None,
        stream: torch.Stream | None = None,
        thread_local_errors: bool = False,
    ) -> AbstractContextManager[ReplayableGraph]:
        """Open a capture and yield the graph it records into."""
        ...

    def graph_pool_handle(self) -> DeviceGraphPool:
        """A pool handle several captures of this backend can share."""
        ...


class CudaDeviceGraphBackend:
    """CUDA graph capture."""

    def graph_pool_handle(self) -> CudaGraphPoolHandle:
        return torch.cuda.graph_pool_handle()

    @contextmanager
    def capture(
        self,
        *,
        pool: CudaGraphPoolHandle | None = None,
        stream: torch.cuda.Stream | None = None,
        thread_local_errors: bool = False,
    ) -> Iterator[torch.cuda.CUDAGraph]:
        graph = torch.cuda.CUDAGraph()
        kwargs: CudaCaptureKwargs = {}
        if pool is not None:
            kwargs["pool"] = pool
        else:
            pass
        if stream is not None:
            kwargs["stream"] = stream
        else:
            pass
        if thread_local_errors:
            kwargs["capture_error_mode"] = "thread_local"
        else:
            pass
        with torch.cuda.graph(cuda_graph=graph, **kwargs):
            yield graph


__all__ = [
    "CudaDeviceGraphBackend",
    "DeviceGraphBackend",
    "DeviceGraphPool",
    "ReplayableGraph",
]
