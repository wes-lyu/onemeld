# SPDX-License-Identifier: Apache-2.0
"""Distributed forwards and blocked cleanup for scheduler lifecycle tests."""

from __future__ import annotations

import threading
from datetime import timedelta
from multiprocessing.synchronize import Event
from typing import Literal

import torch
import torch.distributed as distributed

from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.tensor_parallel_scheduler import TensorParallelScheduler
from tests.unit_test.fixtures.pipeline_fakes import make_stage_payload


def blocked_failure_scheduler(
    *,
    fail_forward: bool,
    forward_gate: Event,
    cleanup_entered: Event,
    peer_collective_entered: Event,
    backend: Literal["gloo", "nccl"],
    rank: int,
    rendezvous: str,
) -> TensorParallelScheduler:
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    else:
        pass
    distributed.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    cleanup_blocker = threading.Event()

    def compute(payload: StagePayload) -> StagePayload:
        assert forward_gate.wait(60)
        distributed.barrier()
        if fail_forward:
            assert peer_collective_entered.wait(30)
            raise RuntimeError("injected rank failure")
        else:
            value = torch.ones(1, device=device)
            collective = distributed.all_reduce(value, async_op=True)
            peer_collective_entered.set()
            collective.wait()
            return payload

    def cleanup() -> None:
        cleanup_entered.set()
        assert cleanup_blocker.wait(60)

    scheduler = TensorParallelScheduler(
        compute, shutdown_join_timeout_s=0.1, shutdown_callback=cleanup
    )
    scheduler.enqueue(
        IncomingMessage(
            request_id="pending",
            type="new_request",
            data=make_stage_payload(request_id="pending"),
        )
    )
    return scheduler


def drain_collective_worker(
    rank: int,
    backend: Literal["gloo", "nccl"],
    rendezvous: str,
    collective_entered: Event,
    stop_requested: Event,
) -> None:
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    else:
        pass
    distributed.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    completed: list[str] = []

    def compute(payload: StagePayload) -> StagePayload:
        value = torch.tensor([rank + 1.0], device=device)
        collective = distributed.all_reduce(value, async_op=True)
        collective_entered.set()
        collective.wait()
        assert value.item() == 3.0
        completed.append(payload.request_id)
        return payload

    def cleanup() -> None:
        assert completed == ["cancelled", "queued"]
        completed.append("cleanup")
        distributed.destroy_process_group()

    scheduler = TensorParallelScheduler(
        compute, shutdown_join_timeout_s=5.0, shutdown_callback=cleanup
    )
    for request_id in ("cancelled", "queued"):
        scheduler.enqueue(
            IncomingMessage(
                request_id=request_id,
                type="new_request",
                data=make_stage_payload(request_id=request_id),
            )
        )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    if rank == 0:
        thread.start()
    else:
        pass
    assert stop_requested.wait(30)
    scheduler.abort("cancelled")
    scheduler.stop()
    scheduler.stop()
    if rank == 1:
        thread.start()
    else:
        pass
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert completed == ["cancelled", "queued", "cleanup"]
    for request_id in ("cancelled", "queued"):
        result = scheduler.outbox.get(timeout=1)
        assert result.type != "error"
        assert result.request_id == request_id
        assert result.data.request_id == request_id
    assert scheduler.outbox.empty()
