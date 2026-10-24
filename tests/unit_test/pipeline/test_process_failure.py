# SPDX-License-Identifier: Apache-2.0
"""A live worker must not hide a fatal error behind blocked cleanup."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import threading
from functools import partial
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.tensor_parallel_scheduler import TensorParallelScheduler
from tests.unit_test.fixtures.pipeline_fakes import (
    RecordingCoordinatorControlPlane,
    make_stage_payload,
)
from tests.unit_test.pipeline.helpers import make_stage

FACTORY = "tests.unit_test.fixtures.process_failure.blocked_failure_scheduler"


def make_running_runner() -> MultiProcessPipelineRunner:
    runner = MultiProcessPipelineRunner(
        PipelineConfig(
            model_path="unused",
            entry_stage="worker",
            stages=[
                StageConfig(
                    name="worker", process="worker", factory_path=FACTORY, terminal=True
                )
            ],
        )
    )
    coordinator = Coordinator(
        "inproc://complete", "inproc://abort", entry_stage="worker"
    )
    coordinator.control_plane = RecordingCoordinatorControlPlane()
    coordinator.register_stage("worker", "inproc://worker")
    runner._coordinator = coordinator  # noqa: leading-underscore  # Existing runner state has no public setter.
    runner.started = True
    runner.fatal_event = asyncio.Event()
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_rank", [0, 1])
@pytest.mark.parametrize(
    "backend", ["gloo", pytest.param("nccl", marks=pytest.mark.accelerator)]
)
async def test_live_failed_worker_triggers_recovery_of_all_processes(
    monkeypatch: pytest.MonkeyPatch,
    failed_rank: int,
    tmp_path: Path,
    backend: Literal["gloo", "nccl"],
) -> None:
    """A worker stuck in cleanup used to remain invisible to death-only monitoring."""
    if backend == "nccl" and torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    else:
        pass
    context = multiprocessing.get_context("spawn")
    forward_gate = context.Event()
    cleanup_entered = context.Event()
    peer_collective_entered = context.Event()
    specs = [
        StageWorkerProcessSpec(
            process_name=f"worker-{rank}",
            stage_specs=[
                StageLaunchConfig(
                    stage_name=f"worker-{rank}",
                    role="follower",
                    factory=FACTORY,
                    factory_kwargs={
                        "fail_forward": rank == failed_rank,
                        "forward_gate": forward_gate,
                        "cleanup_entered": cleanup_entered,
                        "peer_collective_entered": peer_collective_entered,
                        "backend": backend,
                        "rank": rank,
                        "rendezvous": (tmp_path / "rendezvous").as_uri(),
                    },
                    internal_work_queue=context.Queue(),
                    internal_abort_queue=context.Queue(),
                    is_terminal=True,
                )
            ],
        )
        for rank in range(2)
    ]
    group = StageGroup("workers", specs)
    runner = make_running_runner()
    runner.groups = [group]
    coordinator = runner.coordinator
    await coordinator.submit_request("pending", "input")
    result = coordinator.completion_futures["pending"]
    shutdown = AsyncMock(wraps=partial(group.shutdown, join_timeout=0.05))
    monkeypatch.setattr(group, "shutdown", shutdown)
    processes = []
    try:
        group.spawn(context)
        processes = group.processes
        await asyncio.wait_for(group.wait_ready(timeout=30), timeout=40)
        assert group.failure_summary() == ""
        forward_gate.set()
        assert await asyncio.to_thread(cleanup_entered.wait, 10)
        assert peer_collective_entered.is_set()
        assert group.failure_events[failed_rank].is_set()
        assert all(process.is_alive() for process in processes)
        await asyncio.wait_for(runner.monitor_children(), timeout=15)
        with pytest.raises(RuntimeError, match=f"worker-{failed_rank}"):
            await result
        with pytest.raises(RuntimeError, match="Stage process failure reported"):
            await runner.wait_failed()
        with pytest.raises(RuntimeError, match="Stage process failure reported"):
            await coordinator.submit_request("late", "input")
        assert all(
            not process.is_alive() and process.exitcode is not None
            for process in processes
        )
        assert shutdown.await_count == 1
        await runner.stop()
        assert shutdown.await_count == 1
    finally:
        forward_gate.set()
        await runner.stop()
        for process in processes:
            if process.is_alive():
                process.kill()
            else:
                pass
            process.join(timeout=5)
            process.close()
        for spec in specs:
            for launch in spec.stage_specs:
                launch.internal_work_queue.close()
                launch.internal_work_queue.join_thread()
                launch.internal_abort_queue.close()
                launch.internal_abort_queue.join_thread()
        if result.done() and not result.cancelled():
            result.exception()
        else:
            result.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("dead", [False, True])
async def test_monitor_keeps_healthy_workers_and_detects_unreported_death(
    dead: bool,
) -> None:
    runner = make_running_runner()
    group = Mock(spec=StageGroup)
    group.failure_summary.return_value = ""
    group.any_dead.return_value = dead
    group.dead_summary.return_value = "worker (exit=1)"
    runner.groups = [group]
    failure = AsyncMock()
    runner.fail_runtime = failure
    monitor = asyncio.create_task(runner.monitor_children())
    await asyncio.sleep(0)
    if dead:
        await monitor
        assert failure.await_count == 1
        assert "Dead stage process" in str(failure.await_args.args[0])
    else:
        assert not monitor.done()
        failure.assert_not_awaited()
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
    runner.started = False


@pytest.mark.asyncio
async def test_receive_failure_is_reported_before_blocked_stage_stop() -> None:
    """A receive failure used to stay invisible while Stage.stop was blocked."""
    failure_event = threading.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    stage = make_stage(process_failure_callback=failure_event.set)
    stage.start = AsyncMock()
    stage.running = True
    stage.control_plane.recv = AsyncMock(
        side_effect=RuntimeError("control receive failed")
    )

    async def stop() -> None:
        cleanup_entered.set()
        await release_cleanup.wait()

    stage.stop = stop
    running = asyncio.create_task(stage.run())
    try:
        await asyncio.wait_for(cleanup_entered.wait(), timeout=2)
        assert not running.done()
        assert failure_event.is_set()
    finally:
        release_cleanup.set()
        with pytest.raises(RuntimeError, match="control receive failed"):
            await asyncio.wait_for(running, timeout=2)


@pytest.mark.parametrize("partially_started", [False, True])
def test_process_failure_is_reported_before_outer_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    partially_started: bool,
) -> None:
    """Partial startup used to enter Stage.stop before notifying the parent."""
    failure_event = multiprocessing.get_context("spawn").Event()
    startup_error_channel = multiprocessing.get_context("spawn").Queue()
    spec = StageWorkerProcessSpec(
        "worker", [StageLaunchConfig("worker")], failure_event=failure_event
    )
    constructed = Mock(spec=Stage)
    constructed.name = "worker"
    constructed.start = AsyncMock(side_effect=RuntimeError("startup failure"))
    constructed.running = partially_started

    async def stop() -> None:
        assert failure_event.is_set()
        constructed.running = False

    constructed.stop = AsyncMock(side_effect=stop)
    monkeypatch.setattr(
        stage_workers, "construct_stage", Mock(return_value=constructed)
    )
    cleanup_called = []

    def cleanup(stages: list[Stage], log: logging.Logger, *, reason: str) -> None:
        assert failure_event.is_set()
        assert "startup failure" in startup_error_channel.get(timeout=1)
        assert stages == [constructed]
        cleanup_called.append(reason)

    monkeypatch.setattr(stage_workers, "cleanup_constructed_stages", cleanup)
    try:
        with pytest.raises(RuntimeError, match="startup failure"):
            stage_workers.run_process(
                spec,
                multiprocessing.get_context("spawn").Event(),
                logging.getLogger(__name__),
                startup_error_channel,
            )
    finally:
        startup_error_channel.close()
        startup_error_channel.join_thread()
    assert cleanup_called == ["stage process worker failure"]
    assert constructed.stop.await_count == int(partially_started)


@pytest.mark.asyncio
async def test_ready_worker_failure_during_startup_does_not_wait_for_death() -> None:
    group = StageGroup(
        "worker", [StageWorkerProcessSpec("worker", [StageLaunchConfig("worker")])]
    )
    context = multiprocessing.get_context("spawn")
    failure_event = context.Event()
    failure_event.set()
    group.failure_events = [failure_event]
    group.ready_events = [context.Event()]
    group.ready_events[0].set()
    group.startup_error_channels = [context.Queue()]
    process = Mock()
    process.pid = 123
    process.is_alive.return_value = True
    group._processes = [
        process
    ]  # noqa: leading-underscore  # Existing worker lifecycle state has no public setter.
    try:
        with pytest.raises(RuntimeError, match="failure during startup: worker"):
            await group.wait_ready(timeout=1)
    finally:
        group.close_control_channels()


@pytest.mark.asyncio
@pytest.mark.parametrize("request_id", ["invalid", "cancelled", "valid"])
async def test_request_outcomes_and_normal_stop_do_not_report_process_failure(
    request_id: str,
) -> None:
    failure_event = threading.Event()
    cleaned = threading.Event()

    def validate(payload: StagePayload) -> None:
        if payload.request_id == "invalid":
            raise ValueError("invalid input")
        else:
            pass

    def compute(payload: StagePayload) -> StagePayload:
        return payload

    scheduler = TensorParallelScheduler(
        compute,
        shutdown_join_timeout_s=1,
        validate_request=validate,
        shutdown_callback=cleaned.set,
    )
    stage = make_stage(
        role="leader", scheduler=scheduler, process_failure_callback=failure_event.set
    )
    stage.send_failure = AsyncMock()
    await stage.start()
    try:
        if request_id == "cancelled":
            stage.on_abort(request_id)
        else:
            pass
        await stage.execute(make_stage_payload(request_id=request_id))
        if request_id == "valid":
            result = await asyncio.to_thread(scheduler.outbox.get, True, 2)
            assert result.request_id == request_id
            assert result.type == "result"
        elif request_id == "invalid":
            stage.send_failure.assert_awaited_once_with(request_id, "invalid input")
            assert scheduler.inbox.empty()
        else:
            stage.send_failure.assert_not_awaited()
            assert scheduler.inbox.empty()
    finally:
        await stage.stop()
    assert cleaned.is_set()
    assert not failure_event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["leader", "follower"])
async def test_stage_notifies_parent_before_blocked_scheduler_cleanup(
    role: Literal["leader", "follower"]
) -> None:
    failure_event = threading.Event()
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()

    def compute(payload: StagePayload) -> StagePayload:
        raise RuntimeError("rank-local failure")

    def cleanup() -> None:
        cleanup_entered.set()
        assert release_cleanup.wait(5)

    scheduler = TensorParallelScheduler(
        compute, shutdown_join_timeout_s=1, shutdown_callback=cleanup
    )
    stage = make_stage(
        role=role, scheduler=scheduler, process_failure_callback=failure_event.set
    )
    await stage.start()
    try:
        await stage.execute(make_stage_payload())
        assert await asyncio.to_thread(cleanup_entered.wait, 2)
        assert failure_event.is_set()
        assert stage.scheduler_thread.is_alive()
    finally:
        release_cleanup.set()
        await stage.stop()
    assert stage.scheduler_thread is None
