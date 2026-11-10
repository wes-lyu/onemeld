# SPDX-License-Identifier: Apache-2.0
"""A real worker subprocess: startup, steps, cancellation, failures and shutdown."""

import asyncio
import os
import signal
import sys

import pytest
import pytest_asyncio

from onemeld.backends.base import BackendSpec
from onemeld.backends.client import WorkerClient, WorkerError
from onemeld.composition.types import StepRequest
from tests.unit_test.fixtures.echo_backend import CRASH_EXIT_CODE

ECHO = "tests.unit_test.fixtures.echo_backend:EchoBackend"
# Half-second polls for an orphaned worker, several times its one-second parent check.
ORPHAN_EXIT_POLLS = 20


def spec_of(adapter: str = ECHO, **fields: object) -> BackendSpec:
    return BackendSpec.model_validate(
        {
            "name": "echo",
            "framework": "test",
            "task": "chat",
            "model": "echo-model",
            "adapter": adapter,
            "startup_timeout_s": 60.0,
            **fields,
        }
    )


@pytest_asyncio.fixture
async def worker() -> WorkerClient:
    client = WorkerClient(spec_of(gpu=5, env={"ONEMELD_TEST": "1"}))
    await client.start()
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_steps_round_trip_text_and_bytes(worker: WorkerClient) -> None:
    text, audio = await asyncio.gather(
        worker.run("a", StepRequest(inputs="hello")),
        worker.run("b", StepRequest(inputs=b"raw audio")),
    )
    assert text == {"text": "hello", "model": "echo-model"}
    assert audio["text"] == "raw audio"


@pytest.mark.asyncio
async def test_worker_sees_its_gpu(worker: WorkerClient) -> None:
    report = await worker.run("r", StepRequest(inputs="", params={"action": "report"}))
    assert report["gpu"] == "5"


@pytest.mark.asyncio
async def test_cancelled_step_reaches_the_backend(worker: WorkerClient) -> None:
    waiting = asyncio.create_task(
        worker.run("slow", StepRequest(inputs="", params={"action": "wait"}))
    )
    await asyncio.sleep(0.5)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    report = await worker.run("r", StepRequest(inputs="", params={"action": "report"}))
    assert report["cancelled"] == ["slow"]
    assert worker.healthy


@pytest.mark.asyncio
async def test_step_failure_is_reported_and_the_worker_keeps_serving(
    worker: WorkerClient,
) -> None:
    with pytest.raises(WorkerError, match="ValueError: scripted failure"):
        await worker.run("f", StepRequest(inputs="", params={"action": "fail"}))
    assert (await worker.run("ok", StepRequest(inputs="again")))["text"] == "again"


@pytest.mark.asyncio
async def test_worker_crash_fails_in_flight_steps(worker: WorkerClient) -> None:
    waiting = asyncio.create_task(
        worker.run("slow", StepRequest(inputs="", params={"action": "wait"}))
    )
    await asyncio.sleep(0.5)
    with pytest.raises(WorkerError, match=f"exited with code {CRASH_EXIT_CODE}"):
        await worker.run("x", StepRequest(inputs="", params={"action": "exit"}))
    with pytest.raises(WorkerError, match=f"exited with code {CRASH_EXIT_CODE}"):
        await waiting
    assert not worker.healthy
    with pytest.raises(WorkerError, match="not running"):
        await worker.run("after", StepRequest(inputs="late"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend, message",
    [
        ("FailingLoadBackend", "scripted load failure"),
        ("FailingBackend", "scripted startup failure"),
    ],
)
async def test_startup_failure_carries_the_worker_traceback(
    backend: str, message: str
) -> None:
    client = WorkerClient(spec_of(f"tests.unit_test.fixtures.echo_backend:{backend}"))
    with pytest.raises(WorkerError, match=message):
        await client.start()
    assert not client.healthy
    await client.close()


@pytest.mark.asyncio
async def test_unknown_framework_names_the_known_adapters() -> None:
    spec = spec_of(adapter=None, framework="vllm")
    with pytest.raises(ValueError, match="'vllm' has no chat adapter"):
        spec.adapter_path


ORPHANING_PARENT = f"""
import asyncio, os
from onemeld.backends.base import BackendSpec
from onemeld.backends.client import WorkerClient

async def main():
    spec = {spec_of(args={"block_load": True}).model_dump_json()!r}
    client = WorkerClient(BackendSpec.model_validate_json(spec))
    starting = asyncio.create_task(client.start())
    while client.process is None:
        await asyncio.sleep(0.05)
    print(client.process.pid, flush=True)
    os._exit(0)

asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_worker_stops_when_its_orchestrator_dies() -> None:
    """A worker still loading its model when the orchestrator exits must stop itself."""
    parent = await asyncio.create_subprocess_exec(
        sys.executable, "-c", ORPHANING_PARENT, stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await parent.communicate()
    worker_pid = int(stdout.decode().strip())
    for _ in range(ORPHAN_EXIT_POLLS):
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.5)
    os.kill(worker_pid, signal.SIGKILL)
    raise AssertionError(f"orphaned worker {worker_pid} kept running")


@pytest.mark.asyncio
async def test_close_stops_the_worker_process(worker: WorkerClient) -> None:
    process = worker.process
    assert process is not None
    await worker.close()
    assert process.returncode == 0
    assert not worker.healthy
