# SPDX-License-Identifier: Apache-2.0
"""Start a backend worker process and run steps on it."""

import asyncio
import logging
import os
import shutil
import sys
import tempfile

import zmq
import zmq.asyncio

from onemeld.backends.base import BackendSpec, StepResult
from onemeld.backends.wire import WireMessage, pack, pack_request, unpack
from onemeld.composition.types import StepRequest

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT_S = 30.0
KILL_TIMEOUT_S = 5.0


class WorkerError(RuntimeError):
    """A backend worker failed to start, failed a step, or exited."""


class WorkerClient:
    """The orchestrator's handle on one backend worker process."""

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.directory = tempfile.mkdtemp(prefix=f"onemeld-{spec.name}-")
        self.process: asyncio.subprocess.Process | None = None
        self.pending: dict[str, asyncio.Future[StepResult]] = {}
        self.reader: asyncio.Task[None] | None = None
        self.watcher: asyncio.Task[None] | None = None
        self.ready = False
        self.closing = False
        self.closed = asyncio.Event()

    @property
    def healthy(self) -> bool:
        return (
            self.ready and self.process is not None and self.process.returncode is None
        )

    def worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.spec.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.spec.gpu)
        else:
            pass
        env.update(self.spec.env)
        return env

    async def start(self) -> None:
        endpoint = f"ipc://{self.directory}/worker.sock"
        self.socket.bind(endpoint)
        self.process = await asyncio.create_subprocess_exec(
            self.spec.python or sys.executable,
            "-m",
            "onemeld.backends.worker",
            "--endpoint",
            endpoint,
            "--spec",
            self.spec.model_dump_json(),
            "--parent-pid",
            str(os.getpid()),
            env=self.worker_env(),
        )
        try:
            message = await self.first_message()
        except BaseException:
            await self.terminate()
            raise
        if message["kind"] == "ready":
            self.ready = True
            self.reader = asyncio.create_task(self.read())
            self.watcher = asyncio.create_task(self.watch())
            logger.info(f"Backend {self.spec.name!r} is ready")
        else:
            await self.terminate()
            raise WorkerError(
                f"Backend {self.spec.name!r} failed to start:\n{message.get('error')}"
            )

    async def first_message(self) -> WireMessage:
        """The worker's ready or failed message, unless it exits or times out first."""
        assert self.process is not None
        received = asyncio.ensure_future(self.socket.recv())
        exited = asyncio.ensure_future(self.process.wait())
        try:
            done, _ = await asyncio.wait(
                {received, exited},
                timeout=self.spec.startup_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for waiter in (received, exited):
                if not waiter.done():
                    waiter.cancel()
                else:
                    pass
        if received in done:
            return unpack(received.result())
        elif done:
            raise WorkerError(
                f"Backend {self.spec.name!r} worker exited with code "
                f"{self.process.returncode} before it was ready"
            )
        else:
            raise WorkerError(
                f"Backend {self.spec.name!r} was not ready after "
                f"startup_timeout_s={self.spec.startup_timeout_s}"
            )

    async def read(self) -> None:
        while True:
            message = unpack(await self.socket.recv())
            if message["kind"] == "closed":
                self.closed.set()
                return
            else:
                self.resolve(message)

    def resolve(self, message: WireMessage) -> None:
        future = self.pending.get(str(message.get("id")))
        if message["kind"] != "result":
            logger.warning(f"Backend {self.spec.name!r} sent {message['kind']!r}")
        elif future is None or future.done():
            # A reply to a step the orchestrator already gave up on.
            pass
        elif message.get("ok") is True and isinstance(message.get("result"), dict):
            future.set_result(message["result"])
        else:
            future.set_exception(
                WorkerError(f"Backend {self.spec.name!r}: {message.get('error')}")
            )

    async def watch(self) -> None:
        assert self.process is not None
        code = await self.process.wait()
        self.ready = False
        if not self.closing:
            logger.error(f"Backend {self.spec.name!r} worker exited with code {code}")
        else:
            pass
        for future in self.pending.values():
            if not future.done():
                future.set_exception(
                    WorkerError(
                        f"Backend {self.spec.name!r} worker exited with code {code}"
                    )
                )
            else:
                pass

    async def run(self, request_id: str, request: StepRequest) -> StepResult:
        if not self.healthy:
            raise WorkerError(f"Backend {self.spec.name!r} is not running")
        else:
            pass
        future: asyncio.Future[StepResult] = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self.socket.send(
                pack(
                    {"kind": "run", "id": request_id, "request": pack_request(request)}
                )
            )
            return await future
        except asyncio.CancelledError:
            if self.healthy:
                await self.socket.send(pack({"kind": "cancel", "id": request_id}))
            else:
                pass
            raise
        finally:
            self.pending.pop(request_id, None)

    async def close(self) -> None:
        """Ask the worker to stop, then terminate it if it does not."""
        self.closing = True
        if self.healthy:
            await self.socket.send(pack({"kind": "shutdown"}))
            try:
                await asyncio.wait_for(self.closed.wait(), SHUTDOWN_TIMEOUT_S)
                await asyncio.wait_for(self.process.wait(), KILL_TIMEOUT_S)
            except TimeoutError:
                logger.warning(f"Backend {self.spec.name!r} did not stop in time")
        else:
            pass
        await self.terminate()
        for task in (self.reader, self.watcher):
            if task is not None:
                task.cancel()
            else:
                pass
        await asyncio.gather(
            *(task for task in (self.reader, self.watcher) if task is not None),
            return_exceptions=True,
        )
        self.socket.close(linger=0)
        self.context.term()
        shutil.rmtree(self.directory, ignore_errors=True)

    async def terminate(self) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return
        else:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), KILL_TIMEOUT_S)
        except TimeoutError:
            process.kill()
            await process.wait()
