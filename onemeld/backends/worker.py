# SPDX-License-Identifier: Apache-2.0
"""Host one backend in its own process: python -m onemeld.backends.worker."""

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
import traceback

import zmq
import zmq.asyncio

from onemeld.backends.base import Backend, BackendSpec, load_backend
from onemeld.backends.wire import WireMessage, pack, unpack, unpack_request
from onemeld.composition.types import StepRequest

logger = logging.getLogger(__name__)

# How often a worker checks that the orchestrator that started it lives.
PARENT_POLL_INTERVAL_S = 1.0


class Worker:
    def __init__(self, socket: zmq.asyncio.Socket, backend: Backend) -> None:
        self.socket = socket
        self.backend = backend
        self.running: dict[str, asyncio.Task[None]] = {}

    async def execute(self, request_id: str, request: StepRequest) -> None:
        try:
            result = await self.backend.run(request_id, request)
            reply: WireMessage = {
                "kind": "result",
                "id": request_id,
                "ok": True,
                "result": result,
            }
        except asyncio.CancelledError:
            # The orchestrator cancelled this step and awaits no reply.
            raise
        except Exception as error:
            logger.exception(f"Backend step {request_id} failed")
            reply = {
                "kind": "result",
                "id": request_id,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            self.running.pop(request_id, None)
        await self.socket.send(pack(reply))

    async def handle(self, message: WireMessage) -> bool:
        """Act on one orchestrator message; False once the worker should stop."""
        kind = message["kind"]
        request_id = message.get("id")
        if kind == "shutdown":
            return False
        elif not isinstance(request_id, str):
            raise ValueError(f"Worker message {kind!r} has no request id")
        elif kind == "run":
            self.running[request_id] = asyncio.create_task(
                self.execute(request_id, unpack_request(message.get("request")))
            )
            return True
        elif kind == "cancel":
            task = self.running.get(request_id)
            if task is not None:
                await self.backend.cancel(request_id)
                task.cancel()
            else:
                pass
            return True
        else:
            raise ValueError(f"Unknown worker message kind {kind!r}")

    async def serve(self) -> None:
        """Serve orchestrator messages until it asks for shutdown."""
        while await self.handle(unpack(await self.socket.recv())):
            pass

    async def stop(self) -> None:
        for task in self.running.values():
            task.cancel()
        await asyncio.gather(*self.running.values(), return_exceptions=True)
        await self.backend.close()


async def lifetime(
    socket: zmq.asyncio.Socket, backend: Backend | None, load_error: str | None
) -> int:
    """Start the loaded backend, serve until shutdown, and report each transition."""
    if backend is None:
        await socket.send(pack({"kind": "failed", "error": load_error}))
        return 1
    else:
        pass
    try:
        await backend.start()
    except Exception:
        await socket.send(pack({"kind": "failed", "error": traceback.format_exc()}))
        return 1
    await socket.send(pack({"kind": "ready"}))
    worker = Worker(socket, backend)
    try:
        await worker.serve()
    finally:
        await worker.stop()
    await socket.send(pack({"kind": "closed"}))
    return 0


async def run_worker(
    endpoint: str, backend: Backend | None, load_error: str | None
) -> int:
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(endpoint)
    try:
        return await lifetime(socket, backend, load_error)
    finally:
        socket.close(linger=1000)
        context.term()


def watch_parent(parent_pid: int) -> None:
    """End this process once the orchestrator that started it has exited."""
    while os.getppid() == parent_pid:
        time.sleep(PARENT_POLL_INTERVAL_S)
    logger.error("The orchestrator exited; stopping the worker")
    # note (Wes): a worker can be blocked in a model load, a framework thread or
    # a send to an orchestrator that no longer exists, so nothing short of
    # exiting the process reliably stops it.
    os._exit(1)  # noqa: leading-underscore  # stdlib name; exits without cleanup


def main() -> None:
    parser = argparse.ArgumentParser(prog="onemeld-worker")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--spec", required=True, help="BackendSpec as JSON")
    parser.add_argument("--parent-pid", required=True, type=int)
    options = parser.parse_args()
    spec = BackendSpec.model_validate_json(options.spec)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{spec.name}] %(levelname)s %(name)s: %(message)s",
    )
    # The orchestrator passes its pid because it may exit before this process
    # first reads getppid.
    threading.Thread(
        target=watch_parent, args=(options.parent_pid,), daemon=True
    ).start()
    backend: Backend | None = None
    load_error: str | None = None
    try:
        backend = load_backend(spec)
        backend.load()
    except Exception:
        backend = None
        load_error = traceback.format_exc()
    sys.exit(asyncio.run(run_worker(options.endpoint, backend, load_error)))


if __name__ == "__main__":
    main()
else:
    pass
