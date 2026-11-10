# SPDX-License-Identifier: Apache-2.0
"""Host one backend in its own process: python -m onemeld.backends.worker."""

import argparse
import asyncio
import logging
import os
import sys
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


async def lifetime(socket: zmq.asyncio.Socket, spec: BackendSpec) -> int:
    """Start the backend, serve until shutdown, and report each transition."""
    try:
        backend = load_backend(spec)
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


async def orphaned(parent_pid: int) -> None:
    """Return once the orchestrator that started this worker has exited."""
    while os.getppid() == parent_pid:
        await asyncio.sleep(PARENT_POLL_INTERVAL_S)


async def run_worker(endpoint: str, spec: BackendSpec, parent_pid: int) -> int:
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(endpoint)
    # note (Wes): watch the parent for the whole lifetime, since a worker can block
    # in a model load or a send to an orchestrator that no longer exists. The parent
    # passes its pid because it may exit before this process first reads getppid.
    served = asyncio.create_task(lifetime(socket, spec))
    watchdog = asyncio.create_task(orphaned(parent_pid))
    try:
        await asyncio.wait({served, watchdog}, return_when=asyncio.FIRST_COMPLETED)
        if served.done():
            return served.result()
        else:
            logger.error("The orchestrator exited; stopping the worker")
            served.cancel()
            await asyncio.gather(served, return_exceptions=True)
            return 1
    finally:
        watchdog.cancel()
        socket.close(linger=1000)
        context.term()


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
    sys.exit(asyncio.run(run_worker(options.endpoint, spec, options.parent_pid)))


if __name__ == "__main__":
    main()
