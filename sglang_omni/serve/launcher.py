# SPDX-License-Identifier: Apache-2.0
"""Launch OneMeld with the shared native worker runtime."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import threading
from collections.abc import Generator
from contextlib import contextmanager
from types import FrameType

import uvicorn

from sglang_omni.client.client import Client
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.serve.openai_api import create_app
from sglang_omni.utils.gpu_compat import apply_gpu_compat_env_defaults

logger = logging.getLogger(__name__)
HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class PipelineUvicornServer(uvicorn.Server):
    """Let the launcher clean up native workers before the process exits."""

    @contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        else:
            pass
        previous_handlers = {
            signum: signal.signal(signum, self.handle_exit)
            for signum in HANDLED_SIGNALS
        }
        try:
            yield
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            self._captured_signals.clear()  # noqa: leading-underscore


def find_available_port(host: str, port: int) -> int:
    """Return *port* if available, otherwise find a free port and warn.

    SGLANG_OMNI_STRICT_PORT=1 requires the requested port to be available.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # note (Richard Wang): probe the way the server binds. Uvicorn sets
            # SO_REUSEADDR, so a port whose last connections are still in
            # TIME_WAIT after a restart is free for it, while a live listener
            # still fails this bind.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return port
    except OSError as exc:
        if (os.environ.get("SGLANG_OMNI_STRICT_PORT") or "").strip() == "1":
            raise RuntimeError(
                f"port {port} is already in use on {host} and "
                "SGLANG_OMNI_STRICT_PORT=1 forbids falling back"
            ) from exc
        else:
            pass
    logger.warning(f"Port {port} is already in use on {host}.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        free_port = s.getsockname()[1]
    logger.warning(f"Using port {free_port} instead.")
    return free_port


async def run_server(
    pipeline_config: PipelineConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str | None = None,
    log_level: str = "info",
) -> None:
    port = find_available_port(host, port)
    runner = MultiProcessPipelineRunner(pipeline_config)
    await runner.start(
        timeout=float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "600"))
    )
    try:
        app = create_app(
            Client(runner.coordinator), model_name=model_name or pipeline_config.name
        )
        server = PipelineUvicornServer(
            uvicorn.Config(app, host=host, port=port, log_level=log_level)
        )
        server_task = asyncio.create_task(server.serve())
        failure_task = asyncio.create_task(runner.wait_failed())
        try:
            done, _ = await asyncio.wait(
                {server_task, failure_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if failure_task in done:
                server.should_exit = True
                await server_task
                await failure_task
                raise RuntimeError("Pipeline failure monitor exited unexpectedly")
            else:
                await server_task
        finally:
            for task in (server_task, failure_task):
                if not task.done():
                    task.cancel()
                else:
                    pass
            await asyncio.gather(server_task, failure_task, return_exceptions=True)
    finally:
        await runner.stop()


def launch_server(
    pipeline_config: PipelineConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_name: str | None = None,
    log_level: str = "info",
) -> None:
    """Run the server and stop its workers on SIGTERM."""
    apply_gpu_compat_env_defaults()
    sigterm_received = False
    cleanup_failed = False

    async def run_server_with_sigterm_cleanup() -> None:
        """Let SIGTERM cancel run_server while HTTP serving is not running.

        PipelineUvicornServer replaces this handler while it serves. Once runner
        cleanup succeeds, launch_server raises SIGTERM again for the previous handler.
        """
        nonlocal cleanup_failed
        server_run = run_server(
            pipeline_config,
            host=host,
            port=port,
            model_name=model_name,
            log_level=log_level,
        )
        previous_handler = signal.getsignal(signal.SIGTERM)
        if (
            threading.current_thread() is not threading.main_thread()
            or previous_handler in (None, signal.SIG_IGN)
        ):
            await server_run
            return
        else:
            pass
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()

        def cancel_main_task() -> None:
            nonlocal sigterm_received
            logger.warning("Received SIGTERM, stopping the pipeline")
            if not sigterm_received:
                sigterm_received = True
                main_task.cancel()
            else:
                pass

        def handle_sigterm(signal_number: int, frame: FrameType | None) -> None:
            loop.call_soon_threadsafe(cancel_main_task)

        signal.signal(signal.SIGTERM, handle_sigterm)
        try:
            await server_run
        except asyncio.CancelledError as cancelled:
            # note (Richard Wang): Python 3.10 drops the cause outside asyncio.run
            cleanup_failed = cancelled.__cause__ is not None
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)

    try:
        asyncio.run(run_server_with_sigterm_cleanup())
    except asyncio.CancelledError:
        if sigterm_received and not cleanup_failed:
            signal.raise_signal(signal.SIGTERM)
        else:
            pass
        raise
